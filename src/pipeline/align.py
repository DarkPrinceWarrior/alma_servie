"""
Align cleaned AGZU and SU datasets on a common time grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG_PATH, load_config


def load_clean_tables(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(config["paths"]["processed_dir"])
    agzu_path = processed_dir / "agzu_clean.parquet"
    su_path = processed_dir / "su_clean.parquet"
    if not agzu_path.exists() or not su_path.exists():
        raise FileNotFoundError("Cleaned parquet tables not found. Run `pipeline.clean` first.")
    return pd.read_parquet(agzu_path), pd.read_parquet(su_path)


def _resample_with_limit(
    df: pd.DataFrame,
    datetime_col: str,
    well_col: str,
    freq: str,
    max_hours: float,
) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    value_cols = [col for col in df.columns if col not in {datetime_col, well_col}]
    for well, part in df.groupby(well_col):
        single = (
            part.sort_values(datetime_col)
            .set_index(datetime_col)[value_cols]
        )
        resampled = single.resample(freq).ffill()
        last_valid = single.index.to_series().reindex(resampled.index, method="ffill")
        hours_since = (resampled.index.to_series() - last_valid).dt.total_seconds() / 3600.0
        mask = hours_since > max_hours
        resampled.loc[mask, value_cols] = np.nan
        resampled = resampled.reset_index().assign(**{well_col: well})
        dfs.append(resampled)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=[datetime_col, well_col] + value_cols)


def resample_agzu(agzu: pd.DataFrame, config: Dict) -> pd.DataFrame:
    agzu_conf = config["agzu"]
    alignment = config.get("alignment", {})
    freq = alignment.get("frequency", "1H").lower()
    limit = alignment.get("agzu_max_ffill_hours", 24)
    return _resample_with_limit(agzu, agzu_conf["datetime_column"], agzu_conf["well_column"], freq, limit)


def resample_su(su: pd.DataFrame, config: Dict) -> pd.DataFrame:
    su_conf = config["su"]
    alignment = config.get("alignment", {})
    freq = alignment.get("frequency", "1H").lower()
    limit = alignment.get("su_max_ffill_hours", 6)

    # Average within resample window before limiting propagation
    dfs: List[pd.DataFrame] = []
    value_cols = [col for col in su.columns if col not in {su_conf["datetime_column"], su_conf["well_column"]}]
    for well, part in su.groupby(su_conf["well_column"]):
        single = (
            part.sort_values(su_conf["datetime_column"])
            .set_index(su_conf["datetime_column"])[value_cols]
        )
        resampled = single.resample(freq).mean()
        resampled = resampled.reset_index()
        resampled[su_conf["well_column"]] = well
        dfs.append(resampled)
    averaged = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=[su_conf["datetime_column"], su_conf["well_column"]] + value_cols)

    return _resample_with_limit(averaged, su_conf["datetime_column"], su_conf["well_column"], freq, limit)


def merge_datasets(agzu_resampled: pd.DataFrame, su_resampled: pd.DataFrame, config: Dict) -> pd.DataFrame:
    agzu_conf = config["agzu"]
    su_conf = config["su"]
    datetime_col = "timestamp"

    agzu_tmp = agzu_resampled.rename(columns={agzu_conf["datetime_column"]: datetime_col})
    su_tmp = su_resampled.rename(columns={su_conf["datetime_column"]: datetime_col})

    merged = pd.merge(
        su_tmp,
        agzu_tmp,
        on=[su_conf["well_column"], datetime_col],
        how="outer",
        suffixes=("_su", "_agzu"),
    )
    merged = merged.sort_values([su_conf["well_column"], datetime_col]).reset_index(drop=True)
    return merged


def save_aligned_tables(agzu_resampled: pd.DataFrame, su_resampled: pd.DataFrame, merged: pd.DataFrame, config: Dict) -> None:
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    agzu_resampled.to_parquet(processed_dir / "agzu_resampled.parquet", index=False)
    su_resampled.to_parquet(processed_dir / "su_resampled.parquet", index=False)
    merged.to_parquet(processed_dir / "merged_hourly.parquet", index=False)


def run_alignment(config_path: Path) -> Dict[str, int]:
    config = load_config(config_path)
    agzu_clean, su_clean = load_clean_tables(config)
    agzu_resampled = resample_agzu(agzu_clean, config)
    su_resampled = resample_su(su_clean, config)
    merged = merge_datasets(agzu_resampled, su_resampled, config)
    save_aligned_tables(agzu_resampled, su_resampled, merged, config)
    return {
        "agzu_resampled_rows": len(agzu_resampled),
        "su_resampled_rows": len(su_resampled),
        "merged_rows": len(merged),
    }


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample and align cleaned datasets to a common timeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to pipeline configuration.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or [])
    stats = run_alignment(args.config)
    print(f"Resampled AGZU rows: {stats['agzu_resampled_rows']}")
    print(f"Resampled SU rows: {stats['su_resampled_rows']}")
    print(f"Merged rows: {stats['merged_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
