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


def _normalise_frequency(freq: str | None) -> str:
    """Convert deprecated pandas offset aliases to their modern form."""
    if not freq:
        return "1h"
    freq_str = str(freq).strip()
    if not freq_str:
        return "1h"
    suffix_map = {"t": "min", "T": "min", "H": "h"}
    last_char = freq_str[-1]
    mapped = suffix_map.get(last_char)
    if mapped:
        prefix = freq_str[:-1]
        if prefix and prefix[-1].isdigit():
            return f"{prefix}{mapped}"
        return mapped
    return freq_str


def load_clean_tables(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(config["paths"]["processed_dir"])
    agzu_path = processed_dir / "agzu_clean.parquet"
    su_path = processed_dir / "su_clean.parquet"
    if not agzu_path.exists() or not su_path.exists():
        raise FileNotFoundError("Cleaned parquet tables not found. Run `pipeline.clean` first.")
    return pd.read_parquet(agzu_path), pd.read_parquet(su_path)


def _hourly_interpolate(
    df: pd.DataFrame,
    datetime_col: str,
    well_col: str,
    freq: str,
) -> pd.DataFrame:
    """Build hourly series per well, fill temporal gaps via interpolation, preserve original NaNs."""
    value_cols = [col for col in df.columns if col not in {datetime_col, well_col}]
    results: List[pd.DataFrame] = []

    for well, part in df.groupby(well_col):
        if part.empty:
            continue

        ordered = part.sort_values(datetime_col)
        indexed = ordered.set_index(datetime_col)[value_cols]
        if indexed.empty:
            continue

        aggregated = indexed.resample(freq).agg(["mean", "size"])
        hourly_mean = aggregated.xs("mean", axis=1, level=1, drop_level=True)
        hourly_counts = aggregated.xs("size", axis=1, level=1, drop_level=True)
        if isinstance(hourly_mean, pd.Series):
            hourly_mean = hourly_mean.to_frame(name=hourly_mean.name)
        if isinstance(hourly_counts, pd.Series):
            hourly_counts = hourly_counts.to_frame(name=hourly_counts.name)
        interpolated = hourly_mean.interpolate(method="time")
        # Restore NaNs that were present in original data (hour had rows but all were NaN)
        original_nan_mask = (hourly_counts > 0) & hourly_mean.isna()
        interpolated[original_nan_mask] = np.nan

        interpolated = interpolated.reset_index()
        interpolated[well_col] = well
        results.append(interpolated)

    columns = [datetime_col] + value_cols + [well_col]
    return pd.concat(results, ignore_index=True)[columns] if results else pd.DataFrame(columns=columns)


def resample_agzu(agzu: pd.DataFrame, config: Dict) -> pd.DataFrame:
    agzu_conf = config["agzu"]
    alignment = config.get("alignment", {})
    freq = _normalise_frequency(alignment.get("frequency", "1H"))
    return _hourly_interpolate(agzu, agzu_conf["datetime_column"], agzu_conf["well_column"], freq)


def resample_su(su: pd.DataFrame, config: Dict) -> pd.DataFrame:
    su_conf = config["su"]
    alignment = config.get("alignment", {})
    freq = _normalise_frequency(alignment.get("frequency", "1H"))
    return _hourly_interpolate(su, su_conf["datetime_column"], su_conf["well_column"], freq)


def _align_time_ranges(
    agzu_df: pd.DataFrame,
    su_df: pd.DataFrame,
    config: Dict,
    freq: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agzu_conf = config["agzu"]
    su_conf = config["su"]
    datetime_agzu = agzu_conf["datetime_column"]
    datetime_su = su_conf["datetime_column"]
    well_col = agzu_conf["well_column"]

    agzu_values = [col for col in agzu_df.columns if col not in {datetime_agzu, well_col}]
    su_values = [col for col in su_df.columns if col not in {datetime_su, well_col}]

    wells = sorted(set(agzu_df[well_col].unique()).union(su_df[well_col].unique()))
    aligned_agzu: List[pd.DataFrame] = []
    aligned_su: List[pd.DataFrame] = []

    for well in wells:
        agzu_part = agzu_df[agzu_df[well_col] == well]
        su_part = su_df[su_df[well_col] == well]

        agzu_indexed = (
            agzu_part.sort_values(datetime_agzu).set_index(datetime_agzu)[agzu_values]
            if not agzu_part.empty
            else pd.DataFrame(columns=agzu_values)
        )
        su_indexed = (
            su_part.sort_values(datetime_su).set_index(datetime_su)[su_values]
            if not su_part.empty
            else pd.DataFrame(columns=su_values)
        )

        candidates_start = []
        candidates_end = []
        if not agzu_indexed.empty:
            candidates_start.append(agzu_indexed.index.min())
            candidates_end.append(agzu_indexed.index.max())
        if not su_indexed.empty:
            candidates_start.append(su_indexed.index.min())
            candidates_end.append(su_indexed.index.max())
        if not candidates_start:
            continue

        start = min(candidates_start)
        end = max(candidates_end)
        full_index = pd.date_range(start=start, end=end, freq=freq)

        agzu_reindexed = (
            agzu_indexed.reindex(full_index)
            if not agzu_indexed.empty
            else pd.DataFrame(index=full_index, columns=agzu_values, dtype=float)
        )
        su_reindexed = (
            su_indexed.reindex(full_index)
            if not su_indexed.empty
            else pd.DataFrame(index=full_index, columns=su_values, dtype=float)
        )

        agzu_ready = agzu_reindexed.reset_index().rename(columns={"index": datetime_agzu})
        su_ready = su_reindexed.reset_index().rename(columns={"index": datetime_su})
        agzu_ready[well_col] = well
        su_ready[well_col] = well

        agzu_cols = [datetime_agzu, well_col] + agzu_values
        su_cols = [datetime_su, well_col] + su_values

        aligned_agzu.append(agzu_ready[agzu_cols])
        aligned_su.append(su_ready[su_cols])

    agzu_aligned = pd.concat(aligned_agzu, ignore_index=True) if aligned_agzu else agzu_df.copy()
    su_aligned = pd.concat(aligned_su, ignore_index=True) if aligned_su else su_df.copy()
    return agzu_aligned, su_aligned


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
    alignment = config.get("alignment", {})
    freq = _normalise_frequency(alignment.get("frequency", "1H"))

    agzu_resampled = resample_agzu(agzu_clean, config)
    su_resampled = resample_su(su_clean, config)
    agzu_aligned, su_aligned = _align_time_ranges(agzu_resampled, su_resampled, config, freq)
    merged = merge_datasets(agzu_aligned, su_aligned, config)
    save_aligned_tables(agzu_aligned, su_aligned, merged, config)
    return {
        "agzu_resampled_rows": len(agzu_aligned),
        "su_resampled_rows": len(su_aligned),
        "merged_rows": len(merged),
    }


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample and align cleaned datasets to a common timeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to pipeline configuration.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or [])
    config = load_config(args.config)
    alignment = config.get("alignment", {})
    freq_raw = alignment.get("frequency", "1H")
    freq_display = freq_raw.replace('T', 'min').replace('t', 'min')
    stats = run_alignment(args.config)
    print(f"Resampled AGZU rows: {stats['agzu_resampled_rows']} (freq={freq_display})")
    print(f"Resampled SU rows: {stats['su_resampled_rows']} (freq={freq_display})")
    print(f"Merged rows: {stats['merged_rows']} (freq={freq_display})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
