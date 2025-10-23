"""
Data cleaning routines for AGZU and SU datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG_PATH, load_config


def load_raw_tables(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    interim_dir = Path(config["paths"]["interim_dir"])
    agzu_path = interim_dir / "agzu_raw.parquet"
    su_path = interim_dir / "su_raw.parquet"

    if not agzu_path.exists() or not su_path.exists():
        raise FileNotFoundError("Raw parquet tables not found. Run `pipeline.extract` first.")

    agzu_df = pd.read_parquet(agzu_path)
    su_df = pd.read_parquet(su_path)
    return agzu_df, su_df


def clean_agzu(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    datetime_col = config["agzu"]["datetime_column"]
    well_col = config["agzu"]["well_column"]

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.sort_values([well_col, datetime_col]).reset_index(drop=True)

    summary: Dict[str, Dict[str, float]] = {}

    for col in config["agzu"].get("negative_sensitive_columns", []):
        if col in df.columns:
            mask = pd.to_numeric(df[col], errors="coerce") < 0
            summary[f"{col}_negatives"] = {"count": int(mask.sum())}
            df.loc[mask, col] = np.nan

    for col, threshold in config["agzu"].get("extreme_thresholds", {}).items():
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            mask = values > threshold
            summary[f"{col}_above_{threshold}"] = {"count": int(mask.sum())}
            df.loc[mask, col] = np.nan

    return df, summary


def clean_su(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    datetime_col = config["su"]["datetime_column"]
    well_col = config["su"]["well_column"]

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.sort_values([well_col, datetime_col]).reset_index(drop=True)

    fill_cols = [col for col in df.columns if col not in {datetime_col, well_col}]
    summary: Dict[str, Dict[str, float]] = {}

    for col in fill_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        df[col] = numeric
        summary[col] = {
            "nan_before": int(numeric.isna().sum()),
            "zero_ratio": float((numeric == 0).mean()),
        }

    for method in config["su"].get("fill_method", []):
        if method == "ffill":
            df[fill_cols] = df.groupby(well_col)[fill_cols].ffill()
        elif method == "bfill":
            df[fill_cols] = df.groupby(well_col)[fill_cols].bfill()

    for col in fill_cols:
        summary[col]["nan_after"] = int(df[col].isna().sum())

    return df, summary


def save_clean_tables(agzu_df: pd.DataFrame, su_df: pd.DataFrame, config: Dict) -> None:
    processed_dir = Path(config["paths"]["processed_dir"])
    save_path_agzu = processed_dir / "agzu_clean.parquet"
    save_path_su = processed_dir / "su_clean.parquet"
    processed_dir.mkdir(parents=True, exist_ok=True)
    agzu_df.to_parquet(save_path_agzu, index=False)
    su_df.to_parquet(save_path_su, index=False)


def write_summary(summary: Dict, config: Dict) -> None:
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "cleaning_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_cleaning(config_path: Path) -> Dict:
    config = load_config(config_path)
    agzu_raw, su_raw = load_raw_tables(config)

    agzu_clean, summary_agzu = clean_agzu(agzu_raw, config)
    su_clean, summary_su = clean_su(su_raw, config)

    save_clean_tables(agzu_clean, su_clean, config)

    summary = {
        "agzu": summary_agzu,
        "su": summary_su,
        "agzu_rows": len(agzu_clean),
        "su_rows": len(su_clean),
    }
    write_summary(summary, config)
    return summary


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean AGZU and SU datasets according to pipeline rules.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to pipeline configuration.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or [])
    summary = run_cleaning(args.config)
    print(f"Saved cleaned tables with {summary['agzu_rows']} AGZU rows and {summary['su_rows']} SU rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
