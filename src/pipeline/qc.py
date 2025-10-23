"""
Quality control utilities for interim datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .config import DEFAULT_CONFIG_PATH, load_config


def load_interim_tables(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agzu_path = Path(config["paths"]["interim_dir"]) / "agzu_raw.parquet"
    su_path = Path(config["paths"]["interim_dir"]) / "su_raw.parquet"
    agzu_df = pd.read_parquet(agzu_path) if agzu_path.exists() else pd.DataFrame()
    su_df = pd.read_parquet(su_path) if su_path.exists() else pd.DataFrame()
    return agzu_df, su_df


def gaps_report(df: pd.DataFrame, datetime_col: str, well_col: str, threshold_hours: float) -> pd.DataFrame:
    if df.empty or datetime_col not in df.columns:
        return pd.DataFrame(columns=[well_col, "prev_timestamp", "next_timestamp", "gap_hours"])

    df = df.sort_values([well_col, datetime_col]).copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df["prev_timestamp"] = df.groupby(well_col)[datetime_col].shift(1)
    df["gap_hours"] = (df[datetime_col] - df["prev_timestamp"]).dt.total_seconds() / 3600.0
    report = df.loc[df["gap_hours"] > threshold_hours, [well_col, "prev_timestamp", datetime_col, "gap_hours"]]
    report = report.rename(columns={datetime_col: "next_timestamp"})
    return report.reset_index(drop=True)


def duplicates_report(df: pd.DataFrame, datetime_col: str, well_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[well_col, datetime_col, "count"])
    dup_counts = (
        df.groupby([well_col, datetime_col])
        .size()
        .reset_index(name="count")
        .query("count > 1")
        .sort_values(["count"], ascending=False)
    )
    return dup_counts


def negative_values_report(df: pd.DataFrame, columns: List[str], well_col: str) -> pd.DataFrame:
    problems: List[pd.DataFrame] = []
    for col in columns:
        if col not in df.columns:
            continue
        mask = pd.to_numeric(df[col], errors="coerce") < 0
        subset = df.loc[mask, [well_col, col]]
        if not subset.empty:
            subset = subset.rename(columns={col: "value"}).assign(metric=col)
            problems.append(subset)
    if not problems:
        return pd.DataFrame(columns=[well_col, "metric", "value"])
    merged = pd.concat(problems, ignore_index=True)
    return merged[[well_col, "metric", "value"]]


def threshold_report(df: pd.DataFrame, thresholds: Dict[str, float], well_col: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for col, limit in thresholds.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        mask = values > limit
        if mask.any():
            temp = df.loc[mask, [well_col, col]].copy()
            temp["metric"] = col
            temp["value"] = values[mask]
            temp["threshold"] = limit
            rows.append(temp[[well_col, "metric", "value", "threshold"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[well_col, "metric", "value", "threshold"])


def zero_ratio_report(df: pd.DataFrame, columns: List[str], well_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[well_col, "metric", "zero_ratio"])

    reports = []
    for col in columns:
        if col not in df.columns:
            continue
        zero_ratio = df.groupby(well_col)[col].apply(lambda s: (pd.to_numeric(s, errors="coerce") == 0).mean())
        reports.append(zero_ratio.rename("zero_ratio").reset_index().assign(metric=col))
    if not reports:
        return pd.DataFrame(columns=[well_col, "metric", "zero_ratio"])
    return pd.concat(reports, ignore_index=True)[[well_col, "metric", "zero_ratio"]]


def nan_share_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["column", "nan_ratio"])
    ratios = df.isna().mean().reset_index()
    ratios.columns = ["column", "nan_ratio"]
    return ratios


def write_html_report(sections: List[Tuple[str, pd.DataFrame]], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    parts = ["<html><head><meta charset='utf-8'><title>Quality report</title></head><body>"]
    for title, df in sections:
        parts.append(f"<h2>{title}</h2>")
        if df.empty:
            parts.append("<p>Данные отсутствуют.</p>")
        else:
            parts.append(df.to_html(index=False, border=0))
    parts.append("</body></html>")
    dest.write_text("\n".join(parts), encoding="utf-8")


def generate_quality_reports(config_path: Path) -> None:
    config = load_config(config_path)
    agzu_df, su_df = load_interim_tables(config)

    agzu_datetime = config["agzu"]["datetime_column"]
    agzu_well = config["agzu"]["well_column"]
    agzu_gap = config["agzu"].get("max_gap_hours_flag", 24)

    agzu_sections = [
        ("Разрывы по времени (> {} ч)".format(agzu_gap), gaps_report(agzu_df, agzu_datetime, agzu_well, agzu_gap)),
        ("Дубликаты по времени", duplicates_report(agzu_df, agzu_datetime, agzu_well)),
        ("Отрицательные значения", negative_values_report(agzu_df, config["agzu"].get("negative_sensitive_columns", []), agzu_well)),
        ("Превышение порогов", threshold_report(agzu_df, config["agzu"].get("extreme_thresholds", {}), agzu_well)),
        ("Доля пропусков", nan_share_report(agzu_df)),
    ]

    su_datetime = config["su"]["datetime_column"]
    su_well = config["su"]["well_column"]
    su_gap_hours = config["su"].get("gap_minutes_flag", 60) / 60

    su_sections = [
        ("Разрывы по времени (> {} ч)".format(su_gap_hours), gaps_report(su_df, su_datetime, su_well, su_gap_hours)),
        ("Дубликаты по времени", duplicates_report(su_df, su_datetime, su_well)),
        ("Доля нулевых значений", zero_ratio_report(su_df, config["su"].get("zero_sensitive_columns", []), su_well)),
        ("Доля пропусков", nan_share_report(su_df)),
    ]

    reports_dir = Path(config["paths"]["reports_dir"]) / "qc"
    write_html_report(agzu_sections, reports_dir / "agzu_quality.html")
    write_html_report(su_sections, reports_dir / "su_quality.html")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate quality control reports for interim datasets.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to pipeline configuration file.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or [])
    generate_quality_reports(args.config)
    print("QC reports generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
