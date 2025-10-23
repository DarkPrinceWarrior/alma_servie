"""
Tools for aggregating knowledge from the reference workbook (нормальная/ненормальная работа).

This module parses both sheets, computes the same feature set as the anomaly
analysis step, and exports consolidated datasets and percentile summaries to
help calibrate anomaly conditions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .anomalies import AnomalyEvent, compute_metrics_for_event, parse_datetime_range
from .config import DEFAULT_CONFIG_PATH, load_config


def load_reference_events(workbook_path: Path) -> pd.DataFrame:
    """Load sheets 'Ненормальная работа' and 'Нормальная работа' into a unified DataFrame."""
    xl = pd.ExcelFile(workbook_path)
    data_frames: List[pd.DataFrame] = []

    sheet_mapping = [
        ("Ненормальная работа", "abnormal"),
        ("Нормальная работа", "normal"),
    ]

    for sheet_name, label in sheet_mapping:
        if sheet_name not in xl.sheet_names:
            continue
        df = xl.parse(sheet_name)
        df = df.copy()
        df["label"] = label
        df["sheet_name"] = sheet_name
        data_frames.append(df)

    if not data_frames:
        raise ValueError("Reference workbook does not contain expected sheets: 'Ненормальная работа'/'Нормальная работа'")

    combined = pd.concat(data_frames, ignore_index=True)
    return combined


def normalise_events(df: pd.DataFrame) -> pd.DataFrame:
    """Parse datetime ranges, normalise well identifiers, and preserve comments."""
    df = df.copy()
    starts: List[Optional[pd.Timestamp]] = []
    ends: List[Optional[pd.Timestamp]] = []
    for value in df["Дата и время"]:
        start, end = parse_datetime_range(value)
        starts.append(start)
        ends.append(end)

    df["start"] = pd.to_datetime(starts)
    df["end"] = pd.to_datetime(ends)
    df["well"] = df["Скважина"].astype(str).str.strip()
    df["event_name"] = df.get("Название предупреждения", df["label"])
    df["comment"] = df.get("Комментарий")
    df["details"] = df.filter(like="Unnamed").bfill(axis=1).iloc[:, 0].where(lambda s: s.notna())
    return df


def compute_event_features(reference_df: pd.DataFrame, merged: pd.DataFrame, config: Dict) -> pd.DataFrame:
    records: List[Dict] = []
    merged = merged.copy()
    timestamp_col = "timestamp"
    merged[timestamp_col] = pd.to_datetime(merged[timestamp_col])
    merged[config["su"]["well_column"]] = merged[config["su"]["well_column"]].astype(str)

    for _, row in reference_df.iterrows():
        start = row["start"]
        end = row["end"]
        if pd.isna(start) or pd.isna(end):
            continue
        event = AnomalyEvent(
            well=row["well"],
            name=str(row["event_name"]),
            start=start,
            end=end,
            raw_row=row.to_dict(),
        )
        metrics = compute_metrics_for_event(merged, event, config)
        if not metrics:
            continue
        metrics.update(
            {
                "label": row["label"],
                "well": row["well"],
                "event_name": row["event_name"],
                "start": start,
                "end": end,
                "comment": row["comment"],
                "details": row["details"],
            }
        )
        records.append(metrics)

    return pd.DataFrame(records)


def compute_percentile_summary(features: pd.DataFrame, percentiles: Iterable[float]) -> Dict:
    summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    metric_columns = [col for col in features.columns if col.endswith("_изм%")]

    for label, group in features.groupby("label"):
        numeric = group[metric_columns].apply(pd.to_numeric, errors="coerce")
        for col in metric_columns:
            series = numeric[col].dropna()
            if series.empty:
                continue
            stats = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
            for p in percentiles:
                stats[f"p{int(p*100)}"] = float(series.quantile(p))
            summary[label][col] = stats
    return summary


def compute_event_type_summary(features: pd.DataFrame, percentiles: Iterable[float]) -> Dict:
    metric_columns = [col for col in features.columns if col.endswith("_изм%")]
    summaries: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)

    for event_name, group in features.groupby("event_name"):
        numeric = group[metric_columns].apply(pd.to_numeric, errors="coerce")
        for col in metric_columns:
            series = numeric[col].dropna()
            if series.empty:
                continue
            stats = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
            for p in percentiles:
                stats[f"p{int(p*100)}"] = float(series.quantile(p))
            summaries[event_name][col] = stats
    return summaries


def run_reference_analysis(config_path: Path, workbook_override: Optional[Path] = None) -> Dict:
    config = load_config(config_path)
    workbook_path = workbook_override or Path(config["anomalies"]["source_workbook"])
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook for reference analysis not found: {workbook_path}")

    merged_path = Path(config["paths"]["processed_dir"]) / "merged_hourly.parquet"
    if not merged_path.exists():
        raise FileNotFoundError("Merged dataset not found. Run alignment step before reference analysis.")

    merged = pd.read_parquet(merged_path)
    reference_raw = load_reference_events(workbook_path)
    reference_norm = normalise_events(reference_raw)
    features = compute_event_features(reference_norm, merged, config)

    reports_dir = Path(config["paths"]["reports_dir"]) / "anomalies"
    reports_dir.mkdir(parents=True, exist_ok=True)

    features_path = reports_dir / "events_features.parquet"
    features.to_parquet(features_path, index=False)

    percentiles = config["anomalies"].get("calibration_percentiles", [0.05, 0.5, 0.95])
    label_summary = compute_percentile_summary(features, percentiles)
    event_summary = compute_event_type_summary(features, percentiles)

    summary_payload = {
        "label_summary": label_summary,
        "event_summary": event_summary,
        "records": len(features),
    }

    summary_path = reports_dir / "events_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = reports_dir / "events_features.csv"
    features.to_csv(csv_path, index=False)

    return summary_payload


def parse_args(args: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse reference workbook (normal vs abnormal periods).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source", type=Path, help="Override path to reference workbook.")
    return parser.parse_args(list(args) if args is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_reference_analysis(args.config, workbook_override=args.source)
    print(f"Reference analysis completed. Records: {summary['records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
