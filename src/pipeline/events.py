"""
Generate summary statistics for reference intervals (нормальная/аномальная работа)
using the new alma/Общая_таблица.xlsx workbook.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .anomalies import (
    InterpretationThresholds,
    classify_direction,
    compute_feature_frame,
    load_detection_settings,
    load_svod_sheet,
    load_well_series,
    parse_reference_intervals,
)
from .config import DEFAULT_CONFIG_PATH, load_config


def summarise_series(series: pd.Series, percentiles: List[float]) -> Dict[str, float]:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return {}
    result = {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
    }
    for p in percentiles:
        result[f"p{int(p * 100)}"] = float(series.quantile(p))
    return result


def build_interval_records(
    intervals: List,
    well_data: Dict[str, pd.DataFrame],
    features_map: Dict[str, pd.DataFrame],
    interpretation: InterpretationThresholds,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for interval in intervals:
        well_frame = well_data.get(interval.well)
        feature_frame = features_map.get(interval.well)
        if well_frame is None or feature_frame is None:
            continue

        segment_data = well_frame.loc[interval.start : interval.end]
        segment_features = feature_frame.loc[interval.start : interval.end]
        if segment_data.empty or segment_features.empty:
            continue

        pressure_delta_median = float(segment_features["pressure_delta"].median())
        current_delta_median = float(segment_features["current_delta"].median())
        temperature_delta_median = float(segment_features["temperature_delta"].median())

        record = {
            "well": interval.well,
            "label": interval.label,
            "start": interval.start,
            "end": interval.end,
            "duration_minutes": (interval.end - interval.start).total_seconds() / 60.0,
            "pressure_mean": float(segment_data["Intake_Pressure"].mean()),
            "pressure_std": float(segment_data["Intake_Pressure"].std(ddof=0)),
            "pressure_min": float(segment_data["Intake_Pressure"].min()),
            "pressure_max": float(segment_data["Intake_Pressure"].max()),
            "pressure_delta_median": pressure_delta_median,
            "pressure_delta_mean": float(segment_features["pressure_delta"].mean()),
            "pressure_direction": classify_direction(pressure_delta_median, interpretation),
            "current_mean": float(segment_data["Current"].mean()) if "Current" in segment_data else float("nan"),
            "current_std": float(segment_data["Current"].std(ddof=0)) if "Current" in segment_data else float("nan"),
            "current_delta_median": current_delta_median,
            "current_direction": classify_direction(current_delta_median, interpretation),
            "temperature_mean": float(segment_data["Motor_Temperature"].mean())
            if "Motor_Temperature" in segment_data
            else float("nan"),
            "temperature_std": float(segment_data["Motor_Temperature"].std(ddof=0))
            if "Motor_Temperature" in segment_data
            else float("nan"),
            "temperature_delta_median": temperature_delta_median,
            "temperature_direction": classify_direction(temperature_delta_median, interpretation),
            "frequency_mean": float(segment_data["Frequency"].mean()) if "Frequency" in segment_data else float("nan"),
            "notes": "; ".join(interval.notes) if interval.notes else "",
        }
        records.append(record)
    return pd.DataFrame(records)


def compute_label_summary(
    features_df: pd.DataFrame,
    percentiles: List[float],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    metric_columns = [
        "pressure_mean",
        "pressure_delta_median",
        "current_mean",
        "current_delta_median",
        "temperature_mean",
        "temperature_delta_median",
    ]

    for label, group in features_df.groupby("label"):
        summary[label] = {}
        for column in metric_columns:
            stats = summarise_series(group[column], percentiles)
            if stats:
                summary[label][column] = stats
    return summary


def run_reference_analysis(config_path: Path, workbook_override: Optional[Path] = None) -> Dict:
    config = load_config(config_path)
    workbook_path = workbook_override or Path(config["anomalies"]["source_workbook"])
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    settings, interpretation = load_detection_settings(config)
    svod_sheet = config["anomalies"].get("svod_sheet", "svod")
    anomaly_cause = config["anomalies"].get("anomaly_cause", "Негерметичность НКТ")
    normal_cause = config["anomalies"].get("normal_cause", "Нормальная работа при изменении частоты")
    percentiles = config["anomalies"].get("calibration_percentiles", [0.1, 0.5, 0.9])

    xl = pd.ExcelFile(workbook_path)
    svod = load_svod_sheet(xl, svod_sheet)
    well_data = load_well_series(xl, svod_sheet)
    if not well_data:
        raise ValueError("Не удалось загрузить временные ряды из workbook для справочного анализа.")

    reference_intervals = parse_reference_intervals(svod, well_data, anomaly_cause, normal_cause)
    if not reference_intervals:
        raise ValueError("В листе svod отсутствуют интервалы нормальной/аномальной работы.")

    features_map = {well: compute_feature_frame(df, settings) for well, df in well_data.items()}
    features_df = build_interval_records(reference_intervals, well_data, features_map, interpretation)
    if features_df.empty:
        raise ValueError("Не удалось вычислить статистики по предоставленным интервалам.")

    label_summary = compute_label_summary(features_df, percentiles)

    reports_dir = Path(config["paths"]["reports_dir"]) / "anomalies"
    reports_dir.mkdir(parents=True, exist_ok=True)

    features_path = reports_dir / "events_features.parquet"
    features_df.to_parquet(features_path, index=False)
    features_df.to_csv(features_path.with_suffix(".csv"), index=False)

    summary_payload = {
        "records": len(features_df),
        "percentiles": percentiles,
        "label_summary": label_summary,
    }
    summary_path = reports_dir / "events_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary_payload


def parse_args(args: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise reference intervals from alma/Общая_таблица.xlsx.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source", type=Path, help="Override workbook path.")
    return parser.parse_args(list(args) if args is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_reference_analysis(args.config, workbook_override=args.source)
    print(f"Reference analysis completed. Records: {summary['records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
