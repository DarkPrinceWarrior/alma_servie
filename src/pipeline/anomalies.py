"""
Compute anomaly metrics by comparing pre-window averages with anomaly-period averages.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG_PATH, load_config


@dataclass
class AnomalyEvent:
    well: str
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    raw_row: Dict


def parse_datetime_range(value: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    pattern = r"(\d{2})\.(\d{2})\.(\d{2})\s+(\d{2}):(\d{2})\s*-\s*(\d{2})\.(\d{2})\.(\d{2})\s+(\d{2}):(\d{2})"
    match = re.search(pattern, str(value))
    if not match:
        return None, None
    day1, month1, year1, hour1, minute1, day2, month2, year2, hour2, minute2 = match.groups()
    start = pd.Timestamp(f"20{year1}-{month1}-{day1} {hour1}:{minute1}:00")
    end = pd.Timestamp(f"20{year2}-{month2}-{day2} {hour2}:{minute2}:00")
    return start, end


def load_anomaly_table(config: Dict, override_path: Optional[Path] = None) -> pd.DataFrame:
    workbook_path = override_path or Path(config["anomalies"]["source_workbook"])
    if not workbook_path.exists():
        raise FileNotFoundError(
            f"Anomaly workbook not found at {workbook_path}. "
            "Provide the file or use --source to override the path."
        )
    sheet = config["anomalies"]["sheet_name"]
    return pd.read_excel(workbook_path, sheet_name=sheet)


def extract_events(anomalies_df: pd.DataFrame) -> List[AnomalyEvent]:
    events: List[AnomalyEvent] = []
    for _, row in anomalies_df.iterrows():
        start, end = parse_datetime_range(row.get("Дата и время"))
        if start is None or end is None:
            continue
        events.append(
            AnomalyEvent(
                well=str(row.get("Скважина")),
                name=str(row.get("Название предупреждения")),
                start=start,
                end=end,
                raw_row=row.to_dict(),
            )
        )
    return events


def compute_percent_change(before: float, during: float) -> Optional[float]:
    if pd.isna(before) or pd.isna(during):
        return None
    if before == 0:
        if during == 0:
            return 0.0
        return 100.0
    return round(((during - before) / before) * 100.0, 2)


def compute_metrics_for_event(
    merged: pd.DataFrame,
    event: AnomalyEvent,
    config: Dict,
) -> Dict[str, Optional[float]]:
    anomalies_conf = config["anomalies"]
    pre_window_days = anomalies_conf.get("pre_window_days", 3)
    agzu_metrics = anomalies_conf.get("agzu_metrics", [])
    su_metrics = anomalies_conf.get("su_metrics", [])
    agzu_prefix = anomalies_conf.get("metrics_agzu_prefix", "АГЗУ")
    su_prefix = anomalies_conf.get("metrics_su_prefix", "СУ")

    df_well = merged[merged[config["su"]["well_column"]].astype(str) == event.well]
    if df_well.empty:
        return {}

    timestamp_col = "timestamp"
    df_well = df_well.sort_values(timestamp_col)

    pre_start = event.start - pd.Timedelta(days=pre_window_days)
    window_before = df_well[(df_well[timestamp_col] >= pre_start) & (df_well[timestamp_col] < event.start)]
    window_anomaly = df_well[(df_well[timestamp_col] >= event.start) & (df_well[timestamp_col] <= event.end)]

    metrics: Dict[str, Optional[float]] = {}

    def _window_mean(frame: pd.DataFrame, column: str) -> float:
        return pd.to_numeric(frame[column], errors="coerce").mean()

    for metric in su_metrics:
        if metric not in df_well.columns:
            continue
        before_mean = _window_mean(window_before, metric)
        during_mean = _window_mean(window_anomaly, metric)
        metrics[f"{su_prefix}_{metric}_изм%"] = compute_percent_change(before_mean, during_mean)

    for metric in agzu_metrics:
        if metric not in df_well.columns:
            continue
        before_mean = _window_mean(window_before, metric)
        during_mean = _window_mean(window_anomaly, metric)
        metrics[f"{agzu_prefix}_{metric}_изм%"] = compute_percent_change(before_mean, during_mean)

    metrics[f"{agzu_prefix}_точек_до"] = int(len(window_before))
    metrics[f"{agzu_prefix}_точек_во_время"] = int(len(window_anomaly))
    return metrics


def evaluate_conditions(metrics: Dict[str, Optional[float]], conditions: List[Dict]) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for condition in conditions:
        name = condition.get("name", "condition")
        rules = condition.get("rules", [])
        condition_met = True
        for rule in rules:
            metric_name = rule.get("metric")
            if metric_name not in metrics or metrics[metric_name] is None:
                condition_met = False
                break
            value = metrics[metric_name]
            operator = rule.get("operator", ">=")
            threshold = rule.get("threshold", 0)
            if operator == ">=":
                if not (value >= threshold):
                    condition_met = False
                    break
            elif operator == "<=":
                if not (value <= threshold):
                    condition_met = False
                    break
            elif operator == ">":
                if not (value > threshold):
                    condition_met = False
                    break
            elif operator == "<":
                if not (value < threshold):
                    condition_met = False
                    break
            else:
                condition_met = False
                break
        results[name] = condition_met
    return results


def run_anomaly_analysis(config_path: Path, workbook_override: Optional[Path] = None) -> pd.DataFrame:
    config = load_config(config_path)
    anomalies_df = load_anomaly_table(config, override_path=workbook_override)
    events = extract_events(anomalies_df)
    if not events:
        raise ValueError("No valid anomaly events parsed from workbook.")

    merged = pd.read_parquet(Path(config["paths"]["processed_dir"]) / "merged_hourly.parquet")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged[config["su"]["well_column"]] = merged[config["su"]["well_column"]].astype(str)

    records: List[Dict] = []
    for event in events:
        metrics = compute_metrics_for_event(merged, event, config)
        condition_flags = evaluate_conditions(metrics, config["anomalies"].get("conditions", []))
        record = {
            "Номер скважины": event.well,
            "Причина": event.name,
            "Начало": event.start,
            "Конец": event.end,
        }
        record.update(metrics)
        record.update({f"cond_{name}": flag for name, flag in condition_flags.items()})
        records.append(record)

    result_df = pd.DataFrame(records)
    output_path = Path(config["paths"]["reports_dir"]) / "anomalies" / "anomaly_analysis.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)

    excel_path = output_path.with_suffix(".xlsx")
    result_df.to_excel(excel_path, index=False)
    summary_records: List[Dict] = []
    for record in records:
        converted = {}
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                converted[key] = value.isoformat()
            else:
                converted[key] = value
        summary_records.append(converted)

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary_records, ensure_ascii=False, indent=2), encoding="utf-8")
    return result_df


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute anomaly deviations based on merged datasets.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to pipeline configuration.")
    parser.add_argument("--source", type=Path, help="Override path to anomaly workbook.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else None)
    result = run_anomaly_analysis(args.config, workbook_override=args.source)
    print(f"Computed anomalies for {len(result)} events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
