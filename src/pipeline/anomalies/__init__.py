"""
Anomaly detection pipeline orchestration and public API surface.

The logic is structured across dedicated modules:
    - models.py: domain dataclasses and settings containers.
    - settings.py: configuration loaders that initialise dataclasses.
    - preprocessing.py: helpers for workbook ingestion and data cleanup.
    - detection.py: feature engineering, baseline modelling, and segment detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..config import DEFAULT_CONFIG_PATH, load_config
from .detection import (
    aggregate_feature_values,
    build_frequency_baseline,
    build_residual_detection_model,
    classify_direction,
    compute_feature_frame,
    derive_thresholds,
    detect_segments_for_well,
    overlaps_interval,
)
from .models import (
    DetectionSettings,
    FrequencyBaseline,
    InterpretationThresholds,
    ReferenceInterval,
    ResidualDetectionModel,
    ResidualDetectionSettings,
    WellTimeseries,
)
from .preprocessing import (
    clip_interval_to_data,
    load_svod_sheet,
    load_well_series,
    parse_reference_intervals,
    preprocess_well_data,
)
from .settings import load_detection_settings, load_residual_settings
from .simulation import (
    DetectionContext,
    StepwiseResult,
    build_detection_context,
    evaluate_stepwise_for_intervals,
    run_stepwise_evaluation,
)


def run_anomaly_analysis(config_path: Path, workbook_override: Optional[Path] = None) -> pd.DataFrame:
    config = load_config(config_path)
    workbook_path = workbook_override or Path(config["anomalies"]["source_workbook"])
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook with well data not found: {workbook_path}")

    xl = pd.ExcelFile(workbook_path)
    context = build_detection_context(config, xl, use_streaming_calibration=True)

    reports_dir = Path(config["paths"]["reports_dir"]) / "anomalies"
    reports_dir.mkdir(parents=True, exist_ok=True)

    base_export_frames: List[pd.DataFrame] = []
    for well, frame in context.base_well_data.items():
        if frame.empty:
            continue
        export = frame.copy()
        export.index.name = "timestamp"
        export = export.reset_index()
        export.insert(0, "well", well)
        base_export_frames.append(export)
    if base_export_frames:
        base_export_df = pd.concat(base_export_frames, ignore_index=True)
        base_export_path = reports_dir / "timeseries_15min.parquet"
        base_export_df.to_parquet(base_export_path, index=False)
        base_export_df.to_csv(base_export_path.with_suffix(".csv"), index=False)

    pressure_export_frames: List[pd.DataFrame] = []
    for well, frame in context.pressure_fast_data.items():
        if frame.empty:
            continue
        export = frame.copy()
        export.index.name = "timestamp"
        export = export.reset_index()
        export.insert(0, "well", well)
        pressure_export_frames.append(export)
    if pressure_export_frames:
        pressure_export_df = pd.concat(pressure_export_frames, ignore_index=True)
        pressure_export_path = reports_dir / "pressure_fast.parquet"
        pressure_export_df.to_parquet(pressure_export_path, index=False)
        pressure_export_df.to_csv(pressure_export_path.with_suffix(".csv"), index=False)

    if context.preprocess_summary:
        print("Preprocessing summary (anomalies):")
        for well, stats in sorted(context.preprocess_summary.items()):
            removed_total = sum(values.get("removed_outliers", 0) for values in stats.values())
            filled_total = sum(values.get("ffill_values", 0) for values in stats.values())
            print(f"  {well}: removed={removed_total}, ffilled={filled_total}")

    if context.frequency_baseline is None:
        print("Frequency baseline: недостаточно данных для построения модели.")
    else:
        baseline_path = reports_dir / "frequency_baseline.parquet"
        context.frequency_baseline.summary.to_parquet(baseline_path, index=False)
        context.frequency_baseline.summary.to_csv(baseline_path.with_suffix(".csv"), index=False)

    features_map = context.features_map
    reference_intervals = context.reference_intervals
    holdout_wells = context.holdout_wells

    anomaly_intervals = [interval for interval in reference_intervals if interval.label == "anomaly"]
    anomaly_intervals_train = [interval for interval in anomaly_intervals if interval.well not in holdout_wells]
    normal_intervals_full = [interval for interval in reference_intervals if interval.label == "normal"]
    normal_intervals_train = [interval for interval in normal_intervals_full if interval.well not in holdout_wells]

    if anomaly_intervals_train:
        anomaly_features = aggregate_feature_values(anomaly_intervals_train, features_map)
        training_anomaly_intervals = anomaly_intervals_train
    else:
        anomaly_features = aggregate_feature_values(anomaly_intervals, features_map)
        training_anomaly_intervals = anomaly_intervals

    if normal_intervals_train:
        normal_features_train = aggregate_feature_values(normal_intervals_train, features_map)
        normal_training_intervals = normal_intervals_train
    else:
        normal_features_train = aggregate_feature_values(normal_intervals_full, features_map)
        normal_training_intervals = normal_intervals_full

    detection_records: List[Dict[str, object]] = []
    for well, well_df in context.base_well_data.items():
        features = features_map.get(well)
        if features is None or features.empty:
            continue
        detection_records.extend(
            detect_segments_for_well(
                well=well,
                well_df=well_df,
                features=features,
                thresholds=context.thresholds,
                settings=context.settings,
                interpretation=context.interpretation,
                reference_intervals=reference_intervals,
                residual_model=context.residual_model,
                holdout_wells=holdout_wells,
            )
        )

    detections = pd.DataFrame(detection_records)
    if not detections.empty:
        detections.sort_values(["start", "well", "segment_type"], inplace=True)
        detections.reset_index(drop=True, inplace=True)

    output_parquet = reports_dir / "anomaly_analysis.parquet"
    detections.to_parquet(output_parquet, index=False)
    detections.to_excel(output_parquet.with_suffix(".xlsx"), index=False)

    json_records: List[Dict[str, object]] = []
    for record in detections.to_dict(orient="records"):
        converted: Dict[str, object] = {}
        for key, value in record.items():
            if value is pd.NaT:
                converted[key] = None
            elif isinstance(value, pd.Timestamp):
                converted[key] = None if pd.isna(value) else value.isoformat()
            elif isinstance(value, (np.floating, float)):
                converted[key] = None if np.isnan(value) else float(value)
            else:
                converted[key] = value
        json_records.append(converted)

    summary_path = output_parquet.with_suffix(".json")
    summary_payload = {
        "thresholds": context.thresholds,
        "training": {
            "anomaly_points": int(len(anomaly_features["pressure_delta"])),
            "normal_points": int(len(normal_features_train["pressure_delta"])),
            "anomaly_wells": sorted({interval.well for interval in training_anomaly_intervals}),
            "normal_wells": sorted({interval.well for interval in normal_training_intervals}),
        },
        "settings": {
            "window_minutes": context.settings.window_minutes,
            "shift_minutes": context.settings.shift_minutes,
            "min_samples": context.settings.min_samples,
            "delta_factor": context.settings.delta_factor,
            "delta_quantile": context.settings.delta_quantile,
            "slope_margin": context.settings.slope_margin,
            "slope_quantile": context.settings.slope_quantile,
            "min_duration_minutes": context.settings.min_duration_minutes,
            "gap_minutes": context.settings.gap_minutes,
        },
        "detections": json_records,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return detections


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect anomalies based on alma/Общая_таблица.xlsx reference data."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to pipeline configuration.")
    parser.add_argument(
        "--source",
        type=Path,
        help="Override path to workbook (по умолчанию -- alma/Общая_таблица.xlsx из конфигурации).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    detections = run_anomaly_analysis(args.config, workbook_override=args.source)
    event_count = len(detections)
    print(f"Detected {event_count} anomaly events.")
    return 0


__all__ = [
    "DetectionSettings",
    "FrequencyBaseline",
    "InterpretationThresholds",
    "ReferenceInterval",
    "ResidualDetectionModel",
    "ResidualDetectionSettings",
    "WellTimeseries",
    "DetectionContext",
    "StepwiseResult",
    "build_detection_context",
    "aggregate_feature_values",
    "build_frequency_baseline",
    "build_residual_detection_model",
    "classify_direction",
    "clip_interval_to_data",
    "compute_feature_frame",
    "derive_thresholds",
    "detect_segments_for_well",
    "overlaps_interval",
    "evaluate_stepwise_for_intervals",
    "load_detection_settings",
    "load_residual_settings",
    "load_svod_sheet",
    "load_well_series",
    "parse_args",
    "parse_reference_intervals",
    "preprocess_well_data",
    "run_anomaly_analysis",
    "run_stepwise_evaluation",
    "main",
]
