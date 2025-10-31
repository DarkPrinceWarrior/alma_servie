from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .detection import (
    aggregate_feature_values,
    build_frequency_baseline,
    build_residual_detection_model,
    compute_feature_frame,
    derive_thresholds,
    detect_segments_for_well,
)
from .models import (
    DetectionSettings,
    FrequencyBaseline,
    InterpretationThresholds,
    ReferenceInterval,
    ResidualDetectionModel,
    WellTimeseries,
)
from .preprocessing import (
    load_svod_sheet,
    load_well_series,
    parse_reference_intervals,
    preprocess_well_data,
)
from .settings import load_detection_settings, load_residual_settings


@dataclass
class DetectionContext:
    settings: DetectionSettings
    interpretation: InterpretationThresholds
    reference_intervals: List[ReferenceInterval]
    holdout_wells: Set[str]
    base_well_data: Dict[str, pd.DataFrame]
    features_map: Dict[str, pd.DataFrame]
    thresholds: Dict[str, Dict[str, float]]
    residual_model: Optional[ResidualDetectionModel]


@dataclass
class StepwiseResult:
    well: str
    reference_start: Optional[pd.Timestamp]
    reference_end: Optional[pd.Timestamp]
    detection_start: Optional[pd.Timestamp]
    notification_time: Optional[pd.Timestamp]
    evaluated_points: int
    delay_minutes: Optional[float]
    triggers: Dict[str, object]


def build_detection_context(
    config: Dict,
    xl: pd.ExcelFile,
) -> DetectionContext:
    settings, interpretation = load_detection_settings(config)
    holdout_wells = set(config["anomalies"].get("holdout_wells", []) or [])

    svod_sheet = config["anomalies"].get("svod_sheet", "svod")
    svod = load_svod_sheet(xl, svod_sheet)

    alignment_cfg = config.get("alignment", {})
    base_frequency = alignment_cfg.get("frequency", "15T")
    base_aggregation = alignment_cfg.get("base_aggregation", "mean")
    pressure_metrics_cfg = alignment_cfg.get("pressure_fast_metrics", ["Intake_Pressure"])
    if isinstance(pressure_metrics_cfg, str):
        pressure_metrics = [pressure_metrics_cfg]
    else:
        pressure_metrics = list(pressure_metrics_cfg)

    well_series_map: Dict[str, WellTimeseries] = load_well_series(
        xl,
        svod_sheet,
        base_frequency=base_frequency,
        pressure_metrics=pressure_metrics,
        base_aggregation=base_aggregation,
    )
    base_well_data: Dict[str, pd.DataFrame] = {
        well: series.base
        for well, series in well_series_map.items()
        if series.base is not None and not series.base.empty
    }

    if not base_well_data:
        raise ValueError("Не удалось подготовить базовые временные ряды для пошаговой проверки.")

    preprocessing_cfg = config["anomalies"].get("preprocessing", {})
    window_minutes = float(preprocessing_cfg.get("hampel_window_minutes", 45))
    n_sigma = float(preprocessing_cfg.get("hampel_n_sigma", 3.0))
    ffill_limit_minutes = float(preprocessing_cfg.get("ffill_limit_minutes", 30))

    if window_minutes > 0 and n_sigma > 0:
        for well, frame in list(base_well_data.items()):
            processed, _ = preprocess_well_data(
                frame,
                window_minutes=window_minutes,
                n_sigma=n_sigma,
                ffill_limit_minutes=ffill_limit_minutes,
            )
            base_well_data[well] = processed

    reference_intervals = parse_reference_intervals(
        svod=svod,
        well_data=base_well_data,
        anomaly_label=config["anomalies"].get("anomaly_cause", "Негерметичность НКТ"),
        normal_label=config["anomalies"].get("normal_cause", "Нормальная работа при изменении частоты"),
    )

    anomaly_intervals = [interval for interval in reference_intervals if interval.label == "anomaly"]
    anomaly_intervals_train = [interval for interval in anomaly_intervals if interval.well not in holdout_wells]
    normal_intervals_full = [interval for interval in reference_intervals if interval.label == "normal"]
    normal_intervals_train = [interval for interval in normal_intervals_full if interval.well not in holdout_wells]

    frequency_cfg = config["anomalies"].get("frequency_baseline", {})
    baseline: Optional[FrequencyBaseline] = None
    if frequency_cfg.get("enabled", True):
        metrics = frequency_cfg.get("metrics", ["Intake_Pressure", "Current", "Motor_Temperature"])
        bin_width = float(frequency_cfg.get("bin_width_hz", 2.0))
        min_points = int(frequency_cfg.get("min_points", 10))
        baseline = build_frequency_baseline(
            normal_intervals_train,
            base_well_data,
            metrics=metrics,
            bin_width=bin_width,
            min_points=min_points,
        )

    features_map: Dict[str, pd.DataFrame] = {
        well: compute_feature_frame(df, settings, baseline=baseline) for well, df in base_well_data.items()
    }

    residual_settings = load_residual_settings(config)
    residual_model = build_residual_detection_model(features_map, normal_intervals_train, residual_settings)

    if not anomaly_intervals_train:
        anomaly_features = aggregate_feature_values(anomaly_intervals, features_map)
        training_anomaly_intervals = anomaly_intervals
    else:
        anomaly_features = aggregate_feature_values(anomaly_intervals_train, features_map)
        training_anomaly_intervals = anomaly_intervals_train

    if normal_intervals_train:
        normal_features_train = aggregate_feature_values(normal_intervals_train, features_map)
    else:
        normal_features_train = aggregate_feature_values(normal_intervals_full, features_map)

    thresholds = derive_thresholds(normal_features_train, anomaly_features, settings)

    # ensure training intervals referenced in thresholds remain available (unused variable but clarifies parity)
    _ = training_anomaly_intervals  # noqa: F841

    return DetectionContext(
        settings=settings,
        interpretation=interpretation,
        reference_intervals=reference_intervals,
        holdout_wells=holdout_wells,
        base_well_data=base_well_data,
        features_map=features_map,
        thresholds=thresholds,
        residual_model=residual_model,
    )


def _simulate_detection_for_well(
    context: DetectionContext,
    well: str,
) -> StepwiseResult:
    base_df = context.base_well_data.get(well)
    features = context.features_map.get(well)
    if base_df is None or features is None or base_df.empty or features.empty:
        return StepwiseResult(
            well=well,
            reference_start=None,
            reference_end=None,
            detection_start=None,
            notification_time=None,
            evaluated_points=0,
            delay_minutes=None,
            triggers={},
        )

    well_intervals = [ref for ref in context.reference_intervals if ref.well == well and ref.label == "anomaly"]
    reference_start = well_intervals[0].start if well_intervals else None
    reference_end = well_intervals[0].end if well_intervals else None

    sorted_index = features.index.sort_values()
    evaluated_points = 0
    detection_record: Optional[Dict[str, object]] = None
    notification_time: Optional[pd.Timestamp] = None

    for ts in sorted_index:
        evaluated_points += 1
        truncated_features = features.loc[:ts].copy()
        truncated_well_df = base_df.loc[:ts].copy()
        records = detect_segments_for_well(
            well=well,
            well_df=truncated_well_df,
            features=truncated_features,
            thresholds=context.thresholds,
            settings=context.settings,
            interpretation=context.interpretation,
            reference_intervals=context.reference_intervals,
            residual_model=context.residual_model,
            holdout_wells=context.holdout_wells,
        )
        if not records:
            continue
        earliest = min(records, key=lambda item: item["start"])
        if earliest["start"] <= ts:
            detection_record = earliest
            notification_time = ts
            break

    if detection_record is None:
        return StepwiseResult(
            well=well,
            reference_start=reference_start,
            reference_end=reference_end,
            detection_start=None,
            notification_time=None,
            evaluated_points=evaluated_points,
            delay_minutes=None,
            triggers={},
        )

    detection_start = detection_record["start"]
    delay_minutes: Optional[float] = None
    if reference_start is not None and detection_start is not None:
        delay_minutes = (detection_start - reference_start).total_seconds() / 60.0

    triggers = {
        "pressure_delta": float(detection_record.get("pressure_delta_median", float("nan"))),
        "pressure_slope": float(detection_record.get("pressure_slope_median", float("nan"))),
        "residual": bool(detection_record.get("residual_triggered", False)),
        "ewma": bool(detection_record.get("ewma_triggered", False)),
        "spike": bool(detection_record.get("spike_triggered", False)),
        "current_delta": float(detection_record.get("current_delta_median", float("nan"))),
        "temperature_delta": float(detection_record.get("temperature_delta_median", float("nan"))),
    }

    triggers["notification_lag_minutes"] = (
        (notification_time - detection_start).total_seconds() / 60.0 if notification_time else None
    )

    return StepwiseResult(
        well=well,
        reference_start=reference_start,
        reference_end=reference_end,
        detection_start=detection_start,
        notification_time=notification_time,
        evaluated_points=evaluated_points,
        delay_minutes=delay_minutes,
        triggers=triggers,
    )


def run_stepwise_evaluation(
    config: Dict,
    xl: pd.ExcelFile,
    wells: Sequence[str],
) -> Dict[str, StepwiseResult]:
    if not wells:
        return {}

    context = build_detection_context(config, xl)
    results: Dict[str, StepwiseResult] = {}
    for well in wells:
        results[well] = _simulate_detection_for_well(context, well)
    return results


__all__ = [
    "DetectionContext",
    "StepwiseResult",
    "build_detection_context",
    "run_stepwise_evaluation",
]
