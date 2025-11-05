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
    overlaps_interval,
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
from .workbook import WorkbookSource


@dataclass
class DetectionContext:
    settings: DetectionSettings
    interpretation: InterpretationThresholds
    reference_intervals: List[ReferenceInterval]
    holdout_wells: Set[str]
    base_well_data: Dict[str, pd.DataFrame]
    pressure_fast_data: Dict[str, pd.DataFrame]
    preprocess_summary: Dict[str, Dict[str, Dict[str, int]]]
    features_map: Dict[str, pd.DataFrame]
    thresholds: Dict[str, Dict[str, float]]
    residual_model: Optional[ResidualDetectionModel]
    frequency_baseline: Optional[FrequencyBaseline]


@dataclass
class StepwiseResult:
    well: str
    interval_start: Optional[pd.Timestamp]
    interval_end: Optional[pd.Timestamp]
    detection_start: Optional[pd.Timestamp]
    notification_time: Optional[pd.Timestamp]
    evaluated_points: int
    delay_minutes: Optional[float]
    triggers: Dict[str, object]


def build_detection_context(
    config: Dict,
    workbook: WorkbookSource,
    *,
    use_streaming_calibration: bool = False,
) -> DetectionContext:
    settings, interpretation = load_detection_settings(config)
    holdout_wells = set(config["anomalies"].get("holdout_wells", []) or [])

    svod_sheet = config["anomalies"].get("svod_sheet", "svod")
    svod = load_svod_sheet(workbook, svod_sheet)

    alignment_cfg = config.get("alignment", {})
    base_frequency = alignment_cfg.get("frequency", "15T")
    base_aggregation = alignment_cfg.get("base_aggregation", "mean")
    pressure_metrics_cfg = alignment_cfg.get("pressure_fast_metrics", ["Intake_Pressure"])
    if isinstance(pressure_metrics_cfg, str):
        pressure_metrics = [pressure_metrics_cfg]
    else:
        pressure_metrics = list(pressure_metrics_cfg)

    well_series_map: Dict[str, WellTimeseries] = load_well_series(
        workbook,
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
    pressure_fast_map: Dict[str, pd.DataFrame] = {
        well: series.pressure_fast
        for well, series in well_series_map.items()
        if series.pressure_fast is not None and not series.pressure_fast.empty
    }

    if not base_well_data:
        raise ValueError("Не удалось подготовить базовые временные ряды для пошаговой проверки.")

    preprocessing_cfg = config["anomalies"].get("preprocessing", {})
    window_minutes = float(preprocessing_cfg.get("hampel_window_minutes", 45))
    n_sigma = float(preprocessing_cfg.get("hampel_n_sigma", 3.0))
    ffill_limit_minutes = float(preprocessing_cfg.get("ffill_limit_minutes", 30))

    preprocess_summary: Dict[str, Dict[str, Dict[str, int]]] = {}
    if window_minutes > 0 and n_sigma > 0:
        for well, frame in list(base_well_data.items()):
            processed, stats = preprocess_well_data(
                frame,
                window_minutes=window_minutes,
                n_sigma=n_sigma,
                ffill_limit_minutes=ffill_limit_minutes,
            )
            base_well_data[well] = processed
            if stats:
                preprocess_summary[well] = stats

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

    context = DetectionContext(
        settings=settings,
        interpretation=interpretation,
        reference_intervals=reference_intervals,
        holdout_wells=holdout_wells,
        base_well_data=base_well_data,
        pressure_fast_data=pressure_fast_map,
        preprocess_summary=preprocess_summary,
        features_map=features_map,
        thresholds=thresholds,
        residual_model=residual_model,
        frequency_baseline=baseline,
    )

    if use_streaming_calibration and anomaly_intervals_train:
        streaming_results = evaluate_stepwise_for_intervals(context, anomaly_intervals_train)
        anomaly_delta_values = [
            result.triggers.get("pressure_delta")
            for result in streaming_results
            if result.detection_start is not None
            and result.triggers
            and result.triggers.get("pressure_delta") is not None
            and not pd.isna(result.triggers.get("pressure_delta"))
        ]
        anomaly_slope_values = [
            result.triggers.get("pressure_slope")
            for result in streaming_results
            if result.detection_start is not None
            and result.triggers
            and result.triggers.get("pressure_slope") is not None
            and not pd.isna(result.triggers.get("pressure_slope"))
        ]
        if anomaly_delta_values and anomaly_slope_values:
            anomaly_features_stream = {
                "pressure_delta": pd.Series(anomaly_delta_values, dtype=float),
                "pressure_slope": pd.Series(anomaly_slope_values, dtype=float),
            }
            context.thresholds = derive_thresholds(normal_features_train, anomaly_features_stream, settings)

    return context


def evaluate_stepwise_for_intervals(
    context: DetectionContext,
    intervals: Sequence[ReferenceInterval],
) -> List[StepwiseResult]:
    if not intervals:
        return []
    results: List[StepwiseResult] = []
    for interval in sorted(intervals, key=lambda it: (it.well, it.start)):
        results.append(_simulate_interval(context, interval))
    return results


def _simulate_interval(context: DetectionContext, interval: ReferenceInterval) -> StepwiseResult:
    well = interval.well
    base_df = context.base_well_data.get(well)
    features = context.features_map.get(well)
    if base_df is None or features is None or base_df.empty or features.empty:
        return StepwiseResult(
            well=well,
            interval_start=interval.start,
            interval_end=interval.end,
            detection_start=None,
            notification_time=None,
            evaluated_points=0,
            delay_minutes=None,
            triggers={},
        )

    warmup_minutes = max(context.settings.window_minutes, context.settings.shift_minutes)
    warmup = pd.Timedelta(minutes=warmup_minutes)
    start_bound = interval.start - warmup if interval.start is not None else features.index.min()
    end_bound = interval.end if interval.end is not None else features.index.max()

    feature_slice = features.loc[start_bound:end_bound]
    base_slice = base_df.loc[start_bound:end_bound]
    if feature_slice.empty or base_slice.empty:
        return StepwiseResult(
            well=well,
            interval_start=interval.start,
            interval_end=interval.end,
            detection_start=None,
            notification_time=None,
            evaluated_points=0,
            delay_minutes=None,
            triggers={},
        )

    tolerance = pd.Timedelta(minutes=context.settings.gap_minutes)
    detection_record: Optional[Dict[str, object]] = None
    notification_time: Optional[pd.Timestamp] = None
    evaluated_points = 0

    for evaluated_points, ts in enumerate(feature_slice.index, start=1):
        current_features = feature_slice.loc[:ts].copy()
        current_base = base_slice.loc[:ts].copy()
        records = detect_segments_for_well(
            well=well,
            well_df=current_base,
            features=current_features,
            thresholds=context.thresholds,
            settings=context.settings,
            interpretation=context.interpretation,
            reference_intervals=context.reference_intervals,
            residual_model=context.residual_model,
            holdout_wells=context.holdout_wells,
        )
        if not records:
            continue
        candidates: List[Dict[str, object]] = []
        for record in records:
            if record.get("segment_type") != "anomaly":
                continue
            record_start = record.get("start")
            record_end = record.get("end", record_start)
            if record_start is None:
                continue
            if interval.end is not None and record_start > interval.end + tolerance:
                continue
            ref_start = record.get("reference_start")
            matches_reference = False
            if ref_start is not None and interval.start is not None:
                matches_reference = ref_start == interval.start
            elif record_end is not None:
                matches_reference = overlaps_interval(record_start, record_end, interval)
            if not matches_reference:
                continue
            candidates.append(record)
        if candidates:
            detection_record = min(candidates, key=lambda item: item["start"])
            notification_time = ts
            break

    detection_start = None
    delay_minutes: Optional[float] = None
    triggers: Dict[str, object] = {}
    if detection_record is not None:
        detection_start = detection_record.get("start")
        if interval.start is not None and detection_start is not None and detection_start < interval.start:
            detection_start = interval.start
        if detection_start is not None and interval.start is not None:
            delay_minutes = max(0.0, (detection_start - interval.start).total_seconds() / 60.0)
        triggers = {
            "pressure_delta": float(detection_record.get("pressure_delta_median", float("nan"))),
            "pressure_slope": float(detection_record.get("pressure_slope_median", float("nan"))),
            "residual": bool(detection_record.get("residual_triggered", False)),
            "ewma": bool(detection_record.get("ewma_triggered", False)),
            "spike": bool(detection_record.get("spike_triggered", False)),
            "current_delta": float(detection_record.get("current_delta_median", float("nan"))),
            "temperature_delta": float(detection_record.get("temperature_delta_median", float("nan"))),
        }
        if notification_time is not None and detection_start is not None:
            notification_time = max(notification_time, detection_start)
            triggers["notification_lag_minutes"] = (notification_time - detection_start).total_seconds() / 60.0
        else:
            triggers["notification_lag_minutes"] = None

    return StepwiseResult(
        well=well,
        interval_start=interval.start,
        interval_end=interval.end,
        detection_start=detection_start,
        notification_time=notification_time,
        evaluated_points=evaluated_points,
        delay_minutes=delay_minutes,
        triggers=triggers,
    )


def run_stepwise_evaluation(
    config: Dict,
    workbook: WorkbookSource,
    wells: Sequence[str],
) -> List[StepwiseResult]:
    if not wells:
        return []

    context = build_detection_context(config, workbook, use_streaming_calibration=True)
    target_intervals = [
        interval
        for interval in context.reference_intervals
        if interval.label == "anomaly" and interval.well in wells
    ]
    return evaluate_stepwise_for_intervals(context, target_intervals)


__all__ = [
    "DetectionContext",
    "StepwiseResult",
    "build_detection_context",
    "evaluate_stepwise_for_intervals",
    "run_stepwise_evaluation",
]
