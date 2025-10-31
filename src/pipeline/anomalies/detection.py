from __future__ import annotations

import math
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .models import (
    DetectionSettings,
    FrequencyBaseline,
    InterpretationThresholds,
    ReferenceInterval,
    ResidualDetectionModel,
    ResidualDetectionSettings,
)


def build_frequency_baseline(
    intervals: Sequence[ReferenceInterval],
    well_data: Dict[str, pd.DataFrame],
    *,
    metrics: Sequence[str],
    bin_width: float,
    min_points: int,
) -> Optional[FrequencyBaseline]:
    if not intervals or not well_data or bin_width <= 0:
        return None

    metric_buckets: Dict[str, Dict[int, List[float]]] = {metric: {} for metric in metrics}
    for interval in intervals:
        frame = well_data.get(interval.well)
        if frame is None or "Frequency" not in frame.columns:
            continue
        segment = frame.loc[interval.start : interval.end]
        if segment.empty:
            continue
        freq_series = pd.to_numeric(segment["Frequency"], errors="coerce")
        if freq_series.dropna().empty:
            continue
        for metric in metrics:
            if metric not in segment:
                continue
            values = pd.to_numeric(segment[metric], errors="coerce")
            mask = freq_series.notna() & values.notna()
            if not mask.any():
                continue
            freq_valid = freq_series[mask].to_numpy(dtype=float)
            values_valid = values[mask].to_numpy(dtype=float)
            bin_indices = np.floor(freq_valid / bin_width).astype(int)
            buckets = metric_buckets[metric]
            for bin_idx, value in zip(bin_indices, values_valid):
                if not np.isfinite(value):
                    continue
                buckets.setdefault(int(bin_idx), []).append(float(value))

    baseline_metrics: Dict[str, Dict[int, Dict[str, float]]] = {}
    records: List[Dict[str, float]] = []
    for metric, buckets in metric_buckets.items():
        metric_model: Dict[int, Dict[str, float]] = {}
        for bin_idx, values in buckets.items():
            if len(values) < min_points:
                continue
            array = np.asarray(values, dtype=float)
            median = float(np.median(array))
            mad = float(np.median(np.abs(array - median)) * 1.4826)
            if not np.isfinite(median):
                continue
            if not np.isfinite(mad) or mad <= 0.0:
                mad = 1e-6
            metric_model[bin_idx] = {"median": median, "mad": mad, "count": len(values)}
            records.append(
                {
                    "metric": metric,
                    "bin_index": bin_idx,
                    "frequency_center": (bin_idx + 0.5) * bin_width,
                    "count": len(values),
                    "median": median,
                    "mad": mad,
                }
            )
        if metric_model:
            baseline_metrics[metric] = metric_model

    if not baseline_metrics:
        return None

    summary = pd.DataFrame(records)
    summary.sort_values(["metric", "frequency_center"], inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return FrequencyBaseline(bin_width=bin_width, metrics=baseline_metrics, summary=summary)


def _apply_frequency_baseline(
    metric: str,
    series: pd.Series,
    frequency: pd.Series,
    baseline: Optional[FrequencyBaseline],
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    expected = pd.Series(np.nan, index=series.index, dtype=float)
    residual = pd.Series(series, dtype=float)
    scale = pd.Series(np.nan, index=series.index, dtype=float)
    if baseline is None or metric not in baseline.metrics or frequency.dropna().empty:
        return expected, residual, scale

    models = baseline.metrics.get(metric, {})
    if not models:
        return expected, residual, scale
    freq_values = pd.to_numeric(frequency, errors="coerce")
    if freq_values.dropna().empty:
        return expected, residual, scale
    bin_width = baseline.bin_width
    bin_indices = np.floor(freq_values / bin_width).astype("Int64")
    for bin_idx, stats in models.items():
        mask = bin_indices == bin_idx
        if mask.any():
            median = stats["median"]
            mad = stats.get("mad", 1e-6)
            if not np.isfinite(mad) or mad <= 0:
                mad = 1e-6
            expected.loc[mask] = median
            scale.loc[mask] = mad
    residual = series - expected
    missing_expected = expected.isna()
    if missing_expected.any():
        residual.loc[missing_expected] = series.loc[missing_expected]
    non_null_scale = scale.dropna()
    fallback_mad = (
        float(np.nanmedian(np.abs(series - np.nanmedian(series)))) if series.dropna().size else float("nan")
    )
    if not np.isfinite(fallback_mad) or fallback_mad <= 0:
        fallback_std = float(np.nanstd(series)) if series.dropna().size else float("nan")
        fallback_mad = fallback_std if np.isfinite(fallback_std) and fallback_std > 0 else 1.0
    fallback_mad *= 1.4826
    if non_null_scale.empty:
        scale = scale.fillna(fallback_mad)
    else:
        median_scale = float(non_null_scale.median())
        if not np.isfinite(median_scale) or median_scale <= 0:
            median_scale = fallback_mad
        scale = scale.fillna(median_scale)
    min_scale = max(fallback_mad * 0.1, 0.05)
    scale = scale.clip(lower=min_scale)
    return expected, residual, scale


def _chi2_ppf(probability: float, dof: int) -> float:
    probability = max(min(probability, 1 - 1e-9), 1e-9)
    if dof <= 0:
        raise ValueError("Degrees of freedom must be positive for chi-square quantile")
    z = NormalDist().inv_cdf(probability)
    term = 1 - (2.0 / (9.0 * dof)) + z * math.sqrt(2.0 / (9.0 * dof))
    term = max(term, 1e-9)
    return dof * term**3


def build_residual_detection_model(
    features_map: Dict[str, pd.DataFrame],
    normal_intervals: Sequence[ReferenceInterval],
    settings: ResidualDetectionSettings,
) -> Optional[ResidualDetectionModel]:
    if not settings.enabled or not normal_intervals:
        return None

    collected_frames: List[pd.DataFrame] = []
    for interval in normal_intervals:
        frame = features_map.get(interval.well)
        if frame is None:
            continue
        segment = frame.loc[interval.start : interval.end]
        if segment.empty:
            continue
        subset = segment[
            [
                "pressure_z",
                "current_z",
                "temperature_z",
            ]
        ]
        collected_frames.append(subset)

    if not collected_frames:
        return None

    combined = pd.concat(collected_frames, axis=0, ignore_index=False)
    metric_columns = {
        "pressure": "pressure_z",
        "current": "current_z",
        "temperature": "temperature_z",
    }

    active_metrics: List[str] = []
    for metric, column in metric_columns.items():
        if column in combined and combined[column].dropna().shape[0] >= settings.min_t2_points:
            active_metrics.append(metric)

    if not active_metrics:
        return None

    columns = [metric_columns[m] for m in active_metrics]
    z_matrix = combined[columns].dropna()
    if z_matrix.shape[0] < settings.min_t2_points:
        return None

    cov = np.cov(z_matrix.to_numpy().T, ddof=0)
    if not np.isfinite(cov).all():
        return None

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        epsilon = 1e-6 * np.eye(len(columns))
        inv_cov = np.linalg.inv(cov + epsilon)
        cov = cov + epsilon

    dof = len(columns)
    try:
        t2_threshold = float(_chi2_ppf(1 - settings.t2_alpha, dof))
    except Exception:
        t2_threshold = float(_chi2_ppf(0.99, dof))

    return ResidualDetectionModel(
        metrics=active_metrics,
        covariance=cov,
        inv_covariance=inv_cov,
        t2_threshold=t2_threshold,
        settings=settings,
    )


def _compute_ewma_series(series: pd.Series, lambda_: float) -> pd.Series:
    if series.dropna().empty:
        return pd.Series(np.nan, index=series.index)
    ewma_values = []
    prev = 0.0
    initialized = False
    for value in series:
        if not np.isfinite(value):
            ewma_values.append(np.nan)
            continue
        if not initialized:
            prev = value
            initialized = True
        else:
            prev = lambda_ * value + (1 - lambda_) * prev
        ewma_values.append(prev)
    return pd.Series(ewma_values, index=series.index, dtype=float)


def _compute_t2_series(
    z_frame: pd.DataFrame,
    model: ResidualDetectionModel,
) -> pd.Series:
    metrics = model.metrics
    column_map = {
        "pressure": "pressure_z",
        "current": "current_z",
        "temperature": "temperature_z",
    }
    columns = [column_map[m] for m in metrics]
    if not set(columns).issubset(z_frame.columns):
        return pd.Series(np.nan, index=z_frame.index, dtype=float)
    subset = z_frame[columns]
    values = subset.to_numpy(dtype=float)
    mask = np.isfinite(values).all(axis=1)
    t2 = pd.Series(np.nan, index=z_frame.index, dtype=float)
    if not mask.any():
        return t2
    valid_values = values[mask]
    inv = model.inv_covariance
    t2_values = np.einsum("ij,jk,ik->i", valid_values, inv, valid_values)
    t2.loc[mask] = t2_values
    return t2


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


def compute_feature_frame(
    well_df: pd.DataFrame,
    settings: DetectionSettings,
    baseline: Optional[FrequencyBaseline] = None,
) -> pd.DataFrame:
    def _compute_window_params(series: pd.Series) -> Tuple[int, int, int]:
        valid = series.dropna()
        if len(valid) < 2:
            return 1, 1, 1
        diffs = valid.index.to_series().diff().dropna().dt.total_seconds()
        if diffs.empty:
            median_seconds = 60.0
        else:
            median_seconds = float(diffs.median())
            if median_seconds <= 0:
                median_seconds = 60.0
        window_points = max(1, int(round((settings.window_minutes * 60) / median_seconds)))
        shift_points = max(1, int(round((settings.shift_minutes * 60) / median_seconds)))
        fraction = min(1.0, settings.min_samples / max(1, settings.window_minutes))
        min_periods = max(1, min(window_points, int(math.ceil(window_points * fraction))))
        return window_points, shift_points, min_periods

    features = pd.DataFrame(index=well_df.index)

    frequency_series = (
        pd.to_numeric(well_df["Frequency"], errors="coerce")
        if "Frequency" in well_df
        else pd.Series(np.nan, index=well_df.index)
    )

    pressure_raw = pd.to_numeric(well_df["Intake_Pressure"], errors="coerce")
    pressure_expected, pressure_residual, pressure_scale = _apply_frequency_baseline(
        "Intake_Pressure",
        pressure_raw,
        frequency_series,
        baseline,
    )
    pressure_base = pressure_residual
    pressure_window, pressure_shift, pressure_min_periods = _compute_window_params(pressure_base)
    rolling_pressure = pressure_base.rolling(window=pressure_window, min_periods=pressure_min_periods).mean()
    baseline_pressure = (
        pressure_base.shift(pressure_shift)
        .rolling(window=pressure_window, min_periods=pressure_min_periods)
        .mean()
    )
    baseline_pressure = baseline_pressure.reindex(rolling_pressure.index).interpolate(method="time")
    pressure_delta = _safe_divide(rolling_pressure - baseline_pressure, baseline_pressure)
    pressure_slope = pressure_base - pressure_base.shift(pressure_shift)

    features["pressure_delta"] = pressure_delta
    features["pressure_slope"] = pressure_slope

    current_raw = (
        pd.to_numeric(well_df["Current"], errors="coerce") if "Current" in well_df else pd.Series(np.nan, index=well_df.index)
    )
    current_expected = pd.Series(np.nan, index=well_df.index, dtype=float)
    current_residual = current_raw
    current_scale = pd.Series(np.nan, index=well_df.index, dtype=float)
    if "Current" in well_df.columns:
        if baseline is not None:
            ce, cr, cs = _apply_frequency_baseline("Current", current_raw, frequency_series, baseline)
            current_expected = ce
            current_residual = cr
            current_scale = cs
        current_window, current_shift, current_min_periods = _compute_window_params(current_residual)
        rolling_current = current_residual.rolling(window=current_window, min_periods=current_min_periods).mean()
        baseline_current = (
            current_residual.shift(current_shift)
            .rolling(window=current_window, min_periods=current_min_periods)
            .mean()
        )
        baseline_current = baseline_current.reindex(rolling_current.index).interpolate(method="time")
        features["current_delta"] = _safe_divide(rolling_current - baseline_current, baseline_current)
    else:
        features["current_delta"] = np.nan

    temperature_raw = (
        pd.to_numeric(well_df["Motor_Temperature"], errors="coerce")
        if "Motor_Temperature" in well_df
        else pd.Series(np.nan, index=well_df.index)
    )
    temperature_expected = pd.Series(np.nan, index=well_df.index, dtype=float)
    temperature_residual = temperature_raw
    temperature_scale = pd.Series(np.nan, index=well_df.index, dtype=float)
    if "Motor_Temperature" in well_df.columns:
        if baseline is not None:
            te, tr, ts = _apply_frequency_baseline("Motor_Temperature", temperature_raw, frequency_series, baseline)
            temperature_expected = te
            temperature_residual = tr
            temperature_scale = ts
        temp_window, temp_shift, temp_min_periods = _compute_window_params(temperature_residual)
        rolling_temperature = temperature_residual.rolling(window=temp_window, min_periods=temp_min_periods).mean()
        baseline_temperature = (
            temperature_residual.shift(temp_shift)
            .rolling(window=temp_window, min_periods=temp_min_periods)
            .mean()
        )
        baseline_temperature = baseline_temperature.reindex(rolling_temperature.index).interpolate(method="time")
        features["temperature_delta"] = _safe_divide(
            rolling_temperature - baseline_temperature, baseline_temperature
        )
    else:
        features["temperature_delta"] = np.nan

    features["pressure"] = pressure_raw
    features["pressure_expected"] = pressure_expected
    features["pressure_residual"] = pressure_residual
    features["pressure_scale"] = pressure_scale
    features["pressure_z"] = _safe_divide(pressure_residual, pressure_scale)
    features["current"] = current_raw
    features["current_expected"] = current_expected
    features["current_residual"] = current_residual
    features["current_scale"] = current_scale
    features["current_z"] = _safe_divide(current_residual, current_scale)
    features["temperature"] = temperature_raw
    features["temperature_expected"] = temperature_expected
    features["temperature_residual"] = temperature_residual
    features["temperature_scale"] = temperature_scale
    features["temperature_z"] = _safe_divide(temperature_residual, temperature_scale)
    features["frequency"] = frequency_series
    return features


def aggregate_feature_values(
    intervals: Sequence[ReferenceInterval],
    features_map: Dict[str, pd.DataFrame],
) -> Dict[str, pd.Series]:
    collected: Dict[str, List[pd.Series]] = {
        "pressure_delta": [],
        "pressure_slope": [],
        "current_delta": [],
        "temperature_delta": [],
    }
    for interval in intervals:
        frame = features_map.get(interval.well)
        if frame is None:
            continue
        segment = frame.loc[interval.start : interval.end]
        if segment.empty:
            continue
        for key in collected.keys():
            if key in segment:
                collected[key].append(segment[key].dropna())

    feature_limits = {
        "pressure_delta": 10.0,
        "pressure_slope": 20.0,
        "current_delta": 10.0,
        "temperature_delta": 10.0,
    }
    aggregated = {}
    for key, parts in collected.items():
        if parts:
            combined = pd.concat(parts).dropna()
            limit = feature_limits.get(key)
            if limit is not None and not combined.empty:
                combined = combined[combined.abs() <= limit]
            aggregated[key] = combined
        else:
            aggregated[key] = pd.Series(dtype=float)
    return aggregated


def derive_thresholds(
    normal_features: Dict[str, pd.Series],
    anomaly_features: Dict[str, pd.Series],
    settings: DetectionSettings,
) -> Dict[str, Dict[str, float]]:
    normal_delta_series = normal_features["pressure_delta"]
    normal_slope_series = normal_features["pressure_slope"]
    anomaly_delta_series = anomaly_features["pressure_delta"]
    anomaly_slope_series = anomaly_features["pressure_slope"]

    normal_delta_abs = normal_delta_series.abs() if not normal_delta_series.empty else pd.Series(dtype=float)
    normal_slope_abs = normal_slope_series.abs() if not normal_slope_series.empty else pd.Series(dtype=float)

    normal_delta_high = (
        float(normal_delta_abs.quantile(0.75, interpolation="linear")) if not normal_delta_abs.empty else 0.0
    )
    anomaly_delta_high = (
        float(anomaly_delta_series.abs().quantile(0.8, interpolation="linear"))
        if not anomaly_delta_series.empty
        else 0.0
    )
    delta_threshold = max(normal_delta_high * settings.delta_factor, anomaly_delta_high)

    normal_slope_high = (
        float(normal_slope_abs.quantile(0.9, interpolation="linear")) if not normal_slope_abs.empty else 0.0
    )
    anomaly_slope_high = (
        float(anomaly_slope_series.abs().quantile(0.7, interpolation="linear"))
        if not anomaly_slope_series.empty
        else 0.0
    )
    slope_threshold = max(normal_slope_high + settings.slope_margin, anomaly_slope_high)

    normal_delta_threshold = (
        float(normal_delta_abs.quantile(0.95, interpolation="linear")) if not normal_delta_abs.empty else 0.01
    )
    normal_slope_threshold = (
        float(normal_slope_abs.quantile(0.95, interpolation="linear")) if not normal_slope_abs.empty else 0.5
    )

    return {
        "anomaly": {
            "pressure_delta": delta_threshold,
            "pressure_slope": slope_threshold,
        },
        "normal": {
            "pressure_delta": normal_delta_threshold,
            "pressure_slope": normal_slope_threshold,
        },
    }


def _find_event_times(mask: pd.Series, min_gap: pd.Timedelta) -> List[pd.Timestamp]:
    events: List[pd.Timestamp] = []
    mask = mask.fillna(False)
    previous = False
    last_event_time: Optional[pd.Timestamp] = None
    for timestamp, is_true in mask.items():
        if pd.isna(timestamp):
            continue
        current = bool(is_true)
        if current and not previous:
            if last_event_time is None or timestamp - last_event_time >= min_gap:
                events.append(timestamp)
                last_event_time = timestamp
        previous = current
    return events


def classify_direction(delta_value: float, thresholds: InterpretationThresholds) -> str:
    if np.isnan(delta_value):
        return "?"
    if delta_value >= thresholds.strong_increase:
        return "↑↑"
    if delta_value >= thresholds.mild_increase:
        return "↑"
    if delta_value <= thresholds.strong_drop:
        return "↓↓"
    if delta_value <= thresholds.mild_drop:
        return "↓"
    return "="


def overlaps_interval(
    start_a: pd.Timestamp,
    end_a: pd.Timestamp,
    ref: ReferenceInterval,
) -> bool:
    return not (end_a < ref.start or start_a > ref.end)


def detect_segments_for_well(
    well: str,
    well_df: pd.DataFrame,
    features: pd.DataFrame,
    thresholds: Dict[str, Dict[str, float]],
    settings: DetectionSettings,
    interpretation: InterpretationThresholds,
    reference_intervals: List[ReferenceInterval],
    residual_model: Optional[ResidualDetectionModel] = None,
    holdout_wells: Optional[Set[str]] = None,
) -> List[Dict[str, object]]:
    gap = pd.Timedelta(minutes=settings.gap_minutes)

    anomaly_thresholds = thresholds["anomaly"]
    normal_thresholds = thresholds["normal"]

    slope_condition = features["pressure_slope"] >= anomaly_thresholds["pressure_slope"]
    delta_condition = features["pressure_delta"].abs() >= anomaly_thresholds["pressure_delta"]
    anomaly_mask = (delta_condition & slope_condition).fillna(False)

    reference_normal_mask = pd.Series(False, index=features.index)
    is_holdout = holdout_wells is not None and well in holdout_wells
    anomaly_refs: List[ReferenceInterval] = []
    normal_refs: List[ReferenceInterval] = []
    for ref in reference_intervals:
        if ref.well != well:
            continue
        if ref.label == "anomaly":
            anomaly_refs.append(ref)
        elif ref.label == "normal":
            normal_refs.append(ref)
            if not is_holdout:
                reference_normal_mask.loc[ref.start : ref.end] = True

    if not anomaly_refs:
        return []

    residual_flag = pd.Series(False, index=features.index, dtype=bool)
    ewma_flag_any = pd.Series(False, index=features.index, dtype=bool)
    spike_flag_any = pd.Series(False, index=features.index, dtype=bool)
    t2_flag = pd.Series(False, index=features.index, dtype=bool)
    t2_series = pd.Series(np.nan, index=features.index, dtype=float)

    metric_column_map = {
        "pressure": "pressure_z",
        "current": "current_z",
        "temperature": "temperature_z",
    }

    if residual_model is not None:
        ewma_flags: List[pd.Series] = []
        spike_flags: List[pd.Series] = []
        for metric in residual_model.metrics:
            column = metric_column_map.get(metric)
            if column is None or column not in features:
                continue
            z_series = pd.to_numeric(features[column], errors="coerce")
            ewma_series = _compute_ewma_series(z_series, residual_model.settings.ewma_lambda)
            features[f"{metric}_ewma"] = ewma_series
            ewma_flag_metric = ewma_series.abs() > residual_model.settings.ewma_l_multiplier
            ewma_flag_metric = ewma_flag_metric.fillna(False)
            ewma_flags.append(ewma_flag_metric)
            spike_flag_metric = z_series.abs() > residual_model.settings.spike_threshold
            spike_flag_metric = spike_flag_metric.fillna(False)
            spike_flags.append(spike_flag_metric)

        if ewma_flags:
            ewma_flag_any = ewma_flags[0].copy()
            for flag in ewma_flags[1:]:
                ewma_flag_any |= flag
        if spike_flags:
            spike_flag_any = spike_flags[0].copy()
            for flag in spike_flags[1:]:
                spike_flag_any |= flag

        z_frame = features[
            [metric_column_map[m] for m in residual_model.metrics if metric_column_map[m] in features]
        ]
        t2_series = _compute_t2_series(z_frame, residual_model)
        t2_flag = t2_series > residual_model.t2_threshold
        t2_flag = t2_flag.fillna(False)

        residual_flag = t2_flag.fillna(False)

    features["t2_stat"] = t2_series
    features["ewma_flag"] = ewma_flag_any
    features["spike_flag"] = spike_flag_any
    features["t2_flag"] = t2_flag

    residual_signal = (residual_flag & spike_flag_any & slope_condition.fillna(False)).fillna(False)
    if not is_holdout:
        residual_signal &= ~reference_normal_mask
    features["residual_flag"] = residual_signal

    anomaly_signal = (anomaly_mask.fillna(False) | residual_signal).fillna(False)
    event_times = _find_event_times(anomaly_signal, gap)

    def _locate_timestamp(frame: pd.DataFrame, timestamp: pd.Timestamp) -> Optional[pd.Timestamp]:
        if timestamp in frame.index:
            return timestamp
        indexer = frame.index.get_indexer([timestamp], method="nearest")
        if indexer.size == 0 or indexer[0] == -1:
            return None
        return frame.index[indexer[0]]

    def _find_nearest_unassigned_ref(
        target_time: pd.Timestamp, assigned: Set[Tuple[pd.Timestamp, pd.Timestamp]]
    ) -> Optional[ReferenceInterval]:
        nearest: Optional[ReferenceInterval] = None
        min_abs_diff: Optional[float] = None
        for ref in anomaly_refs:
            if ref.start is None or pd.isna(ref.start):
                continue
            ref_key = (ref.start, ref.end)
            if ref_key in assigned:
                continue
            diff = abs((target_time - ref.start).total_seconds())
            if min_abs_diff is None or diff < min_abs_diff:
                min_abs_diff = diff
                nearest = ref
        return nearest

    def _build_event_record(
        event_time: pd.Timestamp, assigned_refs: Set[Tuple[pd.Timestamp, pd.Timestamp]]
    ) -> Optional[Tuple[Dict[str, object], Optional[Tuple[pd.Timestamp, pd.Timestamp]]]]:
        located_time = _locate_timestamp(features, event_time)
        if located_time is None:
            return None
        located_data_time = _locate_timestamp(well_df, located_time)
        if located_data_time is None:
            located_data_time = located_time

        feature_row = features.loc[located_time]
        pressure_delta_value = float(
            pd.to_numeric(pd.Series([feature_row.get("pressure_delta")]), errors="coerce").iloc[0]
        )
        pressure_slope_value = float(
            pd.to_numeric(pd.Series([feature_row.get("pressure_slope")]), errors="coerce").iloc[0]
        )
        if np.isnan(pressure_delta_value) or np.isnan(pressure_slope_value):
            return None

        current_delta_value = float(
            pd.to_numeric(pd.Series([feature_row.get("current_delta")]), errors="coerce").iloc[0]
        )
        temperature_delta_value = float(
            pd.to_numeric(pd.Series([feature_row.get("temperature_delta")]), errors="coerce").iloc[0]
        )
        residual_triggered = bool(feature_row.get("residual_flag", False))
        ewma_triggered = bool(feature_row.get("ewma_flag", False))
        spike_triggered = bool(feature_row.get("spike_flag", False))
        t2_value = float(pd.to_numeric(pd.Series([feature_row.get("t2_stat")]), errors="coerce").iloc[0])

        well_row = well_df.loc[located_data_time] if located_data_time in well_df.index else None

        def _series_value(source: Optional[pd.Series], key: str) -> float:
            if source is None:
                return float("nan")
            if key not in source:
                return float("nan")
            return float(pd.to_numeric(pd.Series([source.get(key)]), errors="coerce").iloc[0])

        pressure_mean = _series_value(well_row, "Intake_Pressure")
        pressure_min_15m = _series_value(well_row, "Intake_Pressure_min")
        pressure_max_15m = _series_value(well_row, "Intake_Pressure_max")
        current_mean = _series_value(well_row, "Current")
        temperature_mean = _series_value(well_row, "Motor_Temperature")
        frequency_mean = _series_value(well_row, "Frequency")

        event_end = located_time
        duration_minutes = 0.0

        reference_match = "none"
        reference_notes: List[str] = []
        matched_ref_key: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        reference_start: Optional[pd.Timestamp] = None
        for ref in anomaly_refs:
            if ref.start <= located_time <= ref.end:
                reference_match = "anomaly"
                reference_notes = ref.notes
                matched_ref_key = (ref.start, ref.end)
                reference_start = ref.start
                break
        if reference_match == "none":
            for ref in normal_refs:
                if ref.start <= located_time <= ref.end:
                    reference_match = "normal"
                    reference_notes = ref.notes
                    break
        if reference_match != "anomaly":
            nearest_ref = _find_nearest_unassigned_ref(located_time, assigned_refs)
            if nearest_ref is not None:
                matched_ref_key = (nearest_ref.start, nearest_ref.end)
                reference_start = nearest_ref.start

        reference_delta_minutes = (
            float((located_time - reference_start).total_seconds() / 60.0)
            if reference_start is not None
            else float("nan")
        )

        score = (
            pressure_delta_value * pressure_slope_value
            if not np.isnan(pressure_delta_value) and not np.isnan(pressure_slope_value)
            else float("nan")
        )

        reference_start_value: Optional[pd.Timestamp] = (
            reference_start if reference_start is not None and pd.notna(reference_start) else None
        )

        record: Dict[str, object] = {
            "well": well,
            "start": located_time,
            "end": event_end,
            "duration_minutes": duration_minutes,
            "segment_type": "anomaly",
            "pressure_delta_median": pressure_delta_value,
            "pressure_slope_median": pressure_slope_value,
            "current_delta_median": current_delta_value,
            "temperature_delta_median": temperature_delta_value,
            "pressure_direction": classify_direction(pressure_delta_value, interpretation),
            "current_direction": classify_direction(current_delta_value, interpretation),
            "temperature_direction": classify_direction(temperature_delta_value, interpretation),
            "pressure_mean": pressure_mean,
            "pressure_min_15m": pressure_min_15m,
            "pressure_max_15m": pressure_max_15m,
            "ewma_triggered": ewma_triggered,
            "spike_triggered": spike_triggered,
            "t2_max": t2_value,
            "residual_triggered": residual_triggered,
            "current_mean": current_mean,
            "temperature_mean": temperature_mean,
            "frequency_mean": frequency_mean,
            "score": score,
            "reference_match": reference_match,
            "reference_notes": "; ".join(reference_notes) if reference_notes else "",
            "reference_start": reference_start_value,
            "reference_delta_minutes": reference_delta_minutes,
            "threshold_pressure_delta": anomaly_thresholds["pressure_delta"],
            "threshold_pressure_slope": anomaly_thresholds["pressure_slope"],
            "threshold_normal_pressure_delta": normal_thresholds["pressure_delta"],
            "threshold_normal_pressure_slope": normal_thresholds["pressure_slope"],
        }
        return record, matched_ref_key

    records: List[Dict[str, object]] = []
    recorded_times: List[pd.Timestamp] = []
    seen_ref_keys: Set[Tuple[pd.Timestamp, pd.Timestamp]] = set()
    total_ref_keys: Set[Tuple[pd.Timestamp, pd.Timestamp]] = {
        (ref.start, ref.end)
        for ref in anomaly_refs
        if ref.start is not None and not pd.isna(ref.start)
    }
    for timestamp in event_times:
        result = _build_event_record(timestamp, seen_ref_keys)
        if result:
            record, ref_key = result
            if ref_key and ref_key in seen_ref_keys:
                continue
            if ref_key is None and total_ref_keys and seen_ref_keys == total_ref_keys:
                continue
            records.append(record)
            recorded_times.append(record["start"])
            if ref_key:
                seen_ref_keys.add(ref_key)

    if not is_holdout:
        for ref in anomaly_refs:
            if any(ref.start <= ts <= ref.end for ts in recorded_times):
                continue
            supplemental = _build_event_record(ref.start, seen_ref_keys)
            if supplemental:
                record, ref_key = supplemental
                if ref_key and ref_key in seen_ref_keys:
                    continue
                if ref_key is None and total_ref_keys and seen_ref_keys == total_ref_keys:
                    continue
                records.append(record)
                recorded_times.append(record["start"])
                if ref_key:
                    seen_ref_keys.add(ref_key)

    return records


__all__ = [
    "aggregate_feature_values",
    "build_frequency_baseline",
    "build_residual_detection_model",
    "classify_direction",
    "compute_feature_frame",
    "derive_thresholds",
    "detect_segments_for_well",
    "overlaps_interval",
]
