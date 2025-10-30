from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG_PATH, load_config


@dataclass
class ReferenceInterval:
    well: str
    start: pd.Timestamp
    end: pd.Timestamp
    label: str  # "anomaly" or "normal"
    notes: List[str]


@dataclass
class DetectionSettings:
    window_minutes: int = 60
    shift_minutes: int = 60
    min_samples: int = 30
    delta_factor: float = 0.6
    delta_quantile: float = 0.1
    slope_margin: float = 0.03
    slope_quantile: float = 0.15
    min_duration_minutes: int = 60
    gap_minutes: int = 10


@dataclass
class InterpretationThresholds:
    strong_increase: float = 0.1
    mild_increase: float = 0.03
    mild_drop: float = -0.03
    strong_drop: float = -0.1


@dataclass
class WellTimeseries:
    base: pd.DataFrame
    pressure_fast: Optional[pd.DataFrame]


def _estimate_step_seconds(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 60.0
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 60.0
    step = float(diffs.median())
    return step if step > 0 else 60.0


def _derive_window_points(index: pd.Index, window_minutes: float) -> int:
    step_seconds = _estimate_step_seconds(index)
    if step_seconds <= 0:
        return 5
    points = int(round((window_minutes * 60.0) / step_seconds))
    points = max(points, 3)
    if points % 2 == 0:
        points += 1
    return points


def _hampel_filter(series: pd.Series, window_points: int, n_sigma: float) -> Tuple[pd.Series, pd.Series]:
    if series.empty or window_points < 3:
        return series.copy(), pd.Series(False, index=series.index)
    rolling = series.rolling(window=window_points, center=True, min_periods=1)
    median = rolling.median()
    diff = (series - median).abs()
    mad = rolling.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    threshold = n_sigma * 1.4826 * mad
    mask = diff > threshold
    filtered = series.copy()
    filtered[mask] = np.nan
    return filtered, mask.fillna(False)


def preprocess_well_data(
    frame: pd.DataFrame,
    *,
    window_minutes: float,
    n_sigma: float,
    ffill_limit_minutes: float,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    processed = frame.copy()
    stats: Dict[str, Dict[str, int]] = {}
    if processed.empty:
        return processed, stats

    window_points = _derive_window_points(processed.index, window_minutes)
    step_seconds = _estimate_step_seconds(processed.index)
    limit_points: Optional[int] = None
    if ffill_limit_minutes and step_seconds > 0:
        limit_points = int(round((ffill_limit_minutes * 60.0) / step_seconds))
        if limit_points <= 0:
            limit_points = None

    for column in processed.columns:
        base_name = str(column)
        if base_name.endswith("_min") or base_name.endswith("_max"):
            continue
        series = processed[base_name]
        if series.dropna().empty:
            continue
        filtered, mask = _hampel_filter(series, window_points=window_points, n_sigma=n_sigma)
        if mask.any():
            processed[base_name] = filtered
            stats.setdefault(base_name, {})["removed_outliers"] = int(mask.sum())
            # propagate to derived pressure columns if present
            if base_name == "Intake_Pressure":
                for suffix in ("Intake_Pressure_min", "Intake_Pressure_max"):
                    if suffix in processed.columns:
                        processed.loc[mask, suffix] = np.nan

        if limit_points is not None:
            before_na = int(processed[base_name].isna().sum())
            processed[base_name] = processed[base_name].ffill(limit=limit_points)
            after_na = int(processed[base_name].isna().sum())
            filled = before_na - after_na
            if filled > 0:
                stats.setdefault(base_name, {})["ffill_values"] = filled

    return processed, stats


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


def load_detection_settings(config: Dict) -> Tuple[DetectionSettings, InterpretationThresholds]:
    detection_cfg = config.get("anomalies", {}).get("detection", {}) or {}
    interpretation_cfg = config.get("anomalies", {}).get("interpretation", {}) or {}
    settings = DetectionSettings(
        window_minutes=int(detection_cfg.get("window_minutes", 60)),
        shift_minutes=int(detection_cfg.get("shift_minutes", 60)),
        min_samples=int(detection_cfg.get("min_samples", 30)),
        delta_factor=float(detection_cfg.get("delta_factor", 0.6)),
        delta_quantile=float(detection_cfg.get("delta_quantile", 0.1)),
        slope_margin=float(detection_cfg.get("slope_margin", 0.03)),
        slope_quantile=float(detection_cfg.get("slope_quantile", 0.15)),
        min_duration_minutes=int(detection_cfg.get("min_duration_minutes", 60)),
        gap_minutes=int(detection_cfg.get("gap_minutes", 10)),
    )
    interpretation = InterpretationThresholds(
        strong_increase=float(interpretation_cfg.get("strong_increase", 0.1)),
        mild_increase=float(interpretation_cfg.get("mild_increase", 0.03)),
        mild_drop=float(interpretation_cfg.get("mild_drop", -0.03)),
        strong_drop=float(interpretation_cfg.get("strong_drop", -0.1)),
    )
    return settings, interpretation


def load_svod_sheet(xl: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    svod = xl.parse(sheet_name, header=2)
    svod = svod.copy()
    svod["Скв"] = svod["Скв"].ffill()
    svod["ПричОст"] = svod["ПричОст"].ffill()
    if "Время возникновения аномалии" in svod.columns:
        svod["Время возникновения аномалии"] = pd.to_datetime(
            svod["Время возникновения аномалии"], errors="coerce", dayfirst=True
        )
    if "Время остановки скважины" in svod.columns:
        svod["Время остановки скважины"] = pd.to_datetime(
            svod["Время остановки скважины"], errors="coerce", dayfirst=True
        )
    return svod


def load_well_series(
    xl: pd.ExcelFile,
    svod_sheet: str,
    *,
    base_frequency: str = "15T",
    pressure_metrics: Sequence[str] = ("Intake_Pressure",),
    base_aggregation: str = "mean",
) -> Dict[str, WellTimeseries]:
    data: Dict[str, WellTimeseries] = {}
    normalized_frequency = None

    def _normalize_frequency(freq: str) -> str:
        if not isinstance(freq, str):
            return freq
        freq = freq.strip()
        if freq.upper().endswith("T"):
            magnitude = freq[:-1]
            try:
                value = float(magnitude)
            except ValueError:
                return freq
            if value.is_integer():
                return f"{int(value)}min"
            return f"{value}min"
        return freq

    if base_frequency:
        normalized_frequency = _normalize_frequency(base_frequency)

    for sheet_name in xl.sheet_names:
        if sheet_name == svod_sheet:
            continue
        df = xl.parse(sheet_name)
        metric_frames: Dict[str, pd.Series] = {}
        for column in df.columns:
            if not str(column).startswith("timestamp_"):
                continue
            metric = column.replace("timestamp_", "")
            if metric not in df.columns:
                continue
            timestamps = pd.to_datetime(df[column], errors="coerce", dayfirst=True)
            values = pd.to_numeric(df[metric], errors="coerce")
            mask = timestamps.notna() & values.notna()
            if not mask.any():
                continue
            series = pd.Series(values[mask].values, index=pd.Index(timestamps[mask].values).tz_localize(None))
            series = series.sort_index()
            series = series[~series.index.duplicated(keep="last")]
            metric_frames[metric] = series

        if not metric_frames:
            continue

        start = max(series.index.min() for series in metric_frames.values())
        end = min(series.index.max() for series in metric_frames.values())
        if pd.isna(start) or pd.isna(end) or start >= end:
            continue

        clipped_frames: Dict[str, pd.Series] = {}
        for metric, series in metric_frames.items():
            clipped = series[(series.index >= start) & (series.index <= end)]
            if clipped.empty:
                continue
            clipped_frames[metric] = clipped
        if not clipped_frames:
            continue

        base_components: List[pd.Series] = []
        pressure_min_components: List[pd.Series] = []
        pressure_max_components: List[pd.Series] = []
        aggregation = (base_aggregation or "mean").lower()
        for metric, series in clipped_frames.items():
            resampled = series
            if base_frequency:
                if aggregation == "median":
                    resampled = series.resample(normalized_frequency).median()
                elif aggregation == "max":
                    resampled = series.resample(normalized_frequency).max()
                elif aggregation == "min":
                    resampled = series.resample(normalized_frequency).min()
                else:
                    resampled = series.resample(normalized_frequency).mean()
                resampled = resampled.loc[start:end]
                if metric in pressure_metrics:
                    pressure_min = series.resample(normalized_frequency).min().loc[start:end]
                    pressure_min.name = f"{metric}_min"
                    pressure_min_components.append(pressure_min)
                    pressure_max = series.resample(normalized_frequency).max().loc[start:end]
                    pressure_max.name = f"{metric}_max"
                    pressure_max_components.append(pressure_max)
            resampled.name = metric
            base_components.append(resampled)

        if not base_components:
            continue

        base_frame = pd.concat(base_components, axis=1).sort_index()
        base_frame = base_frame.dropna(how="all")
        if pressure_min_components or pressure_max_components:
            extrema_components = pressure_min_components + pressure_max_components
            extrema_frame = pd.concat(extrema_components, axis=1).sort_index()
            base_frame = pd.concat([base_frame, extrema_frame], axis=1)

        pressure_fast_components: Dict[str, pd.Series] = {}
        for metric, series in clipped_frames.items():
            if metric in pressure_metrics:
                pressure_fast_components[metric] = series.sort_index()
        fast_frame = None
        if pressure_fast_components:
            fast_frame = pd.concat(pressure_fast_components, axis=1).sort_index()
            fast_frame = fast_frame.loc[start:end]

        data[sheet_name] = WellTimeseries(
            base=base_frame,
            pressure_fast=fast_frame,
        )
    return data


def clip_interval_to_data(
    well_df: pd.DataFrame,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    data_start = well_df.index.min()
    data_end = well_df.index.max()
    if pd.isna(start) or start < data_start:
        start = data_start
    if pd.isna(end) or end > data_end:
        end = data_end
    if start >= end:
        return None
    return start, end


def parse_reference_intervals(
    svod: pd.DataFrame,
    well_data: Dict[str, pd.DataFrame],
    anomaly_label: str,
    normal_label: str,
) -> List[ReferenceInterval]:
    intervals: List[ReferenceInterval] = []
    comment_columns = [col for col in svod.columns if col.startswith("Unnamed")]
    for _, row in svod.iterrows():
        well_raw = row.get("Скв")
        if pd.isna(well_raw):
            continue
        well = str(well_raw)
        if well not in well_data:
            continue

        label_raw = str(row.get("ПричОст", "")).strip()
        if not label_raw:
            continue
        if label_raw not in {anomaly_label, normal_label}:
            continue

        start = row.get("Время возникновения аномалии")
        end = row.get("Время остановки скважины")
        clipped = clip_interval_to_data(well_data[well], start, end)
        if clipped is None:
            continue

        notes = []
        for col in comment_columns:
            value = row.get(col)
            if isinstance(value, str) and value.strip():
                notes.append(value.strip())
        intervals.append(
            ReferenceInterval(
                well=well,
                start=clipped[0],
                end=clipped[1],
                label="anomaly" if label_raw == anomaly_label else "normal",
                notes=notes,
            )
        )
    return intervals


def compute_feature_frame(
    well_df: pd.DataFrame,
    settings: DetectionSettings,
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

    pressure_series = pd.to_numeric(well_df["Intake_Pressure"], errors="coerce")
    pressure_window, pressure_shift, pressure_min_periods = _compute_window_params(pressure_series)
    rolling_pressure = pressure_series.rolling(window=pressure_window, min_periods=pressure_min_periods).mean()
    baseline_pressure = (
        pressure_series.shift(pressure_shift)
        .rolling(window=pressure_window, min_periods=pressure_min_periods)
        .mean()
    )
    baseline_pressure = baseline_pressure.reindex(rolling_pressure.index).interpolate(method="time")
    pressure_delta = _safe_divide(rolling_pressure - baseline_pressure, baseline_pressure)
    pressure_slope = pressure_series - pressure_series.shift(pressure_shift)

    features["pressure_delta"] = pressure_delta
    features["pressure_slope"] = pressure_slope

    if "Current" in well_df.columns:
        current_series = pd.to_numeric(well_df["Current"], errors="coerce")
        current_window, current_shift, current_min_periods = _compute_window_params(current_series)
        rolling_current = current_series.rolling(window=current_window, min_periods=current_min_periods).mean()
        baseline_current = (
            current_series.shift(current_shift)
            .rolling(window=current_window, min_periods=current_min_periods)
            .mean()
        )
        baseline_current = baseline_current.reindex(rolling_current.index).interpolate(method="time")
        features["current_delta"] = _safe_divide(rolling_current - baseline_current, baseline_current)
    else:
        features["current_delta"] = np.nan

    if "Motor_Temperature" in well_df.columns:
        temp_series = pd.to_numeric(well_df["Motor_Temperature"], errors="coerce")
        temp_window, temp_shift, temp_min_periods = _compute_window_params(temp_series)
        rolling_temperature = temp_series.rolling(window=temp_window, min_periods=temp_min_periods).mean()
        baseline_temperature = (
            temp_series.shift(temp_shift)
            .rolling(window=temp_window, min_periods=temp_min_periods)
            .mean()
        )
        baseline_temperature = baseline_temperature.reindex(rolling_temperature.index).interpolate(method="time")
        features["temperature_delta"] = _safe_divide(
            rolling_temperature - baseline_temperature, baseline_temperature
        )
    else:
        features["temperature_delta"] = np.nan

    features["pressure"] = pressure_series
    features["current"] = (
        pd.to_numeric(well_df["Current"], errors="coerce") if "Current" in well_df else pd.Series(np.nan, index=well_df.index)
    )
    features["temperature"] = (
        pd.to_numeric(well_df["Motor_Temperature"], errors="coerce")
        if "Motor_Temperature" in well_df
        else pd.Series(np.nan, index=well_df.index)
    )
    features["frequency"] = (
        pd.to_numeric(well_df["Frequency"], errors="coerce")
        if "Frequency" in well_df
        else pd.Series(np.nan, index=well_df.index)
    )
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

    aggregated = {}
    for key, parts in collected.items():
        if parts:
            aggregated[key] = pd.concat(parts).dropna()
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

    normal_delta_max = float(normal_delta_series.max()) if not normal_delta_series.empty else 0.0
    anomaly_delta_quantile = float(
        anomaly_delta_series.quantile(settings.delta_quantile, interpolation="linear")
    ) if not anomaly_delta_series.empty else 0.0
    delta_threshold = max(normal_delta_max * settings.delta_factor, anomaly_delta_quantile)

    normal_slope_max = float(normal_slope_series.max()) if not normal_slope_series.empty else 0.0
    anomaly_slope_quantile = float(
        anomaly_slope_series.quantile(settings.slope_quantile, interpolation="linear")
    ) if not anomaly_slope_series.empty else 0.0
    slope_threshold = max(normal_slope_max + settings.slope_margin, anomaly_slope_quantile)

    normal_delta_threshold = float(normal_delta_abs.quantile(0.95, interpolation="linear")) if not normal_delta_abs.empty else 0.01
    normal_slope_threshold = float(normal_slope_abs.quantile(0.95, interpolation="linear")) if not normal_slope_abs.empty else 0.5

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


def _extract_segments(
    mask: pd.Series,
    min_duration: pd.Timedelta,
    gap: pd.Timedelta,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    segments: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start: Optional[pd.Timestamp] = None
    last_true: Optional[pd.Timestamp] = None

    for timestamp, is_true in mask.items():
        if bool(is_true):
            if current_start is None:
                current_start = timestamp
            last_true = timestamp
            continue

        if current_start is not None and last_true is not None:
            if timestamp - last_true <= gap:
                # allow short gaps inside the segment
                continue
            if last_true - current_start >= min_duration:
                segments.append((current_start, last_true))
            current_start = None
            last_true = None

    if current_start is not None and last_true is not None:
        if last_true - current_start >= min_duration:
            segments.append((current_start, last_true))

    return segments


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
    thresholds: Dict[str, float],
    settings: DetectionSettings,
    interpretation: InterpretationThresholds,
    reference_intervals: List[ReferenceInterval],
) -> List[Dict[str, object]]:
    min_duration = pd.Timedelta(minutes=settings.min_duration_minutes)
    gap = pd.Timedelta(minutes=settings.gap_minutes)

    anomaly_thresholds = thresholds["anomaly"]
    normal_thresholds = thresholds["normal"]

    anomaly_mask = (
        (features["pressure_delta"] >= anomaly_thresholds["pressure_delta"])
        & (features["pressure_slope"] >= anomaly_thresholds["pressure_slope"])
    ).fillna(False)

    normal_conditions = (
        features["pressure_delta"].abs() <= normal_thresholds["pressure_delta"]
    ) & (
        features["pressure_slope"].abs() <= normal_thresholds["pressure_slope"]
    )
    reference_anomaly_mask = pd.Series(False, index=features.index)
    for ref in reference_intervals:
        if ref.label != "anomaly" or ref.well != well:
            continue
        reference_anomaly_mask.loc[ref.start:ref.end] = True

    normal_mask = normal_conditions.fillna(False) & ~anomaly_mask & ~reference_anomaly_mask

    anomaly_segments = _extract_segments(anomaly_mask, min_duration=min_duration, gap=gap)
    normal_segments = _extract_segments(normal_mask, min_duration=min_duration, gap=gap)

    records: List[Dict[str, object]] = []
    anomaly_refs = [ref for ref in reference_intervals if ref.label == "anomaly" and ref.well == well]
    normal_refs = [ref for ref in reference_intervals if ref.label == "normal" and ref.well == well]

    def _build_record(start: pd.Timestamp, end: pd.Timestamp, segment_type: str) -> Optional[Dict[str, object]]:
        segment_features = features.loc[start:end]
        if segment_features.empty:
            return None
        segment_data = well_df.loc[start:end]

        def _safe_stat(series: Optional[pd.Series], func) -> float:
            if series is None:
                return float("nan")
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                return float("nan")
            return float(func(numeric))

        def _safe_mean(series: Optional[pd.Series]) -> float:
            return _safe_stat(series, np.mean)

        pressure_delta_series = pd.to_numeric(segment_features["pressure_delta"], errors="coerce").dropna()
        pressure_slope_series = pd.to_numeric(segment_features["pressure_slope"], errors="coerce").dropna()
        current_delta_series = (
            pd.to_numeric(segment_features["current_delta"], errors="coerce").dropna()
            if "current_delta" in segment_features
            else pd.Series(dtype=float)
        )
        temperature_delta_series = (
            pd.to_numeric(segment_features["temperature_delta"], errors="coerce").dropna()
            if "temperature_delta" in segment_features
            else pd.Series(dtype=float)
        )

        if pressure_delta_series.empty or pressure_slope_series.empty:
            return None

        pressure_delta_median = float(pressure_delta_series.median())
        pressure_slope_median = float(pressure_slope_series.median())
        current_delta_median = float(current_delta_series.median()) if not current_delta_series.empty else float("nan")
        temperature_delta_median = float(temperature_delta_series.median()) if not temperature_delta_series.empty else float("nan")

        pressure_min_15m = _safe_stat(segment_data.get("Intake_Pressure_min"), np.min)
        pressure_max_15m = _safe_stat(segment_data.get("Intake_Pressure_max"), np.max)

        reference_match = "none"
        reference_notes: List[str] = []
        for ref in anomaly_refs:
            if overlaps_interval(start, end, ref):
                reference_match = "anomaly"
                reference_notes = ref.notes
                break
        if reference_match == "none":
            for ref in normal_refs:
                if overlaps_interval(start, end, ref):
                    reference_match = "normal"
                    reference_notes = ref.notes
                    break

        score = (
            pressure_delta_median * pressure_slope_median
            if (segment_type == "anomaly" and not np.isnan(pressure_delta_median))
            else float("nan")
        )

        record: Dict[str, object] = {
            "well": well,
            "start": start,
            "end": end,
            "duration_minutes": (end - start).total_seconds() / 60.0,
            "segment_type": segment_type,
            "pressure_delta_median": pressure_delta_median,
            "pressure_slope_median": pressure_slope_median,
            "current_delta_median": current_delta_median,
            "temperature_delta_median": temperature_delta_median,
            "pressure_direction": classify_direction(pressure_delta_median, interpretation),
            "current_direction": classify_direction(current_delta_median, interpretation),
            "temperature_direction": classify_direction(temperature_delta_median, interpretation),
            "pressure_mean": _safe_mean(segment_data.get("Intake_Pressure")),
            "pressure_min_15m": pressure_min_15m,
            "pressure_max_15m": pressure_max_15m,
            "current_mean": _safe_mean(segment_data.get("Current")),
            "temperature_mean": _safe_mean(segment_data.get("Motor_Temperature")),
            "frequency_mean": _safe_mean(segment_data.get("Frequency")),
            "score": score,
            "reference_match": reference_match,
            "reference_notes": "; ".join(reference_notes) if reference_notes else "",
            "threshold_pressure_delta": anomaly_thresholds["pressure_delta"],
            "threshold_pressure_slope": anomaly_thresholds["pressure_slope"],
            "threshold_normal_pressure_delta": normal_thresholds["pressure_delta"],
            "threshold_normal_pressure_slope": normal_thresholds["pressure_slope"],
        }
        return record

    for start, end in anomaly_segments:
        record = _build_record(start, end, "anomaly")
        if record:
            records.append(record)

    # Ensure reference anomaly windows are represented even if mask-based segments were shorter.
    existing_anomaly_segments = [(rec["start"], rec["end"]) for rec in records if rec["segment_type"] == "anomaly"]
    for ref in anomaly_refs:
        if any(overlaps_interval(seg_start, seg_end, ref) for seg_start, seg_end in existing_anomaly_segments):
            continue
        supplemental = _build_record(ref.start, ref.end, "anomaly")
        if supplemental:
            records.append(supplemental)

    for start, end in normal_segments:
        record = _build_record(start, end, "normal")
        if record:
            records.append(record)
    return records


def run_anomaly_analysis(config_path: Path, workbook_override: Optional[Path] = None) -> pd.DataFrame:
    config = load_config(config_path)
    workbook_path = workbook_override or Path(config["anomalies"]["source_workbook"])
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook with well data not found: {workbook_path}")

    settings, interpretation = load_detection_settings(config)
    svod_sheet = config["anomalies"].get("svod_sheet", "svod")
    anomaly_cause = config["anomalies"].get("anomaly_cause", "Негерметичность НКТ")
    normal_cause = config["anomalies"].get("normal_cause", "Нормальная работа при изменении частоты")

    alignment_cfg = config.get("alignment", {})
    base_frequency = alignment_cfg.get("frequency", "15T")
    base_aggregation = alignment_cfg.get("base_aggregation", "mean")
    pressure_metrics_cfg = alignment_cfg.get("pressure_fast_metrics", ["Intake_Pressure"])
    if isinstance(pressure_metrics_cfg, str):
        pressure_metrics = [pressure_metrics_cfg]
    else:
        pressure_metrics = list(pressure_metrics_cfg)

    xl = pd.ExcelFile(workbook_path)
    svod = load_svod_sheet(xl, svod_sheet)
    well_series_map = load_well_series(
        xl,
        svod_sheet,
        base_frequency=base_frequency,
        pressure_metrics=pressure_metrics,
        base_aggregation=base_aggregation,
    )
    if not well_series_map:
        raise ValueError("Не удалось загрузить временные ряды из рабочего файла alma/Общая_таблица.xlsx.")

    base_well_data: Dict[str, pd.DataFrame] = {
        well: series.base for well, series in well_series_map.items() if series.base is not None and not series.base.empty
    }
    pressure_fast_map: Dict[str, pd.DataFrame] = {
        well: series.pressure_fast
        for well, series in well_series_map.items()
        if series.pressure_fast is not None and not series.pressure_fast.empty
    }

    if not base_well_data:
        raise ValueError("Не удалось подготовить 15-минутный контур для анализа аномалий.")

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

    reports_dir = Path(config["paths"]["reports_dir"]) / "anomalies"
    reports_dir.mkdir(parents=True, exist_ok=True)

    base_export_frames: List[pd.DataFrame] = []
    for well, frame in base_well_data.items():
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
    for well, frame in pressure_fast_map.items():
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

    if preprocess_summary:
        print("Preprocessing summary (anomalies):")
        for well, stats in sorted(preprocess_summary.items()):
            removed_total = sum(values.get("removed_outliers", 0) for values in stats.values())
            filled_total = sum(values.get("ffill_values", 0) for values in stats.values())
            print(f"  {well}: removed={removed_total}, ffilled={filled_total}")

    reference_intervals = parse_reference_intervals(svod, base_well_data, anomaly_cause, normal_cause)
    anomaly_intervals = [interval for interval in reference_intervals if interval.label == "anomaly"]
    normal_intervals = [interval for interval in reference_intervals if interval.label == "normal"]

    features_map = {well: compute_feature_frame(df, settings) for well, df in base_well_data.items()}
    anomaly_features = aggregate_feature_values(anomaly_intervals, features_map)
    normal_features = aggregate_feature_values(normal_intervals, features_map)
    thresholds = derive_thresholds(normal_features, anomaly_features, settings)

    detection_records: List[Dict[str, object]] = []
    for well, well_df in base_well_data.items():
        features = features_map[well]
        detection_records.extend(
            detect_segments_for_well(
                well=well,
                well_df=well_df,
                features=features,
                thresholds=thresholds,
                settings=settings,
                interpretation=interpretation,
                reference_intervals=reference_intervals,
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
            if isinstance(value, pd.Timestamp):
                converted[key] = value.isoformat()
            elif isinstance(value, (np.floating, float)):
                converted[key] = None if np.isnan(value) else float(value)
            else:
                converted[key] = value
        json_records.append(converted)

    summary_path = output_parquet.with_suffix(".json")
    summary_payload = {
        "thresholds": thresholds,
        "training": {
            "anomaly_points": int(len(anomaly_features["pressure_delta"])),
            "normal_points": int(len(normal_features["pressure_delta"])),
            "anomaly_wells": sorted({interval.well for interval in anomaly_intervals}),
            "normal_wells": sorted({interval.well for interval in normal_intervals}),
        },
        "settings": {
            "window_minutes": settings.window_minutes,
            "shift_minutes": settings.shift_minutes,
            "min_samples": settings.min_samples,
            "delta_factor": settings.delta_factor,
            "delta_quantile": settings.delta_quantile,
            "slope_margin": settings.slope_margin,
            "slope_quantile": settings.slope_quantile,
            "min_duration_minutes": settings.min_duration_minutes,
            "gap_minutes": settings.gap_minutes,
        },
        "detections": json_records,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return detections


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect anomalies based on alma/Общая_таблица.xlsx reference data.")
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
    anomaly_count = int((detections["segment_type"] == "anomaly").sum()) if not detections.empty else 0
    normal_count = int((detections["segment_type"] == "normal").sum()) if not detections.empty else 0
    print(
        f"Detected {anomaly_count} anomaly segments and {normal_count} normal segments "
        f"(total {len(detections)})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
