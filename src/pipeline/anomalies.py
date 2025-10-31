from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import math

import numpy as np
import pandas as pd
from statistics import NormalDist

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


@dataclass
class FrequencyBaseline:
    bin_width: float
    metrics: Dict[str, Dict[int, Dict[str, float]]]
    summary: pd.DataFrame


@dataclass
class ResidualDetectionSettings:
    enabled: bool = True
    ewma_lambda: float = 0.2
    ewma_l_multiplier: float = 3.0
    spike_threshold: float = 4.0
    t2_alpha: float = 0.01
    min_t2_points: int = 30


@dataclass
class ResidualDetectionModel:
    metrics: List[str]
    covariance: np.ndarray
    inv_covariance: np.ndarray
    t2_threshold: float
    settings: ResidualDetectionSettings


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

    metric_buckets: Dict[str, Dict[int, List[float]]] = {
        metric: {} for metric in metrics
    }
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
    fallback_mad = float(np.nanmedian(np.abs(series - np.nanmedian(series)))) if series.dropna().size else float("nan")
    if not np.isfinite(fallback_mad) or fallback_mad <= 0:
        fallback_std = float(np.nanstd(series)) if series.dropna().size else float("nan")
        fallback_mad = fallback_std if np.isfinite(fallback_std) and fallback_std > 0 else 1.0
    fallback_mad *= 1.4826  # consistent with MAD scale
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
    return dof * term ** 3


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

    # regularize covariance if needed
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


def load_residual_settings(config: Dict) -> ResidualDetectionSettings:
    residual_cfg = config.get("anomalies", {}).get("detection_residual", {}) or {}
    return ResidualDetectionSettings(
        enabled=bool(residual_cfg.get("enabled", True)),
        ewma_lambda=float(residual_cfg.get("ewma_lambda", 0.2)),
        ewma_l_multiplier=float(residual_cfg.get("ewma_l_multiplier", 3.0)),
        spike_threshold=float(residual_cfg.get("spike_threshold", 4.0)),
        t2_alpha=float(residual_cfg.get("t2_alpha", 0.01)),
        min_t2_points=int(residual_cfg.get("min_t2_points", 30)),
    )


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

    normal_delta_abs = (
        normal_delta_series.abs() if not normal_delta_series.empty else pd.Series(dtype=float)
    )
    normal_slope_abs = (
        normal_slope_series.abs() if not normal_slope_series.empty else pd.Series(dtype=float)
    )

    normal_delta_high = float(
        normal_delta_abs.quantile(0.75, interpolation="linear")
    ) if not normal_delta_abs.empty else 0.0
    anomaly_delta_high = float(
        anomaly_delta_series.abs().quantile(0.8, interpolation="linear")
    ) if not anomaly_delta_series.empty else 0.0
    delta_threshold = max(normal_delta_high * settings.delta_factor, anomaly_delta_high)

    normal_slope_high = float(
        normal_slope_abs.quantile(0.9, interpolation="linear")
    ) if not normal_slope_abs.empty else 0.0
    anomaly_slope_high = float(
        anomaly_slope_series.abs().quantile(0.7, interpolation="linear")
    ) if not anomaly_slope_series.empty else 0.0
    slope_threshold = max(normal_slope_high + settings.slope_margin, anomaly_slope_high)

    normal_delta_threshold = float(
        normal_delta_abs.quantile(0.95, interpolation="linear")
    ) if not normal_delta_abs.empty else 0.01
    normal_slope_threshold = float(
        normal_slope_abs.quantile(0.95, interpolation="linear")
    ) if not normal_slope_abs.empty else 0.5

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
    thresholds: Dict[str, float],
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
                reference_normal_mask.loc[ref.start:ref.end] = True

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

        z_frame = features[[metric_column_map[m] for m in residual_model.metrics if metric_column_map[m] in features]]
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

    def _find_nearest_unassigned_ref(target_time: pd.Timestamp, assigned: Set[Tuple[pd.Timestamp, pd.Timestamp]]) -> Optional[ReferenceInterval]:
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

        reference_start_value: Optional[pd.Timestamp] = reference_start if reference_start is not None and pd.notna(reference_start) else None

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

    holdout_wells = set(config["anomalies"].get("holdout_wells", []) or [])
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
        if baseline is None:
            print("Frequency baseline: недостаточно данных для построения модели.")
        else:
            baseline_path = reports_dir / "frequency_baseline.parquet"
            baseline.summary.to_parquet(baseline_path, index=False)
            baseline.summary.to_csv(baseline_path.with_suffix(".csv"), index=False)

    features_map = {
        well: compute_feature_frame(df, settings, baseline=baseline) for well, df in base_well_data.items()
    }

    residual_settings = load_residual_settings(config)
    residual_model = build_residual_detection_model(features_map, normal_intervals_train, residual_settings)
    if residual_settings.enabled and residual_model is None:
        print("Residual detection: недостаточно данных для построения ковариационной модели.")
    if not anomaly_intervals_train:
        anomaly_features = aggregate_feature_values(anomaly_intervals, features_map)
        training_anomaly_intervals = anomaly_intervals
    else:
        anomaly_features = aggregate_feature_values(anomaly_intervals_train, features_map)
        training_anomaly_intervals = anomaly_intervals_train

    if normal_intervals_train:
        normal_features_train = aggregate_feature_values(normal_intervals_train, features_map)
        normal_training_intervals = normal_intervals_train
    else:
        normal_features_train = aggregate_feature_values(normal_intervals_full, features_map)
        normal_training_intervals = normal_intervals_full
    thresholds = derive_thresholds(normal_features_train, anomaly_features, settings)

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
                residual_model=residual_model,
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
        "thresholds": thresholds,
        "training": {
            "anomaly_points": int(len(anomaly_features["pressure_delta"])),
            "normal_points": int(len(normal_features_train["pressure_delta"])),
            "anomaly_wells": sorted({interval.well for interval in training_anomaly_intervals}),
            "normal_wells": sorted({interval.well for interval in normal_training_intervals}),
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
    event_count = len(detections)
    print(f"Detected {event_count} anomaly events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
