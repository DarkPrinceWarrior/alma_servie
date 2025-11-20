from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .models import ReferenceInterval, WellTimeseries
from .workbook import WorkbookSource


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


def load_svod_sheet(workbook: WorkbookSource, sheet_name: str) -> pd.DataFrame:
    svod = workbook.parse(sheet_name, header=2)
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
    workbook: WorkbookSource,
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

    for sheet_name in workbook.sheet_names:
        if sheet_name == svod_sheet:
            continue
        df = workbook.parse(sheet_name)
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
    anomaly_labels: Set[str],
    normal_labels: Set[str],
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
        if label_raw not in anomaly_labels and label_raw not in normal_labels:
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
                label="anomaly" if label_raw in anomaly_labels else "normal",
                cause=label_raw,
                notes=notes,
            )
        )
    return intervals


__all__ = [
    "clip_interval_to_data",
    "load_svod_sheet",
    "load_well_series",
    "parse_reference_intervals",
    "preprocess_well_data",
]
