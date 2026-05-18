from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alma_service.onset_detection import infer_step_seconds
from alma_service.operational_events import build_event_zone_masks
from alma_service.zone_labels import Zone, label_zones


QUALITY_OK = "ok"
QUALITY_GAP = "gap"
QUALITY_MISSING = "missing"
QUALITY_SENSOR_STUCK = "sensor_stuck"

REGIME_NORMAL = "normal"
REGIME_STOP_START = "stop_start"
REGIME_FREQUENCY_CHANGE = "frequency_change"
REGIME_SHIFT = "regime_shift"

EVENT_NORMAL = "normal_context"
EVENT_BAD_DATA = "bad_data"
EVENT_REGIME = "regime_event"
EVENT_PRE_ANOMALY = "pre_anomaly_zone"
EVENT_LABELLED_ANOMALY = "labelled_anomaly"
EVENT_ANOMALY_CANDIDATE = "anomaly_candidate"
ZONE_OPERATIONAL_EVENT = "operational_event"

FREQUENCY_ALIASES = ("выходная частота", "частота")
POWER_ALIASES = ("активная выходная мощность", "полная выходная мощность", "мощность")
CURRENT_ALIASES = ("выходной ток", "ток на фазе", "ток")
LOAD_ALIASES = ("коэффициент загрузки", "загрузка")


@dataclass(frozen=True)
class TelemetryStatus:
    frame: pd.DataFrame

    def row_at(self, index: int) -> dict[str, Any]:
        if self.frame.empty:
            return {
                "quality_status": QUALITY_OK,
                "regime_status": REGIME_NORMAL,
                "zone_status": "unknown",
                "event_class": EVENT_NORMAL,
            }
        idx = min(max(int(index), 0), len(self.frame) - 1)
        return self.frame.iloc[idx].to_dict()


def _contains_any(name: str, aliases: tuple[str, ...]) -> bool:
    lower = name.casefold()
    return any(alias in lower for alias in aliases)


def _column_indices(raw_columns: list[str], aliases: tuple[str, ...]) -> list[int]:
    return [idx for idx, column in enumerate(raw_columns) if _contains_any(column, aliases)]


def _expand(mask: np.ndarray, back_steps: int, forward_steps: int) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    expanded = mask.astype(bool).copy()
    for idx in np.flatnonzero(mask):
        start = max(0, int(idx) - back_steps)
        end = min(len(mask), int(idx) + forward_steps + 1)
        expanded[start:end] = True
    return expanded


def _stale_mask(values: np.ndarray, *, min_run: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    finite = np.isfinite(values)
    same = np.zeros(len(values), dtype=bool)
    same[1:] = finite[1:] & finite[:-1] & (np.abs(np.diff(values)) <= 1e-8)
    run = np.zeros(len(values), dtype=np.int32)
    for idx in range(1, len(values)):
        run[idx] = run[idx - 1] + 1 if same[idx] else 0
    return run >= max(1, int(min_run) - 1)


def _robust_step_mask(values: np.ndarray, *, min_sigma: float = 8.0) -> np.ndarray:
    if len(values) < 3:
        return np.zeros(len(values), dtype=bool)
    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=float)
    if not np.isfinite(filled).any():
        return np.zeros(len(values), dtype=bool)
    delta = np.diff(filled, prepend=filled[0])
    abs_delta = np.abs(delta[1:])
    median = float(np.nanmedian(abs_delta)) if len(abs_delta) else 0.0
    mad = float(np.nanmedian(np.abs(abs_delta - median))) if len(abs_delta) else 0.0
    if not np.isfinite(mad) or mad <= 1e-8:
        positive = abs_delta[abs_delta > 1e-8]
        if not len(positive):
            threshold = np.inf
        elif len(positive) > max(2, int(round(0.05 * len(abs_delta)))):
            typical = float(np.nanmedian(positive))
            high = float(np.nanquantile(positive, 0.95))
            threshold = high if high > max(typical * min_sigma, 1e-8) else np.inf
        else:
            threshold = float(np.nanquantile(positive, 0.95))
    else:
        threshold = median + min_sigma * 1.4826 * mad
    if not np.isfinite(threshold):
        return np.zeros(len(values), dtype=bool)
    return np.abs(delta) >= max(threshold, 1e-8)


def _low_operation_mask(values: np.ndarray) -> np.ndarray:
    finite_abs = np.abs(values[np.isfinite(values)])
    positive = finite_abs[finite_abs > 1e-8]
    if len(positive) == 0:
        return np.zeros(len(values), dtype=bool)
    low_thr = max(float(np.nanquantile(positive, 0.10)) * 0.25, 1e-8)
    return np.abs(values) <= low_thr


def _zone_status(
    timestamps: np.ndarray,
    anomaly_intervals: pd.DataFrame | None,
    *,
    patch_size: int,
    anomaly_key: str,
) -> np.ndarray:
    labels = label_zones(
        timestamps=timestamps,
        anomaly_intervals=anomaly_intervals,
        patch_size=patch_size,
        anomaly_key=anomaly_key,
    )
    names = {
        int(Zone.CLEAN_NORMAL): "clean_normal",
        int(Zone.PRE_ANOMALY_BUFFER): EVENT_PRE_ANOMALY,
        int(Zone.ANOMALY): EVENT_LABELLED_ANOMALY,
        int(Zone.POST_ANOMALY_RECOVERY): "post_anomaly_recovery",
    }
    return np.asarray([names.get(int(label), "unknown") for label in labels], dtype=object)


def build_telemetry_status(
    *,
    timestamps: np.ndarray,
    raw_columns: list[str],
    raw_matrix: np.ndarray,
    anomaly_key: str = "",
    anomaly_intervals: pd.DataFrame | None = None,
    operational_events: pd.DataFrame | None = None,
    operational_pre_window_hours: float = 24.0,
    patch_size: int = 96,
) -> TelemetryStatus:
    n_points = int(len(timestamps))
    ts = pd.to_datetime(timestamps)
    quality = np.full(n_points, QUALITY_OK, dtype=object)
    regime = np.full(n_points, REGIME_NORMAL, dtype=object)

    if n_points == 0:
        return TelemetryStatus(pd.DataFrame())

    step_seconds = infer_step_seconds(timestamps)
    back_steps = max(1, int(round(15 * 60 / step_seconds)))
    forward_steps = max(1, int(round(30 * 60 / step_seconds)))
    stale_steps = max(6, int(round(6 * 60 * 60 / step_seconds)))

    deltas = pd.Series(ts).diff().dt.total_seconds().to_numpy()
    gap_mask = np.zeros(n_points, dtype=bool)
    if len(deltas) > 1:
        gap_mask = np.asarray(deltas > max(step_seconds * 3.0, 30 * 60), dtype=bool)
        gap_mask[0] = False
        gap_mask = _expand(gap_mask, back_steps=0, forward_steps=forward_steps)

    raw = np.asarray(raw_matrix, dtype=float)
    missing_mask = np.zeros(n_points, dtype=bool)
    stale_mask = np.zeros(n_points, dtype=bool)
    if raw.ndim == 2 and raw.shape[0] == n_points and raw.shape[1] == len(raw_columns):
        freq_indices = _column_indices(raw_columns, FREQUENCY_ALIASES)
        current_indices = _column_indices(raw_columns, CURRENT_ALIASES)
        power_indices = _column_indices(raw_columns, POWER_ALIASES)
        load_indices = _column_indices(raw_columns, LOAD_ALIASES)
        operation_indices = set(freq_indices + current_indices + power_indices + load_indices)

        missing_mask = np.isnan(raw).mean(axis=1) >= 0.25
        for idx in range(raw.shape[1]):
            column_stale = _stale_mask(raw[:, idx], min_run=stale_steps)
            if idx in operation_indices:
                column_stale &= _low_operation_mask(raw[:, idx])
            stale_mask |= column_stale

        frequency_change = np.zeros(n_points, dtype=bool)
        for idx in freq_indices:
            frequency_change |= _robust_step_mask(raw[:, idx], min_sigma=6.0)
        frequency_change = _expand(frequency_change, back_steps=back_steps, forward_steps=forward_steps)

        stop_start = np.zeros(n_points, dtype=bool)
        for idx in (*freq_indices, *current_indices, *power_indices):
            low = _low_operation_mask(raw[:, idx])
            transition = np.zeros(n_points, dtype=bool)
            transition[1:] = low[1:] != low[:-1]
            stop_start |= transition
        stop_start = _expand(stop_start, back_steps=back_steps, forward_steps=forward_steps)

        regime_shift = np.zeros(n_points, dtype=bool)
        for idx in (*current_indices, *power_indices, *load_indices):
            regime_shift |= _robust_step_mask(raw[:, idx], min_sigma=10.0)
        regime_shift = _expand(regime_shift, back_steps=back_steps, forward_steps=forward_steps)

        regime[regime_shift] = REGIME_SHIFT
        regime[frequency_change] = REGIME_FREQUENCY_CHANGE
        regime[stop_start] = REGIME_STOP_START

    quality[stale_mask] = QUALITY_SENSOR_STUCK
    quality[missing_mask] = QUALITY_MISSING
    quality[gap_mask] = QUALITY_GAP

    zone = _zone_status(
        timestamps=timestamps,
        anomaly_intervals=anomaly_intervals,
        patch_size=patch_size,
        anomaly_key=anomaly_key,
    )
    event_masks = build_event_zone_masks(
        timestamps=timestamps,
        events=operational_events,
        default_pre_window_hours=operational_pre_window_hours,
    )
    clean_zone = zone == "clean_normal"
    zone[clean_zone & event_masks.pre_anomaly_mask] = EVENT_PRE_ANOMALY
    zone[clean_zone & event_masks.event_mask] = ZONE_OPERATIONAL_EVENT

    event_class = np.full(n_points, EVENT_NORMAL, dtype=object)
    event_class[zone == EVENT_PRE_ANOMALY] = EVENT_PRE_ANOMALY
    event_class[zone == EVENT_LABELLED_ANOMALY] = EVENT_LABELLED_ANOMALY
    event_class[zone == ZONE_OPERATIONAL_EVENT] = EVENT_REGIME
    event_class[regime != REGIME_NORMAL] = EVENT_REGIME
    event_class[quality != QUALITY_OK] = EVENT_BAD_DATA

    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "quality_status": quality,
            "regime_status": regime,
            "zone_status": zone,
            "event_class": event_class,
            "is_bad_data": quality != QUALITY_OK,
            "is_regime_event": regime != REGIME_NORMAL,
            "is_pre_anomaly_zone": zone == EVENT_PRE_ANOMALY,
            "is_labelled_anomaly": zone == EVENT_LABELLED_ANOMALY,
            "is_operational_event": zone == ZONE_OPERATIONAL_EVENT,
        }
    )
    return TelemetryStatus(frame)
