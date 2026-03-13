"""
Utilities for causal anomaly-onset detection from point-wise anomaly scores.

Core design goals:
- Use only historical information (causal processing).
- Calibrate thresholds on a historical reference window.
- Target operational false-alarm rate per day (FAR/day).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional runtime acceleration
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@dataclass
class CausalThresholds:
    score_threshold: float
    ema_z_threshold: float
    cusum_threshold: float
    drift: float
    baseline_median: float
    baseline_mad: float
    quantile: float


def infer_step_seconds(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 60.0
    ts = pd.to_datetime(timestamps)
    dt = pd.Series(ts).diff().dropna().dt.total_seconds()
    if dt.empty:
        return 60.0
    step = float(dt.median())
    if not np.isfinite(step) or step <= 0:
        return 60.0
    return step


def choose_reference_end_index(
    timestamps: np.ndarray,
    patch_size: int,
    min_ratio: float = 0.15,
    max_ratio: float = 0.5,
    min_days: float = 1.0,
) -> int:
    n = len(timestamps)
    if n <= patch_size * 4:
        return min(n, patch_size * 2)

    step_sec = infer_step_seconds(timestamps)
    min_points_by_days = int(np.ceil((min_days * 86400.0) / max(step_sec, 1.0)))
    idx_by_ratio = int(np.ceil(n * min_ratio))
    idx_cap = int(np.floor(n * max_ratio))

    ref_end = max(patch_size * 2, min_points_by_days, idx_by_ratio)
    ref_end = min(ref_end, idx_cap, n - patch_size)
    ref_end = max(ref_end, patch_size * 2)
    return int(ref_end)


def robust_stats(values: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(values, dtype=np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    mad = max(mad, 1e-6)
    return med, mad


def robust_z(values: np.ndarray, median: float, mad: float) -> np.ndarray:
    scale = 1.4826 * max(mad, 1e-6)
    z = (np.asarray(values, dtype=np.float32) - median) / scale
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


@njit(cache=True, fastmath=True)
def _ema_impl(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def ema(values: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if len(x) == 0:
        return x
    return _ema_impl(x, float(alpha))


@njit(cache=True, fastmath=True)
def _positive_cusum_impl(x: np.ndarray, drift: float) -> np.ndarray:
    out = np.empty_like(x)
    c = 0.0
    for i in range(len(x)):
        c = max(0.0, c + x[i] - drift)
        out[i] = c
    return out


def positive_cusum(values: np.ndarray, drift: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    return _positive_cusum_impl(x, float(drift))


@njit(cache=True, fastmath=True)
def _positive_cusum_masked_impl(x: np.ndarray, drift: float, m: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    c = 0.0
    for i in range(len(x)):
        if not m[i]:
            c = 0.0
            out[i] = 0.0
            continue
        c = max(0.0, c + x[i] - drift)
        out[i] = c
    return out


def positive_cusum_masked(values: np.ndarray, drift: float, mask: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if len(x) != len(m):
        raise ValueError("mask length must match values length.")
    return _positive_cusum_masked_impl(x, float(drift), m)


def _quantile_for_target_far(
    points_per_day: float,
    target_far_per_day: float,
    min_run_points: int,
    min_q: float = 0.90,
    max_q: float = 0.9990,
) -> float:
    alarms_per_day_possible = max(points_per_day / max(min_run_points, 1), 1.0)
    tail_prob = target_far_per_day / alarms_per_day_possible
    q = 1.0 - tail_prob
    return float(np.clip(q, min_q, max_q))


def _scaled_down_threshold(threshold: float, scale: float) -> float:
    value = float(threshold)
    factor = float(np.clip(scale, 1e-3, 1.0))
    if value >= 0.0:
        return value * factor
    return value / factor


def _build_gate_condition(
    scores: np.ndarray,
    ema_z: np.ndarray,
    cusum: np.ndarray,
    *,
    score_threshold: float,
    ema_threshold: float,
    cusum_threshold: float,
    gate_mode: str,
) -> np.ndarray:
    score_cond = np.asarray(scores, dtype=np.float32) >= float(score_threshold)
    ema_cond = np.asarray(ema_z, dtype=np.float32) >= float(ema_threshold)
    cusum_cond = np.asarray(cusum, dtype=np.float32) >= float(cusum_threshold)

    if gate_mode == "strict":
        return score_cond & ema_cond & cusum_cond
    if gate_mode == "score_ema":
        return score_cond & ema_cond
    return score_cond & (ema_cond | cusum_cond)


def _detect_onsets_stateful(
    entry_cond: np.ndarray,
    sustain_cond: np.ndarray,
    timestamps: np.ndarray,
    *,
    start_i: int,
    min_run_points: int,
    cooldown_hours: float,
    rearm_window_minutes: float,
) -> List[pd.Timestamp]:
    starts: List[pd.Timestamp] = []
    if len(entry_cond) == 0:
        return starts

    ts = pd.to_datetime(timestamps)
    step_seconds = infer_step_seconds(timestamps)
    clear_points = max(
        int(np.ceil((float(rearm_window_minutes) * 60.0) / max(step_seconds, 1.0))),
        int(min_run_points),
    )
    cooldown = pd.Timedelta(hours=float(cooldown_hours))

    armed = True
    run_start = None
    run_len = 0
    clear_len = 0
    last_start: pd.Timestamp | None = None
    rearmed_after_clear = False

    for i in range(int(np.clip(start_i, 0, len(entry_cond))), len(entry_cond)):
        current_ts = pd.Timestamp(ts[i])
        if armed:
            if entry_cond[i]:
                if run_start is None:
                    run_start = i
                    run_len = 1
                else:
                    run_len += 1
                if run_len >= int(min_run_points):
                    start_ts = pd.Timestamp(ts[run_start])
                    if (
                        last_start is None
                        or rearmed_after_clear
                        or start_ts - last_start >= cooldown
                    ):
                        starts.append(start_ts)
                        last_start = start_ts
                    armed = False
                    rearmed_after_clear = False
                    run_start = None
                    run_len = 0
                    clear_len = 0
            else:
                run_start = None
                run_len = 0
            continue

        if sustain_cond[i]:
            clear_len = 0
        else:
            clear_len += 1

        if clear_len >= clear_points:
            armed = True
            rearmed_after_clear = True
            run_start = None
            run_len = 0
            clear_len = 0

    return starts


def calibrate_causal_thresholds(
    scores: np.ndarray,
    timestamps: np.ndarray,
    reference_end_idx: int,
    target_far_per_day: float,
    min_run_points: int,
    ema_alpha: float = 0.08,
) -> Tuple[CausalThresholds, Dict[str, np.ndarray]]:
    x = np.asarray(scores, dtype=np.float32)
    n = len(x)
    if n == 0:
        raise ValueError("Empty scores passed to threshold calibration.")

    ref_end = int(np.clip(reference_end_idx, 10, n))
    ref_scores = x[:ref_end]
    b_med, b_mad = robust_stats(ref_scores)
    z = robust_z(x, b_med, b_mad)
    ema_z = ema(z, alpha=ema_alpha)

    ref_ema = ema_z[:ref_end]
    drift = float(np.quantile(np.clip(ref_ema, 0.0, None), 0.5) * 0.5)
    cusum = positive_cusum(ema_z, drift=drift)
    ref_cusum = cusum[:ref_end]

    step_sec = infer_step_seconds(timestamps)
    points_per_day = 86400.0 / max(step_sec, 1.0)
    q = _quantile_for_target_far(
        points_per_day=points_per_day,
        target_far_per_day=target_far_per_day,
        min_run_points=min_run_points,
    )

    score_thr = float(np.quantile(ref_scores, q))
    ema_thr = float(np.quantile(ref_ema, q))
    cusum_q = float(np.clip(q - 0.02, 0.90, 0.995))
    cusum_thr = float(np.quantile(ref_cusum, cusum_q))

    thresholds = CausalThresholds(
        score_threshold=score_thr,
        ema_z_threshold=ema_thr,
        cusum_threshold=cusum_thr,
        drift=drift,
        baseline_median=b_med,
        baseline_mad=b_mad,
        quantile=q,
    )
    diagnostics = {
        "z": z,
        "ema_z": ema_z,
        "cusum": cusum,
    }
    return thresholds, diagnostics


def detect_causal_onsets(
    scores: np.ndarray,
    timestamps: np.ndarray,
    diagnostics: Dict[str, np.ndarray],
    thresholds: CausalThresholds,
    reference_end_idx: int,
    min_run_points: int,
    cooldown_hours: float,
    gate_mode: str = "relaxed",
    rearm_window_minutes: float = 60.0,
    hysteresis_scale: float = 0.60,
) -> List[pd.Timestamp]:
    x = np.asarray(scores, dtype=np.float32)
    z_ema = diagnostics["ema_z"]
    cusum = diagnostics["cusum"]

    entry_cond = _build_gate_condition(
        x,
        z_ema,
        cusum,
        score_threshold=thresholds.score_threshold,
        ema_threshold=thresholds.ema_z_threshold,
        cusum_threshold=thresholds.cusum_threshold,
        gate_mode=gate_mode,
    )
    sustain_cond = _build_gate_condition(
        x,
        z_ema,
        cusum,
        score_threshold=_scaled_down_threshold(thresholds.score_threshold, hysteresis_scale),
        ema_threshold=_scaled_down_threshold(thresholds.ema_z_threshold, hysteresis_scale),
        cusum_threshold=_scaled_down_threshold(thresholds.cusum_threshold, hysteresis_scale),
        gate_mode=gate_mode,
    )
    start_i = int(np.clip(reference_end_idx, 0, len(entry_cond)))
    return _detect_onsets_stateful(
        entry_cond,
        sustain_cond,
        timestamps,
        start_i=start_i,
        min_run_points=min_run_points,
        cooldown_hours=cooldown_hours,
        rearm_window_minutes=rearm_window_minutes,
    )


def detect_causal_onsets_masked(
    scores: np.ndarray,
    timestamps: np.ndarray,
    diagnostics: Dict[str, np.ndarray],
    thresholds: CausalThresholds,
    reference_mask: np.ndarray,
    onset_mask: np.ndarray,
    min_run_points: int,
    cooldown_hours: float,
    gate_mode: str = "relaxed",
    rearm_window_minutes: float = 60.0,
    hysteresis_scale: float = 0.60,
) -> List[pd.Timestamp]:
    x = np.asarray(scores, dtype=np.float32)
    ref_mask = np.asarray(reference_mask, dtype=bool)
    valid_mask = np.asarray(onset_mask, dtype=bool)
    if len(ref_mask) != len(x) or len(valid_mask) != len(x):
        raise ValueError("reference_mask and onset_mask must match scores length.")

    z_ema = diagnostics["ema_z"]
    cusum = diagnostics["cusum"]

    entry_cond = _build_gate_condition(
        x,
        z_ema,
        cusum,
        score_threshold=thresholds.score_threshold,
        ema_threshold=thresholds.ema_z_threshold,
        cusum_threshold=thresholds.cusum_threshold,
        gate_mode=gate_mode,
    ) & valid_mask
    sustain_cond = _build_gate_condition(
        x,
        z_ema,
        cusum,
        score_threshold=_scaled_down_threshold(thresholds.score_threshold, hysteresis_scale),
        ema_threshold=_scaled_down_threshold(thresholds.ema_z_threshold, hysteresis_scale),
        cusum_threshold=_scaled_down_threshold(thresholds.cusum_threshold, hysteresis_scale),
        gate_mode=gate_mode,
    ) & valid_mask

    ref_indices = np.flatnonzero(ref_mask)
    start_i = int(ref_indices[-1] + 1) if len(ref_indices) else 0
    return _detect_onsets_stateful(
        entry_cond,
        sustain_cond,
        timestamps,
        start_i=start_i,
        min_run_points=min_run_points,
        cooldown_hours=cooldown_hours,
        rearm_window_minutes=rearm_window_minutes,
    )


def robust_scale_for_fusion(scores: np.ndarray, reference_end_idx: int) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float32)
    if len(x) == 0:
        return x
    ref_end = int(np.clip(reference_end_idx, 10, len(x)))
    med, mad = robust_stats(x[:ref_end])
    return robust_z(x, med, mad)


def robust_scale_for_fusion_mask(scores: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float32)
    mask = np.asarray(reference_mask, dtype=bool)
    if len(x) == 0:
        return x
    if len(mask) != len(x):
        raise ValueError("reference_mask length must match scores length.")
    ref = x[mask]
    if len(ref) < 10:
        ref = x
    med, mad = robust_stats(ref)
    return robust_z(x, med, mad)


def calibrate_causal_thresholds_from_reference_mask(
    scores: np.ndarray,
    timestamps: np.ndarray,
    reference_mask: np.ndarray,
    target_far_per_day: float,
    min_run_points: int,
    ema_alpha: float = 0.08,
) -> Tuple[CausalThresholds, Dict[str, np.ndarray]]:
    x = np.asarray(scores, dtype=np.float32)
    n = len(x)
    if n == 0:
        raise ValueError("Empty scores passed to threshold calibration.")

    mask = np.asarray(reference_mask, dtype=bool)
    if len(mask) != n:
        raise ValueError("reference_mask length must match scores length.")
    if mask.sum() < 10:
        mask = np.ones(n, dtype=bool)

    ref_scores = x[mask]
    b_med, b_mad = robust_stats(ref_scores)
    z = robust_z(x, b_med, b_mad)
    ema_z = ema(z, alpha=ema_alpha)

    ref_ema = ema_z[mask]
    drift = float(np.quantile(np.clip(ref_ema, 0.0, None), 0.5) * 0.5)
    # For masked reference calibration, reset CUSUM outside reference to avoid
    # threshold inflation from unrelated segments.
    cusum = positive_cusum(ema_z, drift=drift)
    cusum_ref = positive_cusum_masked(ema_z, drift=drift, mask=mask)
    ref_cusum = cusum_ref[mask]

    step_sec = infer_step_seconds(timestamps)
    points_per_day = 86400.0 / max(step_sec, 1.0)
    q = _quantile_for_target_far(
        points_per_day=points_per_day,
        target_far_per_day=target_far_per_day,
        min_run_points=min_run_points,
    )

    score_thr = float(np.quantile(ref_scores, q))
    ema_thr = float(np.quantile(ref_ema, q))
    cusum_q = float(np.clip(q - 0.02, 0.90, 0.995))
    cusum_thr = float(np.quantile(ref_cusum, cusum_q))

    thresholds = CausalThresholds(
        score_threshold=score_thr,
        ema_z_threshold=ema_thr,
        cusum_threshold=cusum_thr,
        drift=drift,
        baseline_median=b_med,
        baseline_mad=b_mad,
        quantile=q,
    )
    diagnostics = {
        "z": z,
        "ema_z": ema_z,
        "cusum": cusum,
    }
    return thresholds, diagnostics
