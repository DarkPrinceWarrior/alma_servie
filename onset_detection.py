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


def ema(values: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if len(x) == 0:
        return x
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def positive_cusum(values: np.ndarray, drift: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    out = np.empty_like(x)
    c = 0.0
    for i in range(len(x)):
        c = max(0.0, c + float(x[i]) - drift)
        out[i] = c
    return out


def positive_cusum_masked(values: np.ndarray, drift: float, mask: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if len(x) != len(m):
        raise ValueError("mask length must match values length.")
    out = np.empty_like(x)
    c = 0.0
    for i in range(len(x)):
        if not m[i]:
            c = 0.0
            out[i] = 0.0
            continue
        c = max(0.0, c + float(x[i]) - drift)
        out[i] = c
    return out


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
) -> List[pd.Timestamp]:
    x = np.asarray(scores, dtype=np.float32)
    z_ema = diagnostics["ema_z"]
    cusum = diagnostics["cusum"]

    score_cond = x >= thresholds.score_threshold
    ema_cond = z_ema >= thresholds.ema_z_threshold
    cusum_cond = cusum >= thresholds.cusum_threshold

    if gate_mode == "strict":
        cond = score_cond & ema_cond & cusum_cond
    elif gate_mode == "score_ema":
        cond = score_cond & ema_cond
    else:
        # Relaxed: score must be high, then either smooth level (EMA) or shift evidence (CUSUM).
        cond = score_cond & (ema_cond | cusum_cond)

    starts: List[pd.Timestamp] = []
    cooldown = pd.Timedelta(hours=float(cooldown_hours))
    run_start = None
    run_len = 0

    start_i = int(np.clip(reference_end_idx, 0, len(cond)))
    for i in range(start_i, len(cond)):
        if cond[i]:
            if run_start is None:
                run_start = i
                run_len = 1
            else:
                run_len += 1
            if run_len == min_run_points:
                ts = pd.Timestamp(timestamps[run_start])
                if not starts or ts - starts[-1] >= cooldown:
                    starts.append(ts)
        else:
            run_start = None
            run_len = 0

    return starts


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
