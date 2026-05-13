from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alma_service.engineered_features import PreparedWellData

PRESSURE_COL = "Давление на приеме насоса кгс/см²"
EPS = 1e-6


@dataclass
class PressureTrendOutput:
    score: np.ndarray
    components: dict[str, np.ndarray]
    detail: dict[str, object]


def _as_float_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)


def _robust_stats(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float32)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if not np.isfinite(mad) or mad < EPS:
        q75, q25 = np.percentile(x, [75, 25])
        mad = float((q75 - q25) / 1.349)
    if not np.isfinite(mad) or mad < EPS:
        mad = float(np.std(x))
    return med, max(mad, EPS)


def empirical_tail_score(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    """Map values to 0..100 by their empirical upper-tail rank in reference data."""
    x = np.asarray(values, dtype=np.float32)
    ref = np.asarray(reference_values, dtype=np.float32)
    ref = ref[np.isfinite(ref)]
    out = np.zeros(len(x), dtype=np.float32)
    if len(ref) < 8:
        return out
    ref_sorted = np.sort(ref)
    finite = np.isfinite(x)
    ranks = np.searchsorted(ref_sorted, x[finite], side="right")
    out[finite] = (100.0 * ranks / len(ref_sorted)).astype(np.float32)
    return np.clip(out, 0.0, 100.0)


def _robust_excess_score(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    ref = np.asarray(reference_values, dtype=np.float32)
    ref = ref[np.isfinite(ref)]
    out = np.zeros(len(x), dtype=np.float32)
    if len(ref) < 8:
        return out
    med, scale = _robust_stats(ref)
    finite = np.isfinite(x)
    out[finite] = np.maximum(0.0, (x[finite] - med) / scale).astype(np.float32)
    return out


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    min_periods = max(3, int(np.ceil(window * 0.5)))
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=min_periods)
        .median()
        .to_numpy(dtype=np.float32)
    )


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    min_periods = max(3, int(np.ceil(window * 0.5)))
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .to_numpy(dtype=np.float32)
    )


def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling least-squares slope of values over `window` samples.

    Slope at index i is OLS fit of values[i-window+1..i] against local time
    indices 0..window-1. Indices < window-1 yield 0 (insufficient data).
    Mirrors scripts.detection.physical_branches_3w._rolling_slope so the
    physical-trend signal stays consistent between 3W class 5 and ALMA pritok.
    """
    n = len(values)
    w = int(window)
    out = np.zeros(n, dtype=np.float64)
    if n < w or w < 3:
        return out.astype(np.float32)
    t = np.arange(w, dtype=np.float64)
    t_mean = t.mean()
    t_centered = t - t_mean
    denom = float(np.sum(t_centered * t_centered))
    if denom < 1e-12:
        return out.astype(np.float32)
    x = np.where(np.isnan(values), 0.0, values).astype(np.float64)
    idx = np.arange(n, dtype=np.float64)
    cs_x = np.concatenate(([0.0], np.cumsum(x)))
    cs_tx = np.concatenate(([0.0], np.cumsum(idx * x)))
    for i in range(w - 1, n):
        s = i - w + 1
        sum_x = cs_x[i + 1] - cs_x[s]
        sum_tx_global = cs_tx[i + 1] - cs_tx[s]
        sum_t_local_x = sum_tx_global - s * sum_x
        x_mean = sum_x / w
        slope_num = sum_t_local_x - t_mean * w * x_mean
        out[i] = slope_num / denom
    return out.astype(np.float32)


def _adaptive_horizons(reference_points: int, total_points: int) -> list[int]:
    if reference_points < 16 or total_points < 32:
        return []
    ref = float(reference_points)
    candidates = [
        np.sqrt(ref) * 0.5,
        np.sqrt(ref),
        ref ** 0.60,
        ref / 16.0,
        ref / 8.0,
        ref / 4.0,
    ]
    lower = max(4, int(np.floor(np.sqrt(ref) * 0.25)))
    upper = max(lower, min(int(reference_points * 0.50), int(total_points * 0.25)))
    horizons = {
        int(np.clip(round(candidate), lower, upper))
        for candidate in candidates
        if np.isfinite(candidate)
    }
    return sorted(h for h in horizons if h >= lower)


def _best_by_score(
    current_score: np.ndarray,
    best_score: np.ndarray,
    arrays: dict[str, np.ndarray],
    best_arrays: dict[str, np.ndarray],
) -> None:
    mask = np.isfinite(current_score) & (current_score > best_score)
    if not np.any(mask):
        return
    best_score[mask] = current_score[mask]
    for name, values in arrays.items():
        best_arrays[name][mask] = values[mask]


def build_pressure_trend_branch(prepared: PreparedWellData) -> PressureTrendOutput:
    """Build a causal, per-well calibrated pressure trend score for pritok."""
    n = len(prepared.timestamps)
    zero = np.zeros(n, dtype=np.float32)
    if PRESSURE_COL not in prepared.raw_columns:
        return PressureTrendOutput(
            score=zero.copy(),
            components={
                "pressure_trend_score": zero.copy(),
                "pressure_trend_level": zero.copy(),
                "pressure_trend_change": zero.copy(),
                "pressure_trend_persistence": zero.copy(),
                "pressure_trend_practical": zero.copy(),
                "pressure_trend_slope": zero.copy(),
                "pressure_trend_direction": zero.copy(),
                "pressure_trend_horizon": zero.copy(),
            },
            detail={"pressure_trend_enabled": False, "reason": "pressure_column_missing"},
        )

    pressure_idx = prepared.raw_columns.index(PRESSURE_COL)
    pressure = _as_float_array(prepared.raw_matrix[:, pressure_idx])
    finite = np.isfinite(pressure)
    if finite.sum() < 32:
        return PressureTrendOutput(
            score=zero.copy(),
            components={
                "pressure_trend_score": zero.copy(),
                "pressure_trend_level": zero.copy(),
                "pressure_trend_change": zero.copy(),
                "pressure_trend_persistence": zero.copy(),
                "pressure_trend_practical": zero.copy(),
                "pressure_trend_slope": zero.copy(),
                "pressure_trend_direction": zero.copy(),
                "pressure_trend_horizon": zero.copy(),
            },
            detail={"pressure_trend_enabled": False, "reason": "not_enough_pressure_points"},
        )

    pressure = (
        pd.Series(pressure)
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(dtype=np.float32)
    )
    reference_mask = np.asarray(prepared.reference_mask, dtype=bool) & np.isfinite(pressure)
    if int(reference_mask.sum()) < 16:
        reference_mask = np.zeros(n, dtype=bool)
        reference_mask[: max(1, int(prepared.reference_end_idx))] = True
        reference_mask &= np.isfinite(pressure)

    ref_values = pressure[reference_mask]
    ref_median, ref_scale = _robust_stats(ref_values)
    horizons = _adaptive_horizons(int(reference_mask.sum()), n)
    if not horizons:
        return PressureTrendOutput(
            score=zero.copy(),
            components={
                "pressure_trend_score": zero.copy(),
                "pressure_trend_level": zero.copy(),
                "pressure_trend_change": zero.copy(),
                "pressure_trend_persistence": zero.copy(),
                "pressure_trend_practical": zero.copy(),
                "pressure_trend_slope": zero.copy(),
                "pressure_trend_direction": zero.copy(),
                "pressure_trend_horizon": zero.copy(),
            },
            detail={"pressure_trend_enabled": False, "reason": "no_adaptive_horizons"},
        )

    best_score = np.zeros(n, dtype=np.float32)
    best_arrays = {
        "pressure_trend_level": zero.copy(),
        "pressure_trend_change": zero.copy(),
        "pressure_trend_persistence": zero.copy(),
        "pressure_trend_practical": zero.copy(),
        "pressure_trend_slope": zero.copy(),
        "pressure_trend_direction": zero.copy(),
        "pressure_trend_horizon": zero.copy(),
    }

    for horizon in horizons:
        current = _rolling_median(pressure, horizon)
        previous = np.roll(current, horizon)
        previous[:horizon] = np.nan

        level_delta = current - ref_median
        change_delta = current - previous
        level_abs = np.abs(level_delta) / ref_scale
        change_abs = np.abs(change_delta) / ref_scale
        practical_abs = np.abs(level_delta) / max(abs(ref_median), ref_scale, EPS)

        level_tail = empirical_tail_score(level_abs, level_abs[reference_mask])
        change_tail = empirical_tail_score(change_abs, change_abs[reference_mask])
        practical_tail = empirical_tail_score(practical_abs, practical_abs[reference_mask])
        level_excess = _robust_excess_score(level_abs, level_abs[reference_mask])
        change_excess = _robust_excess_score(change_abs, change_abs[reference_mask])
        practical_excess = _robust_excess_score(practical_abs, practical_abs[reference_mask])
        instant = np.maximum(change_excess, 0.50 * level_excess)
        persistence_raw = _rolling_mean(instant, max(4, int(round(np.sqrt(horizon)))))
        persistence_tail = empirical_tail_score(
            persistence_raw,
            persistence_raw[reference_mask],
        )
        persistence_excess = _robust_excess_score(
            persistence_raw,
            persistence_raw[reference_mask],
        )

        # Sprint 4 #2: rolling regression slope as DIAGNOSTIC component only.
        # Fusion via max with change_excess was tried but degraded test hit
        # 1.000 -> 0.500 due to per-well reference-mask calibration shift.
        # Kept as a stored component for future Optuna-tuned separate weight.
        slope_raw = _rolling_slope(pressure, horizon)
        slope_abs = np.abs(slope_raw) / ref_scale
        slope_tail = empirical_tail_score(slope_abs, slope_abs[reference_mask])

        change_core = np.maximum(change_excess, persistence_excess)
        level_support = np.sqrt((level_excess + 1.0) * (practical_excess + 1.0))
        # The branch must describe an onset/change, not a permanently elevated
        # level. Level and practical effect support the alarm, while contrast
        # between adaptive windows is what generates the score.
        horizon_score = np.power(change_core, 0.85) * np.power(level_support, 0.25)
        horizon_score = np.nan_to_num(horizon_score, nan=0.0, posinf=0.0, neginf=0.0)
        direction = np.sign(level_delta).astype(np.float32)
        direction[~np.isfinite(level_delta)] = 0.0

        _best_by_score(
            horizon_score,
            best_score,
            {
                "pressure_trend_level": level_tail.astype(np.float32),
                "pressure_trend_change": change_tail.astype(np.float32),
                "pressure_trend_persistence": persistence_tail.astype(np.float32),
                "pressure_trend_practical": practical_tail.astype(np.float32),
                "pressure_trend_slope": slope_tail.astype(np.float32),
                "pressure_trend_direction": direction,
                "pressure_trend_horizon": np.full(n, horizon, dtype=np.float32),
            },
            best_arrays,
        )

    best_score = np.maximum(best_score, 0.0).astype(np.float32)
    components = {"pressure_trend_score": best_score.copy(), **best_arrays}
    detail = {
        "pressure_trend_enabled": True,
        "pressure_column": PRESSURE_COL,
        "reference_points": int(reference_mask.sum()),
        "reference_median": ref_median,
        "reference_scale": ref_scale,
        "adaptive_horizons": [int(h) for h in horizons],
        "score_scale": "robust_reference_excess",
    }
    return PressureTrendOutput(score=best_score, components=components, detail=detail)


def fuse_model_with_pressure_trend(
    model_score: np.ndarray,
    pressure_output: PressureTrendOutput,
    reference_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    model = np.asarray(model_score, dtype=np.float32)
    ref_mask = np.asarray(reference_mask, dtype=bool)
    model_tail = empirical_tail_score(model, model[ref_mask])
    # The runtime fusion weight is selected by generic_detection._score_for_config.
    # Here primary stays equal to PaAno for safe defaults, while pressure evidence
    # is persisted as components for tuning, reports, and diagnostics.
    fused = model.astype(np.float32)
    pressure_dominates = pressure_output.score > model_tail

    components = {
        "paano_score": model.astype(np.float32),
        "paano_tail_score": model_tail.astype(np.float32),
        **pressure_output.components,
        "pressure_trend_dominates": pressure_dominates.astype(np.float32),
    }
    detail = {
        "fusion": "paano_plus_tuned_pressure_trend",
        "score_scale": "paano_score_with_optional_pressure_boost",
        "pressure_trend": pressure_output.detail,
    }
    return fused, components, detail
