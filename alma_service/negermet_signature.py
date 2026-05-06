from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alma_service.engineered_features import PreparedWellData
from alma_service.anomaly_physics import NEGERMET_PHYSICAL_SPEC
from alma_service.pressure_trend import PRESSURE_COL, empirical_tail_score

EPS = 1e-6

LOAD_COLUMNS = NEGERMET_PHYSICAL_SPEC.groups["load_response"]
FREQUENCY_COLUMNS = NEGERMET_PHYSICAL_SPEC.groups["regime_context"]
THERMAL_COLUMNS = tuple(
    name
    for name in NEGERMET_PHYSICAL_SPEC.groups["thermal_vibration_support"]
    if name.startswith("Температура")
)
VIBRATION_COLUMNS = tuple(
    name
    for name in NEGERMET_PHYSICAL_SPEC.groups["thermal_vibration_support"]
    if name.startswith("Вибрация")
)


@dataclass
class NegermetSignatureOutput:
    score: np.ndarray
    components: dict[str, np.ndarray]
    detail: dict[str, object]


def _zero_output(n: int, reason: str) -> NegermetSignatureOutput:
    zero = np.zeros(n, dtype=np.float32)
    return NegermetSignatureOutput(
        score=zero.copy(),
        components={
            "negermet_signature_score": zero.copy(),
            "negermet_pressure_step": zero.copy(),
            "negermet_load_step": zero.copy(),
            "negermet_frequency_step": zero.copy(),
            "negermet_thermal_step": zero.copy(),
            "negermet_vibration_step": zero.copy(),
            "negermet_pressure_direction": zero.copy(),
            "negermet_signature_horizon": zero.copy(),
        },
        detail={"negermet_signature_enabled": False, "reason": reason},
    )


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


def _adaptive_horizons(reference_points: int, total_points: int) -> list[int]:
    if reference_points < 16 or total_points < 32:
        return []
    ref = float(reference_points)
    candidates = [
        np.sqrt(ref) * 0.35,
        np.sqrt(ref) * 0.70,
        np.sqrt(ref) * 1.40,
        ref / 32.0,
        ref / 16.0,
        ref / 8.0,
    ]
    lower = max(4, int(np.floor(np.sqrt(ref) * 0.20)))
    upper = max(lower, min(int(reference_points * 0.35), int(total_points * 0.20)))
    horizons = {
        int(np.clip(round(candidate), lower, upper))
        for candidate in candidates
        if np.isfinite(candidate)
    }
    return sorted(h for h in horizons if h >= lower)


def _column_indices(raw_columns: list[str], candidates: tuple[str, ...]) -> list[int]:
    return [raw_columns.index(name) for name in candidates if name in raw_columns]


def _filled_signal(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if np.isfinite(x).sum() == 0:
        return np.full(len(x), np.nan, dtype=np.float32)
    return (
        pd.Series(x)
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(dtype=np.float32)
    )


def _best_step_score(
    values: np.ndarray,
    reference_mask: np.ndarray,
    horizons: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(values)
    best_score = np.zeros(n, dtype=np.float32)
    best_direction = np.zeros(n, dtype=np.float32)
    best_horizon = np.zeros(n, dtype=np.float32)
    x = _filled_signal(values)
    finite_ref = reference_mask & np.isfinite(x)
    if int(finite_ref.sum()) < 16:
        return best_score, best_direction, best_horizon

    ref_values = x[finite_ref]
    ref_median, ref_scale = _robust_stats(ref_values)
    denom = max(abs(ref_median), ref_scale, EPS)

    for horizon in horizons:
        current = _rolling_median(x, horizon)
        previous = np.roll(current, horizon)
        previous[:horizon] = np.nan

        change_delta = current - previous
        level_delta = current - ref_median
        change_abs = np.abs(change_delta) / ref_scale
        level_abs = np.abs(level_delta) / ref_scale
        practical_abs = np.abs(change_delta) / denom

        change_excess = _robust_excess_score(change_abs, change_abs[finite_ref])
        level_excess = _robust_excess_score(level_abs, level_abs[finite_ref])
        practical_tail = empirical_tail_score(practical_abs, practical_abs[finite_ref]) / 100.0
        horizon_score = np.maximum(change_excess, 0.45 * level_excess)
        horizon_score = np.sqrt((horizon_score + 1.0) * (practical_tail + 1.0)) - 1.0
        horizon_score = np.nan_to_num(horizon_score, nan=0.0, posinf=0.0, neginf=0.0)

        mask = horizon_score > best_score
        if np.any(mask):
            best_score[mask] = horizon_score[mask]
            best_direction[mask] = np.sign(change_delta[mask]).astype(np.float32)
            best_horizon[mask] = float(horizon)

    return best_score, best_direction, best_horizon


def _group_score(
    prepared: PreparedWellData,
    indices: list[int],
    reference_mask: np.ndarray,
    horizons: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(prepared.timestamps)
    best_score = np.zeros(n, dtype=np.float32)
    best_direction = np.zeros(n, dtype=np.float32)
    best_horizon = np.zeros(n, dtype=np.float32)
    for idx in indices:
        score, direction, horizon = _best_step_score(
            prepared.raw_matrix[:, idx],
            reference_mask=reference_mask,
            horizons=horizons,
        )
        mask = score > best_score
        if np.any(mask):
            best_score[mask] = score[mask]
            best_direction[mask] = direction[mask]
            best_horizon[mask] = horizon[mask]
    return best_score, best_direction, best_horizon


def build_negermet_signature_branch(prepared: PreparedWellData) -> NegermetSignatureOutput:
    n = len(prepared.timestamps)
    if n < 32:
        return _zero_output(n, "not_enough_points")

    reference_mask = np.asarray(prepared.reference_mask, dtype=bool)
    if int(reference_mask.sum()) < 16:
        reference_mask = np.zeros(n, dtype=bool)
        reference_mask[: max(1, int(prepared.reference_end_idx))] = True
    horizons = _adaptive_horizons(int(reference_mask.sum()), n)
    if not horizons:
        return _zero_output(n, "no_adaptive_horizons")

    raw_columns = list(prepared.raw_columns)
    pressure_idx = _column_indices(raw_columns, (PRESSURE_COL,))
    load_idx = _column_indices(raw_columns, LOAD_COLUMNS)
    freq_idx = _column_indices(raw_columns, FREQUENCY_COLUMNS)
    thermal_idx = _column_indices(raw_columns, THERMAL_COLUMNS)
    vibration_idx = _column_indices(raw_columns, VIBRATION_COLUMNS)
    if not pressure_idx and not load_idx:
        return _zero_output(n, "missing_pressure_and_load_channels")

    pressure_score, pressure_direction, pressure_horizon = _group_score(
        prepared,
        pressure_idx,
        reference_mask,
        horizons,
    )
    load_score, load_direction, load_horizon = _group_score(
        prepared,
        load_idx,
        reference_mask,
        horizons,
    )
    freq_score, _, _ = _group_score(prepared, freq_idx, reference_mask, horizons)
    thermal_score, _, _ = _group_score(prepared, thermal_idx, reference_mask, horizons)
    vibration_score, _, _ = _group_score(prepared, vibration_idx, reference_mask, horizons)

    core_score = np.maximum(pressure_score, load_score)
    support_stack = [
        arr
        for arr, indices in (
            (freq_score, freq_idx),
            (thermal_score, thermal_idx),
            (vibration_score, vibration_idx),
        )
        if indices
    ]
    if support_stack:
        support_score = np.nanmean(np.vstack(support_stack), axis=0).astype(np.float32)
        signature_score = np.sqrt((core_score + 1.0) * (support_score + 1.0)) - 1.0
        signature_score = np.maximum(core_score, signature_score)
    else:
        signature_score = core_score.copy()
    signature_score = np.nan_to_num(signature_score, nan=0.0, posinf=0.0, neginf=0.0)

    signature_horizon = pressure_horizon.copy()
    load_wins = load_score > pressure_score
    signature_horizon[load_wins] = load_horizon[load_wins]
    pressure_direction_out = pressure_direction.astype(np.float32)

    components = {
        "negermet_signature_score": signature_score.astype(np.float32),
        "negermet_pressure_step": pressure_score.astype(np.float32),
        "negermet_load_step": load_score.astype(np.float32),
        "negermet_frequency_step": freq_score.astype(np.float32),
        "negermet_thermal_step": thermal_score.astype(np.float32),
        "negermet_vibration_step": vibration_score.astype(np.float32),
        "negermet_pressure_direction": pressure_direction_out,
        "negermet_signature_horizon": signature_horizon.astype(np.float32),
    }
    detail = {
        "negermet_signature_enabled": True,
        "reference_points": int(reference_mask.sum()),
        "adaptive_horizons": [int(h) for h in horizons],
        "pressure_columns": [raw_columns[idx] for idx in pressure_idx],
        "load_columns": [raw_columns[idx] for idx in load_idx],
        "frequency_columns": [raw_columns[idx] for idx in freq_idx],
        "thermal_columns": [raw_columns[idx] for idx in thermal_idx],
        "vibration_columns": [raw_columns[idx] for idx in vibration_idx],
        "score_scale": "robust_reference_step_signature",
    }
    return NegermetSignatureOutput(score=signature_score.astype(np.float32), components=components, detail=detail)


def fuse_model_with_negermet_signature(
    model_score: np.ndarray,
    signature_output: NegermetSignatureOutput,
    reference_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    model = np.asarray(model_score, dtype=np.float32)
    ref_mask = np.asarray(reference_mask, dtype=bool)
    model_tail = empirical_tail_score(model, model[ref_mask])
    fused = model.astype(np.float32)
    signature_dominates = signature_output.score > model_tail
    components = {
        "paano_score": model.astype(np.float32),
        "paano_tail_score": model_tail.astype(np.float32),
        **signature_output.components,
        "negermet_signature_dominates": signature_dominates.astype(np.float32),
    }
    detail = {
        "fusion": "paano_plus_tuned_negermet_signature",
        "score_scale": "paano_score_with_optional_negermet_signature_boost",
        "negermet_signature": signature_output.detail,
    }
    return fused, components, detail
