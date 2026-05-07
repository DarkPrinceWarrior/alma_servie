from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alma_service.engineered_features import PreparedWellData
from alma_service.anomaly_physics import SALT_PHYSICAL_SPEC

EPS = 1e-6

GROUP_COLUMNS = SALT_PHYSICAL_SPEC.groups


@dataclass
class SaltTrendOutput:
    score: np.ndarray
    components: dict[str, np.ndarray]
    detail: dict[str, object]


def _zero_output(n: int, reason: str) -> SaltTrendOutput:
    zero = np.zeros(n, dtype=np.float32)
    components = {
        "salt_deposition_score": zero.copy(),
        "salt_deposition_fusion_score": zero.copy(),
        "salt_deposition_raw_fusion_score": zero.copy(),
        "salt_group_agreement": zero.copy(),
        "salt_drift_horizon": zero.copy(),
        "salt_multivariate_residual_score": zero.copy(),
        "salt_feature_residual_score": zero.copy(),
        "salt_distribution_shift_score": zero.copy(),
        "salt_distribution_shift_tail_score": zero.copy(),
        "salt_distribution_shift_excess_score": zero.copy(),
        "salt_distribution_shift_raw_score": zero.copy(),
        "salt_distribution_shift_horizon": zero.copy(),
    }
    for group in GROUP_COLUMNS:
        components[f"salt_{group}_drift"] = zero.copy()
        components[f"salt_{group}_direction"] = zero.copy()
    return SaltTrendOutput(
        score=zero.copy(),
        components=components,
        detail={"salt_trend_enabled": False, "reason": reason},
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
    scale = max(scale, abs(med) / np.sqrt(float(len(ref))), 1.0 / np.sqrt(float(len(ref))), EPS)
    finite = np.isfinite(x)
    out[finite] = np.maximum(0.0, (x[finite] - med) / scale).astype(np.float32)
    return out


def _conformal_tail_score(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    ref = np.asarray(reference_values, dtype=np.float32)
    ref = ref[np.isfinite(ref)]
    out = np.zeros(len(x), dtype=np.float32)
    if len(ref) < 8:
        return out
    ref_sorted = np.sort(ref)
    finite = np.isfinite(x)
    ranks = np.searchsorted(ref_sorted, x[finite], side="right").astype(np.float32)
    out[finite] = (100.0 * ranks / (len(ref_sorted) + 1.0)).astype(np.float32)
    return np.clip(out, 0.0, 100.0)


def _two_sample_ks_scaled(window_values: np.ndarray, reference_sorted: np.ndarray) -> float:
    sample = np.asarray(window_values, dtype=np.float32)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 8 or len(reference_sorted) < 8:
        return 0.0
    sample_sorted = np.sort(sample)
    grid = np.union1d(sample_sorted, reference_sorted)
    if len(grid) == 0:
        return 0.0
    sample_cdf = np.searchsorted(sample_sorted, grid, side="right") / float(len(sample_sorted))
    ref_cdf = np.searchsorted(reference_sorted, grid, side="right") / float(len(reference_sorted))
    ks = float(np.max(np.abs(sample_cdf - ref_cdf)))
    scale = np.sqrt((len(sample_sorted) * len(reference_sorted)) / float(len(sample_sorted) + len(reference_sorted)))
    return float(ks * scale)


def _distribution_shift_score(
    values: np.ndarray,
    reference_mask: np.ndarray,
    horizons: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = _filled_signal(np.asarray(values, dtype=np.float32))
    ref_mask = np.asarray(reference_mask, dtype=bool)
    n = len(x)
    ref = x[ref_mask & np.isfinite(x)]
    if len(ref) < 32:
        zero = np.zeros(n, dtype=np.float32)
        return zero.copy(), zero.copy(), zero.copy(), zero.copy(), zero.copy()

    ref_sorted = np.sort(ref.astype(np.float32))
    best_raw = np.zeros(n, dtype=np.float32)
    best_horizon = np.zeros(n, dtype=np.float32)
    min_ref_window = max(8, int(np.floor(np.sqrt(float(len(ref))))))
    for horizon in horizons:
        window = max(int(horizon), min_ref_window)
        if window < 8 or window > n:
            continue
        raw = np.zeros(n, dtype=np.float32)
        min_periods = max(8, int(np.ceil(window * 0.5)))
        for idx in range(window - 1, n):
            current = x[idx - window + 1 : idx + 1]
            current = current[np.isfinite(current)]
            if len(current) < min_periods:
                continue
            raw[idx] = _two_sample_ks_scaled(current, ref_sorted)
        better = raw > best_raw
        best_raw[better] = raw[better]
        best_horizon[better] = float(window)

    tail = _conformal_tail_score(best_raw, best_raw[ref_mask])
    excess = _robust_excess_score(best_raw, best_raw[ref_mask])
    calibrated = tail + excess
    return (
        calibrated.astype(np.float32),
        tail.astype(np.float32),
        excess.astype(np.float32),
        best_raw.astype(np.float32),
        best_horizon.astype(np.float32),
    )


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


def _adaptive_horizons(reference_points: int, total_points: int) -> list[int]:
    if reference_points < 32 or total_points < 64:
        return []
    ref = float(reference_points)
    candidates = [
        np.sqrt(ref),
        np.sqrt(ref) * 2.0,
        np.sqrt(ref) * 4.0,
        ref / 64.0,
        ref / 32.0,
        ref / 16.0,
        ref / 8.0,
    ]
    lower = max(4, int(np.floor(np.sqrt(ref) * 0.5)))
    upper = max(lower, min(int(reference_points * 0.50), int(total_points * 0.30)))
    horizons = {
        int(np.clip(round(candidate), lower, upper))
        for candidate in candidates
        if np.isfinite(candidate)
    }
    return sorted(h for h in horizons if h >= lower)


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


def _signal_map(prepared: PreparedWellData) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for idx, name in enumerate(prepared.raw_columns):
        out[name] = prepared.raw_matrix[:, idx]
    for idx, name in enumerate(prepared.feature_columns):
        if name.endswith("::raw"):
            out.setdefault(name.removesuffix("::raw"), prepared.feature_matrix[:, idx])
    return out


def _multivariate_residual_score(
    signals: dict[str, np.ndarray],
    reference_mask: np.ndarray,
) -> tuple[np.ndarray, list[str], int]:
    names: list[str] = []
    seen: set[str] = set()
    for group_names in GROUP_COLUMNS.values():
        for name in group_names:
            if name in seen or name not in signals:
                continue
            seen.add(name)
            names.append(name)
    if len(names) < 2 or int(reference_mask.sum()) < 32:
        return np.zeros(len(reference_mask), dtype=np.float32), [], 0

    columns: list[np.ndarray] = []
    used: list[str] = []
    for name in names:
        x = _filled_signal(signals[name])
        ref = x[reference_mask & np.isfinite(x)]
        if len(ref) < 32:
            continue
        med, scale = _robust_stats(ref)
        denom = max(abs(med), scale, EPS)
        scale = max(scale, denom / np.sqrt(float(len(ref))), EPS)
        z = (x - med) / scale
        if not np.isfinite(z).any():
            continue
        columns.append(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
        used.append(name)
    if len(columns) < 2:
        return np.zeros(len(reference_mask), dtype=np.float32), [], 0

    x_all = np.column_stack(columns).astype(np.float32)
    x_ref = x_all[reference_mask]
    if x_ref.shape[0] < 32 or x_ref.shape[1] < 2:
        return np.zeros(len(reference_mask), dtype=np.float32), [], 0

    try:
        _, singular, vt = np.linalg.svd(x_ref, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(len(reference_mask), dtype=np.float32), [], 0
    if len(singular) == 0 or not np.isfinite(singular).any():
        return np.zeros(len(reference_mask), dtype=np.float32), [], 0

    variance = np.square(singular)
    total_variance = float(np.sum(variance))
    if total_variance <= EPS:
        return np.zeros(len(reference_mask), dtype=np.float32), [], 0
    explained = np.cumsum(variance) / total_variance
    max_components = max(1, min(10, x_ref.shape[1] - 1, x_ref.shape[0] - 1))
    n_components = int(np.searchsorted(explained, 0.95) + 1)
    n_components = max(1, min(max_components, n_components))

    components = vt[:n_components]
    projected = x_all @ components.T
    reconstructed = projected @ components
    residual = x_all - reconstructed
    spe = np.sum(np.square(residual), axis=1).astype(np.float32)

    score_scale = np.maximum(
        np.square(singular[:n_components]) / max(float(x_ref.shape[0] - 1), 1.0),
        EPS,
    )
    t2 = np.sum(np.square(projected) / score_scale, axis=1).astype(np.float32)
    spe_score = _robust_excess_score(spe, spe[reference_mask])
    t2_score = _robust_excess_score(t2, t2[reference_mask])
    score = np.maximum(spe_score, 0.50 * t2_score)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score.astype(np.float32), used, n_components


def _feature_residual_score(
    prepared: PreparedWellData,
    reference_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    if int(reference_mask.sum()) < 32 or prepared.feature_matrix.shape[1] < 2:
        return np.zeros(len(reference_mask), dtype=np.float32), 0

    from alma_service.generic_detectors import PCASPEDetector

    detector = PCASPEDetector()
    try:
        detector.fit_reference(prepared.feature_matrix[reference_mask])
        output = detector.score_stream(prepared.feature_matrix)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return np.zeros(len(reference_mask), dtype=np.float32), 0

    n_components = int(output.detail.get("n_components", 0))
    score = np.nan_to_num(output.primary, nan=0.0, posinf=0.0, neginf=0.0)
    return score.astype(np.float32), n_components


def _best_drift_score(
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
    effective_scale = max(ref_scale, denom / np.sqrt(float(len(ref_values))), EPS)

    for horizon in horizons:
        current = _rolling_median(x, horizon)
        previous = np.roll(current, horizon)
        previous[:horizon] = np.nan

        level_delta = current - ref_median
        change_delta = current - previous
        level_abs = np.abs(level_delta) / effective_scale
        change_abs = np.abs(change_delta) / effective_scale
        practical_abs = np.abs(level_delta) / denom

        level_excess = _robust_excess_score(level_abs, level_abs[finite_ref])
        change_excess = _robust_excess_score(change_abs, change_abs[finite_ref])
        practical_excess = _robust_excess_score(practical_abs, practical_abs[finite_ref])

        instant = np.maximum(level_excess, 0.75 * change_excess)
        persistence = _rolling_mean(instant, max(4, int(round(np.sqrt(horizon)))))
        persistence_excess = _robust_excess_score(persistence, persistence[finite_ref])
        horizon_score = np.power(np.maximum(level_excess, persistence_excess), 0.80)
        horizon_score *= np.power(practical_excess + 1.0, 0.20)
        horizon_score = np.nan_to_num(horizon_score, nan=0.0, posinf=0.0, neginf=0.0)

        mask = horizon_score > best_score
        if np.any(mask):
            best_score[mask] = horizon_score[mask]
            best_direction[mask] = np.sign(level_delta[mask]).astype(np.float32)
            best_horizon[mask] = float(horizon)

    return best_score, best_direction, best_horizon


def _group_score(
    signals: dict[str, np.ndarray],
    names: tuple[str, ...],
    reference_mask: np.ndarray,
    horizons: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    n = len(reference_mask)
    best_score = np.zeros(n, dtype=np.float32)
    best_direction = np.zeros(n, dtype=np.float32)
    best_horizon = np.zeros(n, dtype=np.float32)
    used: list[str] = []
    for name in names:
        values = signals.get(name)
        if values is None:
            continue
        used.append(name)
        score, direction, horizon = _best_drift_score(values, reference_mask, horizons)
        mask = score > best_score
        if np.any(mask):
            best_score[mask] = score[mask]
            best_direction[mask] = direction[mask]
            best_horizon[mask] = horizon[mask]
    return best_score, best_direction, best_horizon, used


def build_salt_deposition_branch(prepared: PreparedWellData) -> SaltTrendOutput:
    n = len(prepared.timestamps)
    if n < 64:
        return _zero_output(n, "not_enough_points")

    reference_mask = np.asarray(prepared.reference_mask, dtype=bool)
    if int(reference_mask.sum()) < 32:
        reference_mask = np.zeros(n, dtype=bool)
        reference_mask[: max(1, int(prepared.reference_end_idx))] = True
    horizons = _adaptive_horizons(int(reference_mask.sum()), n)
    if not horizons:
        return _zero_output(n, "no_adaptive_horizons")

    signals = _signal_map(prepared)
    group_scores: dict[str, np.ndarray] = {}
    group_directions: dict[str, np.ndarray] = {}
    group_horizons: list[np.ndarray] = []
    used_groups: dict[str, list[str]] = {}

    for group, names in GROUP_COLUMNS.items():
        score, direction, horizon, used = _group_score(signals, names, reference_mask, horizons)
        group_scores[group] = score.astype(np.float32)
        group_directions[group] = direction.astype(np.float32)
        group_horizons.append(horizon.astype(np.float32))
        used_groups[group] = used

    active_scores = [score for group, score in group_scores.items() if used_groups[group]]
    if not active_scores:
        return _zero_output(n, "no_supported_groups")

    multivariate_score, multivariate_columns, multivariate_components = _multivariate_residual_score(
        signals,
        reference_mask,
    )
    feature_score, feature_components = _feature_residual_score(prepared, reference_mask)
    residual_signal = np.maximum(feature_score, multivariate_score).astype(np.float32)
    shift_score, shift_tail, shift_excess, shift_raw, shift_horizon = _distribution_shift_score(
        residual_signal,
        reference_mask,
        horizons,
    )
    active_scores.append(multivariate_score.astype(np.float32))
    score_matrix = np.vstack(active_scores)
    sorted_scores = np.sort(score_matrix, axis=0)
    strongest = sorted_scores[-1]
    second = sorted_scores[-2] if len(active_scores) > 1 else np.zeros(n, dtype=np.float32)
    mean_score = np.mean(score_matrix, axis=0)
    ref_group_medians = np.array(
        [
            np.nanmedian(score[reference_mask]) if np.any(reference_mask) else 0.0
            for score in active_scores
        ],
        dtype=np.float32,
    )
    agreement_threshold = np.maximum(np.nanmedian(ref_group_medians), EPS)
    agreement = (score_matrix > agreement_threshold).sum(axis=0).astype(np.float32)
    agreement_norm = agreement / max(float(len(active_scores)), 1.0)

    deposition_score = strongest + 0.45 * second + 0.20 * mean_score
    deposition_score *= 1.0 + 0.35 * agreement_norm
    deposition_score = np.maximum(deposition_score, multivariate_score)
    deposition_score = np.maximum(deposition_score, feature_score)
    deposition_score = np.nan_to_num(deposition_score, nan=0.0, posinf=0.0, neginf=0.0)
    # Raw grouped drift and PCA/SPE residuals are not comparable across wells.
    # The train-tuned fusion branch therefore receives only reference-calibrated
    # 0..100 tail scores.
    feature_tail = _conformal_tail_score(feature_score, feature_score[reference_mask])
    multivariate_tail = _conformal_tail_score(multivariate_score, multivariate_score[reference_mask])
    drift_tail = _conformal_tail_score(deposition_score, deposition_score[reference_mask])
    fusion_score = np.maximum.reduce([feature_tail, multivariate_tail, shift_score])
    fusion_score = np.nan_to_num(fusion_score, nan=0.0, posinf=0.0, neginf=0.0)
    raw_fusion_score = np.maximum(feature_score, 0.0)
    raw_fusion_score = np.nan_to_num(raw_fusion_score, nan=0.0, posinf=0.0, neginf=0.0)
    drift_horizon = np.nanmax(np.vstack(group_horizons), axis=0) if group_horizons else np.zeros(n)

    components = {
        "salt_deposition_score": deposition_score.astype(np.float32),
        "salt_deposition_fusion_score": fusion_score.astype(np.float32),
        "salt_deposition_raw_fusion_score": raw_fusion_score.astype(np.float32),
        "salt_deposition_conformal_tail_score": drift_tail.astype(np.float32),
        "salt_group_agreement": agreement_norm.astype(np.float32),
        "salt_drift_horizon": drift_horizon.astype(np.float32),
        "salt_multivariate_residual_score": multivariate_score.astype(np.float32),
        "salt_multivariate_residual_tail_score": multivariate_tail.astype(np.float32),
        "salt_feature_residual_score": feature_score.astype(np.float32),
        "salt_feature_residual_tail_score": feature_tail.astype(np.float32),
        "salt_distribution_shift_score": shift_score.astype(np.float32),
        "salt_distribution_shift_tail_score": shift_tail.astype(np.float32),
        "salt_distribution_shift_excess_score": shift_excess.astype(np.float32),
        "salt_distribution_shift_raw_score": shift_raw.astype(np.float32),
        "salt_distribution_shift_horizon": shift_horizon.astype(np.float32),
    }
    for group in GROUP_COLUMNS:
        components[f"salt_{group}_drift"] = group_scores[group].astype(np.float32)
        components[f"salt_{group}_direction"] = group_directions[group].astype(np.float32)

    detail = {
        "salt_trend_enabled": True,
        "reference_points": int(reference_mask.sum()),
        "adaptive_horizons": [int(h) for h in horizons],
        "groups": used_groups,
        "multivariate_columns": multivariate_columns,
        "multivariate_components": int(multivariate_components),
        "feature_residual_components": int(feature_components),
        "score_scale": "robust_reference_grouped_drift_plus_multivariate_and_feature_residual",
        "distribution_shift": "moving_window_ks_on_residual_signal",
        "fusion_score": "conformal_tail_max_of_feature_multivariate_and_distribution_shift",
    }
    return SaltTrendOutput(score=deposition_score.astype(np.float32), components=components, detail=detail)


def fuse_model_with_salt_trend(
    model_score: np.ndarray,
    salt_output: SaltTrendOutput,
    reference_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    model = np.asarray(model_score, dtype=np.float32)
    ref_mask = np.asarray(reference_mask, dtype=bool)
    model_tail = _conformal_tail_score(model, model[ref_mask])
    drift_tail = _conformal_tail_score(salt_output.score, salt_output.score[ref_mask])
    feature_score = salt_output.components.get("salt_feature_residual_score")
    if feature_score is None:
        feature_tail = np.zeros(len(model), dtype=np.float32)
    else:
        feature = np.asarray(feature_score, dtype=np.float32)
        feature_tail = _conformal_tail_score(feature, feature[ref_mask])
    distribution_shift = salt_output.components.get("salt_distribution_shift_score")
    if distribution_shift is None:
        shift_fusion = np.zeros(len(model), dtype=np.float32)
    else:
        shift_fusion = np.asarray(distribution_shift, dtype=np.float32)
    salt_tail = np.maximum.reduce([drift_tail, feature_tail, shift_fusion])
    fusion_score = salt_output.components.get("salt_deposition_fusion_score")
    if fusion_score is None:
        calibrated_fusion = salt_tail
    else:
        calibrated_fusion = np.asarray(fusion_score, dtype=np.float32)
    fused = model.astype(np.float32)
    salt_dominates = salt_tail > model_tail
    components = {
        "paano_score": model.astype(np.float32),
        "paano_tail_score": model_tail.astype(np.float32),
        **salt_output.components,
        "salt_deposition_drift_tail_score": drift_tail.astype(np.float32),
        "salt_feature_residual_tail_score": feature_tail.astype(np.float32),
        "salt_distribution_shift_fusion_score": shift_fusion.astype(np.float32),
        "salt_deposition_tail_score": salt_tail.astype(np.float32),
        "salt_deposition_calibrated_fusion_score": calibrated_fusion.astype(np.float32),
        "salt_deposition_dominates": salt_dominates.astype(np.float32),
    }
    detail = {
        "fusion": "paano_plus_tuned_salt_deposition_residual",
        "score_scale": "paano_score_with_train_tuned_conformal_salt_tail_boost",
        "tail_calibration": "per_well_reference_conformal_rank",
        "salt_trend": salt_output.detail,
    }
    return fused, components, detail
