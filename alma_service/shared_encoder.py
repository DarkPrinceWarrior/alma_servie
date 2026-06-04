"""
Shared PaAno encoder training across wells within an anomaly family.

Provides utilities to:
1. Collect clean-normal data from all train wells into a unified pool
2. Resolve a common feature set (intersection of per-well features)
3. Train a single PaAno encoder on the combined pool
4. Build local memory banks per-well using the shared encoder
5. Score a well using the shared encoder + its local memory bank
"""

from __future__ import annotations

import hashlib
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAANO_ROOT = PROJECT_ROOT / "paano"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PAANO_ROOT) not in sys.path:
    sys.path.insert(0, str(PAANO_ROOT))

from model import PatchEncoder
from train import train_model
from utils.data_preprocess import PatchCreator, preprocess_to_patches
from utils.evaluation import calculate_anomaly_scores, distribute_patch_scores_to_points
from utils.utils import create_memory_bank

from alma_service.generic_detectors import (
    PAANO_BATCH_SIZE,
    PAANO_INFER_BATCH_SIZE,
    PAANO_LR,
    PAANO_MEMORY_BANK_RATIO,
    PAANO_NUM_ITERS,
    PAANO_TOP_K,
    SEED,
    _ensure_2d_float32,
    _maybe_compile_module,
)
from alma_service.feature_schema import feature_root
from alma_service.paths import MODELS_DIR, ensure_parent


MISSING_RAW_CHANNEL_PREFIX = "__missing_raw__::"


def missing_raw_channel_feature(raw_channel: str) -> str:
    return f"{MISSING_RAW_CHANNEL_PREFIX}{raw_channel}"


def _is_missing_raw_channel_feature(channel: str) -> bool:
    return str(channel).startswith(MISSING_RAW_CHANNEL_PREFIX)


def _missing_raw_channel_name(mask_feature: str) -> str:
    return str(mask_feature)[len(MISSING_RAW_CHANNEL_PREFIX):]


@dataclass
class SharedEncoderState:
    """Holds a trained shared encoder with its normalization statistics."""

    model_short: nn.Module
    model_long: nn.Module
    train_mean_short: np.ndarray
    train_std_short: np.ndarray
    train_mean_long: np.ndarray
    train_std_long: np.ndarray
    shared_channels: list[str]
    patch_short: int
    patch_long: int
    anomaly_key: str
    train_wells: list[str]
    detail: dict[str, Any]


def shared_encoder_path(anomaly_key: str) -> Path:
    return MODELS_DIR / f"{anomaly_key}_paano_shared_encoder.pt"


def _state_dict_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def save_shared_encoder_state(state: SharedEncoderState, path: str | Path | None = None) -> Path:
    output_path = ensure_parent(Path(path) if path is not None else shared_encoder_path(state.anomaly_key))
    payload = {
        "anomaly_key": state.anomaly_key,
        "patch_short": int(state.patch_short),
        "patch_long": int(state.patch_long),
        "shared_channels": list(state.shared_channels),
        "train_wells": list(state.train_wells),
        "detail": dict(state.detail),
        "train_mean_short": np.asarray(state.train_mean_short, dtype=np.float32),
        "train_std_short": np.asarray(state.train_std_short, dtype=np.float32),
        "train_mean_long": np.asarray(state.train_mean_long, dtype=np.float32),
        "train_std_long": np.asarray(state.train_std_long, dtype=np.float32),
        "model_short_state_dict": _state_dict_model(state.model_short).state_dict(),
        "model_long_state_dict": _state_dict_model(state.model_long).state_dict(),
    }
    torch.save(payload, output_path)
    return output_path


def load_shared_encoder_state(
    anomaly_key: str,
    path: str | Path | None = None,
    *,
    device: torch.device,
    compile_model: bool = True,
    verbose: bool = False,
) -> SharedEncoderState:
    input_path = Path(path) if path is not None else shared_encoder_path(anomaly_key)
    if not input_path.exists():
        raise FileNotFoundError(f"Shared encoder artifact not found: {input_path}")
    payload = torch.load(input_path, map_location=device, weights_only=False)
    stored_anomaly = str(payload["anomaly_key"])
    if stored_anomaly != anomaly_key:
        raise ValueError(f"Shared encoder artifact is for {stored_anomaly}, requested {anomaly_key}")

    shared_channels = [str(channel) for channel in payload["shared_channels"]]
    model_short = PatchEncoder(in_channels=len(shared_channels), use_revin=True).to(device)
    model_long = PatchEncoder(in_channels=len(shared_channels), use_revin=True).to(device)
    model_short.load_state_dict(payload["model_short_state_dict"])
    model_long.load_state_dict(payload["model_long_state_dict"])
    model_short.eval()
    model_long.eval()
    if compile_model:
        model_short = _maybe_compile_module(model_short, label=f"SharedEncoder_load_patch{payload['patch_short']}", verbose=verbose)
        model_long = _maybe_compile_module(model_long, label=f"SharedEncoder_load_patch{payload['patch_long']}", verbose=verbose)

    return SharedEncoderState(
        model_short=model_short,
        model_long=model_long,
        train_mean_short=np.asarray(payload["train_mean_short"], dtype=np.float32),
        train_std_short=np.asarray(payload["train_std_short"], dtype=np.float32),
        train_mean_long=np.asarray(payload["train_mean_long"], dtype=np.float32),
        train_std_long=np.asarray(payload["train_std_long"], dtype=np.float32),
        shared_channels=shared_channels,
        patch_short=int(payload["patch_short"]),
        patch_long=int(payload["patch_long"]),
        anomaly_key=stored_anomaly,
        train_wells=[str(well_id) for well_id in payload["train_wells"]],
        detail=dict(payload.get("detail", {})),
    )


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _raw_channels_represented_by_features(
    feature_channels: list[str],
    raw_channels: list[str],
) -> list[str]:
    roots = {feature_root(channel) for channel in feature_channels}
    return [raw for raw in raw_channels if raw in roots]


def _append_missing_channel_masks(
    pool: np.ndarray,
    feature_channels: list[str],
    raw_channels: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    if not raw_channels:
        return pool, feature_channels
    represented = _raw_channels_represented_by_features(feature_channels, list(raw_channels))
    if not represented:
        return pool, feature_channels
    mask_channels = [missing_raw_channel_feature(raw) for raw in represented]
    masks = np.zeros((pool.shape[0], len(mask_channels)), dtype=np.float32)
    return (
        np.concatenate([pool.astype(np.float32), masks], axis=1).astype(np.float32),
        [*feature_channels, *mask_channels],
    )


def _apply_channel_dropout(
    pool: np.ndarray,
    shared_channels: list[str],
    *,
    rate: float,
    seed: int,
) -> np.ndarray:
    rate = float(rate)
    if rate <= 0.0:
        return pool
    if rate >= 1.0:
        raise ValueError(f"channel_dropout_rate must be < 1.0, got {rate}")

    out = np.asarray(pool, dtype=np.float32).copy()
    rng = np.random.default_rng(int(seed))
    mask_lookup = {
        _missing_raw_channel_name(channel): idx
        for idx, channel in enumerate(shared_channels)
        if _is_missing_raw_channel_feature(channel)
    }
    for raw_channel, mask_idx in mask_lookup.items():
        feature_indices = [
            idx
            for idx, channel in enumerate(shared_channels)
            if not _is_missing_raw_channel_feature(channel) and feature_root(channel) == raw_channel
        ]
        if not feature_indices:
            continue
        rows = np.flatnonzero(rng.random(out.shape[0]) < rate)
        if len(rows) == 0:
            continue
        out[np.ix_(rows, feature_indices)] = 0.0
        out[rows, mask_idx] = 1.0
    return out


def collect_shared_train_pool(
    prepared_wells: dict[str, Any],
    *,
    only_split: str = "train",
    enable_reduction: bool = True,
    missing_mask_raw_channels: list[str] | None = None,
    channel_dropout_rate: float = 0.0,
    channel_dropout_seed: int = SEED,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Collect clean-normal feature matrices from train wells.

    Parameters
    ----------
    prepared_wells:
        dict of well_id -> PreparedWellData objects (from engineered_features).
    only_split:
        Only include wells with this split designation.

    Returns
    -------
    pool: np.ndarray
        Concatenated clean-normal rows, columns correspond to ``shared_channels``.
    shared_channels: list[str]
        Feature columns used by the encoder. With missing-channel masks enabled,
        this is a schema union projected per well with zeros for absent feature
        families and explicit ``__missing_raw__`` indicators.
    train_well_ids: list[str]
        List of well IDs that contributed to the pool.

    Raises
    ------
    ValueError
        If no common channels exist across train wells.
    """
    train_wells = {
        wid: pw for wid, pw in prepared_wells.items()
        if pw.split == only_split
    }

    if not train_wells:
        raise ValueError(f"No wells with split='{only_split}' found.")

    use_missing_masks = bool(missing_mask_raw_channels)
    if use_missing_masks:
        raw_filter = set(str(channel) for channel in missing_mask_raw_channels or [])
        shared_channels = sorted(
            {
                channel
                for pw in train_wells.values()
                for channel in pw.feature_columns
                if feature_root(channel) in raw_filter
            }
        )
    else:
        # Find intersection of feature columns across all train wells.
        channel_sets = [set(pw.feature_columns) for pw in train_wells.values()]
        shared = channel_sets[0]
        for cs in channel_sets[1:]:
            shared = shared & cs
        shared_channels = sorted(shared)

    if not shared_channels:
        raise ValueError(
            "No common feature channels across train wells. "
            "Cannot build shared encoder."
        )

    pool_parts: list[np.ndarray] = []
    for wid in sorted(train_wells):
        pw = train_wells[wid]
        if use_missing_masks:
            well_matrix = select_shared_columns(
                pw.feature_columns,
                pw.feature_matrix,
                shared_channels,
            )
        else:
            col_indices = [pw.feature_columns.index(ch) for ch in shared_channels]
            well_matrix = pw.feature_matrix[:, col_indices]

        ref_rows = well_matrix[np.asarray(pw.reference_mask, dtype=bool)]
        if len(ref_rows) > 0:
            pool_parts.append(ref_rows)

    if not pool_parts:
        raise ValueError("No clean-normal reference data collected from train wells.")

    pool = np.concatenate(pool_parts, axis=0).astype(np.float32)
    train_well_ids = sorted(train_wells.keys())

    # Feature reduction: remove low-variance and highly correlated features
    if enable_reduction:
        original_count = len(shared_channels)
        pool, shared_channels = reduce_features(pool, shared_channels)
        if len(shared_channels) < original_count:
            print(
                f"    Feature reduction: {original_count} → {len(shared_channels)} channels "
                f"(removed {original_count - len(shared_channels)})"
            )

    if use_missing_masks:
        represented_roots = {feature_root(channel) for channel in shared_channels}
        mask_raw_channels = [
            str(raw_channel)
            for raw_channel in missing_mask_raw_channels or []
            if str(raw_channel) in represented_roots
        ]
        mask_channels = [missing_raw_channel_feature(raw_channel) for raw_channel in mask_raw_channels]
        masked_parts: list[np.ndarray] = []
        for wid in sorted(train_wells):
            pw = train_wells[wid]
            well_matrix = select_shared_columns(
                pw.feature_columns,
                pw.feature_matrix,
                shared_channels,
            )
            available_roots = {feature_root(channel) for channel in pw.feature_columns}
            masks = np.zeros((len(well_matrix), len(mask_raw_channels)), dtype=np.float32)
            for idx, raw_channel in enumerate(mask_raw_channels):
                if raw_channel not in available_roots:
                    masks[:, idx] = 1.0
            if mask_channels:
                well_matrix = np.concatenate([well_matrix, masks], axis=1).astype(np.float32)
            ref_rows = well_matrix[np.asarray(pw.reference_mask, dtype=bool)]
            if len(ref_rows) > 0:
                masked_parts.append(ref_rows)
        if not masked_parts:
            raise ValueError("No clean-normal reference data collected from train wells.")
        pool = np.concatenate(masked_parts, axis=0).astype(np.float32)
        shared_channels = [*shared_channels, *mask_channels]
    else:
        pool, shared_channels = _append_missing_channel_masks(
            pool,
            shared_channels,
            missing_mask_raw_channels,
        )
    pool = _apply_channel_dropout(
        pool,
        shared_channels,
        rate=float(channel_dropout_rate),
        seed=int(channel_dropout_seed),
    )

    return pool, shared_channels, train_well_ids


def reduce_features(
    pool: np.ndarray,
    channels: list[str],
    min_variance_ratio: float = 0.001,
    max_correlation: float = 0.95,
    max_channels: int | None = 80,
    stability_window: int = 32,
) -> tuple[np.ndarray, list[str]]:
    """Two-stage feature reduction for the shared encoder pool.

    Stage 1: Remove low-variance and highly correlated features.
    Stage 2: Score remaining features by reference stability and keep
             the top ``max_channels`` most stable ones.

    Parameters
    ----------
    pool : np.ndarray
        Training pool matrix (N, C).
    channels : list[str]
        Feature channel names corresponding to columns.
    min_variance_ratio : float
        Drop features whose variance is below this fraction of the max variance.
    max_correlation : float
        When two features have |correlation| above this, drop the later one.
    max_channels : int or None
        Maximum number of channels to keep after stability scoring.
        ``None`` disables the stability cap.
    stability_window : int
        Rolling window size for computing residual smoothness.

    Returns
    -------
    pool : np.ndarray
        Filtered pool with fewer columns.
    channels : list[str]
        Surviving channel names.
    """
    if pool.shape[1] <= 1:
        return pool, channels

    # --- Stage 1: variance + correlation filter ---
    # 1a. Remove near-zero variance features
    variances = np.var(pool, axis=0)
    max_var = np.max(variances) if np.max(variances) > 0 else 1.0
    keep_mask = variances >= min_variance_ratio * max_var

    pool = pool[:, keep_mask]
    channels = [ch for ch, k in zip(channels, keep_mask) if k]

    if pool.shape[1] <= 1:
        return pool, channels

    # 1b. Remove highly correlated features (keep first of each pair)
    corr = np.corrcoef(pool.T)
    corr = np.nan_to_num(corr, nan=0.0)
    drop: set[int] = set()
    for i in range(len(corr)):
        if i in drop:
            continue
        for j in range(i + 1, len(corr)):
            if j not in drop and abs(corr[i, j]) > max_correlation:
                drop.add(j)

    if drop:
        keep = [i for i in range(len(channels)) if i not in drop]
        pool = pool[:, keep]
        channels = [channels[i] for i in keep]

    # --- Stage 2: reference stability scoring ---
    if max_channels is not None and pool.shape[1] > max_channels:
        scores = _score_channel_stability(pool, window=stability_window)
        # Lower score = more stable = better for PaAno
        top_indices = np.argsort(scores)[:max_channels]
        top_indices = np.sort(top_indices)  # preserve original order
        pool = pool[:, top_indices]
        channels = [channels[i] for i in top_indices]

    return pool, channels


def _score_channel_stability(pool: np.ndarray, window: int = 32) -> np.ndarray:
    """Score each channel by how noisy it is relative to its trend.

    Lower score = smoother/more predictable on reference data = more useful
    for PaAno encoder training.

    The metric is ``residual_std / global_std`` — the fraction of total
    variance that is *not* captured by a simple moving average.
    """
    n_channels = pool.shape[1]
    scores = np.full(n_channels, np.inf, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window

    for i in range(n_channels):
        col = pool[:, i].astype(np.float64)
        global_std = np.std(col)
        if global_std < 1e-8:
            continue  # constant channel → inf score → will be trimmed
        smoothed = np.convolve(col, kernel, mode="valid")
        residual = col[window - 1 :] - smoothed
        residual_std = np.std(residual)
        scores[i] = residual_std / (global_std + 1e-12)

    return scores


def _train_encoder_single_scale(
    pool: np.ndarray,
    patch_size: int,
    device: torch.device,
    verbose: bool = False,
    num_iter: int = PAANO_NUM_ITERS,
    label_prefix: str = "SharedEncoder",
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """Train PaAno encoder on a combined clean-normal pool at one patch scale."""
    train_mean = np.mean(pool, axis=0, keepdims=True).astype(np.float32)
    train_std = np.std(pool, axis=0, keepdims=True).astype(np.float32)
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)

    pool_norm = (pool - train_mean) / train_std
    dummy_labels = np.zeros(len(pool), dtype=np.float32)

    patch_creator = PatchCreator(L=patch_size, s=1, random_seed=SEED)
    train_loader, _, _ = patch_creator.create_dataloaders(
        pool_norm,
        pool_norm,  # full == train for shared training
        dummy_labels,
        batch_size=PAANO_BATCH_SIZE,
    )

    model = PatchEncoder(in_channels=pool.shape[1], use_revin=True).to(device)
    model = _maybe_compile_module(model, label=f"{label_prefix}_patch{patch_size}", verbose=verbose)

    t0 = time.time()
    train_patches = preprocess_to_patches(pool_norm, patch_size=patch_size, stride=1)
    train_model(
        model,
        train_loader,
        train_patches,
        device,
        num_iter=num_iter,
        pretext_step=patch_size,
        lr=PAANO_LR,
        see_loss=False,
    )
    elapsed = time.time() - t0

    if verbose:
        print(
            f"    Shared encoder patch={patch_size}: pool_pts={len(pool)}, "
            f"channels={pool.shape[1]}, time={elapsed:.1f}s"
        )

    return model, train_mean, train_std


def train_shared_encoder(
    prepared_wells: dict[str, Any],
    patch_short: int,
    patch_long: int,
    anomaly_key: str,
    device: torch.device,
    verbose: bool = False,
    num_iter: int = PAANO_NUM_ITERS,
    enable_reduction: bool | None = None,
    missing_mask_raw_channels: list[str] | None = None,
    channel_dropout_rate: float = 0.0,
    channel_dropout_seed: int = SEED,
) -> SharedEncoderState:
    """Train two shared encoders (short + long scale) on clean-normal data
    from all train wells of the given anomaly family.

    Returns a :class:`SharedEncoderState` with both models and normalization stats.
    """
    _set_seed()
    if enable_reduction is None:
        enable_reduction = anomaly_key != "negermet"

    pool, shared_channels, train_well_ids = collect_shared_train_pool(
        prepared_wells,
        enable_reduction=bool(enable_reduction),
        missing_mask_raw_channels=missing_mask_raw_channels,
        channel_dropout_rate=float(channel_dropout_rate),
        channel_dropout_seed=int(channel_dropout_seed),
    )

    if verbose:
        print(
            f"  Shared encoder pool: {len(pool)} points, "
            f"{len(shared_channels)} channels, {len(train_well_ids)} wells"
        )

    if len(pool) < patch_long * 2:
        raise ValueError(
            f"Shared training pool too small: {len(pool)} points, "
            f"need at least {patch_long * 2}"
        )

    model_short, mean_short, std_short = _train_encoder_single_scale(
        pool, patch_short, device, verbose=verbose, num_iter=num_iter,
    )
    model_long, mean_long, std_long = _train_encoder_single_scale(
        pool, patch_long, device, verbose=verbose, num_iter=num_iter,
    )

    detail = {
        "pool_points": int(len(pool)),
        "shared_channels": len(shared_channels),
        "missing_channel_masks": [
            _missing_raw_channel_name(channel)
            for channel in shared_channels
            if _is_missing_raw_channel_feature(channel)
        ],
        "channel_dropout_rate": float(channel_dropout_rate),
        "channel_dropout_seed": int(channel_dropout_seed),
        "train_wells": train_well_ids,
        "iterations": int(num_iter),
    }

    return SharedEncoderState(
        model_short=model_short,
        model_long=model_long,
        train_mean_short=mean_short,
        train_std_short=std_short,
        train_mean_long=mean_long,
        train_std_long=std_long,
        shared_channels=shared_channels,
        patch_short=patch_short,
        patch_long=patch_long,
        anomaly_key=anomaly_key,
        train_wells=train_well_ids,
        detail=detail,
    )


def _peek_saved_encoder_meta(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  Shared encoder cache: failed to peek {path}: {exc}")
        return None
    return {
        "anomaly_key": str(payload.get("anomaly_key", "")),
        "patch_short": int(payload.get("patch_short", -1)),
        "patch_long": int(payload.get("patch_long", -1)),
        "shared_channels": [str(c) for c in payload.get("shared_channels", [])],
        "detail": dict(payload.get("detail", {})),
    }


def load_or_train_shared_encoder(
    prepared_wells: dict[str, Any],
    patch_short: int,
    patch_long: int,
    anomaly_key: str,
    device: torch.device,
    verbose: bool = False,
    num_iter: int = PAANO_NUM_ITERS,
    enable_reduction: bool | None = None,
    missing_mask_raw_channels: list[str] | None = None,
    channel_dropout_rate: float = 0.0,
    channel_dropout_seed: int = SEED,
) -> SharedEncoderState:
    """Load a saved shared encoder if one exists in models/ and is compatible
    with the current train pool; otherwise train from scratch.

    Compatibility check requires the saved channel list to match the current
    pool's channel list exactly (order-sensitive) and both patch sizes to
    match. Override with ALMA_FORCE_RETRAIN_ENCODER=1.
    """
    force_retrain = os.environ.get("ALMA_FORCE_RETRAIN_ENCODER", "0") == "1"
    cache_path = shared_encoder_path(anomaly_key)
    meta = _peek_saved_encoder_meta(cache_path) if not force_retrain else None
    if enable_reduction is None:
        enable_reduction = anomaly_key != "negermet"

    if meta is not None:
        pool, shared_channels, train_well_ids = collect_shared_train_pool(
            prepared_wells,
            enable_reduction=bool(enable_reduction),
            missing_mask_raw_channels=missing_mask_raw_channels,
            channel_dropout_rate=float(channel_dropout_rate),
            channel_dropout_seed=int(channel_dropout_seed),
        )
        channels_match = list(meta["shared_channels"]) == list(shared_channels)
        patch_match = (
            int(meta["patch_short"]) == int(patch_short)
            and int(meta["patch_long"]) == int(patch_long)
        )
        key_match = str(meta["anomaly_key"]) == anomaly_key
        detail = dict(meta.get("detail", {}))
        dropout_match = float(detail.get("channel_dropout_rate", 0.0)) == float(channel_dropout_rate)
        mask_match = sorted(str(item) for item in detail.get("missing_channel_masks", [])) == sorted(
            _missing_raw_channel_name(channel)
            for channel in shared_channels
            if _is_missing_raw_channel_feature(channel)
        )

        if channels_match and patch_match and key_match and dropout_match and mask_match:
            if verbose:
                src = meta["detail"].get("training_mode", "shared")
                print(
                    f"  Shared encoder cache HIT: loading {cache_path.name} "
                    f"(channels={len(shared_channels)}, patch=({patch_short},{patch_long}), src={src})"
                )
            return load_shared_encoder_state(
                anomaly_key=anomaly_key,
                path=cache_path,
                device=device,
                compile_model=True,
                verbose=verbose,
            )

        reasons = []
        if not key_match:
            reasons.append(f"anomaly_key mismatch ({meta['anomaly_key']} vs {anomaly_key})")
        if not patch_match:
            reasons.append(
                f"patch sizes mismatch (saved=({meta['patch_short']},{meta['patch_long']}) vs current=({patch_short},{patch_long}))"
            )
        if not channels_match:
            reasons.append(
                f"channel set mismatch (saved={len(meta['shared_channels'])} vs current={len(shared_channels)})"
            )
        if not dropout_match:
            reasons.append(
                f"channel dropout mismatch (saved={detail.get('channel_dropout_rate', 0.0)} vs current={channel_dropout_rate})"
            )
        if not mask_match:
            reasons.append("missing-mask contract mismatch")
        print(f"  Shared encoder cache STALE: {'; '.join(reasons)} -> retraining from scratch")
    elif force_retrain and cache_path.exists():
        print(f"  Shared encoder cache OVERRIDE (ALMA_FORCE_RETRAIN_ENCODER=1): retraining {cache_path.name}")

    state = train_shared_encoder(
        prepared_wells=prepared_wells,
        patch_short=patch_short,
        patch_long=patch_long,
        anomaly_key=anomaly_key,
        device=device,
        verbose=verbose,
        num_iter=int(num_iter),
        enable_reduction=bool(enable_reduction),
        missing_mask_raw_channels=missing_mask_raw_channels,
        channel_dropout_rate=float(channel_dropout_rate),
        channel_dropout_seed=int(channel_dropout_seed),
    )
    save_shared_encoder_state(state, cache_path)
    return state


# Кэш memory bank для больших (популяционных) эталонов: банк 48212 точек одинаков на всех
# скважинах батча, а create_memory_bank (эмбеддинг патчей + детерминированный kmeans seed=42)
# даёт бит-в-бит тот же результат — нет смысла пересчитывать на каждую скважину.
_POPULATION_BANK_CACHE: dict[Any, Any] = {}
_BANK_CACHE_MIN_REF = 5000
_BANK_CACHE_MAX_ENTRIES = 8


def _array_fingerprint(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr, dtype=np.float32)
    return hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).hexdigest()


def _model_fingerprint(model: nn.Module) -> str:
    try:
        param = next(model.parameters())
    except StopIteration:
        return "noparams"
    flat = param.detach().to("cpu", dtype=torch.float32).contiguous().view(-1)[:4096]
    return hashlib.blake2b(flat.numpy().tobytes(), digest_size=8).hexdigest()


def _cached_memory_bank(model, ref_norm, patch_creator, patch_size, device):
    use_cache = len(ref_norm) >= _BANK_CACHE_MIN_REF
    key = None
    if use_cache:
        key = (int(patch_size), tuple(ref_norm.shape),
               _array_fingerprint(ref_norm), _model_fingerprint(model))
        cached = _POPULATION_BANK_CACHE.get(key)
        if cached is not None:
            return cached
    ref_loader, _, _ = patch_creator.create_dataloaders(
        ref_norm, ref_norm, np.zeros(len(ref_norm), dtype=np.float32),
        batch_size=PAANO_INFER_BATCH_SIZE,
    )
    memory_bank, _ = create_memory_bank(
        model, ref_loader, device, num_cores=PAANO_MEMORY_BANK_RATIO,
    )
    if use_cache and key is not None:
        if len(_POPULATION_BANK_CACHE) >= _BANK_CACHE_MAX_ENTRIES:
            _POPULATION_BANK_CACHE.clear()
        _POPULATION_BANK_CACHE[key] = memory_bank
    return memory_bank


def _score_well_single_scale(
    model: nn.Module,
    well_data: np.ndarray,
    ref_data: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    patch_size: int,
    device: torch.device,
    verbose: bool = False,
) -> np.ndarray:
    """Score a single well at a single scale using a (shared) encoder.

    - ``ref_data`` is used to build the **local** memory bank.
    - ``well_data`` is the full timeseries to score.
    """
    well_norm = (well_data - train_mean) / train_std
    ref_norm = (ref_data - train_mean) / train_std
    dummy_labels = np.zeros(len(well_data), dtype=np.float32)

    patch_creator = PatchCreator(L=patch_size, s=1, random_seed=SEED)

    # DataLoader for full well scoring
    _, full_loader, _ = patch_creator.create_dataloaders(
        ref_norm, well_norm, dummy_labels,
        batch_size=PAANO_INFER_BATCH_SIZE,
    )

    # Memory bank из эталона (кэшируется для популяционного банка — см. _cached_memory_bank)
    memory_bank = _cached_memory_bank(model, ref_norm, patch_creator, patch_size, device)

    patch_scores = calculate_anomaly_scores(
        model, full_loader, memory_bank, top_k=PAANO_TOP_K, device=device,
    )
    point_scores = distribute_patch_scores_to_points(
        patch_scores, patch_size=patch_size, num_points=len(well_data),
    )

    if verbose:
        print(f"    Shared score patch={patch_size}: well_pts={len(well_data)}, ref_pts={len(ref_data)}")

    return np.asarray(point_scores, dtype=np.float32)


def select_shared_columns(
    feature_columns: list[str],
    feature_matrix: np.ndarray,
    shared_channels: list[str],
) -> np.ndarray:
    """Project a well's feature matrix to the shared channel set."""
    col_lookup = {name: idx for idx, name in enumerate(feature_columns)}
    available_roots = {feature_root(name) for name in feature_columns}
    projected = np.zeros((feature_matrix.shape[0], len(shared_channels)), dtype=np.float32)
    for out_idx, channel in enumerate(shared_channels):
        if _is_missing_raw_channel_feature(channel):
            raw_channel = _missing_raw_channel_name(channel)
            projected[:, out_idx] = 0.0 if raw_channel in available_roots else 1.0
            continue
        src_idx = col_lookup.get(channel)
        if src_idx is not None:
            projected[:, out_idx] = feature_matrix[:, src_idx].astype(np.float32)
    return projected
