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
    PAANO_LR,
    PAANO_MEMORY_BANK_RATIO,
    PAANO_NUM_ITERS,
    PAANO_TOP_K,
    SEED,
    _ensure_2d_float32,
    _maybe_compile_module,
)
from alma_service.paths import MODELS_DIR, ensure_parent


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


def collect_shared_train_pool(
    prepared_wells: dict[str, Any],
    *,
    only_split: str = "train",
    enable_reduction: bool = True,
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
        Feature columns present in ALL train wells (intersection).
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

    # Find intersection of feature columns across all train wells
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
        # Select only shared columns in consistent order
        col_indices = [pw.feature_columns.index(ch) for ch in shared_channels]
        well_matrix = pw.feature_matrix[:, col_indices]

        # Use reference_mask (already zone-filtered if intervals were provided)
        ref_rows = well_matrix[pw.reference_mask]
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
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """Train PaAno encoder on a combined pool at a single patch scale.

    Returns (model, train_mean, train_std).
    """
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
    model = _maybe_compile_module(model, label=f"SharedEncoder_patch{patch_size}", verbose=verbose)

    t0 = time.time()
    train_patches = preprocess_to_patches(pool_norm, patch_size=patch_size, stride=1)
    train_model(
        model,
        train_loader,
        train_patches,
        device,
        num_iter=PAANO_NUM_ITERS,
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
) -> SharedEncoderState:
    """Train two shared encoders (short + long scale) on clean-normal data
    from all train wells of the given anomaly family.

    Returns a :class:`SharedEncoderState` with both models and normalization stats.
    """
    _set_seed()

    pool, shared_channels, train_well_ids = collect_shared_train_pool(
        prepared_wells,
        enable_reduction=anomaly_key != "negermet",
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
        pool, patch_short, device, verbose=verbose,
    )
    model_long, mean_long, std_long = _train_encoder_single_scale(
        pool, patch_long, device, verbose=verbose,
    )

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
        detail={
            "pool_points": int(len(pool)),
            "shared_channels": len(shared_channels),
            "train_wells": train_well_ids,
        },
    )


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

    # DataLoader for local memory bank (from reference data)
    ref_loader, _, _ = patch_creator.create_dataloaders(
        ref_norm, ref_norm, np.zeros(len(ref_data), dtype=np.float32),
        batch_size=PAANO_BATCH_SIZE,
    )
    # DataLoader for full well scoring
    _, full_loader, _ = patch_creator.create_dataloaders(
        ref_norm, well_norm, dummy_labels,
        batch_size=PAANO_BATCH_SIZE,
    )

    # Local memory bank from this well's reference data
    memory_bank, _ = create_memory_bank(
        model, ref_loader, device, num_cores=PAANO_MEMORY_BANK_RATIO,
    )

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
    projected = np.zeros((feature_matrix.shape[0], len(shared_channels)), dtype=np.float32)
    for out_idx, channel in enumerate(shared_channels):
        src_idx = col_lookup.get(channel)
        if src_idx is not None:
            projected[:, out_idx] = feature_matrix[:, src_idx].astype(np.float32)
    return projected
