from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    import torch._inductor.config as torch_inductor_config
except Exception:  # pragma: no cover - optional runtime tuning
    torch_inductor_config = None
from sklearn.decomposition import PCA

from torch import nn

from alma_service.onset_detection import robust_stats, robust_z
from alma_service.paano_defaults import LONG_PATCH, SHORT_PATCH

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

SEED = 2027
PAANO_NUM_ITERS = 200
PAANO_BATCH_SIZE = 512
PAANO_LR = 1e-4
PAANO_TOP_K = 3
PAANO_MEMORY_BANK_RATIO = 0.1

ENABLE_TORCH_COMPILE = os.getenv("ALMA_TORCH_COMPILE", "1").strip().lower() not in {"0", "false", "no"}

try:  # pragma: no branch - simple runtime guard
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

if torch_inductor_config is not None:
    try:
        torch_inductor_config.triton.cudagraph_skip_dynamic_graphs = True
    except Exception:
        pass


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _maybe_compile_module(module: nn.Module, *, label: str, verbose: bool = False) -> nn.Module:
    if not ENABLE_TORCH_COMPILE or not hasattr(torch, "compile"):
        return module
    try:
        compiled = torch.compile(module, mode="reduce-overhead", dynamic=True)
        if verbose:
            print(f"    torch.compile enabled for {label}")
        return compiled
    except Exception as exc:  # pragma: no cover - runtime fallback
        if verbose:
            print(f"    torch.compile skipped for {label}: {exc}")
        return module


@dataclass
class DetectorScoreOutput:
    primary: np.ndarray
    components: dict[str, np.ndarray]
    detail: dict[str, Any]


class BaseDetector:
    detector_key = "base"

    def fit_reference(self, X_ref: np.ndarray, mask_ref: np.ndarray | None = None) -> "BaseDetector":
        raise NotImplementedError

    def score_stream(self, X_all: np.ndarray, mask_all: np.ndarray | None = None) -> DetectorScoreOutput:
        raise NotImplementedError


def _ensure_2d_float32(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_reference_subset(X_ref: np.ndarray, min_rows: int = 16) -> np.ndarray:
    arr = _ensure_2d_float32(X_ref)
    if len(arr) == 0:
        return arr
    if len(arr) >= min_rows:
        return arr
    repeats = int(np.ceil(min_rows / max(len(arr), 1)))
    tiled = np.tile(arr, (repeats, 1))
    return tiled[:min_rows]


def _run_paano_single_scale(
    data: np.ndarray,
    train_data: np.ndarray,
    patch_size: int,
    device: torch.device,
    verbose: bool = False,
) -> np.ndarray:
    train_mean = np.mean(train_data, axis=0, keepdims=True).astype(np.float32)
    train_std = np.std(train_data, axis=0, keepdims=True).astype(np.float32)
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)

    full_norm = (data - train_mean) / train_std
    train_norm = (train_data - train_mean) / train_std
    dummy_labels = np.zeros(len(data), dtype=np.float32)

    patch_creator = PatchCreator(L=patch_size, s=1, random_seed=SEED)
    train_loader, full_loader, _ = patch_creator.create_dataloaders(
        train_norm,
        full_norm,
        dummy_labels,
        batch_size=PAANO_BATCH_SIZE,
    )

    model = PatchEncoder(in_channels=data.shape[1], use_revin=True).to(device)
    model = _maybe_compile_module(model, label="PatchEncoder", verbose=verbose)
    train_patches = preprocess_to_patches(train_norm, patch_size=patch_size, stride=1)
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

    memory_bank, _ = create_memory_bank(model, train_loader, device, num_cores=PAANO_MEMORY_BANK_RATIO)
    patch_scores = calculate_anomaly_scores(model, full_loader, memory_bank, top_k=PAANO_TOP_K, device=device)
    point_scores = distribute_patch_scores_to_points(patch_scores, patch_size=patch_size, num_points=len(data))
    if verbose:
        print(f"    PaAno patch={patch_size}: train_pts={len(train_data)}, channels={data.shape[1]}")
    return np.asarray(point_scores, dtype=np.float32)


class PaAnoFeatureDetector(BaseDetector):
    detector_key = "paano_feat"

    def __init__(
        self,
        device: torch.device,
        patch_short: int = SHORT_PATCH,
        patch_long: int = LONG_PATCH,
        fusion_weight_short: float = 0.60,
        verbose: bool = False,
    ) -> None:
        self.device = device
        self.patch_short = int(patch_short)
        self.patch_long = int(patch_long)
        self.fusion_weight_short = float(fusion_weight_short)
        self.verbose = verbose
        self.train_ref_: np.ndarray | None = None

    def fit_reference(self, X_ref: np.ndarray, mask_ref: np.ndarray | None = None) -> "PaAnoFeatureDetector":
        self.train_ref_ = _ensure_2d_float32(X_ref)
        return self

    def score_stream(self, X_all: np.ndarray, mask_all: np.ndarray | None = None) -> DetectorScoreOutput:
        if self.train_ref_ is None:
            raise RuntimeError("Detector is not fitted.")
        X = _ensure_2d_float32(X_all)
        if len(self.train_ref_) < self.patch_long * 2 or len(X) < self.patch_long * 2:
            zeros = np.zeros(len(X), dtype=np.float32)
            return DetectorScoreOutput(
                primary=zeros,
                components={"paano_short": zeros.copy(), "paano_long": zeros.copy()},
                detail={"reason": "not_enough_points"},
            )
        short_score = _run_paano_single_scale(
            data=X,
            train_data=self.train_ref_,
            patch_size=self.patch_short,
            device=self.device,
            verbose=self.verbose,
        )
        long_score = _run_paano_single_scale(
            data=X,
            train_data=self.train_ref_,
            patch_size=self.patch_long,
            device=self.device,
            verbose=self.verbose,
        )
        fused = self.fusion_weight_short * short_score + (1.0 - self.fusion_weight_short) * long_score
        return DetectorScoreOutput(
            primary=fused.astype(np.float32),
            components={
                "paano_short": short_score.astype(np.float32),
                "paano_long": long_score.astype(np.float32),
            },
            detail={
                "patch_short": self.patch_short,
                "patch_long": self.patch_long,
                "fusion_weight_short": self.fusion_weight_short,
            },
        )


class PCASPEDetector(BaseDetector):
    detector_key = "pca_spe"

    def __init__(self, explained_variance: float = 0.95, max_components: int = 24) -> None:
        self.explained_variance = float(explained_variance)
        self.max_components = int(max_components)
        self.pca_: PCA | None = None
        self.t2_stats_: tuple[float, float] | None = None
        self.spe_stats_: tuple[float, float] | None = None

    def fit_reference(self, X_ref: np.ndarray, mask_ref: np.ndarray | None = None) -> "PCASPEDetector":
        X = _safe_reference_subset(X_ref, min_rows=32)
        max_components = max(1, min(self.max_components, X.shape[1], X.shape[0] - 1))
        pca_full = PCA(n_components=max_components, svd_solver="full", random_state=SEED)
        pca_full.fit(X)

        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        needed = int(np.searchsorted(cumsum, self.explained_variance) + 1)
        n_components = max(1, min(max_components, needed))
        self.pca_ = PCA(n_components=n_components, svd_solver="full", random_state=SEED)
        self.pca_.fit(X)

        ref_scores = self._score_parts(X)
        self.t2_stats_ = robust_stats(ref_scores["t2"])
        self.spe_stats_ = robust_stats(ref_scores["spe"])
        return self

    def _score_parts(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if self.pca_ is None:
            raise RuntimeError("Detector is not fitted.")
        transformed = self.pca_.transform(X)
        reconstructed = self.pca_.inverse_transform(transformed)
        eigenvalues = np.maximum(self.pca_.explained_variance_, 1e-6)
        t2 = np.sum((transformed**2) / eigenvalues[None, :], axis=1).astype(np.float32)
        spe = np.sum((X - reconstructed) ** 2, axis=1).astype(np.float32)
        return {"t2": t2, "spe": spe}

    def score_stream(self, X_all: np.ndarray, mask_all: np.ndarray | None = None) -> DetectorScoreOutput:
        X = _ensure_2d_float32(X_all)
        parts = self._score_parts(X)
        if self.t2_stats_ is None or self.spe_stats_ is None:
            raise RuntimeError("Detector is not fitted.")
        t2_z = robust_z(parts["t2"], *self.t2_stats_)
        spe_z = robust_z(parts["spe"], *self.spe_stats_)
        primary = t2_z + spe_z
        return DetectorScoreOutput(
            primary=primary.astype(np.float32),
            components={
                "pca_t2": parts["t2"].astype(np.float32),
                "pca_spe": parts["spe"].astype(np.float32),
                "pca_t2_z": t2_z.astype(np.float32),
                "pca_spe_z": spe_z.astype(np.float32),
            },
            detail={"n_components": int(self.pca_.n_components_) if self.pca_ is not None else 0},
        )


class SharedPaAnoDetector(BaseDetector):
    """PaAno detector that uses a **shared** pre-trained encoder.

    Instead of training its own encoder from the well's reference data,
    it receives a pre-trained shared encoder (from ``shared_encoder.py``)
    and builds only a **local** memory bank from the well's reference data.
    """

    detector_key = "paano_shared"

    def __init__(
        self,
        shared_state: Any,
        device: torch.device,
        fusion_weight_short: float = 0.60,
        verbose: bool = False,
    ) -> None:
        from alma_service.shared_encoder import SharedEncoderState

        if not isinstance(shared_state, SharedEncoderState):
            raise TypeError(f"Expected SharedEncoderState, got {type(shared_state)}")
        self.shared_state: SharedEncoderState = shared_state
        self.device = device
        self.fusion_weight_short = float(fusion_weight_short)
        self.verbose = verbose
        self.train_ref_: np.ndarray | None = None

    def fit_reference(self, X_ref: np.ndarray, mask_ref: np.ndarray | None = None) -> "SharedPaAnoDetector":
        self.train_ref_ = _ensure_2d_float32(X_ref)
        return self

    def score_stream(self, X_all: np.ndarray, mask_all: np.ndarray | None = None) -> DetectorScoreOutput:
        from alma_service.shared_encoder import _score_well_single_scale

        if self.train_ref_ is None:
            raise RuntimeError("Detector is not fitted.")
        X = _ensure_2d_float32(X_all)
        st = self.shared_state

        if len(self.train_ref_) < st.patch_long * 2 or len(X) < st.patch_long * 2:
            zeros = np.zeros(len(X), dtype=np.float32)
            return DetectorScoreOutput(
                primary=zeros,
                components={"paano_short": zeros.copy(), "paano_long": zeros.copy()},
                detail={"reason": "not_enough_points"},
            )

        short_score = _score_well_single_scale(
            model=st.model_short,
            well_data=X,
            ref_data=self.train_ref_,
            train_mean=st.train_mean_short,
            train_std=st.train_std_short,
            patch_size=st.patch_short,
            device=self.device,
            verbose=self.verbose,
        )
        long_score = _score_well_single_scale(
            model=st.model_long,
            well_data=X,
            ref_data=self.train_ref_,
            train_mean=st.train_mean_long,
            train_std=st.train_std_long,
            patch_size=st.patch_long,
            device=self.device,
            verbose=self.verbose,
        )
        fused = self.fusion_weight_short * short_score + (1.0 - self.fusion_weight_short) * long_score
        return DetectorScoreOutput(
            primary=fused.astype(np.float32),
            components={
                "paano_short": short_score.astype(np.float32),
                "paano_long": long_score.astype(np.float32),
            },
            detail={
                "patch_short": st.patch_short,
                "patch_long": st.patch_long,
                "fusion_weight_short": self.fusion_weight_short,
                "shared_encoder": True,
                "train_wells": st.train_wells,
            },
        )

