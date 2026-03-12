from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from alma_service.onset_detection import robust_scale_for_fusion_mask, robust_stats, robust_z
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
PAANO_BATCH_SIZE = 256
PAANO_LR = 1e-4
PAANO_TOP_K = 3
PAANO_MEMORY_BANK_RATIO = 0.1
FUSED_WEIGHTS = {
    "paano_feat": 0.50,
    "pca_spe": 0.20,
    "lof": 0.15,
    "iforest": 0.15,
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


class LOFDetector(BaseDetector):
    detector_key = "lof"

    def __init__(self, n_neighbors: int = 35) -> None:
        self.n_neighbors = int(n_neighbors)
        self.model_: LocalOutlierFactor | None = None

    def fit_reference(self, X_ref: np.ndarray, mask_ref: np.ndarray | None = None) -> "LOFDetector":
        X = _safe_reference_subset(X_ref, min_rows=32)
        n_neighbors = max(5, min(self.n_neighbors, len(X) - 1))
        self.model_ = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        self.model_.fit(X)
        self.n_neighbors = n_neighbors
        return self

    def score_stream(self, X_all: np.ndarray, mask_all: np.ndarray | None = None) -> DetectorScoreOutput:
        if self.model_ is None:
            raise RuntimeError("Detector is not fitted.")
        X = _ensure_2d_float32(X_all)
        raw = -self.model_.score_samples(X)
        return DetectorScoreOutput(
            primary=np.asarray(raw, dtype=np.float32),
            components={"lof_raw": np.asarray(raw, dtype=np.float32)},
            detail={"n_neighbors": self.n_neighbors},
        )


class IsolationForestDetector(BaseDetector):
    detector_key = "iforest"

    def __init__(self, n_estimators: int = 200, max_samples: str | int = "auto") -> None:
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.model_: IsolationForest | None = None

    def fit_reference(self, X_ref: np.ndarray, mask_ref: np.ndarray | None = None) -> "IsolationForestDetector":
        X = _safe_reference_subset(X_ref, min_rows=32)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination="auto",
            random_state=SEED,
            n_jobs=-1,
        )
        self.model_.fit(X)
        return self

    def score_stream(self, X_all: np.ndarray, mask_all: np.ndarray | None = None) -> DetectorScoreOutput:
        if self.model_ is None:
            raise RuntimeError("Detector is not fitted.")
        X = _ensure_2d_float32(X_all)
        raw = -self.model_.score_samples(X)
        return DetectorScoreOutput(
            primary=np.asarray(raw, dtype=np.float32),
            components={"iforest_raw": np.asarray(raw, dtype=np.float32)},
            detail={"n_estimators": self.n_estimators},
        )


class FusedDetector(BaseDetector):
    detector_key = "fused"

    def __init__(
        self,
        device: torch.device,
        verbose: bool = False,
        weights: dict[str, float] | None = None,
        patch_short: int = SHORT_PATCH,
        patch_long: int = LONG_PATCH,
    ) -> None:
        self.device = device
        self.verbose = verbose
        self.weights = dict(FUSED_WEIGHTS if weights is None else weights)
        self.reference_mask_: np.ndarray | None = None
        self.detectors_: dict[str, BaseDetector] = {
            "paano_feat": PaAnoFeatureDetector(
                device=device,
                patch_short=patch_short,
                patch_long=patch_long,
                verbose=verbose,
            ),
            "pca_spe": PCASPEDetector(),
            "lof": LOFDetector(),
            "iforest": IsolationForestDetector(),
        }

    def fit_reference(self, X_ref: np.ndarray, mask_ref: np.ndarray | None = None) -> "FusedDetector":
        self.reference_mask_ = None if mask_ref is None else np.asarray(mask_ref, dtype=bool)
        for detector in self.detectors_.values():
            detector.fit_reference(X_ref, mask_ref=mask_ref)
        return self

    def score_stream(self, X_all: np.ndarray, mask_all: np.ndarray | None = None) -> DetectorScoreOutput:
        reference_mask = self.reference_mask_
        if reference_mask is None:
            reference_mask = np.ones(len(X_all), dtype=bool)
        component_outputs: dict[str, DetectorScoreOutput] = {}
        for key, detector in self.detectors_.items():
            component_outputs[key] = detector.score_stream(X_all, mask_all=mask_all)

        fused = np.zeros(len(X_all), dtype=np.float32)
        components: dict[str, np.ndarray] = {}
        for key, output in component_outputs.items():
            z = robust_scale_for_fusion_mask(output.primary, reference_mask)
            fused += float(self.weights.get(key, 0.0)) * z
            components[f"{key}_score"] = output.primary.astype(np.float32)
            components[f"{key}_score_z"] = z.astype(np.float32)
            for name, values in output.components.items():
                components[name] = values.astype(np.float32)

        return DetectorScoreOutput(
            primary=fused.astype(np.float32),
            components=components,
            detail={"weights": self.weights},
        )


class _TranADStyleNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        h = self.input_proj(x)
        h = h + self.pos_embed[:, :seq_len, :]
        h = self.encoder(h)
        return self.head(h[:, -1, :])


class TranADGlobalDetector(BaseDetector):
    detector_key = "tranad_global"

    def __init__(
        self,
        device: torch.device,
        window_size: int = 48,
        hidden_dim: int = 64,
        num_layers: int = 2,
        epochs: int = 8,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        max_train_windows: int = 20000,
        inference_batch_size: int = 2048,
        verbose: bool = False,
    ) -> None:
        self.device = device
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.max_train_windows = int(max_train_windows)
        self.inference_batch_size = int(inference_batch_size)
        self.verbose = verbose
        self.model_: _TranADStyleNet | None = None
        self.feature_dim_: int = 0

    @staticmethod
    def build_windows(
        X: np.ndarray,
        valid_mask: np.ndarray,
        window_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        arr = _ensure_2d_float32(X)
        mask = np.asarray(valid_mask, dtype=bool)
        windows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for end_idx in range(window_size - 1, len(arr)):
            start_idx = end_idx - window_size + 1
            if not mask[end_idx]:
                continue
            if not mask[start_idx : end_idx + 1].all():
                continue
            windows.append(arr[start_idx : end_idx + 1])
            targets.append(arr[end_idx])
        if not windows:
            return np.empty((0, window_size, arr.shape[1]), dtype=np.float32), np.empty((0, arr.shape[1]), dtype=np.float32)
        return np.asarray(windows, dtype=np.float32), np.asarray(targets, dtype=np.float32)

    def fit_global(self, train_series: list[np.ndarray], train_masks: list[np.ndarray]) -> "TranADGlobalDetector":
        windows_list: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []
        for X, mask in zip(train_series, train_masks):
            windows, targets = self.build_windows(X, mask, self.window_size)
            if len(windows):
                windows_list.append(windows)
                targets_list.append(targets)

        if not windows_list:
            raise RuntimeError("No valid windows for TranAD global training.")

        train_windows = np.concatenate(windows_list, axis=0)
        train_targets = np.concatenate(targets_list, axis=0)
        if len(train_windows) > self.max_train_windows:
            rng = np.random.default_rng(SEED)
            chosen = rng.choice(len(train_windows), size=self.max_train_windows, replace=False)
            train_windows = train_windows[chosen]
            train_targets = train_targets[chosen]

        self.feature_dim_ = int(train_windows.shape[2])
        self.model_ = _TranADStyleNet(
            input_dim=self.feature_dim_,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(self.device)

        dataset = TensorDataset(torch.from_numpy(train_windows), torch.from_numpy(train_targets))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model_(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu()) * len(batch_x)
            if self.verbose:
                avg_loss = epoch_loss / max(len(dataset), 1)
                print(f"    TranAD epoch {epoch + 1}/{self.epochs}: loss={avg_loss:.6f}")

        return self

    def fit_reference(self, X_ref: np.ndarray, mask_ref: np.ndarray | None = None) -> "TranADGlobalDetector":
        raise RuntimeError("Use fit_global() for TranADGlobalDetector.")

    def score_stream(self, X_all: np.ndarray, mask_all: np.ndarray | None = None) -> DetectorScoreOutput:
        if self.model_ is None:
            raise RuntimeError("Detector is not fitted.")
        X = _ensure_2d_float32(X_all)
        if X.shape[1] != self.feature_dim_:
            raise ValueError("Input feature dimension does not match trained TranAD model.")
        valid_mask = np.ones(len(X), dtype=bool) if mask_all is None else np.asarray(mask_all, dtype=bool)

        scores = np.zeros(len(X), dtype=np.float32)
        end_indices = np.flatnonzero(valid_mask)
        end_indices = end_indices[end_indices >= self.window_size - 1]
        if len(end_indices) == 0:
            return DetectorScoreOutput(
                primary=scores,
                components={"tranad_recon": scores.copy()},
                detail={
                    "window_size": self.window_size,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "epochs": self.epochs,
                    "inference_batch_size": self.inference_batch_size,
                },
            )

        offsets = np.arange(self.window_size, dtype=np.int64) - (self.window_size - 1)
        self.model_.eval()
        autocast_enabled = self.device.type == "cuda"
        with torch.no_grad():
            for start in range(0, len(end_indices), self.inference_batch_size):
                batch_end = end_indices[start : start + self.inference_batch_size]
                batch_indices = batch_end[:, None] + offsets[None, :]
                batch_windows = X[batch_indices]
                batch_targets = X[batch_end]
                batch_tensor = torch.from_numpy(batch_windows).to(self.device)
                with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
                    pred = self.model_(batch_tensor)
                err = torch.mean((pred - torch.from_numpy(batch_targets).to(self.device)) ** 2, dim=1)
                scores[batch_end] = err.detach().cpu().numpy().astype(np.float32)

        primary = scores
        return DetectorScoreOutput(
            primary=primary.astype(np.float32),
            components={"tranad_recon": primary.astype(np.float32)},
            detail={
                "window_size": self.window_size,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "epochs": self.epochs,
                "inference_batch_size": self.inference_batch_size,
            },
        )

    def save(self, path: str | Path) -> None:
        if self.model_ is None:
            raise RuntimeError("Detector is not fitted.")
        payload = {
            "state_dict": self.model_.state_dict(),
            "feature_dim": self.feature_dim_,
            "window_size": self.window_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "epochs": self.epochs,
            "inference_batch_size": self.inference_batch_size,
        }
        torch.save(payload, path)

    def load(self, path: str | Path) -> "TranADGlobalDetector":
        payload = torch.load(path, map_location=self.device)
        self.feature_dim_ = int(payload["feature_dim"])
        self.window_size = int(payload["window_size"])
        self.hidden_dim = int(payload["hidden_dim"])
        self.num_layers = int(payload["num_layers"])
        self.epochs = int(payload.get("epochs", self.epochs))
        self.inference_batch_size = int(payload.get("inference_batch_size", self.inference_batch_size))
        self.model_ = _TranADStyleNet(
            input_dim=self.feature_dim_,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(self.device)
        self.model_.load_state_dict(payload["state_dict"])
        self.model_.eval()
        return self
