from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
try:
    import torch._inductor.config as torch_inductor_config
except Exception:  # pragma: no cover - optional runtime tuning
    torch_inductor_config = None

from torch import nn

SEED = 2027

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


def _ensure_2d_float32(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


class SharedPaAnoDetector:
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
