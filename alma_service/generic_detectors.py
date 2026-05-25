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

PAANO_BATCH_SIZE = 256
PAANO_INFER_BATCH_SIZE = 1024  # увеличенный inference batch для скоринга (PR прошлой сессии)
PAANO_LR = 1e-3
PAANO_NUM_ITERS = 200
# top_k=1 — nearest-neighbor anomaly score без усреднения. На pritok даёт
# 19/22 → 22/22 hits и p90 17.84ч → 6.75ч, на negermet/salt — бит-в-бит.
# k=5 размывал onset усреднением по 5 ближайшим нормальным; k=1 — чистый
# novelty signal. Подтверждено ablation 2026-05-25.
PAANO_TOP_K = int(os.getenv("ALMA_PAANO_TOP_K", "1"))
PAANO_MEMORY_BANK_RATIO = float(os.getenv("ALMA_PAANO_MEMORY_BANK_RATIO", "0.10"))

ENABLE_TORCH_COMPILE = os.getenv("ALMA_TORCH_COMPILE", "1").strip().lower() not in {"0", "false", "no"}
PAANO_INPUT_PADDING_ENV = "ALMA_PAANO_INPUT_PADDING"
PAANO_INPUT_PADDING_NONE = "none"
PAANO_INPUT_PADDING_EDGE_HOLD = "edge_hold"
PAANO_INPUT_PADDING_MODES = {PAANO_INPUT_PADDING_NONE, PAANO_INPUT_PADDING_EDGE_HOLD}

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


def _paano_input_padding_mode(mode: str | None = None) -> str:
    mode = (os.getenv(PAANO_INPUT_PADDING_ENV, PAANO_INPUT_PADDING_NONE) if mode is None else mode).strip().lower()
    if mode not in PAANO_INPUT_PADDING_MODES:
        raise ValueError(
            f"Unsupported {PAANO_INPUT_PADDING_ENV}={mode!r}. "
            f"Expected one of: {', '.join(sorted(PAANO_INPUT_PADDING_MODES))}."
        )
    return mode


def _edge_hold_pad_prefix(values: np.ndarray, target_len: int) -> tuple[np.ndarray, int]:
    arr = _ensure_2d_float32(values)
    if len(arr) == 0 or len(arr) >= int(target_len):
        return arr, 0
    pad_count = int(target_len) - len(arr)
    pad = np.repeat(arr[:1], pad_count, axis=0)
    return np.concatenate([pad, arr], axis=0).astype(np.float32), pad_count


def _trim_prefix_padding(scores: np.ndarray, pad_count: int, original_len: int) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    if pad_count <= 0:
        return arr[:original_len].astype(np.float32)
    return arr[int(pad_count) : int(pad_count) + int(original_len)].astype(np.float32)


def _input_contract_from_padding(padding_detail: dict[str, Any]) -> str:
    if not bool(padding_detail.get("enabled", False)):
        return "real_window"
    short = dict(padding_detail.get("short", {}))
    long = dict(padding_detail.get("long", {}))
    pad_points = [
        int(short.get("well_prefix_points", 0)),
        int(short.get("reference_prefix_points", 0)),
        int(long.get("well_prefix_points", 0)),
        int(long.get("reference_prefix_points", 0)),
    ]
    if any(value > 0 for value in pad_points):
        return "edge_hold_padded"
    return "real_window"


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
        input_padding_mode: str | None = None,
    ) -> None:
        from alma_service.shared_encoder import SharedEncoderState

        if not isinstance(shared_state, SharedEncoderState):
            raise TypeError(f"Expected SharedEncoderState, got {type(shared_state)}")
        self.shared_state: SharedEncoderState = shared_state
        self.device = device
        self.fusion_weight_short = float(fusion_weight_short)
        self.verbose = verbose
        self.input_padding_mode = _paano_input_padding_mode(input_padding_mode) if input_padding_mode is not None else None
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
        padding_mode = _paano_input_padding_mode(self.input_padding_mode)

        if len(self.train_ref_) < st.patch_long * 2 or len(X) < st.patch_long * 2:
            if padding_mode != PAANO_INPUT_PADDING_EDGE_HOLD or len(self.train_ref_) == 0 or len(X) == 0:
                zeros = np.zeros(len(X), dtype=np.float32)
                return DetectorScoreOutput(
                    primary=zeros,
                    components={"paano_short": zeros.copy(), "paano_long": zeros.copy()},
                    detail={"reason": "not_enough_points"},
                )

        padding_detail: dict[str, Any] = {
            "mode": padding_mode,
            "enabled": padding_mode == PAANO_INPUT_PADDING_EDGE_HOLD,
            "short": {"well_prefix_points": 0, "reference_prefix_points": 0},
            "long": {"well_prefix_points": 0, "reference_prefix_points": 0},
        }

        short_well = X
        short_ref = self.train_ref_
        short_well_pad = 0
        short_ref_pad = 0
        if padding_mode == PAANO_INPUT_PADDING_EDGE_HOLD:
            short_target = st.patch_short * 2
            short_well, short_well_pad = _edge_hold_pad_prefix(X, short_target)
            short_ref, short_ref_pad = _edge_hold_pad_prefix(self.train_ref_, short_target)
            padding_detail["short"] = {
                "well_prefix_points": int(short_well_pad),
                "reference_prefix_points": int(short_ref_pad),
            }

        short_score = _score_well_single_scale(
            model=st.model_short,
            well_data=short_well,
            ref_data=short_ref,
            train_mean=st.train_mean_short,
            train_std=st.train_std_short,
            patch_size=st.patch_short,
            device=self.device,
            verbose=self.verbose,
        )

        short_score = _trim_prefix_padding(short_score, short_well_pad, len(X))

        long_well = X
        long_ref = self.train_ref_
        long_well_pad = 0
        long_ref_pad = 0
        if padding_mode == PAANO_INPUT_PADDING_EDGE_HOLD:
            long_target = st.patch_long * 2
            long_well, long_well_pad = _edge_hold_pad_prefix(X, long_target)
            long_ref, long_ref_pad = _edge_hold_pad_prefix(self.train_ref_, long_target)
            padding_detail["long"] = {
                "well_prefix_points": int(long_well_pad),
                "reference_prefix_points": int(long_ref_pad),
            }

        long_score = _score_well_single_scale(
            model=st.model_long,
            well_data=long_well,
            ref_data=long_ref,
            train_mean=st.train_mean_long,
            train_std=st.train_std_long,
            patch_size=st.patch_long,
            device=self.device,
            verbose=self.verbose,
        )
        long_score = _trim_prefix_padding(long_score, long_well_pad, len(X))

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
                "input_padding": padding_detail,
                "input_contract": _input_contract_from_padding(padding_detail),
            },
        )
