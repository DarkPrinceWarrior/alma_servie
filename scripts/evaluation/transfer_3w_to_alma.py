"""Transfer a 3W-trained PaAno encoder into ALMA's per-anomaly shared encoder.

The strategy exploits PatchEncoder's architecture:
- RevIN1d(affine=False) has no learnable parameters tied to channel count.
- convblocks[0] is the only conv whose in_channels is determined by the source
  data; convblocks[1..] take fixed channel counts from `layers` and are fully
  channel-agnostic.
- The projection head, batchnorm running stats and conv biases past the first
  layer are reusable across datasets.

So we initialize a fresh ALMA-shaped encoder (in_channels = N_alma) and copy:
    convblocks[1].*           full
    convblocks[2..N].*        full
    projection_head.*         full
    classification_head.*     full (unused by detector but copied for parity)

Then fine-tune on ALMA train normals (reuses
alma_service.shared_encoder.fine_tune_shared_encoder with the appropriate
prepared_wells dict).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PAANO_ROOT = PROJECT_ROOT / "paano"
if str(PAANO_ROOT) not in sys.path:
    sys.path.insert(0, str(PAANO_ROOT))

from alma_service.anomaly_specs import get_detection_spec
from alma_service.generic_detection import _prepare_all_wells, _resolve_torch_device, load_anomaly_data, load_intervals
from alma_service.generic_detectors import PAANO_BATCH_SIZE, PAANO_LR, PAANO_NUM_ITERS, SEED, _maybe_compile_module
from alma_service.shared_encoder import (
    SharedEncoderState,
    _set_seed,
    collect_shared_train_pool,
    load_shared_encoder_state,
    save_shared_encoder_state,
    shared_encoder_path,
)
from model import PatchEncoder
from train import train_model
from utils.data_preprocess import PatchCreator, preprocess_to_patches


CHANNEL_DEPENDENT_FIRST_CONV = "convblocks.0.0"


def transplant_weights(src: nn.Module, dst: nn.Module) -> dict:
    """Copy parameters from src into dst for all tensors whose names match and
    whose shapes are compatible. Returns a small report dict of copied vs
    skipped names.
    """
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    copied: list[str] = []
    skipped_shape: list[str] = []
    only_in_src: list[str] = []
    only_in_dst: list[str] = []
    for name in src_state:
        if name not in dst_state:
            only_in_src.append(name)
            continue
        if src_state[name].shape == dst_state[name].shape:
            dst_state[name].copy_(src_state[name])
            copied.append(name)
        else:
            skipped_shape.append(name)
    for name in dst_state:
        if name not in src_state:
            only_in_dst.append(name)
    dst.load_state_dict(dst_state)
    return {
        "copied": copied,
        "skipped_due_to_shape": skipped_shape,
        "only_in_src": only_in_src,
        "only_in_dst": only_in_dst,
    }


def build_alma_encoder_from_3w(
    src_state: SharedEncoderState,
    alma_in_channels: int,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, dict, dict]:
    """Create ALMA-shaped short+long encoders and transplant weights."""
    new_short = PatchEncoder(in_channels=alma_in_channels, use_revin=True).to(device)
    new_long = PatchEncoder(in_channels=alma_in_channels, use_revin=True).to(device)
    report_short = transplant_weights(src_state.model_short, new_short)
    report_long = transplant_weights(src_state.model_long, new_long)
    return new_short, new_long, report_short, report_long


def fine_tune_single_scale(
    pool: np.ndarray,
    patch_size: int,
    model: nn.Module,
    device: torch.device,
    num_iter: int,
    verbose: bool = False,
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    train_mean = np.mean(pool, axis=0, keepdims=True).astype(np.float32)
    train_std = np.std(pool, axis=0, keepdims=True).astype(np.float32)
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)
    pool_norm = (pool - train_mean) / train_std

    patch_creator = PatchCreator(L=patch_size, s=1, random_seed=SEED)
    train_loader, _, _ = patch_creator.create_dataloaders(
        pool_norm, pool_norm, np.zeros(len(pool), dtype=np.float32),
        batch_size=PAANO_BATCH_SIZE,
    )
    compiled = _maybe_compile_module(model, label=f"AlmaTransfer_patch{patch_size}", verbose=verbose)
    train_patches = preprocess_to_patches(pool_norm, patch_size=patch_size, stride=1)
    train_model(
        compiled, train_loader, train_patches, device,
        num_iter=num_iter, pretext_step=patch_size, lr=PAANO_LR, see_loss=False,
    )
    return compiled, train_mean, train_std


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer 3W PaAno encoder weights into an ALMA-shaped encoder, then fine-tune.")
    parser.add_argument("--source-class", type=int, required=True, help="3W event class whose encoder is reused.")
    parser.add_argument("--target-anomaly", choices=["negermet", "pritok", "salt"], required=True)
    parser.add_argument("--fine-tune-iters", type=int, default=PAANO_NUM_ITERS)
    parser.add_argument("--output-anomaly-key", default=None,
                        help="If provided, save as this anomaly_key in models/ instead of overwriting ALMA production weights.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = _resolve_torch_device("paano_shared", verbose=True)
    src_path = PROJECT_ROOT / "artifacts" / "3w" / "checkpoints" / f"3w_class_{args.source_class}_paano_shared_encoder.pt"
    if not src_path.exists():
        raise FileNotFoundError(f"3W encoder not found: {src_path}")
    src_state = load_shared_encoder_state(
        anomaly_key=f"3w_class_{args.source_class}",
        path=src_path,
        device=device,
        compile_model=False,
        verbose=args.verbose,
    )
    print(
        f"[transfer] loaded 3W class {args.source_class} encoder: "
        f"channels={len(src_state.shared_channels)} patch_short={src_state.patch_short} patch_long={src_state.patch_long}",
        flush=True,
    )

    print(f"[transfer] preparing ALMA {args.target_anomaly} data ...", flush=True)
    spec = get_detection_spec(args.target_anomaly)
    df_anomaly = load_anomaly_data(spec)
    intervals_df = load_intervals(spec)
    prepared = _prepare_all_wells(spec, df_anomaly, intervals_df, args.verbose)
    pool, shared_channels, train_wells = collect_shared_train_pool(prepared, enable_reduction=args.target_anomaly != "negermet")
    print(f"[transfer] ALMA pool: {len(pool)} pts, {len(shared_channels)} channels, {len(train_wells)} wells", flush=True)

    new_short, new_long, rep_short, rep_long = build_alma_encoder_from_3w(
        src_state, alma_in_channels=len(shared_channels), device=device,
    )
    print(f"[transfer] transplanted weights short: copied={len(rep_short['copied'])} skipped_shape={len(rep_short['skipped_due_to_shape'])}", flush=True)
    print(f"[transfer] transplanted weights long:  copied={len(rep_long['copied'])} skipped_shape={len(rep_long['skipped_due_to_shape'])}", flush=True)

    _set_seed()
    model_short, mean_short, std_short = fine_tune_single_scale(
        pool, src_state.patch_short, new_short, device, args.fine_tune_iters, verbose=args.verbose,
    )
    model_long, mean_long, std_long = fine_tune_single_scale(
        pool, src_state.patch_long, new_long, device, args.fine_tune_iters, verbose=args.verbose,
    )

    output_anomaly_key = args.output_anomaly_key or args.target_anomaly
    out_path = (
        PROJECT_ROOT / "artifacts" / "3w" / "checkpoints" / f"{output_anomaly_key}_3w_transfer_encoder.pt"
        if args.output_anomaly_key
        else shared_encoder_path(args.target_anomaly)
    )
    new_state = SharedEncoderState(
        model_short=model_short,
        model_long=model_long,
        train_mean_short=mean_short,
        train_std_short=std_short,
        train_mean_long=mean_long,
        train_std_long=std_long,
        shared_channels=list(shared_channels),
        patch_short=src_state.patch_short,
        patch_long=src_state.patch_long,
        anomaly_key=output_anomaly_key if args.output_anomaly_key else args.target_anomaly,
        train_wells=train_wells,
        detail={
            "training_mode": "3w_pretrain_alma_finetune",
            "source_3w_class": int(args.source_class),
            "source_3w_train_wells": src_state.train_wells,
            "fine_tune_iters": int(args.fine_tune_iters),
            "transplant_copied_short": len(rep_short["copied"]),
            "transplant_skipped_shape_short": list(rep_short["skipped_due_to_shape"]),
            "transplant_copied_long": len(rep_long["copied"]),
            "transplant_skipped_shape_long": list(rep_long["skipped_due_to_shape"]),
            "alma_pool_points": int(len(pool)),
            "alma_channels": int(len(shared_channels)),
            "alma_train_wells": train_wells,
        },
    )
    save_shared_encoder_state(new_state, out_path)
    print(f"[transfer] saved encoder -> {out_path}", flush=True)

    summary = {
        "source_3w_class": int(args.source_class),
        "target_anomaly": args.target_anomaly,
        "output_path": str(out_path),
        "fine_tune_iters": int(args.fine_tune_iters),
        "alma_channels": int(len(shared_channels)),
        "transplant_copied_short": rep_short["copied"],
        "transplant_skipped_shape_short": rep_short["skipped_due_to_shape"],
        "transplant_copied_long": rep_long["copied"],
        "transplant_skipped_shape_long": rep_long["skipped_due_to_shape"],
    }
    summary_path = PROJECT_ROOT / "artifacts" / "3w" / "metrics" / f"transfer_3w_class_{args.source_class}_to_{args.target_anomaly}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[transfer] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
