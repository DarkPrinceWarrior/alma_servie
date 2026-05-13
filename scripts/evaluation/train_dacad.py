"""Sprint 5: DACAD-lite — MMD-regularized fine-tune of ALMA encoder against
frozen 3W reference embeddings.

Hypothesis: salt was invariant to direct 3W transfer in Sprint 4. DACAD-style
domain alignment via MMD on the latent space should pull ALMA's representation
toward 3W's, potentially yielding salt encoder-side gain that simple weight
transplant did not.

Architecture:
  Source encoder E_S  (frozen, 3W class N) -> emb_S
  Target encoder E_T  (trainable, ALMA)    -> emb_T
  Loss:  contrastive(E_T, target pool)   [keeps recon objective on target]
       + lambda * MMD(emb_S(source pool), emb_T(target pool))

After training, deploy E_T as ALMA encoder via _s4_swap_test.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path("/root/projects/alma_servie")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.anomaly_specs import get_detection_spec
from alma_service.generic_detection import (
    _prepare_all_wells,
    _resolve_torch_device,
    _runtime_config,
    load_anomaly_data,
    load_intervals,
)
from alma_service.generic_detectors import PAANO_BATCH_SIZE, PAANO_LR, SEED
from alma_service.shared_encoder import (
    SharedEncoderState,
    collect_shared_train_pool,
    save_shared_encoder_state,
)

sys.path.insert(0, str(PROJECT_ROOT / "paano"))
from model import PatchEncoder  # noqa: E402
from utils.data_preprocess import PatchCreator, preprocess_to_patches  # noqa: E402


def _gaussian_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Maximum Mean Discrepancy with single Gaussian kernel.

    x: (n, d), y: (m, d). Returns scalar MMD^2.
    """
    n, m = x.size(0), y.size(0)
    xx = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(-1)
    yy = (y.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)
    xy = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)
    bandwidth = 2.0 * sigma * sigma
    k_xx = torch.exp(-xx / bandwidth).mean()
    k_yy = torch.exp(-yy / bandwidth).mean()
    k_xy = torch.exp(-xy / bandwidth).mean()
    return k_xx + k_yy - 2.0 * k_xy


def _build_3w_pool(source_class: int, patch_size: int) -> tuple[np.ndarray, list[str], dict]:
    """Build 3W normal pool for one class, return (pool, channels, encoder_meta)."""
    encoder_path = PROJECT_ROOT / "artifacts" / "3w" / "checkpoints" / f"3w_class_{source_class}_paano_shared_encoder.pt"
    if not encoder_path.exists():
        raise FileNotFoundError(encoder_path)
    src_payload = torch.load(encoder_path, map_location="cpu", weights_only=False)

    features_path = PROJECT_ROOT / "data" / "processed" / "3w" / f"class_{source_class}" / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(features_path)

    import pandas as pd
    df = pd.read_parquet(features_path, engine="pyarrow")
    splits_path = PROJECT_ROOT / "data" / "processed" / "3w" / "splits.parquet"
    splits_df = pd.read_parquet(splits_path, engine="pyarrow")
    if "split" not in df.columns:
        df = df.merge(splits_df[["instance_id", "split"]], on="instance_id", how="left")

    src_channels = [str(c) for c in src_payload["shared_channels"]]
    df_norm = df[df.get("class", 0).eq(0)].copy() if "class" in df.columns else df.copy()
    if "split" in df_norm.columns:
        df_norm = df_norm[df_norm["split"].eq("train")]

    feature_cols = [c for c in df_norm.columns if c not in ("instance_id", "split", "class", "timestamp")]
    available = [c for c in src_channels if c in feature_cols]
    if not available:
        raise ValueError(f"No 3W class {source_class} channels available in features: {src_channels}")
    pool = df_norm[available].to_numpy(dtype=np.float32)
    pool = np.nan_to_num(pool, nan=0.0, posinf=0.0, neginf=0.0)

    return pool, available, src_payload


def _train_dacad_scale(
    alma_pool: np.ndarray,
    alma_channels: list[str],
    source_pool: np.ndarray,
    source_payload: dict,
    patch_size: int,
    device: torch.device,
    num_iter: int,
    mmd_weight: float,
    verbose: bool,
    label: str,
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """Fine-tune ALMA encoder with MMD regularization against frozen 3W embeddings."""

    # Standardize pools
    train_mean = alma_pool.mean(axis=0, keepdims=True).astype(np.float32)
    train_std = alma_pool.std(axis=0, keepdims=True).astype(np.float32)
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)
    alma_norm = (alma_pool - train_mean) / train_std

    src_mean = source_pool.mean(axis=0, keepdims=True).astype(np.float32)
    src_std = source_pool.std(axis=0, keepdims=True).astype(np.float32)
    src_std = np.where(src_std < 1e-8, 1e-8, src_std)
    src_norm = (source_pool - src_mean) / src_std

    # Patches
    alma_patches = preprocess_to_patches(alma_norm, patch_size=patch_size, stride=1)
    src_patches = preprocess_to_patches(src_norm, patch_size=patch_size, stride=1)
    if alma_patches.shape[0] == 0 or src_patches.shape[0] == 0:
        raise ValueError(f"Empty patches: alma={alma_patches.shape}, src={src_patches.shape}")

    alma_patches = alma_patches.to(device)
    src_patches = src_patches.to(device)

    # Target encoder (trainable)
    model_t = PatchEncoder(in_channels=alma_norm.shape[1], use_revin=True).to(device)

    # Source encoder (frozen) -- match source channel count via payload
    src_state_key = "model_short_state_dict" if patch_size <= int(source_payload["patch_short"]) else "model_long_state_dict"
    src_state = source_payload[src_state_key]
    src_channels_n = source_pool.shape[1]
    model_s = PatchEncoder(in_channels=src_channels_n, use_revin=True).to(device)
    # Filter state dict: only load weights with matching shapes (skip convblocks.0 if mismatch)
    own = model_s.state_dict()
    filtered = {k: v for k, v in src_state.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in filtered]
    if verbose:
        print(f"    [{label}] src encoder: loaded {len(filtered)}/{len(own)} keys, missing {len(missing)} (channel/scale-dep)", flush=True)
    own.update(filtered)
    model_s.load_state_dict(own)
    model_s.eval()
    for p in model_s.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model_t.parameters(), lr=PAANO_LR)
    batch_size = min(PAANO_BATCH_SIZE, alma_patches.shape[0], src_patches.shape[0])

    t0 = time.time()
    for it in range(num_iter):
        # Sample
        idx_t = torch.randint(0, alma_patches.shape[0], (batch_size,), device=device)
        idx_s = torch.randint(0, src_patches.shape[0], (batch_size,), device=device)

        x_t = alma_patches[idx_t]
        x_s = src_patches[idx_s]

        # Positive pairs for ALMA: shift by 1 in time, clipped
        idx_pos = (idx_t + 1).clamp(max=alma_patches.shape[0] - 1)
        x_pos = alma_patches[idx_pos]

        emb_t = model_t.embedding(x_t)
        emb_pos = model_t.embedding(x_pos)
        with torch.no_grad():
            emb_s = model_s.embedding(x_s)

        # Contrastive on target: anchor-positive close, anchor-random apart
        # NT-Xent style
        z_t = F.normalize(model_t.projection(emb_t), dim=-1)
        z_pos = F.normalize(model_t.projection(emb_pos), dim=-1)
        logits = z_t @ z_pos.t()  # (B, B)
        labels = torch.arange(batch_size, device=device)
        loss_recon = F.cross_entropy(logits / 0.1, labels)

        # MMD between target embeddings and frozen source embeddings
        # Normalize both to remove scale before MMD
        emb_t_n = F.normalize(emb_t, dim=-1)
        emb_s_n = F.normalize(emb_s, dim=-1)
        loss_mmd = _gaussian_mmd(emb_t_n, emb_s_n, sigma=0.5)

        loss = loss_recon + mmd_weight * loss_mmd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (it % 50 == 0 or it == num_iter - 1):
            print(
                f"    [{label}] it={it:3d}  L={loss.item():.4f}  L_recon={loss_recon.item():.4f}  "
                f"L_mmd={loss_mmd.item():.4f}",
                flush=True,
            )

    elapsed = time.time() - t0
    if verbose:
        print(f"    [{label}] patch={patch_size}: pool_pts={len(alma_pool)}, time={elapsed:.1f}s", flush=True)

    return model_t, train_mean, train_std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-class", type=int, required=True)
    ap.add_argument("--target-anomaly", choices=["negermet", "pritok", "salt"], required=True)
    ap.add_argument("--mmd-weight", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--output-anomaly-key", required=True, help="Output filename suffix in models/")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dacad] device={device} src=class_{args.source_class} tgt={args.target_anomaly} mmd={args.mmd_weight}", flush=True)

    # Target = ALMA prepared wells
    cfg = _runtime_config(args.target_anomaly)
    spec = get_detection_spec(args.target_anomaly)
    df_anomaly = load_anomaly_data(spec)
    intervals_df = load_intervals(spec)
    wells = _prepare_all_wells(spec, df_anomaly, intervals_df, args.verbose)
    pool, shared_channels, _ = collect_shared_train_pool(wells, enable_reduction=args.target_anomaly != "negermet")
    print(f"[dacad] ALMA pool: {len(pool)} pts, {len(shared_channels)} channels", flush=True)

    patch_short = int(cfg["paano_patch_short"])
    patch_long = int(cfg["paano_patch_long"])

    # Source = 3W class N normal pool + frozen encoder
    src_pool, src_channels, src_payload = _build_3w_pool(args.source_class, patch_short)
    print(f"[dacad] 3W class {args.source_class} pool: {len(src_pool)} pts, {len(src_channels)} channels", flush=True)

    # Train two scales
    model_short, mean_short, std_short = _train_dacad_scale(
        pool, shared_channels, src_pool, src_payload,
        patch_short, device, args.iters, args.mmd_weight, args.verbose, "short",
    )
    model_long, mean_long, std_long = _train_dacad_scale(
        pool, shared_channels, src_pool, src_payload,
        patch_long, device, args.iters, args.mmd_weight, args.verbose, "long",
    )

    # Save as SharedEncoderState compatible with P10 hook
    state = SharedEncoderState(
        model_short=model_short,
        model_long=model_long,
        train_mean_short=mean_short,
        train_std_short=std_short,
        train_mean_long=mean_long,
        train_std_long=std_long,
        shared_channels=shared_channels,
        patch_short=patch_short,
        patch_long=patch_long,
        anomaly_key=args.target_anomaly,
        train_wells=[],
        detail={
            "encoder_training_mode": "3w_dacad_mmd_finetune",
            "source_class": int(args.source_class),
            "mmd_weight": float(args.mmd_weight),
            "iters": int(args.iters),
            "alma_pool_pts": int(len(pool)),
            "src_pool_pts": int(len(src_pool)),
            "shared_channels": len(shared_channels),
        },
    )

    out_path = PROJECT_ROOT / "artifacts" / "3w" / "checkpoints" / f"{args.output_anomaly_key}_3w_dacad_encoder.pt"
    save_shared_encoder_state(state, out_path)
    print(f"[dacad] saved -> {out_path}", flush=True)

    summary_path = PROJECT_ROOT / "artifacts" / "3w" / "metrics" / f"dacad_class_{args.source_class}_to_{args.target_anomaly}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(state.detail, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[dacad] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
