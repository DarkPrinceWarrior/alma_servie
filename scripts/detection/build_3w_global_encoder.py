"""Build a global multi-source PaAno encoder from all 3W NORMAL data.

Aggregates clean-normal rows across:
- All 594 class_0 NORMAL instances (every row, since class_value == 0).
- The is_normal-prefix portion of every class_1..9 instance that has one.

Channel set: intersection of canonical features across the participating classes
(non-NaN-aware). Encoder trained at the same patch sizes as per-class 3W
encoders (paano_patch_short / paano_patch_long from configs/3w_paano.json).

Output: artifacts/3w/checkpoints/3w_global_normal_paano_shared_encoder.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.shared_encoder import (
    SharedEncoderState,
    _set_seed,
    _train_encoder_single_scale,
    reduce_features,
    save_shared_encoder_state,
)


def _collect_normal_rows_for_class(features_df: pd.DataFrame, class_id: int) -> tuple[np.ndarray, list[str]]:
    """Return (rows, columns) for the NORMAL portion of one class' features parquet."""
    meta = {"instance_id", "timestamp", "class_value", "is_normal"}
    cols = [c for c in features_df.columns if c not in meta]
    if class_id == 0:
        mask = np.ones(len(features_df), dtype=bool)
    else:
        if "is_normal" in features_df.columns:
            mask = features_df["is_normal"].to_numpy(dtype=bool)
        else:
            mask = np.zeros(len(features_df), dtype=bool)
    if not mask.any():
        return np.zeros((0, len(cols)), dtype=np.float32), cols
    sub = features_df.loc[mask, cols]
    arr = sub.to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 3W-global NORMAL PaAno encoder.")
    parser.add_argument("--config", default="configs/3w_paano.json")
    parser.add_argument("--include-classes", default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated class ids whose NORMAL rows feed the pool.")
    parser.add_argument("--max-rows-per-class", type=int, default=0,
                        help="If >0, randomly subsample to this many NORMAL rows per class.")
    parser.add_argument("--num-iter", type=int, default=400,
                        help="Training iterations (more than the default 200 because the pool is larger).")
    parser.add_argument("--out-key", default="3w_global_normal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-reduction", action="store_true",
                        help="Apply feature-reduction (variance/correlation) to the aggregated pool.")
    args = parser.parse_args()

    cfg = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    processed_dir = PROJECT_ROOT / cfg["dataset"]["processed_dir"]
    patch_short = int(cfg["paano"]["patch_short"])
    patch_long = int(cfg["paano"]["patch_long"])

    class_ids = [int(c.strip()) for c in args.include_classes.split(",") if c.strip()]
    print(f"[global_encoder] classes={class_ids} patch=({patch_short},{patch_long}) iters={args.num_iter}", flush=True)

    rng = np.random.default_rng(args.seed)
    per_class_rows: list[np.ndarray] = []
    per_class_cols: list[list[str]] = []
    rows_summary: dict[int, int] = {}

    t0 = time.time()
    for cls_id in class_ids:
        p = processed_dir / f"class_{cls_id}" / "features.parquet"
        if not p.exists():
            print(f"[global_encoder] missing {p}, skip", flush=True)
            continue
        df = pd.read_parquet(p, engine="pyarrow")
        rows, cols = _collect_normal_rows_for_class(df, cls_id)
        if len(rows) == 0:
            print(f"[global_encoder] class {cls_id}: 0 NORMAL rows, skip", flush=True)
            continue
        if args.max_rows_per_class > 0 and len(rows) > args.max_rows_per_class:
            idx = rng.choice(len(rows), args.max_rows_per_class, replace=False)
            rows = rows[idx]
        per_class_rows.append(rows)
        per_class_cols.append(cols)
        rows_summary[cls_id] = int(len(rows))
        print(f"[global_encoder] class {cls_id}: {len(rows)} NORMAL rows, {len(cols)} cols", flush=True)

    if not per_class_rows:
        raise SystemExit("No NORMAL rows collected from any class. Did you build 3W dataset first?")

    # Intersect channel sets across participating classes.
    shared = set(per_class_cols[0])
    for cols in per_class_cols[1:]:
        shared &= set(cols)
    shared_channels = sorted(shared)
    print(f"[global_encoder] shared channels across classes: {len(shared_channels)}", flush=True)
    if len(shared_channels) < 4:
        raise SystemExit(f"Too few shared channels ({len(shared_channels)}); aborting.")

    # Project each class' rows to the shared channel space (in shared order).
    aligned: list[np.ndarray] = []
    for rows, cols in zip(per_class_rows, per_class_cols):
        idx = [cols.index(c) for c in shared_channels]
        aligned.append(rows[:, idx])
    pool = np.concatenate(aligned, axis=0).astype(np.float32)
    print(f"[global_encoder] aggregated pool: {len(pool)} rows x {pool.shape[1]} channels", flush=True)

    if args.enable_reduction:
        original = len(shared_channels)
        pool, shared_channels = reduce_features(pool, shared_channels)
        print(f"[global_encoder] reduction: {original} -> {len(shared_channels)} channels", flush=True)

    if len(pool) < patch_long * 4:
        raise SystemExit(f"Pool too small ({len(pool)} rows) for patch_long={patch_long}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[global_encoder] device={device}", flush=True)

    _set_seed(args.seed)

    print(f"[global_encoder] training short scale (patch={patch_short}) ...", flush=True)
    model_short, mean_short, std_short = _train_encoder_single_scale(
        pool, patch_short, device, verbose=True,
        num_iter=args.num_iter, label_prefix="3W-Global-Short",
    )

    print(f"[global_encoder] training long scale (patch={patch_long}) ...", flush=True)
    model_long, mean_long, std_long = _train_encoder_single_scale(
        pool, patch_long, device, verbose=True,
        num_iter=args.num_iter, label_prefix="3W-Global-Long",
    )

    out_path = PROJECT_ROOT / "artifacts" / "3w" / "checkpoints" / f"{args.out_key}_paano_shared_encoder.pt"
    state = SharedEncoderState(
        model_short=model_short,
        model_long=model_long,
        train_mean_short=mean_short,
        train_std_short=std_short,
        train_mean_long=mean_long,
        train_std_long=std_long,
        shared_channels=list(shared_channels),
        patch_short=patch_short,
        patch_long=patch_long,
        anomaly_key=args.out_key,
        train_wells=[f"class_{c}_normal" for c in sorted(rows_summary)],
        detail={
            "training_mode": "3w_multi_source_normal_pretrain",
            "included_classes": sorted(rows_summary.keys()),
            "rows_per_class": rows_summary,
            "pool_rows": int(len(pool)),
            "num_iter": int(args.num_iter),
            "seed": int(args.seed),
            "elapsed_s": round(time.time() - t0, 1),
            "reduction_applied": bool(args.enable_reduction),
        },
    )
    save_shared_encoder_state(state, out_path)
    print(f"[global_encoder] saved -> {out_path}  (elapsed {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
