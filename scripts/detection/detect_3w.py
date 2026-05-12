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

from alma_service.engineered_features import PreparedWellData
from alma_service.shared_encoder import (
    SharedEncoderState,
    _score_well_single_scale,
    load_shared_encoder_state,
    reduce_features,
    save_shared_encoder_state,
    shared_encoder_path,
    train_shared_encoder,
)


def load_class_features(processed_dir: Path, folder_label: int) -> pd.DataFrame:
    p = processed_dir / f"class_{folder_label}" / "features.parquet"
    if not p.exists():
        raise FileNotFoundError(f"features parquet not found: {p}")
    return pd.read_parquet(p, engine="pyarrow")


def build_prepared_wells(
    features_df: pd.DataFrame,
    splits: pd.DataFrame,
    manifest: pd.DataFrame,
    folder_label: int,
    cfg: dict,
) -> tuple[dict[str, PreparedWellData], list[str]]:
    feature_columns = [
        c for c in features_df.columns if c not in ("instance_id", "timestamp", "class_value", "is_normal")
    ]
    if cfg["features"].get("max_channels_after_reduction"):
        train_ids = set(splits[(splits.folder_label == folder_label) & (splits["split"] == "train")]["instance_id"])
        train_rows = features_df[features_df["instance_id"].isin(train_ids) & features_df["is_normal"]]
        if len(train_rows) > cfg["paano"]["patch_long"] * 4 and len(feature_columns) > 1:
            pool = train_rows[feature_columns].to_numpy(dtype=np.float32)
            _, kept = reduce_features(
                pool,
                feature_columns,
                max_channels=cfg["features"]["max_channels_after_reduction"],
            )
            feature_columns = kept
    split_by_id = dict(zip(splits["instance_id"], splits["split"]))
    instance_meta = manifest.set_index("instance_id")
    wells: dict[str, PreparedWellData] = {}
    for iid, grp in features_df.groupby("instance_id", sort=False):
        split = split_by_id.get(iid)
        if split is None:
            continue
        grp = grp.sort_values("timestamp")
        timestamps = grp["timestamp"].to_numpy()
        feature_matrix = grp[feature_columns].to_numpy(dtype=np.float32)
        is_normal = grp["is_normal"].to_numpy().astype(bool)
        first_non_normal = int(np.argmax(~is_normal)) if (~is_normal).any() else len(is_normal)
        if not is_normal.any():
            fallback_len = min(max(len(is_normal) // 4, cfg["paano"]["patch_long"] * 4), len(is_normal))
            reference_mask = np.zeros(len(is_normal), dtype=bool)
            reference_mask[:fallback_len] = True
            reference_end_idx = fallback_len
        else:
            reference_mask = is_normal.copy()
            if first_non_normal == 0 and is_normal.any():
                reference_end_idx = len(is_normal)
            elif (~is_normal).any():
                reference_end_idx = first_non_normal
            else:
                reference_end_idx = len(is_normal)
        stability_mask = np.ones(len(is_normal), dtype=bool)
        warmup = max(cfg["paano"]["patch_long"], 2)
        onset_allowed_mask = np.zeros(len(is_normal), dtype=bool)
        if len(is_normal) > warmup:
            onset_allowed_mask[warmup:] = True
        meta_row = instance_meta.loc[iid] if iid in instance_meta.index else None
        wells[iid] = PreparedWellData(
            well_id=iid,
            split=split,
            timestamps=timestamps,
            raw_columns=feature_columns,
            feature_columns=feature_columns,
            raw_matrix=feature_matrix,
            feature_matrix=feature_matrix,
            reference_end_idx=reference_end_idx,
            reference_mask=reference_mask,
            stability_mask=stability_mask,
            onset_allowed_mask=onset_allowed_mask,
            detail={
                "folder_label": folder_label,
                "source_type": str(meta_row["source_type"]) if meta_row is not None else "",
                "has_event": bool(meta_row["has_event"]) if meta_row is not None else False,
                "has_transient": bool(meta_row["has_transient"]) if meta_row is not None else False,
            },
        )
    return wells, feature_columns


def encoder_artifact_path(folder_label: int, anomaly_key: str) -> Path:
    return PROJECT_ROOT / "artifacts" / "3w" / "checkpoints" / f"{anomaly_key}_paano_shared_encoder.pt"


def train_or_load_encoder(
    wells: dict[str, PreparedWellData],
    folder_label: int,
    cfg: dict,
    device: torch.device,
    *,
    force_retrain: bool,
    anomaly_key: str,
) -> SharedEncoderState:
    artifact = encoder_artifact_path(folder_label, anomaly_key)
    if artifact.exists() and not force_retrain:
        print(f"[encoder] loading existing  {artifact}", flush=True)
        try:
            state = load_shared_encoder_state(anomaly_key, artifact, device=device, compile_model=True, verbose=False)
            return state
        except Exception as exc:
            print(f"[encoder] load failed, retraining ({exc})", flush=True)
    t0 = time.time()
    state = train_shared_encoder(
        wells,
        patch_short=cfg["paano"]["patch_short"],
        patch_long=cfg["paano"]["patch_long"],
        anomaly_key=anomaly_key,
        device=device,
        verbose=True,
    )
    print(f"[encoder] trained in {time.time() - t0:.1f}s  channels={len(state.shared_channels)}", flush=True)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    save_shared_encoder_state(state, artifact)
    return state


def fuse_scores(short: np.ndarray, long_: np.ndarray, weight_short: float) -> np.ndarray:
    def _rank_norm(x: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(x))
        denom = max(len(x) - 1, 1)
        return order.astype(np.float32) / denom
    return weight_short * _rank_norm(short) + (1.0 - weight_short) * _rank_norm(long_)


def score_wells(
    wells: dict[str, PreparedWellData],
    state: SharedEncoderState,
    device: torch.device,
    weight_short: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for iid, well in wells.items():
        col_lookup = {c: i for i, c in enumerate(well.feature_columns)}
        col_indices = [col_lookup[c] for c in state.shared_channels if c in col_lookup]
        if len(col_indices) != len(state.shared_channels):
            missing = [c for c in state.shared_channels if c not in col_lookup]
            print(f"[score] {iid} missing channels: {missing[:5]}... (skipping)", flush=True)
            continue
        well_matrix = well.feature_matrix[:, col_indices]
        ref_matrix = well_matrix[well.reference_mask]
        min_ref_required = max(state.patch_long * 8, 512)
        if len(ref_matrix) < min_ref_required:
            ref_matrix = well_matrix[: max(min_ref_required, len(well_matrix) // 2)]
            if len(ref_matrix) < state.patch_long + 2:
                print(f"[score] {iid} too short for ref ({len(ref_matrix)})", flush=True)
                continue
        try:
            score_short = _score_well_single_scale(
                state.model_short, well_matrix, ref_matrix,
                state.train_mean_short, state.train_std_short,
                state.patch_short, device, verbose=False,
            )
        except Exception as exc:
            print(f"[score] {iid} short failed: {exc}", flush=True)
            continue
        try:
            score_long = _score_well_single_scale(
                state.model_long, well_matrix, ref_matrix,
                state.train_mean_long, state.train_std_long,
                state.patch_long, device, verbose=False,
            )
        except Exception as exc:
            print(f"[score] {iid} long failed (using short only): {exc}", flush=True)
            score_long = score_short
        fused = fuse_scores(score_short, score_long, weight_short)
        df = pd.DataFrame(
            {
                "well_id": iid,
                "timestamp": well.timestamps,
                "split": well.split,
                "score": fused.astype(np.float32),
                "paano_short": score_short.astype(np.float32),
                "paano_long": score_long.astype(np.float32),
                "reference_mask": well.reference_mask,
                "stability_mask": well.stability_mask,
                "onset_allowed_mask": well.onset_allowed_mask,
            }
        )
        rows.append(df)
        print(f"[score] {iid}  split={well.split}  n={len(df)}  ref={int(well.reference_mask.sum())}", flush=True)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PaAno encoder and score 3W instances for one event class.")
    parser.add_argument("--config", default="configs/3w_paano.json")
    parser.add_argument("--event-class", type=int, required=True)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--weight-short", type=float, default=0.6)
    args = parser.parse_args()

    cfg = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    processed_dir = PROJECT_ROOT / cfg["dataset"]["processed_dir"]
    manifest = pd.read_parquet(processed_dir / "manifest.parquet", engine="pyarrow")
    splits = pd.read_parquet(processed_dir / "splits.parquet", engine="pyarrow")
    features_df = load_class_features(processed_dir, args.event_class)

    anomaly_key = f"3w_class_{args.event_class}"
    print(f"[detect_3w] event_class={args.event_class} anomaly_key={anomaly_key}", flush=True)

    wells, feature_columns = build_prepared_wells(features_df, splits, manifest, args.event_class, cfg)
    if not wells:
        print("[detect_3w] no wells built", flush=True)
        return
    print(f"[detect_3w] wells={len(wells)} feature_columns={len(feature_columns)}", flush=True)
    by_split: dict[str, int] = {}
    for w in wells.values():
        by_split[w.split] = by_split.get(w.split, 0) + 1
    print(f"[detect_3w] split counts: {by_split}", flush=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["paano"].get("device", "cuda") == "cuda" else "cpu"
    )
    print(f"[detect_3w] device={device}", flush=True)

    state = train_or_load_encoder(
        wells, args.event_class, cfg, device,
        force_retrain=args.force_retrain, anomaly_key=anomaly_key,
    )

    scores = score_wells(wells, state, device, args.weight_short)
    if scores.empty:
        print("[detect_3w] no scores produced", flush=True)
        return

    scores_dir = PROJECT_ROOT / "artifacts" / "3w" / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    scores_path = scores_dir / f"class_{args.event_class}_scores.parquet"
    scores.to_parquet(scores_path, engine="pyarrow", compression="brotli")
    print(f"[detect_3w] scores -> {scores_path}  rows={len(scores)}", flush=True)

    summary = {
        "event_class": int(args.event_class),
        "anomaly_key": anomaly_key,
        "wells": int(len(wells)),
        "split_counts": by_split,
        "scores_rows": int(len(scores)),
        "shared_channels": int(len(state.shared_channels)),
        "patch_short": int(state.patch_short),
        "patch_long": int(state.patch_long),
    }
    summary_path = scores_dir / f"class_{args.event_class}_scores.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[detect_3w] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
