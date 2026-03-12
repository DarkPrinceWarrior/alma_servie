"""
PaAno-based anomaly detection for Изменение притока (inflow change).
Uses causal onset detection and multi-scale PaAno score fusion.
"""

import os
import sys
import time
import random
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "paano"))

import torch
from model import PatchEncoder
from train import train_model
from utils.data_preprocess import PatchCreator, preprocess_to_patches
from utils.utils import create_memory_bank
from utils.evaluation import calculate_anomaly_scores, distribute_patch_scores_to_points

from onset_detection import (
    choose_reference_end_index,
    robust_scale_for_fusion,
    calibrate_causal_thresholds,
    detect_causal_onsets,
)

warnings.filterwarnings("ignore")

SEED = 2027
PATCH_SIZE_SHORT = 32
PATCH_SIZE_LONG = 64
NUM_ITERS = 200
BATCH_SIZE = 256
LR = 1e-4
TOP_K = 3
MEMORY_BANK_RATIO = 0.1

REFERENCE_MIN_RATIO = 0.15
REFERENCE_MIN_DAYS = 1.0
FUSION_SHORT_WEIGHT = 0.65
FUSION_LONG_WEIGHT = 0.35

TARGET_FAR_PER_DAY = 0.30
MIN_RUN_POINTS = 4
COOLDOWN_HOURS = 24
EMA_ALPHA = 0.08


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pritok_data(source_path=None):
    if source_path is not None:
        src = Path(source_path)
    else:
        candidates = [
            Path("db/pritok_anomaly_database_2min.csv"),
            Path("db/pritok_anomaly_database_15s.csv"),
        ]
        src = next((p for p in candidates if p.exists()), None)
    if src is None or not src.exists():
        print(f"File not found: {src}")
        return pd.DataFrame()
    print(f"Loading pritok data from: {src}")
    df = pd.read_csv(src, dtype={"well_id": str}, low_memory=False)
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["well_id", "timestamp"])
    return df


def load_pritok_intervals():
    src = Path("db/pritok_intervals.csv")
    if not src.exists():
        return pd.DataFrame()
    intervals = pd.read_csv(src, dtype={"well_id": str})
    intervals["well_id"] = intervals["well_id"].astype(str).str.strip().str.lower()
    intervals["start_date"] = pd.to_datetime(intervals["start_date"], errors="coerce")
    intervals["end_date"] = pd.to_datetime(intervals["end_date"], errors="coerce")
    intervals["data_start"] = pd.to_datetime(intervals.get("data_start"), errors="coerce")
    intervals["data_end"] = pd.to_datetime(intervals.get("data_end"), errors="coerce")
    if "interval_idx" not in intervals.columns:
        intervals["interval_idx"] = intervals.groupby("well_id").cumcount() + 1
    return intervals.sort_values(["well_id", "start_date", "interval_idx"]).reset_index(drop=True)


def _fill_nans_forward(arr):
    out = arr.copy()
    for col_idx in range(out.shape[1]):
        col = out[:, col_idx]
        mask = np.isnan(col)
        if mask.all():
            out[:, col_idx] = 0.0
            continue
        if mask.any():
            idx = np.where(~mask, np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            out[:, col_idx] = col[idx]
            still_nan = np.isnan(out[:, col_idx])
            if still_nan.any():
                out[still_nan, col_idx] = col[~mask][0]
    return out


def prepare_well_matrix(well_df):
    wd = well_df.sort_values("timestamp").reset_index(drop=True)
    timestamps = wd["timestamp"].to_numpy()
    numeric_cols = [c for c in wd.columns if c not in ("timestamp", "well_id")]
    data = wd[numeric_cols].to_numpy(dtype=np.float32)
    data = _fill_nans_forward(data)
    return data, timestamps


def run_paano_single_scale(
    data,
    patch_size,
    reference_end_idx,
    device,
    verbose=False,
):
    if len(data) < patch_size * 4:
        return None
    if reference_end_idx < patch_size * 2:
        return None

    train_data = data[:reference_end_idx]
    train_mean = np.mean(train_data, axis=0, keepdims=True).astype(np.float32)
    train_std = np.std(train_data, axis=0, keepdims=True).astype(np.float32)
    train_std = np.where(train_std == 0.0, 1e-8, train_std)

    full_norm = (data - train_mean) / train_std
    train_norm = (train_data - train_mean) / train_std
    dummy_labels = np.zeros(len(data), dtype=np.float32)

    patch_creator = PatchCreator(L=patch_size, s=1, random_seed=SEED)
    train_loader, full_loader, _ = patch_creator.create_dataloaders(
        train_norm, full_norm, dummy_labels, batch_size=BATCH_SIZE
    )

    model = PatchEncoder(in_channels=data.shape[1], use_revin=True).to(device)

    t0 = time.time()
    train_patches = preprocess_to_patches(train_norm, patch_size=patch_size, stride=1)
    train_model(
        model,
        train_loader,
        train_patches,
        device,
        num_iter=NUM_ITERS,
        pretext_step=patch_size,
        lr=LR,
        see_loss=False,
    )
    t_train = time.time() - t0

    memory_bank, _ = create_memory_bank(model, train_loader, device, num_cores=MEMORY_BANK_RATIO)
    patch_scores = calculate_anomaly_scores(model, full_loader, memory_bank, top_k=TOP_K, device=device)
    point_scores = distribute_patch_scores_to_points(
        patch_scores, patch_size=patch_size, num_points=len(data)
    )

    if verbose:
        print(
            f"    scale patch={patch_size}: train_pts={len(train_data)}, "
            f"train_time={t_train:.1f}s"
        )
    return {
        "scores": point_scores,
        "train_points": len(train_data),
        "train_time": t_train,
        "patch_size": patch_size,
    }


def detect_pritok_paano_well(well_df, well_id, verbose=True):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, timestamps = prepare_well_matrix(well_df)

    if len(data) < PATCH_SIZE_LONG * 4:
        return None, timestamps, "Not enough data for PaAno"

    reference_end_idx = choose_reference_end_index(
        timestamps=timestamps,
        patch_size=PATCH_SIZE_LONG,
        min_ratio=REFERENCE_MIN_RATIO,
        min_days=REFERENCE_MIN_DAYS,
    )
    if verbose:
        print(
            f"  PaAno: {well_id} | points={len(data)}, channels={data.shape[1]}, "
            f"reference_end_idx={reference_end_idx}"
        )

    short_run = run_paano_single_scale(
        data=data,
        patch_size=PATCH_SIZE_SHORT,
        reference_end_idx=reference_end_idx,
        device=device,
        verbose=verbose,
    )
    long_run = run_paano_single_scale(
        data=data,
        patch_size=PATCH_SIZE_LONG,
        reference_end_idx=reference_end_idx,
        device=device,
        verbose=verbose,
    )
    if short_run is None or long_run is None:
        return None, timestamps, "Not enough data after reference split"

    short_z = robust_scale_for_fusion(short_run["scores"], reference_end_idx)
    long_z = robust_scale_for_fusion(long_run["scores"], reference_end_idx)
    fused_score = FUSION_SHORT_WEIGHT * short_z + FUSION_LONG_WEIGHT * long_z

    thresholds, diagnostics = calibrate_causal_thresholds(
        scores=fused_score,
        timestamps=timestamps,
        reference_end_idx=reference_end_idx,
        target_far_per_day=TARGET_FAR_PER_DAY,
        min_run_points=MIN_RUN_POINTS,
        ema_alpha=EMA_ALPHA,
    )
    starts = detect_causal_onsets(
        scores=fused_score,
        timestamps=timestamps,
        diagnostics=diagnostics,
        thresholds=thresholds,
        reference_end_idx=reference_end_idx,
        min_run_points=MIN_RUN_POINTS,
        cooldown_hours=COOLDOWN_HOURS,
    )

    detail = (
        f"PaAno multi-scale(s={PATCH_SIZE_SHORT},l={PATCH_SIZE_LONG},iters={NUM_ITERS},"
        f"train_ref={reference_end_idx},q={thresholds.quantile:.5f},"
        f"thr_score={thresholds.score_threshold:.3f},thr_ema={thresholds.ema_z_threshold:.3f},"
        f"thr_cusum={thresholds.cusum_threshold:.3f},starts={len(starts)})"
    )
    if verbose:
        print(f"  PaAno done: {detail}")

    return {
        "fused_score": fused_score,
        "short_score": short_run["scores"],
        "long_score": long_run["scores"],
        "starts": starts,
    }, timestamps, detail


def _map_predictions_to_intervals(
    well_id,
    pred_starts,
    well_intervals,
    detail,
    prestart_tolerance_hours=24,
    early_tolerance_hours=24,
):
    prestart_tolerance = pd.Timedelta(hours=prestart_tolerance_hours)
    early_status_tolerance = pd.Timedelta(hours=early_tolerance_hours)

    rows = []
    used = [False] * len(pred_starts)

    for _, row in well_intervals.sort_values(["start_date", "interval_idx"]).iterrows():
        interval_idx = int(row.get("interval_idx", 1))
        start_dt = row["start_date"]
        end_dt = row["end_date"]

        detected_time = None
        for i, ts in enumerate(pred_starts):
            if used[i]:
                continue
            if (start_dt - prestart_tolerance) <= ts <= end_dt:
                detected_time = ts
                used[i] = True
                break

        if detected_time is None:
            status = "Not found"
        elif detected_time < start_dt - early_status_tolerance:
            status = "Early detected"
        else:
            status = "Detected"

        rows.append(
            {
                "well_id": well_id,
                "interval_idx": interval_idx,
                "detected_time": detected_time,
                "actual_start": start_dt,
                "actual_end": end_dt,
                "status": status,
                "detail": detail,
            }
        )
    return rows


def run_pritok_paano_detection(output_path="pritok_paano_results.csv", source_path=None, verbose=True):
    print("=== PaAno Pritok Detection (causal + multi-scale) ===")
    df = load_pritok_data(source_path=source_path)
    if df.empty:
        print("No pritok data found.")
        return

    intervals = load_pritok_intervals()
    if intervals.empty:
        print("No pritok intervals found.")
        return

    results = []
    score_rows = []
    pred_rows = []
    all_wells = sorted(df["well_id"].unique())

    for wid in all_wells:
        well_data = df[df["well_id"] == wid]
        well_intervals = intervals[intervals["well_id"] == wid]

        out, timestamps, detail = detect_pritok_paano_well(well_data, wid, verbose=verbose)
        if out is None:
            for _, row in well_intervals.iterrows():
                results.append(
                    {
                        "well_id": wid,
                        "interval_idx": int(row.get("interval_idx", 1)),
                        "detected_time": None,
                        "actual_start": row["start_date"],
                        "actual_end": row["end_date"],
                        "status": "Not found",
                        "detail": detail,
                    }
                )
            continue

        pred_starts = out["starts"]
        if verbose:
            print(f"  Predictions for {wid}: {len(pred_starts)} starts")

        for ts in pred_starts:
            pred_rows.append({"well_id": wid, "detected_time": ts})

        fused = out["fused_score"]
        short = out["short_score"]
        longv = out["long_score"]
        for ts, sc, sc_s, sc_l in zip(timestamps, fused, short, longv):
            score_rows.append(
                {
                    "well_id": wid,
                    "timestamp": ts,
                    "paano_score": float(sc),
                    "paano_score_short": float(sc_s),
                    "paano_score_long": float(sc_l),
                }
            )

        results.extend(
            _map_predictions_to_intervals(
                well_id=wid,
                pred_starts=pred_starts,
                well_intervals=well_intervals,
                detail=detail,
                prestart_tolerance_hours=24,
                early_tolerance_hours=24,
            )
        )

    res_df = pd.DataFrame(results)
    print("\n=== PaAno Pritok Detection Results ===")
    if not res_df.empty:
        print(
            res_df[["well_id", "interval_idx", "detected_time", "actual_start", "actual_end", "status"]].to_string()
        )
    res_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    scores_output = Path("db/pritok_paano_scores.csv")
    if score_rows:
        pd.DataFrame(score_rows).to_csv(scores_output, index=False)
        print(f"Per-point scores saved to {scores_output}")

    pred_output = Path("db/pritok_paano_predicted_starts.csv")
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(pred_output, index=False)
        print(f"Predicted starts saved to {pred_output}")
    else:
        pd.DataFrame(columns=["well_id", "detected_time"]).to_csv(pred_output, index=False)
        print(f"Predicted starts saved to {pred_output} (empty)")

    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaAno Pritok Detection")
    parser.add_argument("--output", type=str, default="pritok_paano_results.csv")
    parser.add_argument("--source", type=str, default=None)
    args = parser.parse_args()
    run_pritok_paano_detection(output_path=args.output, source_path=args.source)
