import argparse
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "paano"))

import torch
from model import PatchEncoder
from train import train_model
from utils.data_preprocess import PatchCreator, preprocess_to_patches
from utils.evaluation import calculate_anomaly_scores, distribute_patch_scores_to_points
from utils.utils import create_memory_bank

from onset_detection import (
    calibrate_causal_thresholds,
    calibrate_causal_thresholds_from_reference_mask,
    choose_reference_end_index,
    detect_causal_onsets,
    robust_scale_for_fusion,
    robust_scale_for_fusion_mask,
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

DEFAULT_CONFIG = {
    "fusion_weight_short": 0.65,
    "target_far_per_day": 2.0,
    "min_run_points": 2,
    "cooldown_hours": 8,
    "ema_alpha": 0.08,
    "gate_mode": "relaxed",
    "prestart_tolerance_hours": 6,
    "early_tolerance_hours": 6,
    "holdout_buffer_hours": 12,
}

AUTOTUNE_GRID = {
    "fusion_weight_short": [0.35, 0.5, 0.65, 0.8],
    "target_far_per_day": [1.0, 2.0, 4.0, 8.0, 12.0],
    "min_run_points": [1, 2, 3, 4],
    "cooldown_hours": [4, 8, 12, 24],
    "ema_alpha": [0.04, 0.06, 0.08, 0.12],
    "gate_mode": ["relaxed", "score_ema", "strict"],
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_salt_data(source_path=None):
    if source_path is not None:
        src = Path(source_path)
        if not src.exists():
            print(f"Source file not found: {src}")
            return pd.DataFrame()
    else:
        candidates = [
            Path("db/salt_anomaly_database_2min.csv"),
            Path("db/salt_anomaly_database_15s.csv"),
            Path("db/salt_anomaly_database_interpolated_2min.csv"),
            Path("db/salt_anomaly_database_interpolated.csv"),
            Path("db/salt_anomaly_database.csv"),
        ]
        src = next((p for p in candidates if p.exists()), None)
    if src is None:
        return pd.DataFrame()
    print(f"Loading salt data from: {src}")
    df = pd.read_csv(src, dtype={"well_id": str}, low_memory=False)
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["well_id", "timestamp"])
    return df


def load_salt_intervals(required=False):
    src = Path("db/salt_intervals.csv")
    if not src.exists():
        if required:
            raise FileNotFoundError(f"Intervals file not found: {src}")
        return pd.DataFrame(
            columns=["well_id", "start_date", "end_date", "data_start", "data_end", "interval_idx"]
        )

    intervals = pd.read_csv(src, dtype={"well_id": str})
    intervals["well_id"] = intervals["well_id"].astype(str).str.strip().str.lower()
    intervals["start_date"] = pd.to_datetime(intervals["start_date"], errors="coerce")
    intervals["end_date"] = pd.to_datetime(intervals["end_date"], errors="coerce")
    if "data_start" in intervals.columns:
        intervals["data_start"] = pd.to_datetime(intervals["data_start"], errors="coerce")
    else:
        intervals["data_start"] = pd.NaT
    if "data_end" in intervals.columns:
        intervals["data_end"] = pd.to_datetime(intervals["data_end"], errors="coerce")
    else:
        intervals["data_end"] = pd.NaT

    intervals = intervals.dropna(subset=["well_id", "start_date", "end_date"]).copy()
    if "interval_idx" not in intervals.columns:
        intervals["interval_idx"] = intervals.groupby("well_id").cumcount() + 1
    intervals["interval_idx"] = pd.to_numeric(intervals["interval_idx"], errors="coerce").fillna(1).astype(int)
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


def make_prefix_mask(n_points, end_idx):
    mask = np.zeros(n_points, dtype=bool)
    mask[: int(np.clip(end_idx, 0, n_points))] = True
    return mask


def make_dev_train_mask(
    timestamps,
    well_intervals,
    holdout_interval_idx=None,
    holdout_buffer_hours=24,
):
    n = len(timestamps)
    mask = np.ones(n, dtype=bool)

    for _, row in well_intervals.iterrows():
        s = row["start_date"]
        e = row["end_date"]
        inside = (timestamps >= np.datetime64(s)) & (timestamps <= np.datetime64(e))
        mask[inside] = False

    if holdout_interval_idx is not None:
        row = well_intervals[well_intervals["interval_idx"] == int(holdout_interval_idx)]
        if not row.empty:
            r = row.iloc[0]
            delta = pd.Timedelta(hours=float(holdout_buffer_hours))
            hs = r["start_date"] - delta
            he = r["end_date"] + delta
            hold = (timestamps >= np.datetime64(hs)) & (timestamps <= np.datetime64(he))
            mask[hold] = False
    return mask


def collect_train_data_from_mask(data, mask, patch_size):
    parts = []
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        if (not ok) and start is not None:
            if i - start >= patch_size:
                parts.append(data[start:i])
            start = None
    if start is not None and len(data) - start >= patch_size:
        parts.append(data[start:])

    if parts:
        train_data = np.concatenate(parts, axis=0)
    else:
        train_data = data[mask]
    if len(train_data) < patch_size * 2:
        return None
    return train_data


def run_paano_single_scale(data, train_data, patch_size, device, verbose=False):
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
    return point_scores


def compute_multiscale_scores(data, reference_mask, device, verbose=False):
    train_short = collect_train_data_from_mask(data, reference_mask, PATCH_SIZE_SHORT)
    train_long = collect_train_data_from_mask(data, reference_mask, PATCH_SIZE_LONG)
    if train_short is None or train_long is None:
        return None

    short_score = run_paano_single_scale(
        data=data,
        train_data=train_short,
        patch_size=PATCH_SIZE_SHORT,
        device=device,
        verbose=verbose,
    )
    long_score = run_paano_single_scale(
        data=data,
        train_data=train_long,
        patch_size=PATCH_SIZE_LONG,
        device=device,
        verbose=verbose,
    )

    short_z = robust_scale_for_fusion_mask(short_score, reference_mask)
    long_z = robust_scale_for_fusion_mask(long_score, reference_mask)
    return {
        "short_score": short_score,
        "long_score": long_score,
        "short_z": short_z,
        "long_z": long_z,
    }


def detect_starts_from_fused(
    fused_score,
    timestamps,
    reference_mask,
    cfg,
    blind_reference_end_idx=None,
):
    if blind_reference_end_idx is None:
        thresholds, diagnostics = calibrate_causal_thresholds_from_reference_mask(
            scores=fused_score,
            timestamps=timestamps,
            reference_mask=reference_mask,
            target_far_per_day=float(cfg["target_far_per_day"]),
            min_run_points=int(cfg["min_run_points"]),
            ema_alpha=float(cfg["ema_alpha"]),
        )
        start_idx = 0
    else:
        thresholds, diagnostics = calibrate_causal_thresholds(
            scores=fused_score,
            timestamps=timestamps,
            reference_end_idx=int(blind_reference_end_idx),
            target_far_per_day=float(cfg["target_far_per_day"]),
            min_run_points=int(cfg["min_run_points"]),
            ema_alpha=float(cfg["ema_alpha"]),
        )
        start_idx = int(blind_reference_end_idx)

    gate_mode = str(cfg.get("gate_mode", "relaxed")).strip().lower()
    if gate_mode not in {"relaxed", "score_ema", "strict"}:
        gate_mode = "relaxed"

    starts = detect_causal_onsets(
        scores=fused_score,
        timestamps=timestamps,
        diagnostics=diagnostics,
        thresholds=thresholds,
        reference_end_idx=start_idx,
        min_run_points=int(cfg["min_run_points"]),
        cooldown_hours=float(cfg["cooldown_hours"]),
        gate_mode=gate_mode,
    )
    return starts, thresholds


def select_detection_for_interval(starts, start_dt, end_dt, prestart_tolerance_hours):
    starts = sorted(pd.Timestamp(x) for x in starts)
    inside = [x for x in starts if start_dt <= x <= end_dt]
    if inside:
        return inside[0]

    prestart = pd.Timedelta(hours=float(prestart_tolerance_hours))
    before = [x for x in starts if (start_dt - prestart) <= x < start_dt]
    if before:
        return max(before)
    return None


def map_predictions_to_intervals(well_id, starts, well_intervals, cfg, detail):
    rows = []
    prestart_tol = float(cfg["prestart_tolerance_hours"])
    early_tol = pd.Timedelta(hours=float(cfg["early_tolerance_hours"]))

    for _, row in well_intervals.sort_values(["start_date", "interval_idx"]).iterrows():
        start_dt = row["start_date"]
        end_dt = row["end_date"]
        interval_idx = int(row["interval_idx"])

        det = select_detection_for_interval(
            starts=starts,
            start_dt=start_dt,
            end_dt=end_dt,
            prestart_tolerance_hours=prestart_tol,
        )

        if det is None:
            status = "Not found"
        elif det < start_dt - early_tol:
            status = "Early detected"
        else:
            status = "Detected"

        rows.append(
            {
                "well_id": well_id,
                "interval_idx": interval_idx,
                "type": "Salt",
                "detected_time": det,
                "actual_type": "Salt",
                "actual_start": start_dt,
                "actual_end": end_dt,
                "status": status,
                "detail": detail,
            }
        )
    return rows


def evaluate_candidate_on_runs(interval_runs, candidate_cfg):
    hits = 0
    delays = []
    starts_total = 0
    early_hits = 0

    for run in interval_runs:
        w = float(candidate_cfg["fusion_weight_short"])
        fused = w * run["short_z"] + (1.0 - w) * run["long_z"]
        starts, _ = detect_starts_from_fused(
            fused_score=fused,
            timestamps=run["timestamps"],
            reference_mask=run["reference_mask"],
            cfg=candidate_cfg,
            blind_reference_end_idx=None,
        )
        starts_total += len(starts)

        det = select_detection_for_interval(
            starts=starts,
            start_dt=run["start_dt"],
            end_dt=run["end_dt"],
            prestart_tolerance_hours=float(candidate_cfg["prestart_tolerance_hours"]),
        )
        if det is None:
            continue
        hits += 1
        delay_h = (det - run["start_dt"]).total_seconds() / 3600.0
        delays.append(delay_h)
        if delay_h < 0:
            early_hits += 1

    n = len(interval_runs)
    hit_rate = (hits / n) if n else 0.0
    if delays:
        abs_delays = np.abs(np.asarray(delays, dtype=np.float32))
        mae_abs = float(np.mean(abs_delays))
        med_abs = float(np.median(abs_delays))
    else:
        mae_abs = 1e9
        med_abs = 1e9

    metrics = {
        "hit_count": int(hits),
        "hit_rate": float(hit_rate),
        "mae_abs_delay_h": mae_abs,
        "median_abs_delay_h": med_abs,
        "avg_starts_per_interval_run": float(starts_total / max(n, 1)),
        "early_hit_count": int(early_hits),
    }
    score_key = (
        metrics["hit_count"],
        -metrics["median_abs_delay_h"],
        -metrics["mae_abs_delay_h"],
        -metrics["early_hit_count"],
        -metrics["avg_starts_per_interval_run"],
    )
    return score_key, metrics


def auto_tune_dev_config(interval_runs, base_cfg, verbose=True):
    if not interval_runs:
        return base_cfg.copy(), {"message": "No interval runs for auto-tuning."}

    candidates = []
    for w in AUTOTUNE_GRID["fusion_weight_short"]:
        for far in AUTOTUNE_GRID["target_far_per_day"]:
            for mr in AUTOTUNE_GRID["min_run_points"]:
                for cd in AUTOTUNE_GRID["cooldown_hours"]:
                    for ea in AUTOTUNE_GRID["ema_alpha"]:
                        for gm in AUTOTUNE_GRID["gate_mode"]:
                            cfg = base_cfg.copy()
                            cfg["fusion_weight_short"] = float(w)
                            cfg["target_far_per_day"] = float(far)
                            cfg["min_run_points"] = int(mr)
                            cfg["cooldown_hours"] = float(cd)
                            cfg["ema_alpha"] = float(ea)
                            cfg["gate_mode"] = str(gm)
                            candidates.append(cfg)

    best_cfg = None
    best_key = None
    leaderboard = []
    for cfg in candidates:
        key, metrics = evaluate_candidate_on_runs(interval_runs, cfg)
        row = {**metrics, **cfg}
        leaderboard.append((key, row))
        if best_key is None or key > best_key:
            best_key = key
            best_cfg = cfg.copy()

    leaderboard = sorted(leaderboard, key=lambda x: x[0], reverse=True)
    top_rows = [row for _, row in leaderboard[:10]]

    if verbose and top_rows:
        print("  Auto-tune top configs:")
        for i, row in enumerate(top_rows[:5], 1):
            print(
                f"    {i}. hit={row['hit_count']}/{len(interval_runs)}, "
                f"med_abs={row['median_abs_delay_h']:.2f}h, "
                f"mae_abs={row['mae_abs_delay_h']:.2f}h, "
                f"w={row['fusion_weight_short']:.2f}, far={row['target_far_per_day']:.2f}, "
                f"run={row['min_run_points']}, cd={row['cooldown_hours']:.0f}, "
                f"ema={row['ema_alpha']:.2f}, gate={row['gate_mode']}"
            )

    summary = {
        "interval_runs": len(interval_runs),
        "best_score_key": list(best_key) if best_key is not None else None,
        "top10": top_rows,
    }
    return best_cfg, summary


def load_or_default_config(config_path):
    cfg = DEFAULT_CONFIG.copy()
    p = Path(config_path)
    if p.exists():
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                # Support either flat config or wrapped {"config": ...}
                if "config" in loaded and isinstance(loaded["config"], dict):
                    loaded = loaded["config"]
                for k, v in loaded.items():
                    if k in cfg:
                        cfg[k] = v
        except Exception:
            pass
    return cfg


def save_config(config_path, cfg, metadata=None):
    payload = {"config": cfg}
    if metadata is not None:
        payload["metadata"] = metadata
    Path(config_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_interval_runs_for_dev(well_id, data, timestamps, well_intervals, device, cfg, verbose=True):
    runs = []
    for _, row in well_intervals.sort_values(["start_date", "interval_idx"]).iterrows():
        interval_idx = int(row["interval_idx"])
        ref_mask = make_dev_train_mask(
            timestamps=timestamps,
            well_intervals=well_intervals,
            holdout_interval_idx=interval_idx,
            holdout_buffer_hours=float(cfg["holdout_buffer_hours"]),
        )
        ms = compute_multiscale_scores(data=data, reference_mask=ref_mask, device=device, verbose=verbose)
        if ms is None:
            runs.append(
                {
                    "well_id": well_id,
                    "interval_idx": interval_idx,
                    "start_dt": row["start_date"],
                    "end_dt": row["end_date"],
                    "timestamps": timestamps,
                    "reference_mask": ref_mask,
                    "short_z": None,
                    "long_z": None,
                    "not_enough_data": True,
                }
            )
            continue

        runs.append(
            {
                "well_id": well_id,
                "interval_idx": interval_idx,
                "start_dt": row["start_date"],
                "end_dt": row["end_date"],
                "timestamps": timestamps,
                "reference_mask": ref_mask,
                "short_z": ms["short_z"],
                "long_z": ms["long_z"],
                "not_enough_data": False,
            }
        )
    return runs


def run_dev_for_well(well_id, data, timestamps, well_intervals, device, cfg, verbose=True):
    interval_runs = build_interval_runs_for_dev(
        well_id=well_id,
        data=data,
        timestamps=timestamps,
        well_intervals=well_intervals,
        device=device,
        cfg=cfg,
        verbose=verbose,
    )
    valid_runs = [r for r in interval_runs if not r["not_enough_data"]]

    result_rows = []
    for r in interval_runs:
        if r["not_enough_data"]:
            result_rows.append(
                {
                    "well_id": well_id,
                    "interval_idx": r["interval_idx"],
                    "type": "Salt",
                    "detected_time": None,
                    "actual_type": "Salt",
                    "actual_start": r["start_dt"],
                    "actual_end": r["end_dt"],
                    "status": "Not found",
                    "detail": "Not enough data for PaAno in LOIO mode",
                }
            )
            continue

        w = float(cfg["fusion_weight_short"])
        fused = w * r["short_z"] + (1.0 - w) * r["long_z"]
        starts, thr = detect_starts_from_fused(
            fused_score=fused,
            timestamps=timestamps,
            reference_mask=r["reference_mask"],
            cfg=cfg,
            blind_reference_end_idx=None,
        )
        det = select_detection_for_interval(
            starts=starts,
            start_dt=r["start_dt"],
            end_dt=r["end_dt"],
            prestart_tolerance_hours=float(cfg["prestart_tolerance_hours"]),
        )
        if det is None:
            status = "Not found"
        elif det < r["start_dt"] - pd.Timedelta(hours=float(cfg["early_tolerance_hours"])):
            status = "Early detected"
        else:
            status = "Detected"

        detail = (
            f"PaAno dev-loio(s={PATCH_SIZE_SHORT},l={PATCH_SIZE_LONG},"
            f"w={w:.2f},far={cfg['target_far_per_day']},run={cfg['min_run_points']},"
            f"cd={cfg['cooldown_hours']},ema={cfg['ema_alpha']},gate={cfg.get('gate_mode','relaxed')},"
            f"thr_score={thr.score_threshold:.3f},thr_ema={thr.ema_z_threshold:.3f},"
            f"thr_cusum={thr.cusum_threshold:.3f},starts={len(starts)})"
        )

        result_rows.append(
            {
                "well_id": well_id,
                "interval_idx": r["interval_idx"],
                "type": "Salt",
                "detected_time": det,
                "actual_type": "Salt",
                "actual_start": r["start_dt"],
                "actual_end": r["end_dt"],
                "status": status,
                "detail": detail,
            }
        )

    # For report/scoring overlays: one well-level model on all GT-normal points.
    full_ref_mask = make_dev_train_mask(
        timestamps=timestamps,
        well_intervals=well_intervals,
        holdout_interval_idx=None,
        holdout_buffer_hours=0,
    )
    ms_full = compute_multiscale_scores(data=data, reference_mask=full_ref_mask, device=device, verbose=verbose)
    score_rows = []
    pred_rows = []
    if ms_full is not None:
        w = float(cfg["fusion_weight_short"])
        fused = w * ms_full["short_z"] + (1.0 - w) * ms_full["long_z"]
        starts, _ = detect_starts_from_fused(
            fused_score=fused,
            timestamps=timestamps,
            reference_mask=full_ref_mask,
            cfg=cfg,
            blind_reference_end_idx=None,
        )
        for ts in starts:
            pred_rows.append({"well_id": well_id, "detected_time": ts})
        for ts, sc, sc_s, sc_l in zip(timestamps, fused, ms_full["short_score"], ms_full["long_score"]):
            score_rows.append(
                {
                    "well_id": well_id,
                    "timestamp": ts,
                    "paano_score": float(sc),
                    "paano_score_short": float(sc_s),
                    "paano_score_long": float(sc_l),
                }
            )
    return result_rows, score_rows, pred_rows, valid_runs


def run_blind_for_well(well_id, data, timestamps, well_intervals, device, cfg, verbose=True):
    ref_end_idx = choose_reference_end_index(
        timestamps=timestamps,
        patch_size=PATCH_SIZE_LONG,
        min_ratio=REFERENCE_MIN_RATIO,
        min_days=REFERENCE_MIN_DAYS,
    )
    reference_mask = make_prefix_mask(len(data), ref_end_idx)

    if verbose:
        print(
            f"  PaAno blind: {well_id} | points={len(data)}, "
            f"reference_end_idx={ref_end_idx}"
        )

    ms = compute_multiscale_scores(data=data, reference_mask=reference_mask, device=device, verbose=verbose)
    if ms is None:
        if well_intervals.empty:
            return [], [], [], "Not enough data for PaAno"
        rows = []
        for _, row in well_intervals.iterrows():
            rows.append(
                {
                    "well_id": well_id,
                    "interval_idx": int(row["interval_idx"]),
                    "type": "Salt",
                    "detected_time": None,
                    "actual_type": "Salt",
                    "actual_start": row["start_date"],
                    "actual_end": row["end_date"],
                    "status": "Not found",
                    "detail": "Not enough data for PaAno",
                }
            )
        return rows, [], [], "Not enough data for PaAno"

    w = float(cfg["fusion_weight_short"])
    fused = w * robust_scale_for_fusion(ms["short_score"], ref_end_idx) + (1.0 - w) * robust_scale_for_fusion(
        ms["long_score"], ref_end_idx
    )
    starts, thr = detect_starts_from_fused(
        fused_score=fused,
        timestamps=timestamps,
        reference_mask=reference_mask,
        cfg=cfg,
        blind_reference_end_idx=ref_end_idx,
    )

    detail = (
        f"PaAno blind(s={PATCH_SIZE_SHORT},l={PATCH_SIZE_LONG},w={w:.2f},"
        f"far={cfg['target_far_per_day']},run={cfg['min_run_points']},"
        f"cd={cfg['cooldown_hours']},ema={cfg['ema_alpha']},gate={cfg.get('gate_mode','relaxed')},"
        f"ref={ref_end_idx},"
        f"thr_score={thr.score_threshold:.3f},thr_ema={thr.ema_z_threshold:.3f},"
        f"thr_cusum={thr.cusum_threshold:.3f},starts={len(starts)})"
    )

    pred_rows = [{"well_id": well_id, "detected_time": ts} for ts in starts]
    score_rows = [
        {
            "well_id": well_id,
            "timestamp": ts,
            "paano_score": float(sc),
            "paano_score_short": float(sc_s),
            "paano_score_long": float(sc_l),
        }
        for ts, sc, sc_s, sc_l in zip(timestamps, fused, ms["short_score"], ms["long_score"])
    ]

    if well_intervals.empty:
        return [], score_rows, pred_rows, detail
    rows = map_predictions_to_intervals(
        well_id=well_id,
        starts=starts,
        well_intervals=well_intervals,
        cfg=cfg,
        detail=detail,
    )
    return rows, score_rows, pred_rows, detail


def run_salt_paano_detection(
    output_path="salt_paano_results.csv",
    source_path=None,
    mode="dev",
    config_path="db/salt_paano_config.json",
    auto_tune=None,
    verbose=True,
):
    print(f"=== PaAno Salt Detection ({mode}) ===")
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    salt_df = load_salt_data(source_path=source_path)
    if salt_df.empty:
        print("No salt data found.")
        return None

    intervals = load_salt_intervals(required=(mode == "dev"))
    cfg = load_or_default_config(config_path)

    if auto_tune is None:
        auto_tune = False

    all_wells = sorted(salt_df["well_id"].unique())
    results = []
    score_rows = []
    pred_rows = []
    if mode == "dev":
        if intervals.empty:
            print("No salt intervals found for dev mode.")
            return None

        tuning_meta = None
        if auto_tune:
            # Stage 1: build LOIO runs for auto-tuning only when requested.
            tuning_runs = []
            for wid in all_wells:
                well_data = salt_df[salt_df["well_id"] == wid]
                well_intervals = intervals[intervals["well_id"] == wid]
                if well_intervals.empty:
                    continue
                data, timestamps = prepare_well_matrix(well_data)
                local_runs = build_interval_runs_for_dev(
                    well_id=wid,
                    data=data,
                    timestamps=timestamps,
                    well_intervals=well_intervals,
                    device=device,
                    cfg=cfg,
                    verbose=verbose,
                )
                tuning_runs.extend([r for r in local_runs if not r["not_enough_data"]])

            cfg, tune_summary = auto_tune_dev_config(tuning_runs, cfg, verbose=verbose)
            tuning_meta = {"mode": "dev", "auto_tuned": True, "summary": tune_summary}
            save_config(config_path, cfg, metadata=tuning_meta)
            Path("db/salt_dev_tuning.json").write_text(
                json.dumps(tune_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved tuned config to {config_path}")
            print("Saved dev tuning summary to db/salt_dev_tuning.json")
        else:
            tuning_meta = {"mode": "dev", "auto_tuned": False}

        # Stage 2: final dev results with selected config.
        for wid in all_wells:
            well_data = salt_df[salt_df["well_id"] == wid]
            well_intervals = intervals[intervals["well_id"] == wid]
            if well_intervals.empty:
                continue
            data, timestamps = prepare_well_matrix(well_data)
            rows, sc_rows, pr_rows, valid_runs = run_dev_for_well(
                well_id=wid,
                data=data,
                timestamps=timestamps,
                well_intervals=well_intervals,
                device=device,
                cfg=cfg,
                verbose=verbose,
            )
            results.extend(rows)
            score_rows.extend(sc_rows)
            pred_rows.extend(pr_rows)

    else:
        # Blind mode: per-well independent inference, no GT required.
        for wid in all_wells:
            well_data = salt_df[salt_df["well_id"] == wid]
            data, timestamps = prepare_well_matrix(well_data)
            well_intervals = intervals[intervals["well_id"] == wid] if not intervals.empty else pd.DataFrame()
            rows, sc_rows, pr_rows, detail = run_blind_for_well(
                well_id=wid,
                data=data,
                timestamps=timestamps,
                well_intervals=well_intervals,
                device=device,
                cfg=cfg,
                verbose=verbose,
            )
            results.extend(rows)
            score_rows.extend(sc_rows)
            pred_rows.extend(pr_rows)

    res_df = pd.DataFrame(results)
    print("\n=== Salt Detection Results ===")
    if not res_df.empty:
        cols = [c for c in ["well_id", "interval_idx", "detected_time", "actual_start", "actual_end", "status"] if c in res_df.columns]
        print(res_df[cols].to_string())
        res_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    else:
        print("No interval-based results (likely blind mode without GT intervals).")
        pd.DataFrame(columns=[
            "well_id",
            "interval_idx",
            "type",
            "detected_time",
            "actual_type",
            "actual_start",
            "actual_end",
            "status",
            "detail",
        ]).to_csv(output_path, index=False)
        print(f"Empty results table saved to {output_path}")

    scores_output = Path("db/salt_paano_scores.csv")
    if score_rows:
        pd.DataFrame(score_rows).to_csv(scores_output, index=False)
        print(f"Per-point scores saved to {scores_output}")
    else:
        pd.DataFrame(
            columns=["well_id", "timestamp", "paano_score", "paano_score_short", "paano_score_long"]
        ).to_csv(scores_output, index=False)
        print(f"Per-point scores saved to {scores_output} (empty)")

    pred_output = Path("db/salt_paano_predicted_starts.csv")
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(pred_output, index=False)
        print(f"Predicted starts saved to {pred_output}")
    else:
        pd.DataFrame(columns=["well_id", "detected_time"]).to_csv(pred_output, index=False)
        print(f"Predicted starts saved to {pred_output} (empty)")

    return res_df, cfg


def detect_salt_paano_starts(well_data, intervals_df=None, well_id=None, mode="blind"):
    if well_data.empty:
        return [], "PaAno: Data not found"
    if well_id is None:
        well_id = str(well_data["well_id"].iloc[0]).strip().lower()

    cfg = load_or_default_config("db/salt_paano_config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, timestamps = prepare_well_matrix(well_data)

    if mode == "dev" and intervals_df is not None and not intervals_df.empty:
        well_intervals = intervals_df[intervals_df["well_id"] == well_id]
        if not well_intervals.empty:
            ref_mask = make_dev_train_mask(
                timestamps=timestamps,
                well_intervals=well_intervals,
                holdout_interval_idx=None,
                holdout_buffer_hours=0,
            )
            ms = compute_multiscale_scores(data=data, reference_mask=ref_mask, device=device, verbose=False)
            if ms is None:
                return [], "PaAno: Not enough data"
            w = float(cfg["fusion_weight_short"])
            fused = w * ms["short_z"] + (1.0 - w) * ms["long_z"]
            starts, _ = detect_starts_from_fused(
                fused_score=fused,
                timestamps=timestamps,
                reference_mask=ref_mask,
                cfg=cfg,
                blind_reference_end_idx=None,
            )
            return starts, "PaAno dev starts"

    ref_end_idx = choose_reference_end_index(
        timestamps=timestamps,
        patch_size=PATCH_SIZE_LONG,
        min_ratio=REFERENCE_MIN_RATIO,
        min_days=REFERENCE_MIN_DAYS,
    )
    ref_mask = make_prefix_mask(len(data), ref_end_idx)
    ms = compute_multiscale_scores(data=data, reference_mask=ref_mask, device=device, verbose=False)
    if ms is None:
        return [], "PaAno: Not enough data"

    w = float(cfg["fusion_weight_short"])
    fused = w * robust_scale_for_fusion(ms["short_score"], ref_end_idx) + (1.0 - w) * robust_scale_for_fusion(
        ms["long_score"], ref_end_idx
    )
    starts, _ = detect_starts_from_fused(
        fused_score=fused,
        timestamps=timestamps,
        reference_mask=ref_mask,
        cfg=cfg,
        blind_reference_end_idx=ref_end_idx,
    )
    return starts, "PaAno blind starts"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaAno Salt Anomaly Detection")
    parser.add_argument("--output", type=str, default="salt_paano_results.csv")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--mode", choices=["dev", "blind"], default="dev")
    parser.add_argument("--config", type=str, default="db/salt_paano_config.json")
    parser.add_argument("--auto-tune", type=int, choices=[0, 1], default=None,
                        help="Auto-tune dev config on LOIO runs (default: 0)")
    parser.add_argument("--well", type=str, default=None, help="Single well ID to process")
    parser.add_argument("--holdout-buffer-hours", type=float, default=None,
                        help="Override holdout buffer in dev mode")
    args = parser.parse_args()

    if args.well:
        df = load_salt_data(source_path=args.source)
        wid = args.well.strip().lower()
        well_data = df[df["well_id"] == wid]
        if well_data.empty:
            print(f"No data for well {wid}")
            sys.exit(1)

        intervals = load_salt_intervals(required=False)
        starts, detail = detect_salt_paano_starts(
            well_data=well_data,
            intervals_df=intervals,
            well_id=wid,
            mode=args.mode,
        )
        print(f"\nAnomaly starts: {starts}")
        print(f"Detail: {detail}")
        sys.exit(0)

    cfg = load_or_default_config(args.config)
    if args.holdout_buffer_hours is not None:
        cfg["holdout_buffer_hours"] = float(args.holdout_buffer_hours)
        save_config(args.config, cfg, metadata={"mode": args.mode, "manual_override": True})

    auto_tune = None if args.auto_tune is None else bool(args.auto_tune)
    run_salt_paano_detection(
        output_path=args.output,
        source_path=args.source,
        mode=args.mode,
        config_path=args.config,
        auto_tune=auto_tune,
        verbose=True,
    )
