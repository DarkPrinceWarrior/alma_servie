from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAANO_ROOT = PROJECT_ROOT / "paano"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PAANO_ROOT) not in sys.path:
    sys.path.insert(0, str(PAANO_ROOT))

import torch
from model import PatchEncoder
from train import train_model
from utils.data_preprocess import PatchCreator, preprocess_to_patches
from utils.evaluation import calculate_anomaly_scores, distribute_patch_scores_to_points
from utils.utils import create_memory_bank

from alma_service.anomaly_specs import DetectionSpec, get_detection_spec
from alma_service.onset_detection import (
    calibrate_causal_thresholds,
    detect_causal_onsets,
    robust_scale_for_fusion,
)
from alma_service.paano_defaults import (
    DEFAULT_CONFIG,
    LONG_PATCH,
    MIN_REFERENCE_COVERAGE,
    MIN_TOTAL_COVERAGE,
    PRESTART_TOLERANCE_HOURS,
    REFERENCE_MAX_RATIO,
    REFERENCE_MIN_DAYS,
    REFERENCE_MIN_RATIO,
    SHORT_PATCH,
    TUNE_GRID,
)
from alma_service.paths import DB_DIR, ensure_dir, ensure_parent
from alma_service.tabular_io import read_table
from alma_service.well_preprocess import WellMatrix, prepare_blind_well_matrix

warnings.filterwarnings("ignore")

SEED = 2027
NUM_ITERS = 200
BATCH_SIZE = 256
LR = 1e-4
TOP_K = 3
MEMORY_BANK_RATIO = 0.1
ENABLE_TORCH_COMPILE = True


@dataclass
class BlindScoreRun:
    well_id: str
    split: str
    timestamps: np.ndarray
    short_score: np.ndarray
    long_score: np.ndarray
    reference_end_idx: int
    feature_columns: list[str]
    detail: dict[str, Any]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _maybe_compile_module(module: torch.nn.Module) -> torch.nn.Module:
    if not ENABLE_TORCH_COMPILE or not hasattr(torch, "compile"):
        return module
    try:
        return torch.compile(module, mode="reduce-overhead")
    except Exception:  # pragma: no cover - runtime fallback
        return module


def load_anomaly_data(spec: DetectionSpec, source_path: str | None = None) -> pd.DataFrame:
    if source_path is not None:
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")
    else:
        candidates = [DB_DIR / name for name in spec.dataset.source_candidates]
        src = next((path for path in candidates if path.exists()), None)
        if src is None:
            raise FileNotFoundError(f"No source dataset found for {spec.anomaly_key}")

    print(f"Loading {spec.anomaly_key} data from: {src}")
    df = read_table(
        src,
        dtypes={"well_id": str},
        parse_dates=["timestamp"],
        low_memory=False,
    )
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    return df.dropna(subset=["timestamp"]).sort_values(["well_id", "timestamp"]).reset_index(drop=True)


def load_intervals(spec: DetectionSpec, required: bool = True) -> pd.DataFrame:
    src = spec.dataset.intervals_path
    if not src.exists():
        if required:
            raise FileNotFoundError(f"Intervals file not found: {src}")
        return pd.DataFrame(
            columns=["well_id", "start_date", "end_date", "data_start", "data_end", "split", "interval_idx"]
        )

    df = pd.read_csv(src, dtype={"well_id": str})
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["data_start"] = pd.to_datetime(df.get("data_start"), errors="coerce")
    df["data_end"] = pd.to_datetime(df.get("data_end"), errors="coerce")
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.strip().str.lower()
    else:
        df["split"] = "train"
    if "interval_idx" not in df.columns:
        df["interval_idx"] = df.groupby("well_id").cumcount() + 1
    df["interval_idx"] = pd.to_numeric(df["interval_idx"], errors="coerce").fillna(1).astype(int)
    return df.dropna(subset=["well_id", "start_date", "end_date"]).sort_values(
        ["well_id", "start_date", "interval_idx"]
    )


def run_paano_single_scale(data: np.ndarray, train_data: np.ndarray, patch_size: int, device, verbose: bool) -> np.ndarray:
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
    model = _maybe_compile_module(model)

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
    elapsed = time.time() - t0

    memory_bank, _ = create_memory_bank(model, train_loader, device, num_cores=MEMORY_BANK_RATIO)
    patch_scores = calculate_anomaly_scores(model, full_loader, memory_bank, top_k=TOP_K, device=device)
    point_scores = distribute_patch_scores_to_points(patch_scores, patch_size=patch_size, num_points=len(data))

    if verbose:
        print(f"    scale patch={patch_size}: train_pts={len(train_data)}, train_time={elapsed:.1f}s")
    return point_scores


def build_blind_score_run(
    well_id: str,
    well_df: pd.DataFrame,
    split: str,
    device,
    verbose: bool,
) -> BlindScoreRun | None:
    matrix = prepare_blind_well_matrix(
        well_df=well_df,
        patch_size=LONG_PATCH,
        reference_min_ratio=REFERENCE_MIN_RATIO,
        reference_max_ratio=REFERENCE_MAX_RATIO,
        reference_min_days=REFERENCE_MIN_DAYS,
        min_reference_coverage=MIN_REFERENCE_COVERAGE,
        min_total_coverage=MIN_TOTAL_COVERAGE,
    )
    if matrix is None:
        return None

    train_data = matrix.data[: matrix.reference_end_idx]
    if len(train_data) < LONG_PATCH * 2:
        return None

    if verbose:
        print(
            f"  PaAno: {well_id} | points={len(matrix.data)}, channels={matrix.data.shape[1]}, "
            f"reference_end_idx={matrix.reference_end_idx}"
        )

    short_score = run_paano_single_scale(
        data=matrix.data,
        train_data=train_data,
        patch_size=SHORT_PATCH,
        device=device,
        verbose=verbose,
    )
    long_score = run_paano_single_scale(
        data=matrix.data,
        train_data=train_data,
        patch_size=LONG_PATCH,
        device=device,
        verbose=verbose,
    )

    return BlindScoreRun(
        well_id=well_id,
        split=split,
        timestamps=matrix.timestamps,
        short_score=short_score,
        long_score=long_score,
        reference_end_idx=matrix.reference_end_idx,
        feature_columns=matrix.feature_columns,
        detail={
            "points": len(matrix.data),
            "channels": len(matrix.feature_columns),
            "reference_end_idx": matrix.reference_end_idx,
            "trim_start_idx": matrix.trim_start_idx,
        },
    )


def detect_starts_from_run(run: BlindScoreRun, cfg: dict[str, Any]) -> tuple[np.ndarray, Any, np.ndarray]:
    weight = float(cfg["fusion_weight_short"])
    short_z = robust_scale_for_fusion(run.short_score, run.reference_end_idx)
    long_z = robust_scale_for_fusion(run.long_score, run.reference_end_idx)
    fused = weight * short_z + (1.0 - weight) * long_z

    thresholds, diagnostics = calibrate_causal_thresholds(
        scores=fused,
        timestamps=run.timestamps,
        reference_end_idx=run.reference_end_idx,
        target_far_per_day=float(cfg["target_far_per_day"]),
        min_run_points=int(cfg["min_run_points"]),
        ema_alpha=float(cfg["ema_alpha"]),
    )
    starts = detect_causal_onsets(
        scores=fused,
        timestamps=run.timestamps,
        diagnostics=diagnostics,
        thresholds=thresholds,
        reference_end_idx=run.reference_end_idx,
        min_run_points=int(cfg["min_run_points"]),
        cooldown_hours=float(cfg["cooldown_hours"]),
        gate_mode=str(cfg["gate_mode"]),
    )
    return fused, thresholds, np.array(starts, dtype="datetime64[ns]")


def select_interval_detection(
    predicted_starts: list[pd.Timestamp],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.Timestamp | pd.NaT:
    pre_tol = pd.Timedelta(hours=PRESTART_TOLERANCE_HOURS)
    inside = [ts for ts in predicted_starts if start_dt - pre_tol <= ts <= end_dt]
    return min(inside) if inside else pd.NaT


def summarize_predictions(
    intervals: pd.DataFrame,
    predicted_starts: dict[str, list[pd.Timestamp]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    false_alarms = 0
    observed_days = 0.0

    for well_id, well_intervals in intervals.groupby("well_id"):
        starts = sorted(predicted_starts.get(well_id, []))
        obs_start = well_intervals["data_start"].dropna().min()
        obs_end = well_intervals["data_end"].dropna().max()
        if pd.notna(obs_start) and pd.notna(obs_end) and obs_end > obs_start:
            observed_days += (obs_end - obs_start).total_seconds() / 86400.0

        for ts in starts:
            if not ((well_intervals["start_date"] <= ts) & (ts <= well_intervals["end_date"])).any():
                false_alarms += 1

        for _, row in well_intervals.sort_values(["start_date", "interval_idx"]).iterrows():
            det = select_interval_detection(starts, row["start_date"], row["end_date"])
            delay_h = (
                (det - row["start_date"]).total_seconds() / 3600.0 if pd.notna(det) else np.nan
            )
            rows.append(
                {
                    "well_id": well_id,
                    "interval_idx": int(row["interval_idx"]),
                    "split": row.get("split", "train"),
                    "actual_start": row["start_date"],
                    "actual_end": row["end_date"],
                    "detected_time": det,
                    "status": "Detected" if pd.notna(det) else "Not found",
                    "delay_hours": delay_h,
                    "interval_hours": (row["end_date"] - row["start_date"]).total_seconds() / 3600.0,
                    "type": row.get("type", ""),
                }
            )

    result_df = pd.DataFrame(rows)
    hit_rate = float((result_df["status"] == "Detected").mean()) if not result_df.empty else 0.0
    detected_delays = result_df.loc[result_df["status"] == "Detected", "delay_hours"].dropna()
    summary = {
        "interval_count": int(len(result_df)),
        "hit_count": int((result_df["status"] == "Detected").sum()) if not result_df.empty else 0,
        "hit_rate": hit_rate,
        "median_abs_delay_hours": float(np.median(np.abs(detected_delays))) if len(detected_delays) else np.inf,
        "p90_abs_delay_hours": float(np.quantile(np.abs(detected_delays), 0.90)) if len(detected_delays) else np.inf,
        "false_alarms": int(false_alarms),
        "false_alarms_per_day": float(false_alarms / observed_days) if observed_days > 0 else np.inf,
        "avg_starts_per_interval": float(sum(len(v) for v in predicted_starts.values()) / max(len(rows), 1)),
    }
    return rows, summary


def tune_on_train_runs(train_runs: list[BlindScoreRun], train_intervals: pd.DataFrame, verbose: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    if not train_runs:
        return DEFAULT_CONFIG.copy(), {"message": "No train runs available"}

    def candidate_from_trial(trial: Any) -> dict[str, Any]:
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(
            {
                "fusion_weight_short": float(
                    trial.suggest_categorical("fusion_weight_short", TUNE_GRID["fusion_weight_short"])
                ),
                "target_far_per_day": float(
                    trial.suggest_categorical("target_far_per_day", TUNE_GRID["target_far_per_day"])
                ),
                "min_run_points": int(
                    trial.suggest_categorical("min_run_points", TUNE_GRID["min_run_points"])
                ),
                "cooldown_hours": float(
                    trial.suggest_categorical("cooldown_hours", TUNE_GRID["cooldown_hours"])
                ),
                "ema_alpha": float(trial.suggest_categorical("ema_alpha", TUNE_GRID["ema_alpha"])),
                "gate_mode": str(trial.suggest_categorical("gate_mode", TUNE_GRID["gate_mode"])),
            }
        )
        return cfg

    def score_key(summary: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            summary["hit_count"],
            -summary["false_alarms_per_day"],
            -summary["p90_abs_delay_hours"],
            -summary["median_abs_delay_hours"],
            -summary["avg_starts_per_interval"],
        )

    def objective(trial: Any) -> float:
        cfg = candidate_from_trial(trial)
        predicted: dict[str, list[pd.Timestamp]] = {}
        for run in train_runs:
            _, _, starts = detect_starts_from_run(run, cfg)
            predicted[run.well_id] = [pd.Timestamp(x) for x in starts]
        _, summary = summarize_predictions(train_intervals, predicted)
        trial.set_user_attr("config", cfg)
        trial.set_user_attr("summary", summary)
        trial.set_user_attr("score_key", list(score_key(summary)))
        return (
            summary["hit_count"] * 1_000_000.0
            - float(summary["false_alarms_per_day"]) * 1_000.0
            - float(summary["avg_starts_per_interval"]) * 10.0
            - float(summary["p90_abs_delay_hours"])
        )

    leaderboard: list[dict[str, Any]] = []
    if optuna is not None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=36, show_progress_bar=False)
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            cfg = trial.user_attrs.get("config")
            summary = trial.user_attrs.get("summary")
            key = trial.user_attrs.get("score_key")
            if not cfg or not summary or key is None:
                continue
            leaderboard.append({"score_key": tuple(key), **cfg, **summary})
    else:
        candidates: list[dict[str, Any]] = []
        for weight in TUNE_GRID["fusion_weight_short"]:
            for far in TUNE_GRID["target_far_per_day"]:
                for run_points in TUNE_GRID["min_run_points"]:
                    for cooldown in TUNE_GRID["cooldown_hours"]:
                        for ema_alpha in TUNE_GRID["ema_alpha"]:
                            for gate_mode in TUNE_GRID["gate_mode"]:
                                cfg = DEFAULT_CONFIG.copy()
                                cfg.update(
                                    {
                                        "fusion_weight_short": float(weight),
                                        "target_far_per_day": float(far),
                                        "min_run_points": int(run_points),
                                        "cooldown_hours": float(cooldown),
                                        "ema_alpha": float(ema_alpha),
                                        "gate_mode": str(gate_mode),
                                    }
                                )
                                candidates.append(cfg)
        for cfg in candidates:
            predicted: dict[str, list[pd.Timestamp]] = {}
            for run in train_runs:
                _, _, starts = detect_starts_from_run(run, cfg)
                predicted[run.well_id] = [pd.Timestamp(x) for x in starts]
            _, summary = summarize_predictions(train_intervals, predicted)
            leaderboard.append({"score_key": score_key(summary), **cfg, **summary})

    leaderboard = sorted(leaderboard, key=lambda row: row["score_key"], reverse=True)
    best_cfg = DEFAULT_CONFIG.copy()
    best_key = leaderboard[0]["score_key"] if leaderboard else None
    if leaderboard:
        best_cfg.update(
            {
                "fusion_weight_short": float(leaderboard[0]["fusion_weight_short"]),
                "target_far_per_day": float(leaderboard[0]["target_far_per_day"]),
                "min_run_points": int(leaderboard[0]["min_run_points"]),
                "cooldown_hours": float(leaderboard[0]["cooldown_hours"]),
                "ema_alpha": float(leaderboard[0]["ema_alpha"]),
                "gate_mode": str(leaderboard[0]["gate_mode"]),
            }
        )
    summary = {
        "best_score_key": list(best_key) if best_key is not None else None,
        "top10": leaderboard[:10],
    }
    if verbose:
        print("  Auto-tune top configs:")
        for idx, row in enumerate(summary["top10"][:5], 1):
            print(
                f"    {idx}. hit={row['hit_count']}/{row['interval_count']}, "
                f"FAR/day={row['false_alarms_per_day']:.3f}, "
                f"p90_abs_delay={row['p90_abs_delay_hours']:.2f}h, "
                f"w={row['fusion_weight_short']:.2f}, far={row['target_far_per_day']:.2f}, "
                f"run={row['min_run_points']}, cd={row['cooldown_hours']:.0f}, "
                f"ema={row['ema_alpha']:.2f}, gate={row['gate_mode']}"
            )
    return best_cfg, summary


def load_or_build_config(spec: DetectionSpec, train_runs: list[BlindScoreRun], train_intervals: pd.DataFrame, retune: bool, verbose: bool) -> dict[str, Any]:
    if spec.config_path.exists() and not retune:
        payload = json.loads(spec.config_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "config" in payload:
            return payload["config"]
        if isinstance(payload, dict):
            return payload

    cfg, tuning_summary = tune_on_train_runs(train_runs, train_intervals, verbose=verbose)
    spec.config_path.write_text(
        json.dumps({"config": cfg}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    spec.tuning_path.write_text(
        json.dumps(tuning_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved tuned config to {spec.config_path}")
    print(f"Saved tuning summary to {spec.tuning_path}")
    return cfg


def format_detail(run: BlindScoreRun, cfg: dict[str, Any], thresholds: Any, starts: list[pd.Timestamp]) -> str:
    return (
        f"PaAno blind(s={SHORT_PATCH},l={LONG_PATCH},w={cfg['fusion_weight_short']:.2f},"
        f"far={cfg['target_far_per_day']},run={cfg['min_run_points']},"
        f"cd={cfg['cooldown_hours']},ema={cfg['ema_alpha']},gate={cfg['gate_mode']},"
        f"ref={run.reference_end_idx},ch={len(run.feature_columns)},"
        f"thr_score={thresholds.score_threshold:.3f},thr_ema={thresholds.ema_z_threshold:.3f},"
        f"thr_cusum={thresholds.cusum_threshold:.3f},starts={len(starts)})"
    )


def run_detection(
    anomaly_key: str,
    output_path: str | None = None,
    source_path: str | None = None,
    retune: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    spec = get_detection_spec(anomaly_key)
    print(f"=== PaAno {spec.display_name} Detection (blind unified) ===")
    set_seed()
    ensure_dir(DB_DIR)
    output = ensure_parent(Path(output_path) if output_path else spec.results_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_anomaly_data(spec, source_path=source_path)
    intervals = load_intervals(spec, required=True)
    if df.empty or intervals.empty:
        raise RuntimeError("Empty data or intervals for detection.")

    split_map = (
        intervals[["well_id", "split"]]
        .drop_duplicates("well_id")
        .set_index("well_id")["split"]
        .to_dict()
    )

    all_runs: dict[str, BlindScoreRun | None] = {}
    for well_id in sorted(df["well_id"].unique()):
        well_df = df[df["well_id"] == well_id]
        run = build_blind_score_run(
            well_id=well_id,
            well_df=well_df,
            split=split_map.get(well_id, "train"),
            device=device,
            verbose=verbose,
        )
        all_runs[well_id] = run

    train_runs = [run for run in all_runs.values() if run is not None and run.split == "train"]
    train_intervals = intervals[intervals["split"].astype(str).str.lower() == "train"].copy()
    cfg = load_or_build_config(spec, train_runs, train_intervals, retune=retune, verbose=verbose)

    results: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for well_id in sorted(df["well_id"].unique()):
        well_intervals = intervals[intervals["well_id"] == well_id]
        split = split_map.get(well_id, "train")
        run = all_runs.get(well_id)
        if run is None:
            for _, row in well_intervals.iterrows():
                results.append(
                    {
                        "well_id": well_id,
                        "interval_idx": int(row["interval_idx"]),
                        "split": split,
                        "detected_time": None,
                        "actual_start": row["start_date"],
                        "actual_end": row["end_date"],
                        "status": "Not found",
                        "delay_hours": np.nan,
                        "n_predicted_starts": 0,
                        "n_channels": 0,
                        "detail": "Not enough usable data for PaAno",
                    }
                )
            continue

        fused, thresholds, starts = detect_starts_from_run(run, cfg)
        starts_list = [pd.Timestamp(x) for x in starts]
        detail = format_detail(run, cfg, thresholds, starts_list)
        if verbose:
            print(f"  PaAno done: {detail}")

        for ts, fused_score, short_score, long_score in zip(run.timestamps, fused, run.short_score, run.long_score):
            score_rows.append(
                {
                    "well_id": well_id,
                    "timestamp": ts,
                    "split": split,
                    "paano_score": float(fused_score),
                    "paano_score_short": float(short_score),
                    "paano_score_long": float(long_score),
                }
            )

        for ts in starts_list:
            pred_rows.append({"well_id": well_id, "split": split, "detected_time": ts})

        for _, row in well_intervals.sort_values(["start_date", "interval_idx"]).iterrows():
            det = select_interval_detection(starts_list, row["start_date"], row["end_date"])
            results.append(
                {
                    "well_id": well_id,
                    "interval_idx": int(row["interval_idx"]),
                    "split": split,
                    "detected_time": det,
                    "actual_start": row["start_date"],
                    "actual_end": row["end_date"],
                    "status": "Detected" if pd.notna(det) else "Not found",
                    "delay_hours": (det - row["start_date"]).total_seconds() / 3600.0 if pd.notna(det) else np.nan,
                    "n_predicted_starts": len(starts_list),
                    "n_channels": len(run.feature_columns),
                    "detail": detail,
                }
            )

    res_df = pd.DataFrame(results)
    print(f"\n=== {spec.display_name} Detection Results ===")
    if not res_df.empty:
        print(
            res_df[["well_id", "interval_idx", "split", "detected_time", "actual_start", "actual_end", "status"]]
            .to_string(index=False)
        )
    res_df.to_csv(output, index=False)
    print(f"\nResults saved to {output}")

    pd.DataFrame(score_rows).to_csv(spec.scores_path, index=False)
    print(f"Per-point scores saved to {spec.scores_path}")

    pd.DataFrame(pred_rows).to_csv(spec.predicted_starts_path, index=False)
    print(f"Predicted starts saved to {spec.predicted_starts_path}")

    pred_df = pd.DataFrame(pred_rows)
    split_summaries: dict[str, Any] = {}
    for split_name in ["train", "test", "all"]:
        if split_name == "all":
            subset_intervals = intervals
        else:
            subset_intervals = intervals[intervals["split"].astype(str).str.lower() == split_name]
        subset_pred: dict[str, list[pd.Timestamp]] = {}
        for wid in set(subset_intervals["well_id"]):
            if pred_df.empty:
                subset_pred[wid] = []
                continue
            matched = pred_df[pred_df["well_id"] == wid]
            subset_pred[wid] = [pd.Timestamp(ts) for ts in matched["detected_time"].tolist()]
        _, summary = summarize_predictions(subset_intervals, subset_pred)
        split_summaries[split_name] = summary

    summary_path = spec.results_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps({"config": cfg, "splits": split_summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Summary saved to {summary_path}")
    return res_df


def run_single_well(anomaly_key: str, well_id: str, source_path: str | None, retune: bool) -> None:
    spec = get_detection_spec(anomaly_key)
    df = load_anomaly_data(spec, source_path=source_path)
    intervals = load_intervals(spec, required=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    well_id = well_id.strip().lower()
    well_df = df[df["well_id"] == well_id]
    if well_df.empty:
        raise ValueError(f"No data for well {well_id}")

    split = "train"
    if not intervals.empty:
        well_intervals = intervals[intervals["well_id"] == well_id]
        if not well_intervals.empty and "split" in well_intervals.columns:
            split = str(well_intervals["split"].iloc[0]).strip().lower()

    run = build_blind_score_run(well_id=well_id, well_df=well_df, split=split, device=device, verbose=True)
    if run is None:
        print("No usable data for PaAno")
        return

    if spec.config_path.exists() and not retune:
        payload = json.loads(spec.config_path.read_text(encoding="utf-8"))
        cfg = payload["config"] if isinstance(payload, dict) and "config" in payload else payload
    else:
        cfg = DEFAULT_CONFIG.copy()
    _, thresholds, starts = detect_starts_from_run(run, cfg)
    starts_list = [pd.Timestamp(x) for x in starts]
    print(f"\nAnomaly starts: {starts_list}")
    print(format_detail(run, cfg, thresholds, starts_list))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified PaAno detection pipeline.")
    parser.add_argument("anomaly", choices=["negermet", "pritok", "salt"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--well", type=str, default=None)
    parser.add_argument("--retune", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.well:
        run_single_well(args.anomaly, args.well, args.source, args.retune)
        return
    run_detection(
        anomaly_key=args.anomaly,
        output_path=args.output,
        source_path=args.source,
        retune=args.retune,
        verbose=True,
    )


if __name__ == "__main__":
    main()
