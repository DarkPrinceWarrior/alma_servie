from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
try:
    import optuna
except ImportError:  # pragma: no cover - optional fallback for older envs
    optuna = None

from alma_service.anomaly_specs import DetectionSpec, get_detection_spec
from alma_service.benchmark_metrics import (
    evaluate_predictions,
    load_intervals as load_intervals_df,
    predicted_from_mapping,
    select_interval_detection,
    summarize_splits,
)
from alma_service.detection_artifacts import (
    DEFAULT_DETECTOR,
    DETECTOR_KEYS,
    LOCAL_DETECTOR_KEYS,
    benchmark_summary_path,
    config_path,
    legacy_summary_path,
    load_json,
    normalize_detector_key,
    predicted_starts_path,
    report_path,
    results_path,
    scores_path,
    summary_path,
    tuning_path,
)
from alma_service.engineered_features import PreparedWellData, prepare_engineered_well
from alma_service.generic_detectors import (
    DetectorScoreOutput,
    FusedDetector,
    IsolationForestDetector,
    LOFDetector,
    PCASPEDetector,
    PaAnoFeatureDetector,
    set_seed,
)
from alma_service.onset_detection import (
    calibrate_causal_thresholds_from_reference_mask,
    detect_causal_onsets_masked,
)
from alma_service.paano_defaults import (
    LONG_PATCH,
    MIN_REFERENCE_COVERAGE,
    MIN_TOTAL_COVERAGE,
    PRESTART_TOLERANCE_HOURS,
    REFERENCE_MAX_RATIO,
    REFERENCE_MIN_DAYS,
    REFERENCE_MIN_RATIO,
)
from alma_service.paths import DB_DIR, ensure_dir, ensure_parent
from alma_service.tabular_io import read_table, write_table

BASE_ONSET_CONFIG = {
    "target_far_per_day": 0.50,
    "min_run_points": 3,
    "cooldown_hours": 12.0,
    "rearm_window_minutes": 60.0,
    "ema_alpha": 0.08,
    "gate_mode": "score_ema",
    "hysteresis_scale": 0.60,
}

BASE_ONSET_TUNE_GRID = {
    "target_far_per_day": [0.10, 0.25, 0.50],
    "min_run_points": [3, 4, 6],
    "cooldown_hours": [8.0, 12.0, 24.0],
    "rearm_window_minutes": [30.0, 60.0, 120.0],
    "ema_alpha": [0.04, 0.08, 0.12],
    "gate_mode": ["score_ema", "relaxed", "strict"],
}

ANOMALY_ONSET_PROFILES = {
    "negermet": {},
    "pritok": {
        "grid": {
            "min_run_points": [2, 3, 4],
            "cooldown_hours": [8.0, 12.0, 24.0],
            "rearm_window_minutes": [30.0, 60.0, 120.0],
        },
    },
    "salt": {},
}

PAANO_WEIGHT_GRID = [0.40, 0.60, 0.75]
ANOMALY_RUNTIME_CONFIG = {
    "negermet": {
        "prepare_patch_size": 64,
        "paano_patch_short": 32,
        "paano_patch_long": 64,
        "max_far_per_day": 0.25,
        "max_starts_per_interval": 2.0,
        "max_p90_delay_ratio": 0.25,
    },
    "pritok": {
        "prepare_patch_size": 96,
        "paano_patch_short": 48,
        "paano_patch_long": 96,
        "max_far_per_day": 0.25,
        "max_starts_per_interval": 6.0,
        "max_p90_delay_ratio": 0.40,
    },
    "salt": {
        "prepare_patch_size": 96,
        "paano_patch_short": 48,
        "paano_patch_long": 96,
        "max_far_per_day": 0.40,
        "max_starts_per_interval": 10.0,
        "max_p90_delay_ratio": 0.20,
    },
}
LOCAL_DEFAULT_PRIORITY = {
    "fused": 5,
    "pca_spe": 4,
    "paano_feat": 3,
    "lof": 2,
    "iforest": 1,
}


@dataclass
class PreparedDetectorRun:
    prepared: PreparedWellData
    score_output: DetectorScoreOutput


def _runtime_config(anomaly_key: str) -> dict[str, float | int]:
    return ANOMALY_RUNTIME_CONFIG.get(anomaly_key, ANOMALY_RUNTIME_CONFIG["salt"])


def _default_onset_config(anomaly_key: str, detector_key: str) -> dict[str, Any]:
    cfg = BASE_ONSET_CONFIG.copy()
    profile = ANOMALY_ONSET_PROFILES.get(anomaly_key, {})
    cfg.update(profile.get("defaults", {}))
    if detector_key == "paano_feat":
        cfg["fusion_weight_short"] = 0.60
    return cfg


def _onset_tune_grid(anomaly_key: str) -> dict[str, list[Any]]:
    grid = {key: list(values) for key, values in BASE_ONSET_TUNE_GRID.items()}
    profile = ANOMALY_ONSET_PROFILES.get(anomaly_key, {})
    for key, values in profile.get("grid", {}).items():
        grid[key] = list(values)
    return grid


def _safe_metric(value: Any, large: float = 1e9) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return large
    if not np.isfinite(numeric):
        return large
    return numeric


def _operational_score_key(anomaly_key: str, detector_key: str, summary: dict[str, Any]) -> tuple[float, ...]:
    runtime_cfg = _runtime_config(anomaly_key)
    max_far = float(runtime_cfg["max_far_per_day"])
    max_starts = float(runtime_cfg["max_starts_per_interval"])
    max_delay_ratio = float(runtime_cfg["max_p90_delay_ratio"])
    far = _safe_metric(summary.get("false_alarms_per_day"), large=1e9)
    starts = _safe_metric(summary.get("avg_starts_per_interval"), large=1e9)
    p90_ratio = _safe_metric(summary.get("p90_delay_ratio"), large=1e9)
    p90_abs_delay = _safe_metric(summary.get("p90_abs_delay_hours"), large=1e9)
    feasible_far = int(far <= max_far)
    feasible_starts = int(starts <= max_starts)
    feasible_delay = int(p90_ratio <= max_delay_ratio)
    far_over = max(far - max_far, 0.0)
    starts_over = max(starts - max_starts, 0.0)
    delay_over = max(p90_ratio - max_delay_ratio, 0.0)
    priority = LOCAL_DEFAULT_PRIORITY.get(detector_key, 0)
    return (
        float(summary.get("hit_count", 0)),
        float(feasible_delay),
        float(feasible_starts),
        float(feasible_far),
        -delay_over,
        -far_over,
        -starts_over,
        -p90_ratio,
        -starts,
        -p90_abs_delay,
        -far,
        float(priority),
    )


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
    return load_intervals_df(src)


def _build_detector(anomaly_key: str, detector_key: str, device: torch.device, verbose: bool = False):
    runtime_cfg = _runtime_config(anomaly_key)
    if detector_key == "paano_feat":
        return PaAnoFeatureDetector(
            device=device,
            patch_short=int(runtime_cfg["paano_patch_short"]),
            patch_long=int(runtime_cfg["paano_patch_long"]),
            verbose=verbose,
        )
    if detector_key == "pca_spe":
        return PCASPEDetector()
    if detector_key == "lof":
        return LOFDetector()
    if detector_key == "iforest":
        return IsolationForestDetector()
    if detector_key == "fused":
        return FusedDetector(
            device=device,
            verbose=verbose,
            patch_short=int(runtime_cfg["paano_patch_short"]),
            patch_long=int(runtime_cfg["paano_patch_long"]),
        )
    raise ValueError(f"Unsupported local detector: {detector_key}")


def _prepare_all_wells(
    spec: DetectionSpec,
    df: pd.DataFrame,
    intervals: pd.DataFrame,
    verbose: bool,
) -> dict[str, PreparedWellData]:
    runtime_cfg = _runtime_config(spec.anomaly_key)
    split_map = (
        intervals[["well_id", "split"]]
        .drop_duplicates("well_id")
        .set_index("well_id")["split"]
        .to_dict()
    )
    prepared_runs: dict[str, PreparedWellData] = {}
    for well_id in sorted(df["well_id"].unique()):
        well_df = df[df["well_id"] == well_id].copy()
        prepared = prepare_engineered_well(
            anomaly_key=spec.anomaly_key,
            well_id=well_id,
            split=split_map.get(well_id, "train"),
            well_df=well_df,
            patch_size=int(runtime_cfg["prepare_patch_size"]),
            reference_min_ratio=REFERENCE_MIN_RATIO,
            reference_max_ratio=REFERENCE_MAX_RATIO,
            reference_min_days=REFERENCE_MIN_DAYS,
            min_reference_coverage=MIN_REFERENCE_COVERAGE,
            min_total_coverage=MIN_TOTAL_COVERAGE,
        )
        if prepared is None:
            if verbose:
                print(f"  Skip {well_id}: not enough usable data after preprocessing")
            continue
        prepared_runs[well_id] = prepared
        if verbose:
            print(
                f"  Prepared {well_id}: split={prepared.split}, points={prepared.detail['points']}, "
                f"raw={prepared.detail['raw_channels']}, features={prepared.detail['feature_count']}, "
                f"ref={prepared.detail['reference_points']}, masked={prepared.detail['masked_fraction']:.1%}"
            )
    return prepared_runs


def _build_local_runs(
    anomaly_key: str,
    detector_key: str,
    prepared_runs: dict[str, PreparedWellData],
    device: torch.device,
    verbose: bool,
) -> dict[str, PreparedDetectorRun]:
    out: dict[str, PreparedDetectorRun] = {}
    for well_id, prepared in prepared_runs.items():
        detector = _build_detector(anomaly_key, detector_key, device=device, verbose=verbose)
        X_ref = prepared.feature_matrix[prepared.reference_mask]
        detector.fit_reference(X_ref, mask_ref=prepared.reference_mask)
        score_output = detector.score_stream(prepared.feature_matrix, mask_all=prepared.stability_mask)
        out[well_id] = PreparedDetectorRun(prepared=prepared, score_output=score_output)
    return out


def _score_for_config(run: PreparedDetectorRun, detector_key: str, cfg: dict[str, Any]) -> np.ndarray:
    if detector_key != "paano_feat":
        return run.score_output.primary.astype(np.float32)

    short_score = run.score_output.components.get("paano_short")
    long_score = run.score_output.components.get("paano_long")
    if short_score is None or long_score is None:
        return run.score_output.primary.astype(np.float32)
    weight = float(cfg.get("fusion_weight_short", 0.60))
    return (weight * short_score + (1.0 - weight) * long_score).astype(np.float32)


def _detect_starts_for_run(
    detector_key: str,
    run: PreparedDetectorRun,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, Any, list[pd.Timestamp]]:
    cfg = {**BASE_ONSET_CONFIG, **cfg}
    score = _score_for_config(run, detector_key, cfg)
    thresholds, diagnostics = calibrate_causal_thresholds_from_reference_mask(
        scores=score,
        timestamps=run.prepared.timestamps,
        reference_mask=run.prepared.reference_mask,
        target_far_per_day=float(cfg["target_far_per_day"]),
        min_run_points=int(cfg["min_run_points"]),
        ema_alpha=float(cfg["ema_alpha"]),
    )
    starts = detect_causal_onsets_masked(
        scores=score,
        timestamps=run.prepared.timestamps,
        diagnostics=diagnostics,
        thresholds=thresholds,
        reference_mask=run.prepared.reference_mask,
        onset_mask=run.prepared.onset_allowed_mask,
        min_run_points=int(cfg["min_run_points"]),
        cooldown_hours=float(cfg["cooldown_hours"]),
        rearm_window_minutes=float(cfg["rearm_window_minutes"]),
        gate_mode=str(cfg["gate_mode"]),
        hysteresis_scale=float(cfg["hysteresis_scale"]),
    )
    return score, thresholds, starts


def _candidate_configs(anomaly_key: str, detector_key: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    weight_grid = PAANO_WEIGHT_GRID if detector_key == "paano_feat" else [None]
    grid = _onset_tune_grid(anomaly_key)
    default_cfg = _default_onset_config(anomaly_key, detector_key)
    for target_far_per_day in grid["target_far_per_day"]:
        for min_run_points in grid["min_run_points"]:
            for cooldown_hours in grid["cooldown_hours"]:
                for rearm_window_minutes in grid["rearm_window_minutes"]:
                    for ema_alpha in grid["ema_alpha"]:
                        for gate_mode in grid["gate_mode"]:
                            for fusion_weight_short in weight_grid:
                                cfg = default_cfg.copy()
                                cfg.update(
                                    {
                                        "target_far_per_day": float(target_far_per_day),
                                        "min_run_points": int(min_run_points),
                                        "cooldown_hours": float(cooldown_hours),
                                        "rearm_window_minutes": float(rearm_window_minutes),
                                        "ema_alpha": float(ema_alpha),
                                        "gate_mode": str(gate_mode),
                                    }
                                )
                                if fusion_weight_short is not None:
                                    cfg["fusion_weight_short"] = float(fusion_weight_short)
                                candidates.append(cfg)
    return candidates


def _optuna_objective_value(anomaly_key: str, detector_key: str, summary: dict[str, Any]) -> float:
    runtime_cfg = _runtime_config(anomaly_key)
    max_far = float(runtime_cfg["max_far_per_day"])
    max_starts = float(runtime_cfg["max_starts_per_interval"])
    max_delay_ratio = float(runtime_cfg["max_p90_delay_ratio"])
    far = _safe_metric(summary.get("false_alarms_per_day"), large=1e6)
    starts = _safe_metric(summary.get("avg_starts_per_interval"), large=1e6)
    p90_ratio = _safe_metric(summary.get("p90_delay_ratio"), large=1e6)
    p90_abs_delay = _safe_metric(summary.get("p90_abs_delay_hours"), large=1e6)
    hits = float(summary.get("hit_count", 0))
    feasible_far = 1.0 if far <= max_far else 0.0
    feasible_starts = 1.0 if starts <= max_starts else 0.0
    feasible_delay = 1.0 if p90_ratio <= max_delay_ratio else 0.0
    far_over = max(far - max_far, 0.0)
    starts_over = max(starts - max_starts, 0.0)
    delay_over = max(p90_ratio - max_delay_ratio, 0.0)
    priority = float(LOCAL_DEFAULT_PRIORITY.get(detector_key, 0))
    return (
        hits * 1_000_000_000.0
        + feasible_delay * 10_000_000.0
        + feasible_starts * 1_000_000.0
        + feasible_far * 100_000.0
        - delay_over * 1_000_000.0
        - starts_over * 100_000.0
        - far_over * 10_000.0
        - p90_ratio * 10_000.0
        - starts * 100.0
        - p90_abs_delay
        - far * 100.0
        + priority * 1e-3
    )


def _suggest_optuna_config(trial: Any, anomaly_key: str, detector_key: str) -> dict[str, Any]:
    grid = _onset_tune_grid(anomaly_key)
    cfg = _default_onset_config(anomaly_key, detector_key)
    cfg.update(
        {
            "target_far_per_day": float(
                trial.suggest_categorical("target_far_per_day", grid["target_far_per_day"])
            ),
            "min_run_points": int(
                trial.suggest_categorical("min_run_points", grid["min_run_points"])
            ),
            "cooldown_hours": float(
                trial.suggest_categorical("cooldown_hours", grid["cooldown_hours"])
            ),
            "rearm_window_minutes": float(
                trial.suggest_categorical("rearm_window_minutes", grid["rearm_window_minutes"])
            ),
            "ema_alpha": float(trial.suggest_categorical("ema_alpha", grid["ema_alpha"])),
            "gate_mode": str(trial.suggest_categorical("gate_mode", grid["gate_mode"])),
        }
    )
    if detector_key == "paano_feat":
        cfg["fusion_weight_short"] = float(
            trial.suggest_categorical("fusion_weight_short", PAANO_WEIGHT_GRID)
        )
    return cfg


def _tune_config_with_optuna(
    anomaly_key: str,
    detector_key: str,
    train_runs: dict[str, PreparedDetectorRun],
    train_intervals: pd.DataFrame,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if optuna is None:  # pragma: no cover - runtime fallback
        return _tune_config_with_grid(anomaly_key, detector_key, train_runs, train_intervals, verbose)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=2027)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    n_trials = 48 if detector_key == "paano_feat" else 36

    def objective(trial: Any) -> float:
        cfg = _suggest_optuna_config(trial, anomaly_key, detector_key)
        predicted: dict[str, list[pd.Timestamp]] = {}
        for well_id, run in train_runs.items():
            _, _, starts = _detect_starts_for_run(detector_key, run, cfg)
            predicted[well_id] = starts
        pred_df = predicted_from_mapping(predicted)
        summary, _ = evaluate_predictions(
            train_intervals,
            pred_df,
            scores=None,
            prestart_hours=PRESTART_TOLERANCE_HOURS,
        )
        score_key = _operational_score_key(anomaly_key, detector_key, summary)
        trial.set_user_attr("config", cfg)
        trial.set_user_attr("summary", summary)
        trial.set_user_attr("score_key", list(score_key))
        return _optuna_objective_value(anomaly_key, detector_key, summary)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    leaderboard = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        cfg = trial.user_attrs.get("config")
        summary = trial.user_attrs.get("summary")
        score_key = trial.user_attrs.get("score_key")
        if not cfg or not summary or score_key is None:
            continue
        leaderboard.append(
            {
                "score_key": list(score_key),
                "objective": float(trial.value) if trial.value is not None else None,
                "config": cfg,
                "summary": summary,
            }
        )

    leaderboard.sort(key=lambda row: tuple(row["score_key"]), reverse=True)
    if not leaderboard:
        return _tune_config_with_grid(anomaly_key, detector_key, train_runs, train_intervals, verbose)

    best_cfg = dict(leaderboard[0]["config"])
    best_key = tuple(leaderboard[0]["score_key"])
    tuning_summary = {
        "backend": "optuna_tpe",
        "n_trials": n_trials,
        "best_score_key": list(best_key),
        "top10": leaderboard[:10],
    }
    if verbose and tuning_summary["top10"]:
        print("  Auto-tune top configs:")
        for idx, row in enumerate(tuning_summary["top10"][:5], 1):
            summary = row["summary"]
            cfg = row["config"]
            print(
                f"    {idx}. hit={summary['hit_count']}/{summary['interval_count']}, "
                f"p90_delay_ratio={summary['p90_delay_ratio']:.3f}, "
                f"FAR/day={summary['false_alarms_per_day']:.3f}, "
                f"starts/interval={summary['avg_starts_per_interval']:.2f}, "
                f"gate={cfg['gate_mode']}, run={cfg['min_run_points']}, cd={cfg['cooldown_hours']:.0f}, "
                f"rearm={cfg['rearm_window_minutes']:.0f}m, "
                f"ema={cfg['ema_alpha']:.2f}"
                + (
                    f", w={cfg['fusion_weight_short']:.2f}"
                    if "fusion_weight_short" in cfg
                    else ""
                )
            )
    return best_cfg, tuning_summary


def _tune_config_with_grid(
    anomaly_key: str,
    detector_key: str,
    train_runs: dict[str, PreparedDetectorRun],
    train_intervals: pd.DataFrame,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not train_runs:
        cfg = _default_onset_config(anomaly_key, detector_key)
        if detector_key == "paano_feat":
            cfg["fusion_weight_short"] = 0.60
        return cfg, {"message": "No train runs available"}

    best_cfg: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    leaderboard: list[dict[str, Any]] = []

    for cfg in _candidate_configs(anomaly_key, detector_key):
        predicted: dict[str, list[pd.Timestamp]] = {}
        for well_id, run in train_runs.items():
            _, _, starts = _detect_starts_for_run(detector_key, run, cfg)
            predicted[well_id] = starts
        pred_df = predicted_from_mapping(predicted)
        summary, _ = evaluate_predictions(
            train_intervals,
            pred_df,
            scores=None,
            prestart_hours=PRESTART_TOLERANCE_HOURS,
        )
        key = _operational_score_key(anomaly_key, detector_key, summary)
        leaderboard.append({"score_key": list(key), "config": cfg, "summary": summary})
        if best_key is None or key > best_key:
            best_cfg = cfg.copy()
            best_key = key

    leaderboard = sorted(leaderboard, key=lambda row: tuple(row["score_key"]), reverse=True)
    if best_cfg is None:
        best_cfg = _default_onset_config(anomaly_key, detector_key)
    tuning_summary = {
        "best_score_key": list(best_key) if best_key is not None else None,
        "top10": leaderboard[:10],
    }
    if verbose and tuning_summary["top10"]:
        print("  Auto-tune top configs:")
        for idx, row in enumerate(tuning_summary["top10"][:5], 1):
            summary = row["summary"]
            cfg = row["config"]
            print(
                f"    {idx}. hit={summary['hit_count']}/{summary['interval_count']}, "
                f"p90_delay_ratio={summary['p90_delay_ratio']:.3f}, "
                f"FAR/day={summary['false_alarms_per_day']:.3f}, "
                f"starts/interval={summary['avg_starts_per_interval']:.2f}, "
                f"gate={cfg['gate_mode']}, run={cfg['min_run_points']}, cd={cfg['cooldown_hours']:.0f}, "
                f"rearm={cfg['rearm_window_minutes']:.0f}m, "
                f"ema={cfg['ema_alpha']:.2f}"
                + (
                    f", w={cfg['fusion_weight_short']:.2f}"
                    if "fusion_weight_short" in cfg
                    else ""
                )
            )
    return best_cfg, tuning_summary


def _tune_config(
    anomaly_key: str,
    detector_key: str,
    train_runs: dict[str, PreparedDetectorRun],
    train_intervals: pd.DataFrame,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if optuna is None:
        return _tune_config_with_grid(anomaly_key, detector_key, train_runs, train_intervals, verbose)
    return _tune_config_with_optuna(anomaly_key, detector_key, train_runs, train_intervals, verbose)


def _load_or_build_config(
    spec: DetectionSpec,
    detector_key: str,
    train_runs: dict[str, PreparedDetectorRun],
    train_intervals: pd.DataFrame,
    retune: bool,
    verbose: bool,
) -> dict[str, Any]:
    cfg_path = config_path(spec, detector_key)
    tune_path = tuning_path(spec, detector_key)
    if cfg_path.exists() and not retune:
        payload = load_json(cfg_path)
        if isinstance(payload, dict) and "config" in payload:
            return {**_default_onset_config(spec.anomaly_key, detector_key), **payload["config"]}
        if payload:
            return {**_default_onset_config(spec.anomaly_key, detector_key), **payload}

    cfg, tuning_summary = _tune_config(
        spec.anomaly_key,
        detector_key,
        train_runs,
        train_intervals,
        verbose=verbose,
    )
    ensure_parent(cfg_path).write_text(
        json.dumps({"detector": detector_key, "config": cfg}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ensure_parent(tune_path).write_text(
        json.dumps(tuning_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if verbose:
        print(f"Saved tuned config to {cfg_path}")
        print(f"Saved tuning summary to {tune_path}")
    return cfg


def _build_score_rows(
    detector_key: str,
    runs: dict[str, PreparedDetectorRun],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[pd.Timestamp]], dict[str, dict[str, Any]]]:
    score_rows: list[dict[str, Any]] = []
    predicted: dict[str, list[pd.Timestamp]] = {}
    detail_map: dict[str, dict[str, Any]] = {}

    for well_id, run in runs.items():
        score, thresholds, starts = _detect_starts_for_run(detector_key, run, cfg)
        predicted[well_id] = starts
        detail_map[well_id] = {
            "prepared": run.prepared.detail,
            "detector": run.score_output.detail,
            "thresholds": asdict(thresholds),
            "n_predicted_starts": len(starts),
        }
        components = dict(run.score_output.components)
        if detector_key == "paano_feat":
            components["score"] = score
            short_score = components.get("paano_short")
            long_score = components.get("paano_long")
            if short_score is not None and long_score is not None:
                components["paano_fused"] = score
        else:
            components["score"] = score

        for idx, ts in enumerate(run.prepared.timestamps):
            row = {
                "well_id": well_id,
                "timestamp": ts,
                "split": run.prepared.split,
                "score": float(score[idx]),
            }
            for name, values in components.items():
                if len(values) != len(score):
                    continue
                row[name] = float(values[idx])
            score_rows.append(row)

    return score_rows, predicted, detail_map


def _merge_result_details(
    result_df: pd.DataFrame,
    predicted: dict[str, list[pd.Timestamp]],
    detail_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in result_df.iterrows():
        well_id = row["well_id"]
        detail = detail_map.get(well_id, {})
        starts = predicted.get(well_id, [])
        merged = row.to_dict()
        merged["n_predicted_starts"] = len(starts)
        merged["n_channels"] = int(detail.get("prepared", {}).get("raw_channels", 0))
        merged["n_features"] = int(detail.get("prepared", {}).get("feature_count", 0))
        merged["detail"] = json.dumps(detail, ensure_ascii=False)
        rows.append(merged)
    return pd.DataFrame(rows)


def _summary_for_payload(payload: dict[str, Any]) -> dict[str, Any]:
    splits = payload.get("splits", {})
    if isinstance(splits, dict) and "all" in splits:
        return splits["all"]
    return payload


def _choose_default_detector(
    anomaly_key: str,
    detector_summaries: dict[str, dict[str, Any]],
    legacy_payload: dict[str, Any],
) -> str:
    candidates: list[tuple[tuple[float, ...], str]] = []
    for detector_key, payload in detector_summaries.items():
        if detector_key not in LOCAL_DETECTOR_KEYS:
            continue
        summary = _summary_for_payload(payload.get("splits", {}).get("all", payload))
        candidates.append((_operational_score_key(anomaly_key, detector_key, summary), detector_key))

    if not candidates:
        if "paano_feat" in detector_summaries:
            return "paano_feat"
        available = sorted(detector_summaries)
        return available[0] if available else DEFAULT_DETECTOR

    candidates.sort(reverse=True)
    selected = candidates[0][1]

    if legacy_payload:
        legacy_summary = _summary_for_payload(legacy_payload)
        selected_summary = _summary_for_payload(detector_summaries[selected].get("splits", {}).get("all", detector_summaries[selected]))
        legacy_key = _operational_score_key(anomaly_key, "paano_feat", legacy_summary)
        selected_key = _operational_score_key(anomaly_key, selected, selected_summary)
        if legacy_key > selected_key and "paano_feat" in detector_summaries:
            return "paano_feat"
    return selected


def _update_benchmark_summary(spec: DetectionSpec) -> dict[str, Any]:
    detector_payloads: dict[str, dict[str, Any]] = {}
    for detector_key in DETECTOR_KEYS:
        path = summary_path(spec, detector_key)
        if path.exists():
            detector_payloads[detector_key] = load_json(path)
    legacy_payload = load_json(legacy_summary_path(spec))
    selected = _choose_default_detector(spec.anomaly_key, detector_payloads, legacy_payload)
    payload = {
        "anomaly": spec.anomaly_key,
        "selected_default_detector": selected,
        "detectors": detector_payloads,
    }
    if legacy_payload:
        payload["legacy_paano"] = legacy_payload
    out_path = ensure_parent(benchmark_summary_path(spec))
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _print_result_table(spec: DetectionSpec, detector_key: str, result_df: pd.DataFrame) -> None:
    print(f"\n=== {spec.display_name} Results [{detector_key}] ===")
    if result_df.empty:
        print("No interval results.")
        return
    print(
        result_df[
            ["well_id", "interval_idx", "split", "detected_time", "actual_start", "actual_end", "status"]
        ].to_string(index=False)
    )


def run_detection(
    anomaly_key: str,
    detector: str = DEFAULT_DETECTOR,
    output_path: str | None = None,
    source_path: str | None = None,
    retune: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    detector_key = normalize_detector_key(detector)
    spec = get_detection_spec(anomaly_key)
    print(f"=== {spec.display_name} Detection [{detector_key}] ===")
    set_seed()
    ensure_dir(DB_DIR)
    output = ensure_parent(Path(output_path) if output_path else results_path(spec, detector_key))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_anomaly_data(spec, source_path=source_path)
    intervals = load_intervals(spec, required=True)
    if df.empty or intervals.empty:
        raise RuntimeError("Empty data or intervals for detection.")

    prepared_runs = _prepare_all_wells(spec, df, intervals, verbose=verbose)
    if not prepared_runs:
        raise RuntimeError("No wells survived engineered preprocessing.")

    detector_runs = _build_local_runs(spec.anomaly_key, detector_key, prepared_runs, device=device, verbose=verbose)

    train_runs = {well_id: run for well_id, run in detector_runs.items() if run.prepared.split == "train"}
    train_intervals = intervals[intervals["split"].astype(str).str.lower() == "train"].copy()
    cfg = _load_or_build_config(
        spec=spec,
        detector_key=detector_key,
        train_runs=train_runs,
        train_intervals=train_intervals,
        retune=retune,
        verbose=verbose,
    )

    score_rows, predicted, detail_map = _build_score_rows(detector_key, detector_runs, cfg)
    score_df = pd.DataFrame(score_rows)
    pred_df = predicted_from_mapping(predicted)
    if not pred_df.empty:
        split_lookup = {well_id: run.prepared.split for well_id, run in detector_runs.items()}
        pred_df["split"] = pred_df["well_id"].map(split_lookup).fillna("train")

    split_summaries, split_frames = summarize_splits(
        intervals=intervals,
        predictions=pred_df,
        scores=score_df,
        prestart_hours=PRESTART_TOLERANCE_HOURS,
    )
    result_df = split_frames.get("all", pd.DataFrame())
    result_df = _merge_result_details(result_df, predicted=predicted, detail_map=detail_map)
    _print_result_table(spec, detector_key, result_df)

    write_table(result_df, output)
    print(f"\nResults saved to {output}")

    score_output_path = scores_path(spec, detector_key)
    ensure_parent(score_output_path)
    write_table(score_df, score_output_path)
    print(f"Per-point scores saved to {score_output_path}")

    pred_output_path = predicted_starts_path(spec, detector_key)
    ensure_parent(pred_output_path)
    write_table(pred_df, pred_output_path)
    print(f"Predicted starts saved to {pred_output_path}")

    summary_payload = {
        "anomaly": spec.anomaly_key,
        "detector": detector_key,
        "config": cfg,
        "prestart_hours": PRESTART_TOLERANCE_HOURS,
        "splits": split_summaries,
        "artifacts": {
            "results_path": str(output),
            "scores_path": str(score_output_path),
            "predicted_starts_path": str(pred_output_path),
            "report_path": str(report_path(spec, detector_key)),
        },
    }
    summary_output_path = ensure_parent(summary_path(spec, detector_key))
    summary_output_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary saved to {summary_output_path}")

    benchmark_payload = _update_benchmark_summary(spec)
    print(
        f"Benchmark summary updated: {benchmark_summary_path(spec)} "
        f"(default={benchmark_payload['selected_default_detector']})"
    )
    return result_df


def run_single_well(
    anomaly_key: str,
    well_id: str,
    detector: str = DEFAULT_DETECTOR,
    source_path: str | None = None,
    retune: bool = False,
) -> None:
    detector_key = normalize_detector_key(detector)
    spec = get_detection_spec(anomaly_key)
    df = load_anomaly_data(spec, source_path=source_path)
    intervals = load_intervals(spec, required=False)
    well_id = well_id.strip().lower()
    well_df = df[df["well_id"] == well_id]
    if well_df.empty:
        raise ValueError(f"No data for well {well_id}")
    split = "train"
    if not intervals.empty:
        well_intervals = intervals[intervals["well_id"] == well_id]
        if not well_intervals.empty and "split" in well_intervals.columns:
            split = str(well_intervals["split"].iloc[0]).strip().lower()

    prepared = prepare_engineered_well(
        anomaly_key=spec.anomaly_key,
        well_id=well_id,
        split=split,
        well_df=well_df,
        patch_size=int(_runtime_config(spec.anomaly_key)["prepare_patch_size"]),
        reference_min_ratio=REFERENCE_MIN_RATIO,
        reference_max_ratio=REFERENCE_MAX_RATIO,
        reference_min_days=REFERENCE_MIN_DAYS,
        min_reference_coverage=MIN_REFERENCE_COVERAGE,
        min_total_coverage=MIN_TOTAL_COVERAGE,
    )
    if prepared is None:
        print("No usable data after engineered preprocessing.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector_obj = _build_detector(spec.anomaly_key, detector_key, device=device, verbose=True)
    detector_obj.fit_reference(prepared.feature_matrix[prepared.reference_mask], mask_ref=prepared.reference_mask)
    score_output = detector_obj.score_stream(prepared.feature_matrix, mask_all=prepared.stability_mask)
    run = PreparedDetectorRun(prepared=prepared, score_output=score_output)

    cfg_payload = load_json(config_path(spec, detector_key))
    cfg = {**_default_onset_config(spec.anomaly_key, detector_key), **(cfg_payload.get("config", cfg_payload) if cfg_payload else {})}
    if detector_key == "paano_feat" and "fusion_weight_short" not in cfg:
        cfg["fusion_weight_short"] = 0.60

    score, thresholds, starts = _detect_starts_for_run(detector_key, run, cfg)
    print(f"Prepared detail: {json.dumps(run.prepared.detail, ensure_ascii=False, indent=2)}")
    print(f"Detector detail: {json.dumps(run.score_output.detail, ensure_ascii=False, indent=2)}")
    print(f"Thresholds: {json.dumps(asdict(thresholds), ensure_ascii=False, indent=2)}")
    print(f"Detected starts: {[pd.Timestamp(ts) for ts in starts]}")
    print(f"Score summary: min={float(np.min(score)):.4f}, median={float(np.median(score)):.4f}, max={float(np.max(score)):.4f}")

    if not intervals.empty:
        well_intervals = intervals[intervals["well_id"] == well_id].sort_values(["start_date", "interval_idx"])
        for _, row in well_intervals.iterrows():
            det = select_interval_detection(starts, row["start_date"], row["end_date"], PRESTART_TOLERANCE_HOURS)
            print(
                f"  interval={int(row['interval_idx'])} start={row['start_date']} end={row['end_date']} "
                f"detected={det}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified engineered-feature anomaly detection pipeline.")
    parser.add_argument("anomaly", choices=["negermet", "pritok", "salt"])
    parser.add_argument("--detector", choices=sorted(DETECTOR_KEYS), default=DEFAULT_DETECTOR)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--well", type=str, default=None)
    parser.add_argument("--retune", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.well:
        run_single_well(
            anomaly_key=args.anomaly,
            well_id=args.well,
            detector=args.detector,
            source_path=args.source,
            retune=args.retune,
        )
        return
    run_detection(
        anomaly_key=args.anomaly,
        detector=args.detector,
        output_path=args.output,
        source_path=args.source,
        retune=args.retune,
        verbose=True,
    )


if __name__ == "__main__":
    main()
