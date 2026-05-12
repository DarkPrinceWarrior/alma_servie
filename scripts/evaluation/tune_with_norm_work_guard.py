from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alma_service.anomaly_specs import get_detection_spec
from alma_service.benchmark_metrics import evaluate_predictions, predicted_from_mapping, summarize_splits
from alma_service.detection_artifacts import config_path, load_json
from alma_service.engineered_features import PreparedWellData, prepare_engineered_well
from alma_service.generic_detection import (
    MIN_REFERENCE_COVERAGE,
    MIN_TOTAL_COVERAGE,
    PRESTART_TOLERANCE_HOURS,
    REFERENCE_MAX_RATIO,
    REFERENCE_MIN_DAYS,
    REFERENCE_MIN_RATIO,
    _build_local_runs,
    _build_per_well_tuning_intervals,
    _build_score_rows,
    _default_onset_config,
    _detect_starts_for_config,
    _load_or_build_config,
    _per_well_tuning_summaries_from_mapping,
    _prepare_all_wells,
    _resolve_torch_device,
    _robust_optuna_objective_value,
    _robust_tuning_score_key,
    _runtime_config,
    _suggest_optuna_config,
    _onset_tune_grid,
    SALT_SHARED_ONSET_TUNE_GRID,
    load_anomaly_data,
    load_intervals,
)
from alma_service.paths import DB_DIR, RESULTS_DIR, ensure_dir
from alma_service.shared_encoder import train_shared_encoder
from alma_service.tabular_io import read_table

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


DEFAULT_ANOMALY_FREQ = {
    "negermet": "2min",
    "pritok": "10min",
    "salt": "15min",
}
DEFAULT_ANOMALIES = ("negermet", "pritok", "salt")


def _parse_anomalies(value: str) -> list[str]:
    anomalies = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(anomalies) - set(DEFAULT_ANOMALIES))
    if unknown:
        raise ValueError(f"Unknown anomaly keys: {unknown}")
    return anomalies


def _norm_source_path(anomaly_key: str) -> Path:
    return DB_DIR / f"norm_work_database_{DEFAULT_ANOMALY_FREQ[anomaly_key].replace(' ', '')}.parquet"


def _load_saved_config(anomaly_key: str, detector_key: str) -> dict[str, Any]:
    path = config_path(get_detection_spec(anomaly_key), detector_key)
    payload = load_json(path)
    if isinstance(payload, dict) and "config" in payload:
        return {**_default_onset_config(anomaly_key, detector_key), **payload["config"]}
    if isinstance(payload, dict):
        return {**_default_onset_config(anomaly_key, detector_key), **payload}
    raise FileNotFoundError(f"Config not found: {path}")


def _seed_trial_params(anomaly_key: str, detector_key: str, cfg: dict[str, Any]) -> dict[str, Any]:
    grid = (
        SALT_SHARED_ONSET_TUNE_GRID
        if anomaly_key == "salt" and detector_key == "paano_shared"
        else _onset_tune_grid(anomaly_key)
    )
    allowed_keys = {
        "target_far_per_day",
        "min_run_points",
        "cooldown_hours",
        "rearm_window_minutes",
        "ema_alpha",
        "gate_mode",
        "bypass_cooldown_after_clear",
        "pressure_trend_weight",
        "negermet_signature_weight",
        "salt_trend_weight",
    }
    params: dict[str, Any] = {}
    for key, value in cfg.items():
        if key not in allowed_keys:
            continue
        if key in grid and value not in grid[key]:
            continue
        params[key] = value
    return params


def _prepare_norm_wells(anomaly_key: str) -> dict[str, PreparedWellData]:
    source = _norm_source_path(anomaly_key)
    df = read_table(source, parse_dates=["timestamp"])
    runtime_cfg = _runtime_config(anomaly_key)
    prepared: dict[str, PreparedWellData] = {}
    for well_id in sorted(df["well_id"].unique()):
        well_df = df[df["well_id"] == well_id].copy()
        item = prepare_engineered_well(
            anomaly_key=anomaly_key,
            well_id=str(well_id),
            split="screen",
            well_df=well_df,
            patch_size=int(runtime_cfg["prepare_patch_size"]),
            reference_min_ratio=REFERENCE_MIN_RATIO,
            reference_max_ratio=REFERENCE_MAX_RATIO,
            reference_min_days=REFERENCE_MIN_DAYS,
            min_reference_coverage=MIN_REFERENCE_COVERAGE,
            min_total_coverage=MIN_TOTAL_COVERAGE,
        )
        if item is not None:
            prepared[str(well_id)] = item
    if not prepared:
        raise RuntimeError(f"No norm_work wells survived preprocessing for {anomaly_key}")
    return prepared


def _norm_false_alarm_summary(
    norm_runs: dict[str, Any],
    detector_key: str,
    cfg: dict[str, Any],
    starts_cache: dict[tuple[str, tuple[tuple[str, Any], ...]], list[pd.Timestamp]] | None = None,
) -> dict[str, Any]:
    predicted = _detect_starts_for_config(detector_key, norm_runs, cfg, starts_cache)
    total_starts = int(sum(len(starts) for starts in predicted.values()))
    wells_with_starts = int(sum(1 for starts in predicted.values() if starts))
    observed_days = 0.0
    for run in norm_runs.values():
        timestamps = run.prepared.timestamps
        if len(timestamps) < 2:
            continue
        start = pd.Timestamp(timestamps[0])
        end = pd.Timestamp(timestamps[-1])
        if end > start:
            observed_days += (end - start).total_seconds() / 86400.0
    return {
        "wells_total": int(len(norm_runs)),
        "wells_with_false_alarm": wells_with_starts,
        "total_false_alarm_starts": total_starts,
        "false_alarm_starts_per_day": float(total_starts / observed_days) if observed_days > 0 else np.nan,
        "observed_days_total": float(observed_days),
    }


def _evaluate_class_runs(
    *,
    anomaly_key: str,
    detector_key: str,
    runs: dict[str, Any],
    intervals: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    score_rows, predicted, _ = _build_score_rows(detector_key, runs, cfg)
    pred_df = predicted_from_mapping(predicted)
    if not pred_df.empty:
        split_lookup = {well_id: run.prepared.split for well_id, run in runs.items()}
        pred_df["split"] = pred_df["well_id"].map(split_lookup).fillna("train")
    score_df = pd.DataFrame(score_rows)
    split_summaries, split_frames = summarize_splits(
        intervals=intervals,
        predictions=pred_df,
        scores=score_df,
        prestart_hours=PRESTART_TOLERANCE_HOURS,
    )
    return {
        "splits": split_summaries,
        "interval_results": split_frames.get("all", pd.DataFrame()).to_dict("records"),
    }


def _compact_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "interval_count": summary.get("interval_count"),
        "hit_count": summary.get("hit_count"),
        "hit_rate": summary.get("hit_rate"),
        "false_alarms_per_day": summary.get("false_alarms_per_day"),
        "start_count": summary.get("start_count"),
        "median_abs_delay_hours": summary.get("median_abs_delay_hours"),
        "p90_abs_delay_hours": summary.get("p90_abs_delay_hours"),
        "p90_delay_ratio": summary.get("p90_delay_ratio"),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def tune_one_anomaly(
    *,
    anomaly_key: str,
    detector_key: str,
    n_trials: int,
    n_jobs: int,
    norm_start_weight: float,
    norm_well_weight: float,
    output_dir: Path,
    verbose: bool,
) -> dict[str, Any]:
    if optuna is None:
        raise RuntimeError("Optuna is required for norm_work guarded tuning")

    spec = get_detection_spec(anomaly_key)
    device = _resolve_torch_device(detector_key, verbose=True)
    df = load_anomaly_data(spec)
    intervals = load_intervals(spec, required=True)
    intervals = (
        intervals.sort_values(["well_id", "start_date", "interval_idx"])
        .groupby("well_id", as_index=False)
        .first()
    )
    prepared = _prepare_all_wells(spec, df, intervals, verbose=verbose, zone_aware=True)
    runtime_cfg = _runtime_config(anomaly_key)
    shared_state = train_shared_encoder(
        prepared_wells=prepared,
        patch_short=int(runtime_cfg.get("paano_patch_short", 64)),
        patch_long=int(runtime_cfg.get("paano_patch_long", 128)),
        anomaly_key=anomaly_key,
        device=device,
        verbose=verbose,
    )
    runs = _build_local_runs(
        anomaly_key,
        detector_key,
        prepared,
        device=device,
        verbose=False,
        shared_state=shared_state,
    )
    norm_prepared = _prepare_norm_wells(anomaly_key)
    norm_runs = _build_local_runs(
        anomaly_key,
        detector_key,
        norm_prepared,
        device=device,
        verbose=False,
        shared_state=shared_state,
    )

    train_runs = {well_id: run for well_id, run in runs.items() if run.prepared.split == "train"}
    train_intervals = intervals[intervals["split"].astype(str).str.lower() == "train"].copy()
    per_well_intervals = _build_per_well_tuning_intervals(train_intervals)
    starts_cache: dict[tuple[str, tuple[tuple[str, Any], ...]], list[pd.Timestamp]] = {}
    norm_cache: dict[tuple[str, tuple[tuple[str, Any], ...]], list[pd.Timestamp]] = {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=2027, multivariate=True, group=True, n_startup_trials=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    saved_cfg = _load_saved_config(anomaly_key, detector_key)
    seed_params = _seed_trial_params(anomaly_key, detector_key, saved_cfg)
    if seed_params:
        study.enqueue_trial(seed_params)

    def objective(trial: Any) -> float:
        cfg = _suggest_optuna_config(trial, anomaly_key, detector_key)
        predicted = _detect_starts_for_config(detector_key, train_runs, cfg, starts_cache)
        pred_df = predicted_from_mapping(predicted)
        train_summary, _ = evaluate_predictions(
            train_intervals,
            pred_df,
            scores=None,
            prestart_hours=PRESTART_TOLERANCE_HOURS,
        )
        per_well_summaries = _per_well_tuning_summaries_from_mapping(per_well_intervals, predicted)
        base_value = _robust_optuna_objective_value(
            anomaly_key,
            detector_key,
            train_summary,
            per_well_summaries,
        )
        norm_summary = _norm_false_alarm_summary(norm_runs, detector_key, cfg, norm_cache)
        objective_value = (
            base_value
            - norm_start_weight * float(norm_summary["total_false_alarm_starts"])
            - norm_well_weight * float(norm_summary["wells_with_false_alarm"])
        )
        score_key = _robust_tuning_score_key(
            anomaly_key,
            detector_key,
            train_summary,
            per_well_summaries,
        )
        guarded_score_key = (
            *score_key,
            -float(norm_summary["wells_with_false_alarm"]),
            -float(norm_summary["total_false_alarm_starts"]),
            -float(norm_summary["false_alarm_starts_per_day"]),
        )
        trial.set_user_attr("config", cfg)
        trial.set_user_attr("train_summary", train_summary)
        trial.set_user_attr("per_well_summaries", per_well_summaries)
        trial.set_user_attr("norm_summary", norm_summary)
        trial.set_user_attr("score_key", list(score_key))
        trial.set_user_attr("guarded_score_key", list(guarded_score_key))
        return objective_value

    started = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, n_jobs=max(1, n_jobs), show_progress_bar=False)
    elapsed = time.perf_counter() - started

    leaderboard: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        cfg = trial.user_attrs.get("config")
        if not cfg:
            continue
        leaderboard.append(
            {
                "objective": float(trial.value) if trial.value is not None else None,
                "guarded_score_key": trial.user_attrs.get("guarded_score_key", []),
                "score_key": trial.user_attrs.get("score_key", []),
                "config": cfg,
                "train_summary": trial.user_attrs.get("train_summary", {}),
                "norm_summary": trial.user_attrs.get("norm_summary", {}),
            }
        )
    leaderboard.sort(key=lambda row: (float(row["objective"] or -1e300), tuple(row["guarded_score_key"])), reverse=True)
    if not leaderboard:
        raise RuntimeError(f"No complete trials for {anomaly_key}")

    selected = leaderboard[0]
    selected_cfg = dict(selected["config"])
    saved_eval = _evaluate_class_runs(
        anomaly_key=anomaly_key,
        detector_key=detector_key,
        runs=runs,
        intervals=intervals,
        cfg=saved_cfg,
    )
    selected_eval = _evaluate_class_runs(
        anomaly_key=anomaly_key,
        detector_key=detector_key,
        runs=runs,
        intervals=intervals,
        cfg=selected_cfg,
    )
    saved_norm = _norm_false_alarm_summary(norm_runs, detector_key, saved_cfg)
    selected_norm = _norm_false_alarm_summary(norm_runs, detector_key, selected_cfg)

    payload = {
        "anomaly": anomaly_key,
        "detector": detector_key,
        "n_trials": n_trials,
        "n_jobs": n_jobs,
        "elapsed_seconds": round(elapsed, 3),
        "norm_start_weight": norm_start_weight,
        "norm_well_weight": norm_well_weight,
        "saved_config": saved_cfg,
        "selected_config": selected_cfg,
        "saved": {
            "splits": {
                split: _compact_metrics(summary)
                for split, summary in saved_eval["splits"].items()
            },
            "norm_summary": saved_norm,
        },
        "selected": {
            "splits": {
                split: _compact_metrics(summary)
                for split, summary in selected_eval["splits"].items()
            },
            "norm_summary": selected_norm,
        },
        "top10": leaderboard[:10],
    }
    output_path = output_dir / f"norm_guard_tuning_{anomaly_key}_{detector_key}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(
        f"{anomaly_key}: saved norm starts={saved_norm['total_false_alarm_starts']}, "
        f"selected norm starts={selected_norm['total_false_alarm_starts']}, "
        f"saved all hit={payload['saved']['splits'].get('all', {}).get('hit_count')}, "
        f"selected all hit={payload['selected']['splits'].get('all', {}).get('hit_count')}",
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune detector configs with expert norm_work negative validation.")
    parser.add_argument("--anomalies", default=",".join(DEFAULT_ANOMALIES))
    parser.add_argument("--detector", default="paano_shared")
    parser.add_argument("--n-trials", type=int, default=int(os.environ.get("ALMA_NORM_GUARD_TRIALS", "96")))
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("ALMA_NORM_GUARD_N_JOBS", "8")))
    parser.add_argument("--norm-start-weight", type=float, default=5_000_000.0)
    parser.add_argument("--norm-well-weight", type=float, default=10_000_000.0)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "norm_work_false_alarms")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    anomalies = _parse_anomalies(args.anomalies)
    payloads = [
        tune_one_anomaly(
            anomaly_key=anomaly,
            detector_key=str(args.detector),
            n_trials=int(args.n_trials),
            n_jobs=int(args.n_jobs),
            norm_start_weight=float(args.norm_start_weight),
            norm_well_weight=float(args.norm_well_weight),
            output_dir=output_dir,
            verbose=not args.quiet,
        )
        for anomaly in anomalies
    ]
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for variant in ("saved", "selected"):
            all_metrics = payload[variant]["splits"].get("all", {})
            norm_summary = payload[variant]["norm_summary"]
            rows.append(
                {
                    "anomaly": payload["anomaly"],
                    "variant": variant,
                    **all_metrics,
                    "norm_wells_with_false_alarm": norm_summary["wells_with_false_alarm"],
                    "norm_total_false_alarm_starts": norm_summary["total_false_alarm_starts"],
                    "norm_false_alarm_starts_per_day": norm_summary["false_alarm_starts_per_day"],
                }
            )
    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / "norm_guard_tuning_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
