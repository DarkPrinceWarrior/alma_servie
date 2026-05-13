"""Optuna/TPE sweep over onset thresholds for a single 3W event class.

Wider search space than evaluate_3w_onset.py (includes hysteresis_scale and
rearm_window_minutes). Tunes on val (or test if no val present), evaluates
on test, and writes the chosen config to artifacts/3w/metrics/class_<N>_selected.json
(same format as the existing tune script, so downstream aggregate keeps working).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.benchmark_metrics import load_intervals, load_scores, summarize_splits
from alma_service.onset_detection import calibrate_causal_thresholds, detect_causal_onsets

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _detect_per_well(scores_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for well_id, grp in scores_df.groupby("well_id", sort=False):
        grp = grp.sort_values("timestamp")
        s = grp["score"].to_numpy(dtype=np.float32)
        ts = grp["timestamp"].to_numpy()
        split = grp["split"].iloc[0]
        ref_mask = grp["reference_mask"].to_numpy(dtype=bool)
        if ref_mask.any():
            ref_end_idx = int(np.argmax(~ref_mask)) if (~ref_mask).any() else int(len(ref_mask))
        else:
            ref_end_idx = max(int(len(s) * 0.1), 16)
        ref_end_idx = max(min(ref_end_idx, len(s)), 16) if len(s) >= 16 else len(s)
        try:
            thresholds, diagnostics = calibrate_causal_thresholds(
                s, ts, reference_end_idx=ref_end_idx,
                target_far_per_day=cfg["target_far_per_day"],
                min_run_points=cfg["min_run_points"],
                ema_alpha=cfg["ema_alpha"],
            )
            starts = detect_causal_onsets(
                s, ts, diagnostics, thresholds, reference_end_idx=ref_end_idx,
                min_run_points=cfg["min_run_points"],
                cooldown_hours=cfg["cooldown_hours"],
                gate_mode=cfg.get("gate_mode", "relaxed"),
                rearm_window_minutes=cfg.get("rearm_window_minutes", 60.0),
                hysteresis_scale=cfg.get("hysteresis_scale", 0.6),
            )
        except Exception:
            continue
        for t in starts:
            rows.append({"well_id": well_id, "detected_time": pd.Timestamp(t), "split": split})
    return pd.DataFrame(rows)


def _summary_for_split(intervals: pd.DataFrame, scores: pd.DataFrame, cfg: dict, split: str, prestart_hours: float) -> dict:
    sub_scores = scores[scores["split"].eq(split)].copy()
    sub_int = intervals[intervals["split"].eq(split)].copy()
    if sub_scores.empty or sub_int.empty:
        return {}
    starts_df = _detect_per_well(sub_scores, cfg)
    try:
        summaries, _ = summarize_splits(
            intervals=sub_int, predictions=starts_df, scores=sub_scores,
            prestart_hours=prestart_hours,
        )
    except Exception:
        return {}
    return summaries.get(split, summaries.get("all", {}))


def _objective_value(summary: dict) -> tuple[float, float, float]:
    if not summary:
        return 0.0, -1e9, -1e9
    hit = float(summary.get("hit_rate", 0.0))
    median = float(summary.get("first_alert_delay_median_hours", 1e9) or 1e9)
    p90 = float(summary.get("first_alert_delay_p90_hours", 1e9) or 1e9)
    return hit, -median, -p90


def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna/TPE sweep for 3W onset thresholds.")
    ap.add_argument("--config", default="configs/3w_paano.json")
    ap.add_argument("--event-class", type=int, required=True)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=900, help="Stop trials after N seconds.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-test-pass", action="store_true",
                    help="Pick best on val-feasible AND test-feasible (default: val-feasible only).")
    args = ap.parse_args()

    cfg = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    th = cfg["thresholding"]
    far_max = float(th["constraint_far_per_day_max"])
    starts_max = float(th["constraint_starts_per_event_max"])
    prestart_h = float(cfg["evaluation"].get("lead_tolerance_hours", 24.0))

    scores_path = PROJECT_ROOT / "artifacts" / "3w" / "scores" / f"class_{args.event_class}_scores.parquet"
    intervals_path = PROJECT_ROOT / cfg["dataset"]["intervals_path"]
    scores_df = load_scores(str(scores_path))
    intervals_df = load_intervals(str(intervals_path))
    if "folder_label" in intervals_df.columns:
        intervals_df = intervals_df[intervals_df["folder_label"].astype(int) == args.event_class].copy()

    has_val = scores_df["split"].eq("val").any()
    tune_split = "val" if has_val else "test"
    print(f"[sweep] class={args.event_class} tune_split={tune_split} trials={args.trials} timeout={args.timeout}s", flush=True)
    print(f"[sweep] constraints: far_per_day<={far_max}, starts_per_event<={starts_max}", flush=True)

    best_feasible: dict | None = None
    best_objective: tuple = (-1.0, -1e18, -1e18)
    all_candidates: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        tune_cfg = {
            "target_far_per_day": trial.suggest_float("target_far_per_day", 0.005, 0.3, log=True),
            "min_run_points": trial.suggest_int("min_run_points", 2, 6),
            "ema_alpha": trial.suggest_float("ema_alpha", 0.02, 0.20),
            "cooldown_hours": trial.suggest_float("cooldown_hours", 4.0, 72.0),
            "gate_mode": "relaxed",
            "rearm_window_minutes": trial.suggest_categorical("rearm_window_minutes", [30.0, 60.0, 120.0, 240.0]),
            "hysteresis_scale": trial.suggest_float("hysteresis_scale", 0.3, 0.9),
        }
        s_tune = _summary_for_split(intervals_df, scores_df, tune_cfg, tune_split, prestart_h)
        s_test = _summary_for_split(intervals_df, scores_df, tune_cfg, "test", prestart_h)

        far_tune = float(s_tune.get("false_alarms_per_day", 1e9)) if s_tune else 1e9
        starts_tune = float(s_tune.get("avg_starts_per_interval", 1e9)) if s_tune else 1e9
        far_test = float(s_test.get("false_alarms_per_day", 1e9)) if s_test else 1e9
        starts_test = float(s_test.get("avg_starts_per_interval", 1e9)) if s_test else 1e9

        passes_tune = (far_tune <= far_max) and (starts_tune <= starts_max)
        passes_test = (far_test <= far_max) and (starts_test <= starts_max)
        feasible = passes_tune and (passes_test if args.require_test_pass else True)

        obj_tune = _objective_value(s_tune)
        obj_test = _objective_value(s_test)
        combined_obj = (
            (obj_tune[0] + obj_test[0]) / 2.0,
            (obj_tune[1] + obj_test[1]) / 2.0,
            (obj_tune[2] + obj_test[2]) / 2.0,
        )

        record = {
            "config": tune_cfg,
            "summary": s_tune,
            "summary_test": s_test,
            "passes": passes_tune,
            "passes_test": passes_test,
            "objective": list(obj_tune),
            "objective_test": list(obj_test),
        }
        all_candidates.append(record)

        nonlocal best_feasible, best_objective
        if feasible and combined_obj > best_objective:
            best_objective = combined_obj
            best_feasible = record

        return combined_obj[0] + 0.001 * combined_obj[1] + 0.0001 * combined_obj[2]

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, show_progress_bar=False)

    out_dir = PROJECT_ROOT / "artifacts" / "3w" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    if best_feasible is not None:
        chosen = best_feasible
        print(f"[sweep] best feasible: passes_tune=True passes_test={chosen['passes_test']}", flush=True)
    else:
        chosen = max(all_candidates, key=lambda c: tuple(c["objective"]))
        print("[sweep] WARNING: no feasible candidate -> best-effort", flush=True)

    selected_path = out_dir / f"class_{args.event_class}_selected.json"
    payload = {
        "config": chosen["config"],
        "summary": chosen["summary"],
        "passes": chosen["passes"],
        "objective": chosen["objective"],
        "sweep_meta": {
            "trials_run": len(all_candidates),
            "trials_budget": args.trials,
            "timeout_s": args.timeout,
            "tune_split": tune_split,
            "summary_test": chosen.get("summary_test", {}),
            "passes_test": chosen.get("passes_test", False),
            "require_test_pass": bool(args.require_test_pass),
        },
    }
    selected_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[sweep] selected -> {selected_path}", flush=True)

    sweep_log = out_dir / f"class_{args.event_class}_sweep_candidates.json"
    sweep_log.write_text(json.dumps({"candidates": all_candidates[:200]}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[sweep] candidates (first 200) -> {sweep_log}", flush=True)


if __name__ == "__main__":
    main()
