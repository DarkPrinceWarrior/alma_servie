"""Optuna/TPE sweep over ALMA onset thresholds (analog of optuna_sweep_3w).

Reads per-point scores at `db/<anomaly>_paano_shared_scores.parquet` (written by
detect_<anomaly>.py for detector=paano_shared) and intervals from the canonical
ALMA intervals parquet. Tunes onset config to minimize FAR/day while keeping
hit-rate at the baseline ceiling.

Selected config is written to `db/<anomaly>_paano_shared_config.json` so the
next run of `detect_<anomaly>.py --detector paano_shared` picks it up via the
existing `_load_or_build_config` path.
"""
from __future__ import annotations

import argparse
import copy
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
        ref_mask = (
            grp["reference_mask"].to_numpy(dtype=bool)
            if "reference_mask" in grp.columns
            else np.ones(len(s), dtype=bool)
        )
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
    ap = argparse.ArgumentParser(description="Optuna/TPE sweep for ALMA onset thresholds.")
    ap.add_argument("--anomaly", choices=["negermet", "pritok", "salt"], required=True)
    ap.add_argument("--detector", default="paano_shared")
    ap.add_argument("--far-max", type=float, default=0.10)
    ap.add_argument("--starts-max", type=float, default=3.0)
    ap.add_argument("--prestart-hours", type=float, default=2.0)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-test-pass", action="store_true")
    ap.add_argument("--write-config", action="store_true",
                    help="Write selected config to db/<anomaly>_<detector>_config.json on success.")
    args = ap.parse_args()

    scores_path = PROJECT_ROOT / "db" / f"{args.anomaly}_{args.detector}_scores.parquet"
    intervals_path = PROJECT_ROOT / "db" / f"{args.anomaly}_intervals.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing scores: {scores_path}. Run detect_{args.anomaly}.py first.")
    if not intervals_path.exists():
        raise FileNotFoundError(f"Missing intervals: {intervals_path}")

    scores_df = load_scores(str(scores_path))
    intervals_df = load_intervals(str(intervals_path))

    has_val = scores_df["split"].eq("val").any()
    tune_split = "val" if has_val else "train"
    print(f"[sweep_alma] anomaly={args.anomaly} tune_split={tune_split} trials={args.trials} timeout={args.timeout}s", flush=True)
    print(f"[sweep_alma] constraints: far_per_day<={args.far_max}, starts_per_event<={args.starts_max}", flush=True)

    best_feasible: dict | None = None
    best_objective: tuple = (-1.0, -1e18, -1e18)
    all_candidates: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        tune_cfg = {
            "target_far_per_day": trial.suggest_float("target_far_per_day", 0.005, 0.5, log=True),
            "min_run_points": trial.suggest_int("min_run_points", 2, 8),
            "ema_alpha": trial.suggest_float("ema_alpha", 0.02, 0.20),
            "cooldown_hours": trial.suggest_float("cooldown_hours", 4.0, 168.0),
            "gate_mode": trial.suggest_categorical("gate_mode", ["relaxed", "score_ema"]),
            "rearm_window_minutes": trial.suggest_categorical("rearm_window_minutes", [30.0, 60.0, 120.0, 240.0, 480.0, 960.0]),
            "hysteresis_scale": trial.suggest_float("hysteresis_scale", 0.3, 0.9),
        }
        s_tune = _summary_for_split(intervals_df, scores_df, tune_cfg, tune_split, args.prestart_hours)
        s_test = _summary_for_split(intervals_df, scores_df, tune_cfg, "test", args.prestart_hours)

        far_tune = float(s_tune.get("false_alarms_per_day", 1e9)) if s_tune else 1e9
        starts_tune = float(s_tune.get("avg_starts_per_interval", 1e9)) if s_tune else 1e9
        hit_tune = float(s_tune.get("hit_rate", 0.0)) if s_tune else 0.0
        far_test = float(s_test.get("false_alarms_per_day", 1e9)) if s_test else 1e9
        starts_test = float(s_test.get("avg_starts_per_interval", 1e9)) if s_test else 1e9
        hit_test = float(s_test.get("hit_rate", 0.0)) if s_test else 0.0

        passes_tune = (far_tune <= args.far_max) and (starts_tune <= args.starts_max) and hit_tune >= 1.0
        passes_test = (far_test <= args.far_max) and (starts_test <= args.starts_max) and hit_test >= 1.0
        feasible = passes_tune and (passes_test if args.require_test_pass else True)

        obj_tune = _objective_value(s_tune)
        obj_test = _objective_value(s_test)
        composite = (
            (hit_tune + hit_test) / 2.0,
            -(far_tune + far_test),
            -(starts_tune + starts_test),
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
        if feasible and composite > best_objective:
            best_objective = composite
            best_feasible = record

        return composite[0] + 0.01 * composite[1]

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, show_progress_bar=False)

    out_dir = PROJECT_ROOT / "artifacts" / "results" / "transfers" / "alma_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    if best_feasible is not None:
        chosen = best_feasible
        print(f"[sweep_alma] best feasible: passes_tune=True passes_test={chosen['passes_test']}", flush=True)
    else:
        chosen = max(all_candidates, key=lambda c: tuple(c["objective"]))
        print("[sweep_alma] WARNING: no feasible candidate -> best-effort", flush=True)

    selected_path = out_dir / f"{args.anomaly}_alma_sweep_selected.json"
    payload = {
        "anomaly": args.anomaly,
        "detector": args.detector,
        "config": chosen["config"],
        "summary": chosen["summary"],
        "summary_test": chosen.get("summary_test", {}),
        "passes": chosen["passes"],
        "passes_test": chosen.get("passes_test", False),
        "sweep_meta": {
            "trials_run": len(all_candidates),
            "trials_budget": args.trials,
            "timeout_s": args.timeout,
            "tune_split": tune_split,
            "far_max": args.far_max,
            "starts_max": args.starts_max,
        },
    }
    selected_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[sweep_alma] selected -> {selected_path}", flush=True)

    if args.write_config and best_feasible is not None:
        prod_cfg_path = PROJECT_ROOT / "db" / f"{args.anomaly}_{args.detector}_config.json"
        existing = json.loads(prod_cfg_path.read_text(encoding="utf-8")) if prod_cfg_path.exists() else {}
        merged = copy.deepcopy(existing)
        merged.update(chosen["config"])
        # preserve any non-onset keys (fusion weights, signature weights, etc.)
        prod_cfg_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"[sweep_alma] production config written -> {prod_cfg_path}", flush=True)


if __name__ == "__main__":
    main()
