from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.benchmark_metrics import (
    load_intervals,
    load_predicted_starts,
    load_scores,
    summarize_splits,
)
from alma_service.onset_detection import (
    calibrate_causal_thresholds,
    detect_causal_onsets,
)


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
        if ref_end_idx < 16:
            ref_end_idx = min(max(16, ref_end_idx), len(s))
        try:
            thresholds, diagnostics = calibrate_causal_thresholds(
                s,
                ts,
                reference_end_idx=ref_end_idx,
                target_far_per_day=cfg["target_far_per_day"],
                min_run_points=cfg["min_run_points"],
                ema_alpha=cfg["ema_alpha"],
            )
            starts = detect_causal_onsets(
                s,
                ts,
                diagnostics,
                thresholds,
                reference_end_idx=ref_end_idx,
                min_run_points=cfg["min_run_points"],
                cooldown_hours=cfg["cooldown_hours"],
                gate_mode=cfg.get("gate_mode", "relaxed"),
                rearm_window_minutes=cfg.get("rearm_window_minutes", 60.0),
                hysteresis_scale=cfg.get("hysteresis_scale", 0.6),
            )
        except Exception as exc:
            print(f"[onset] {well_id} failed: {exc}", flush=True)
            continue
        for t in starts:
            rows.append({"well_id": well_id, "detected_time": pd.Timestamp(t), "split": split})
    return pd.DataFrame(rows)


def _objective_passes_constraints(
    summary: dict,
    far_max: float,
    starts_max: float,
    norm_far_max_relative: float,
    norm_baseline: float,
) -> tuple[bool, str]:
    if not summary:
        return False, "no_summary"
    far = float(summary.get("false_alarms_per_day", float("inf")))
    starts = float(summary.get("avg_starts_per_interval", float("inf")))
    if far > far_max:
        return False, f"far_per_day={far:.3f}>{far_max}"
    if starts > starts_max:
        return False, f"starts_per_event={starts:.2f}>{starts_max}"
    return True, "ok"


def _objective_value(summary: dict) -> tuple[float, float, float]:
    hit = float(summary.get("hit_rate", 0.0))
    median_delay = float(summary.get("first_alert_delay_median_hours", 1e9) or 1e9)
    p90 = float(summary.get("first_alert_delay_p90_hours", 1e9) or 1e9)
    return hit, -median_delay, -p90


def tune(args, cfg: dict) -> dict:
    event_class = args.event_class
    scores_path = PROJECT_ROOT / "artifacts" / "3w" / "scores" / f"class_{event_class}_scores.parquet"
    intervals_path = PROJECT_ROOT / cfg["dataset"]["intervals_path"]
    scores_df = load_scores(str(scores_path))
    intervals = load_intervals(str(intervals_path))
    intervals = intervals[intervals.get("folder_label", event_class).astype(int) == event_class].copy() if "folder_label" in intervals.columns else intervals
    val_mask = scores_df["split"].eq("val") | scores_df["split"].eq("test")
    eval_split = "val" if scores_df["split"].eq("val").any() else "test"
    print(f"[tune] event_class={event_class}  tuning on split={eval_split}", flush=True)
    sub_scores = scores_df[scores_df["split"].eq(eval_split)].copy()
    sub_intervals = intervals[intervals["split"].eq(eval_split)].copy()
    grids = cfg["thresholding"]["tune_grids"]
    combos = list(
        itertools.product(
            grids["target_far_per_day"],
            grids["min_run_points"],
            grids["ema_alpha"],
            grids["cooldown_hours"],
        )
    )
    print(f"[tune] grid size: {len(combos)}", flush=True)
    far_max = cfg["thresholding"]["constraint_far_per_day_max"]
    starts_max = cfg["thresholding"]["constraint_starts_per_event_max"]
    candidates: list[dict] = []
    for i, (far, runlen, alpha, cooldown) in enumerate(combos, 1):
        tune_cfg = {
            "target_far_per_day": float(far),
            "min_run_points": int(runlen),
            "ema_alpha": float(alpha),
            "cooldown_hours": float(cooldown),
            "gate_mode": "relaxed",
            "rearm_window_minutes": 60.0,
            "hysteresis_scale": 0.6,
        }
        starts_df = _detect_per_well(sub_scores, tune_cfg)
        try:
            summaries, _ = summarize_splits(
                intervals=sub_intervals,
                predictions=starts_df,
                scores=sub_scores,
                prestart_hours=cfg["evaluation"].get("lead_tolerance_hours", 24.0),
            )
        except Exception as exc:
            summaries = {}
            print(f"[tune] combo {i} summarize_splits failed: {exc}", flush=True)
        summary = summaries.get(eval_split, summaries.get("all", {}))
        passes, reason = _objective_passes_constraints(
            summary, far_max=far_max, starts_max=starts_max,
            norm_far_max_relative=cfg["thresholding"]["constraint_norm_far_max_relative"],
            norm_baseline=0.0,
        )
        obj = _objective_value(summary) if summary else (0.0, -1e9, -1e9)
        candidates.append(
            {
                "config": tune_cfg,
                "summary": summary,
                "passes": passes,
                "reason": reason,
                "objective": obj,
            }
        )
        if i % 10 == 0 or i == len(combos):
            print(f"[tune] {i}/{len(combos)} latest_obj={obj} pass={passes}", flush=True)
    passing = [c for c in candidates if c["passes"]]
    if passing:
        best = max(passing, key=lambda c: c["objective"])
        print(f"[tune] selected from {len(passing)} feasible candidates", flush=True)
    else:
        best = max(candidates, key=lambda c: c["objective"])
        print(f"[tune] WARNING: no feasible candidate, picked best-effort", flush=True)
    out_dir = PROJECT_ROOT / "artifacts" / "3w" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"class_{event_class}_selected.json"
    out_path.write_text(json.dumps(best, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[tune] saved -> {out_path}", flush=True)
    return best


def evaluate(args, cfg: dict) -> dict:
    event_class = args.event_class
    scores_path = PROJECT_ROOT / "artifacts" / "3w" / "scores" / f"class_{event_class}_scores.parquet"
    intervals_path = PROJECT_ROOT / cfg["dataset"]["intervals_path"]
    metrics_dir = PROJECT_ROOT / "artifacts" / "3w" / "metrics"
    selected_path = metrics_dir / f"class_{event_class}_selected.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"Run --mode tune first; missing {selected_path}")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    tune_cfg = selected["config"]
    scores_df = load_scores(str(scores_path))
    intervals = load_intervals(str(intervals_path))
    intervals = intervals[intervals.get("folder_label", event_class).astype(int) == event_class].copy() if "folder_label" in intervals.columns else intervals
    starts_df = _detect_per_well(scores_df, tune_cfg)
    starts_path = PROJECT_ROOT / "artifacts" / "3w" / "scores" / f"class_{event_class}_predicted_starts.parquet"
    if starts_df.empty:
        starts_df = pd.DataFrame(columns=["well_id", "detected_time", "split"])
    starts_df.to_parquet(starts_path, engine="pyarrow", compression="brotli")
    print(f"[eval] starts -> {starts_path}  n={len(starts_df)}", flush=True)
    summaries, _ = summarize_splits(
        intervals=intervals,
        predictions=starts_df,
        scores=scores_df,
        prestart_hours=cfg["evaluation"].get("lead_tolerance_hours", 24.0),
    )
    out = {
        "event_class": int(event_class),
        "config": tune_cfg,
        "prestart_hours": cfg["evaluation"].get("lead_tolerance_hours", 24.0),
        "splits": summaries,
    }
    out_path = metrics_dir / f"class_{event_class}_metrics.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[eval] metrics -> {out_path}", flush=True)
    print(json.dumps({k: v for k, v in summaries.items()}, indent=2, ensure_ascii=False, default=str))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune thresholds on val and evaluate on test for one 3W event class.")
    parser.add_argument("--config", default="configs/3w_paano.json")
    parser.add_argument("--event-class", type=int, required=True)
    parser.add_argument("--mode", choices=["tune", "evaluate", "both"], default="both")
    args = parser.parse_args()
    cfg = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    if args.mode in ("tune", "both"):
        tune(args, cfg)
    if args.mode in ("evaluate", "both"):
        evaluate(args, cfg)


if __name__ == "__main__":
    main()
