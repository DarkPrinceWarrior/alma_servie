from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def collect_per_class(cfg: dict) -> list[dict]:
    event_labels = {int(k): v for k, v in cfg["dataset"]["event_labels"].items()}
    metrics_dir = PROJECT_ROOT / "artifacts" / "3w" / "metrics"
    rows: list[dict] = []
    for cls in sorted(event_labels.keys()):
        if cls == 0:
            continue
        metrics_path = metrics_dir / f"class_{cls}_metrics.json"
        if not metrics_path.exists():
            rows.append({"class": cls, "name": event_labels[cls], "status": "missing"})
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        cfg_used = metrics.get("config", {})
        for split_name, summary in metrics.get("splits", {}).items():
            if not summary:
                continue
            far = float(summary.get("false_alarms_per_day", float("nan")))
            starts = float(summary.get("avg_starts_per_interval", float("nan")))
            far_max = cfg["thresholding"]["constraint_far_per_day_max"]
            starts_max = cfg["thresholding"]["constraint_starts_per_event_max"]
            passes = (far <= far_max) and (starts <= starts_max)
            rows.append(
                {
                    "class": cls,
                    "name": event_labels[cls],
                    "split": split_name,
                    "interval_count": int(summary.get("interval_count", 0)),
                    "hit_rate": float(summary.get("hit_rate", 0.0) or 0.0),
                    "far_per_day": far,
                    "starts_per_event": starts,
                    "median_delay_h": float(summary.get("first_alert_delay_median_hours", float("nan")) or float("nan")),
                    "p90_delay_h": float(summary.get("first_alert_delay_p90_hours", float("nan")) or float("nan")),
                    "observed_days": float(summary.get("observed_days_total", 0.0) or 0.0),
                    "passes_constraints": bool(passes),
                    "tune_target_far": cfg_used.get("target_far_per_day"),
                    "tune_min_run": cfg_used.get("min_run_points"),
                    "tune_ema_alpha": cfg_used.get("ema_alpha"),
                    "tune_cooldown_h": cfg_used.get("cooldown_hours"),
                    "status": "ok",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-class 3W metrics into a sweep summary.")
    parser.add_argument("--config", default="configs/3w_paano.json")
    parser.add_argument("--out", default="artifacts/results/3w_benchmark_summary.json")
    args = parser.parse_args()

    cfg = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    rows = collect_per_class(cfg)
    df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    test_df = df[df.get("split", "") == "test"].copy() if len(df) and "split" in df.columns else pd.DataFrame()

    summary = {
        "config": args.config,
        "n_classes_with_metrics": int(test_df["class"].nunique()) if len(test_df) else 0,
        "test_aggregate": {
            "mean_hit_rate": float(test_df["hit_rate"].mean()) if len(test_df) else None,
            "mean_far_per_day": float(test_df["far_per_day"].mean()) if len(test_df) else None,
            "mean_starts_per_event": float(test_df["starts_per_event"].mean()) if len(test_df) else None,
            "median_delay_p50_h_aggregate": float(test_df["median_delay_h"].median()) if len(test_df) else None,
            "classes_passing_constraints": int(test_df["passes_constraints"].sum()) if len(test_df) else 0,
        },
        "rows": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[aggregate] {out_path}", flush=True)

    if len(df):
        cols = ["class", "name", "split", "interval_count", "hit_rate", "far_per_day",
                "starts_per_event", "median_delay_h", "p90_delay_h", "passes_constraints"]
        avail = [c for c in cols if c in df.columns]
        print(df[avail].to_string(index=False))


if __name__ == "__main__":
    main()
