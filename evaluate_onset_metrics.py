"""
Evaluate onset-detection quality from predicted anomaly starts.

Metrics:
- Interval hit rate
- Start-delay MAE / median / P90 (hours)
- False alarms per day
- Early false alarms per day
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_intervals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"well_id": str})
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    if "interval_idx" not in df.columns:
        df["interval_idx"] = df.groupby("well_id").cumcount() + 1
    if "data_start" in df.columns:
        df["data_start"] = pd.to_datetime(df["data_start"], errors="coerce")
    else:
        df["data_start"] = pd.NaT
    if "data_end" in df.columns:
        df["data_end"] = pd.to_datetime(df["data_end"], errors="coerce")
    else:
        df["data_end"] = pd.NaT
    return df.dropna(subset=["well_id", "start_date", "end_date"]).sort_values(
        ["well_id", "start_date", "interval_idx"]
    )


def load_predicted_starts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"well_id": str})
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["detected_time"] = pd.to_datetime(df["detected_time"], errors="coerce")
    df = df.dropna(subset=["well_id", "detected_time"]).sort_values(["well_id", "detected_time"])
    return df


def load_scores(path: str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["well_id", "timestamp"])
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["well_id", "timestamp"])
    df = pd.read_csv(path, dtype={"well_id": str})
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["well_id", "timestamp"])
    return df


def _is_inside_any(ts: pd.Timestamp, intervals: pd.DataFrame) -> bool:
    for _, r in intervals.iterrows():
        if r["start_date"] <= ts <= r["end_date"]:
            return True
    return False


def evaluate(
    intervals: pd.DataFrame,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    interval_rows = []
    false_alarms = 0
    early_false_alarms = 0
    observed_days_total = 0.0
    pre_anomaly_days_total = 0.0

    all_wells = sorted(set(intervals["well_id"].unique()).union(predictions["well_id"].unique()))

    for wid in all_wells:
        wi = intervals[intervals["well_id"] == wid].sort_values(["start_date", "interval_idx"])
        wp = predictions[predictions["well_id"] == wid].sort_values("detected_time")
        ws = scores[scores["well_id"] == wid].sort_values("timestamp")

        if not wi.empty and wi["data_start"].notna().any() and wi["data_end"].notna().any():
            obs_start = wi["data_start"].dropna().min()
            obs_end = wi["data_end"].dropna().max()
        elif not ws.empty:
            obs_start = ws["timestamp"].min()
            obs_end = ws["timestamp"].max()
        elif not wi.empty:
            obs_start = wi["start_date"].min()
            obs_end = wi["end_date"].max()
        elif not wp.empty:
            obs_start = wp["detected_time"].min()
            obs_end = wp["detected_time"].max()
        else:
            obs_start = pd.NaT
            obs_end = pd.NaT

        if pd.notna(obs_start) and pd.notna(obs_end) and obs_end > obs_start:
            observed_days_total += (obs_end - obs_start).total_seconds() / 86400.0

        if not wi.empty and pd.notna(obs_start):
            first_start = wi["start_date"].min()
            if first_start > obs_start:
                pre_anomaly_days_total += (first_start - obs_start).total_seconds() / 86400.0

        pred_list = list(wp["detected_time"].to_numpy())
        for ts in pred_list:
            ts = pd.Timestamp(ts)
            inside = _is_inside_any(ts, wi)
            if not inside:
                false_alarms += 1
                if wi.empty or ts < wi["start_date"].min():
                    early_false_alarms += 1

        for _, row in wi.iterrows():
            start_dt = row["start_date"]
            end_dt = row["end_date"]
            inside = [pd.Timestamp(ts) for ts in pred_list if start_dt <= pd.Timestamp(ts) <= end_dt]
            first_hit = min(inside) if inside else pd.NaT
            delay_h = (
                float((first_hit - start_dt).total_seconds() / 3600.0) if pd.notna(first_hit) else np.nan
            )
            interval_rows.append(
                {
                    "well_id": wid,
                    "interval_idx": int(row.get("interval_idx", 1)),
                    "actual_start": start_dt,
                    "actual_end": end_dt,
                    "detected_time": first_hit,
                    "hit": int(pd.notna(first_hit)),
                    "delay_hours": delay_h,
                }
            )

    interval_df = pd.DataFrame(interval_rows)
    n_intervals = int(len(interval_df))
    n_hits = int(interval_df["hit"].sum()) if n_intervals else 0
    delays = interval_df["delay_hours"].dropna()

    summary = {
        "interval_count": n_intervals,
        "hit_count": n_hits,
        "hit_rate": float(n_hits / n_intervals) if n_intervals else 0.0,
        "delay_mae_hours": float(np.mean(np.abs(delays))) if len(delays) else np.nan,
        "delay_median_hours": float(np.median(delays)) if len(delays) else np.nan,
        "delay_p90_hours": float(np.quantile(delays, 0.9)) if len(delays) else np.nan,
        "false_alarms": int(false_alarms),
        "false_alarms_per_day": float(false_alarms / observed_days_total) if observed_days_total > 0 else np.nan,
        "early_false_alarms": int(early_false_alarms),
        "early_false_alarms_per_day": (
            float(early_false_alarms / pre_anomaly_days_total) if pre_anomaly_days_total > 0 else np.nan
        ),
        "observed_days_total": float(observed_days_total),
        "pre_anomaly_days_total": float(pre_anomaly_days_total),
    }
    return summary, interval_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly onset detection metrics.")
    parser.add_argument("--intervals", required=True, help="Path to *_intervals.csv")
    parser.add_argument("--predicted-starts", required=True, help="Path to *_paano_predicted_starts.csv")
    parser.add_argument("--scores", default=None, help="Path to *_paano_scores.csv (optional)")
    parser.add_argument("--name", default="run", help="Label for outputs")
    parser.add_argument("--output-prefix", default=None, help="If set, writes JSON summary and per-interval CSV")
    args = parser.parse_args()

    intervals = load_intervals(args.intervals)
    predictions = load_predicted_starts(args.predicted_starts)
    scores = load_scores(args.scores)

    summary, interval_df = evaluate(intervals, predictions, scores)
    summary["name"] = args.name

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_prefix:
        prefix = Path(args.output_prefix)
        summary_path = prefix.with_suffix(".json")
        interval_path = prefix.with_name(prefix.name + "_intervals.csv")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        interval_df.to_csv(interval_path, index=False)
        print(f"Saved: {summary_path}")
        print(f"Saved: {interval_path}")


if __name__ == "__main__":
    main()
