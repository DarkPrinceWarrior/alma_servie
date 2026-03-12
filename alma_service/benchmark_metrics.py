from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_intervals(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"well_id": str})
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    if "interval_idx" not in df.columns:
        df["interval_idx"] = df.groupby("well_id").cumcount() + 1
    df["interval_idx"] = pd.to_numeric(df["interval_idx"], errors="coerce").fillna(1).astype(int)
    if "data_start" in df.columns:
        df["data_start"] = pd.to_datetime(df["data_start"], errors="coerce")
    else:
        df["data_start"] = pd.NaT
    if "data_end" in df.columns:
        df["data_end"] = pd.to_datetime(df["data_end"], errors="coerce")
    else:
        df["data_end"] = pd.NaT
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.strip().str.lower()
    else:
        df["split"] = "train"
    return df.dropna(subset=["well_id", "start_date", "end_date"]).sort_values(
        ["well_id", "start_date", "interval_idx"]
    )


def load_predicted_starts(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"well_id": str})
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["detected_time"] = pd.to_datetime(df["detected_time"], errors="coerce")
    if "split" not in df.columns:
        df["split"] = "train"
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    return df.dropna(subset=["well_id", "detected_time"]).sort_values(["well_id", "detected_time"])


def load_scores(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["well_id", "timestamp"])
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["well_id", "timestamp"])
    df = pd.read_csv(p, dtype={"well_id": str})
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "split" not in df.columns:
        df["split"] = "train"
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    return df.dropna(subset=["well_id", "timestamp"]).sort_values(["well_id", "timestamp"])


def predicted_from_mapping(predicted: dict[str, list[pd.Timestamp]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for well_id, starts in predicted.items():
        for ts in starts:
            rows.append({"well_id": str(well_id).strip().lower(), "detected_time": pd.Timestamp(ts)})
    if not rows:
        return pd.DataFrame(columns=["well_id", "detected_time"])
    return pd.DataFrame(rows).sort_values(["well_id", "detected_time"]).reset_index(drop=True)


def _is_inside_any(ts: pd.Timestamp, intervals: pd.DataFrame, prestart_hours: float) -> bool:
    pre_tol = pd.Timedelta(hours=float(prestart_hours))
    for _, row in intervals.iterrows():
        if row["start_date"] - pre_tol <= ts <= row["end_date"]:
            return True
    return False


def select_interval_detection(
    predicted_times: list[pd.Timestamp],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    prestart_hours: float,
) -> pd.Timestamp | pd.NaT:
    pre_tol = pd.Timedelta(hours=float(prestart_hours))
    inside = [ts for ts in predicted_times if start_dt - pre_tol <= ts <= end_dt]
    return min(inside) if inside else pd.NaT


def _safe_metric(value: Any, large: float = 1e9) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return large
    if not np.isfinite(numeric):
        return large
    return numeric


def summary_score_key(summary: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(summary.get("hit_count", 0)),
        -_safe_metric(summary.get("p90_delay_ratio"), large=1e9),
        -_safe_metric(summary.get("false_alarms_per_day"), large=1e9),
        -_safe_metric(summary.get("avg_starts_per_interval"), large=1e9),
        -_safe_metric(summary.get("p90_abs_delay_hours"), large=1e9),
    )


def evaluate_predictions(
    intervals: pd.DataFrame,
    predictions: pd.DataFrame,
    scores: pd.DataFrame | None = None,
    prestart_hours: float = 2.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    scores_df = scores if scores is not None else pd.DataFrame(columns=["well_id", "timestamp"])
    interval_rows: list[dict[str, Any]] = []
    false_alarms = 0
    observed_days_total = 0.0
    start_count = int(len(predictions))

    all_wells = sorted(set(intervals["well_id"].unique()).union(predictions["well_id"].unique()))
    for well_id in all_wells:
        wi = intervals[intervals["well_id"] == well_id].sort_values(["start_date", "interval_idx"])
        wp = predictions[predictions["well_id"] == well_id].sort_values("detected_time")
        ws = scores_df[scores_df["well_id"] == well_id].sort_values("timestamp")

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

        predicted_times = [pd.Timestamp(ts) for ts in wp["detected_time"].tolist()]
        for ts in predicted_times:
            if not _is_inside_any(ts, wi, prestart_hours=prestart_hours):
                false_alarms += 1

        for _, row in wi.iterrows():
            start_dt = row["start_date"]
            end_dt = row["end_date"]
            first_hit = select_interval_detection(predicted_times, start_dt, end_dt, prestart_hours)
            delay_h = (
                float((first_hit - start_dt).total_seconds() / 3600.0) if pd.notna(first_hit) else np.nan
            )
            interval_hours = max(float((end_dt - start_dt).total_seconds() / 3600.0), 0.0)
            delay_ratio = np.nan
            if np.isfinite(delay_h):
                delay_ratio = max(delay_h, 0.0) / max(interval_hours, 1.0)
            interval_rows.append(
                {
                    "well_id": well_id,
                    "interval_idx": int(row.get("interval_idx", 1)),
                    "actual_start": start_dt,
                    "actual_end": end_dt,
                    "data_start": row.get("data_start", pd.NaT),
                    "data_end": row.get("data_end", pd.NaT),
                    "detected_time": first_hit,
                    "split": str(row.get("split", "train")).strip().lower(),
                    "hit": int(pd.notna(first_hit)),
                    "status": "Detected" if pd.notna(first_hit) else "Not found",
                    "delay_hours": delay_h,
                    "abs_delay_hours": abs(delay_h) if np.isfinite(delay_h) else np.nan,
                    "delay_ratio": delay_ratio,
                    "interval_hours": interval_hours,
                }
            )

    interval_df = pd.DataFrame(interval_rows)
    if interval_df.empty:
        summary = {
            "interval_count": 0,
            "hit_count": 0,
            "hit_rate": 0.0,
            "start_count": start_count,
            "false_alarms": int(false_alarms),
            "false_alarms_per_day": float(false_alarms / observed_days_total) if observed_days_total > 0 else np.nan,
            "observed_days_total": float(observed_days_total),
            "avg_starts_per_interval": 0.0,
            "delay_mae_hours": np.nan,
            "median_abs_delay_hours": np.nan,
            "p90_abs_delay_hours": np.nan,
            "median_delay_ratio": np.nan,
            "p90_delay_ratio": np.nan,
        }
        return summary, interval_df

    detected = interval_df[interval_df["hit"] == 1].copy()
    abs_delays = detected["abs_delay_hours"].dropna().astype(float)
    delay_ratios = detected["delay_ratio"].dropna().astype(float)
    interval_count = int(len(interval_df))
    hit_count = int(interval_df["hit"].sum())
    summary = {
        "interval_count": interval_count,
        "hit_count": hit_count,
        "hit_rate": float(hit_count / interval_count) if interval_count else 0.0,
        "start_count": start_count,
        "false_alarms": int(false_alarms),
        "false_alarms_per_day": float(false_alarms / observed_days_total) if observed_days_total > 0 else np.nan,
        "observed_days_total": float(observed_days_total),
        "avg_starts_per_interval": float(start_count / interval_count) if interval_count else 0.0,
        "delay_mae_hours": float(np.mean(abs_delays)) if len(abs_delays) else np.nan,
        "median_abs_delay_hours": float(np.median(abs_delays)) if len(abs_delays) else np.nan,
        "p90_abs_delay_hours": float(np.quantile(abs_delays, 0.90)) if len(abs_delays) else np.nan,
        "median_delay_ratio": float(np.median(delay_ratios)) if len(delay_ratios) else np.nan,
        "p90_delay_ratio": float(np.quantile(delay_ratios, 0.90)) if len(delay_ratios) else np.nan,
    }
    return summary, interval_df


def summarize_splits(
    intervals: pd.DataFrame,
    predictions: pd.DataFrame,
    scores: pd.DataFrame | None = None,
    prestart_hours: float = 2.0,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    split_summaries: dict[str, Any] = {}
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("all", "train", "test"):
        if split_name == "all":
            subset_intervals = intervals
        else:
            subset_intervals = intervals[intervals["split"].astype(str).str.lower() == split_name]
        if subset_intervals.empty:
            continue
        well_ids = subset_intervals["well_id"].unique()
        subset_predictions = predictions[predictions["well_id"].isin(well_ids)]
        subset_scores = None if scores is None else scores[scores["well_id"].isin(well_ids)]
        split_summaries[split_name], split_frames[split_name] = evaluate_predictions(
            subset_intervals,
            subset_predictions,
            scores=subset_scores,
            prestart_hours=prestart_hours,
        )
    return split_summaries, split_frames


def write_summary(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
