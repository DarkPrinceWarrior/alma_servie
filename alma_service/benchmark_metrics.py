from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alma_service.tabular_io import read_table, write_table


def load_intervals(path: str | Path) -> pd.DataFrame:
    df = read_table(
        path,
        dtypes={"well_id": str},
        parse_dates=["start_date", "end_date", "data_start", "data_end"],
    )
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
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


def load_predicted_starts(path: str | Path, *, actionable_only: bool = True) -> pd.DataFrame:
    df = read_table(path, dtypes={"well_id": str}, parse_dates=["detected_time"])
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    if "split" not in df.columns:
        df["split"] = "train"
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    if actionable_only and "actionable_alert" in df.columns:
        df = df[df["actionable_alert"].fillna(True).astype(bool)].copy()
    return df.dropna(subset=["well_id", "detected_time"]).sort_values(["well_id", "detected_time"])


def load_scores(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["well_id", "timestamp"])
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["well_id", "timestamp"])
    df = read_table(p, dtypes={"well_id": str}, parse_dates=["timestamp"])
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
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


def _median_step(timestamps: pd.Series) -> pd.Timedelta | None:
    values = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()
    if len(values) < 2:
        return None
    deltas = values.diff().dropna()
    if deltas.empty:
        return None
    seconds = float(deltas.dt.total_seconds().median())
    if not np.isfinite(seconds) or seconds <= 0:
        return None
    return pd.Timedelta(seconds=seconds)


def _false_alarm_episode_gap(
    score_timestamps: pd.Series,
    predicted_timestamps: pd.Series,
    prestart_hours: float,
) -> pd.Timedelta:
    step = _median_step(score_timestamps)
    if step is None:
        step = _median_step(predicted_timestamps)
    if step is None:
        return pd.Timedelta(hours=float(prestart_hours))
    return max(pd.Timedelta(hours=float(prestart_hours)), step * 6)


def _count_time_episodes(times: list[pd.Timestamp], max_gap: pd.Timedelta) -> int:
    if not times:
        return 0
    ordered = sorted(pd.Timestamp(ts) for ts in times)
    episodes = 1
    previous = ordered[0]
    for ts in ordered[1:]:
        if ts - previous > max_gap:
            episodes += 1
        previous = ts
    return episodes


def _safe_metric(value: Any, large: float = 1e9) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return large
    if not np.isfinite(numeric):
        return large
    return numeric


def _bool_series(values: pd.Series, *, default: bool) -> pd.Series:
    if values.dtype == object:
        normalized = values.fillna(str(default)).astype(str).str.strip().str.lower()
        return normalized.isin({"1", "true", "yes", "y", "да"})
    return values.fillna(default).astype(bool)


def _score_assessment(scores: pd.DataFrame) -> tuple[bool, str]:
    if scores.empty or "score_valid" not in scores.columns:
        return True, ""
    valid = _bool_series(scores["score_valid"], default=True)
    if bool(valid.any()):
        return True, ""
    if "score_unavailable_reason" not in scores.columns:
        return False, "unknown"
    reasons = (
        scores["score_unavailable_reason"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    reasons = reasons[reasons != ""]
    if reasons.empty:
        return False, "unknown"
    return False, str(reasons.mode().iloc[0])


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
    false_alarm_episodes = 0
    duplicate_starts_inside_interval = 0
    alerts_inside_intervals = 0
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

        well_assessed, score_unavailable_reason = _score_assessment(ws)
        if pd.notna(obs_start) and pd.notna(obs_end) and obs_end > obs_start:
            observed_days_total += (obs_end - obs_start).total_seconds() / 86400.0

        predicted_times = [pd.Timestamp(ts) for ts in wp["detected_time"].tolist()]
        false_alarm_times: list[pd.Timestamp] = []
        for ts in predicted_times:
            if not _is_inside_any(ts, wi, prestart_hours=prestart_hours):
                false_alarms += 1
                false_alarm_times.append(ts)
        false_alarm_episodes += _count_time_episodes(
            false_alarm_times,
            _false_alarm_episode_gap(
                score_timestamps=ws["timestamp"] if "timestamp" in ws.columns else pd.Series(dtype="datetime64[ns]"),
                predicted_timestamps=wp["detected_time"]
                if "detected_time" in wp.columns
                else pd.Series(dtype="datetime64[ns]"),
                prestart_hours=prestart_hours,
            ),
        )

        for _, row in wi.iterrows():
            start_dt = row["start_date"]
            end_dt = row["end_date"]
            pre_tol = pd.Timedelta(hours=float(prestart_hours))
            interval_starts = [ts for ts in predicted_times if start_dt - pre_tol <= ts <= end_dt]
            interval_start_count = len(interval_starts)
            interval_duplicates = max(interval_start_count - 1, 0)
            alerts_inside_intervals += interval_start_count
            duplicate_starts_inside_interval += interval_duplicates
            first_hit = min(interval_starts) if well_assessed and interval_starts else pd.NaT
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
                    "status": (
                        "Detected"
                        if pd.notna(first_hit)
                        else ("Not assessed" if not well_assessed else "Not found")
                    ),
                    "assessed": bool(well_assessed),
                    "score_valid": bool(well_assessed),
                    "score_unavailable_reason": score_unavailable_reason,
                    "delay_hours": delay_h,
                    "abs_delay_hours": abs(delay_h) if np.isfinite(delay_h) else np.nan,
                    "delay_ratio": delay_ratio,
                    "first_alert_delay_hours": delay_h,
                    "first_alert_abs_delay_hours": abs(delay_h) if np.isfinite(delay_h) else np.nan,
                    "first_alert_delay_ratio": delay_ratio,
                    "alert_count_inside_interval": int(interval_start_count),
                    "duplicate_starts_inside_interval": int(interval_duplicates),
                    "interval_hours": interval_hours,
                }
            )

    interval_df = pd.DataFrame(interval_rows)
    if interval_df.empty:
        summary = {
            "interval_count": 0,
            "hit_count": 0,
            "hit_rate": 0.0,
            "assessed_interval_count": 0,
            "not_assessed_interval_count": 0,
            "coverage_rate": 0.0,
            "hit_rate_on_assessed": 0.0,
            "start_count": start_count,
            "false_alarms": int(false_alarms),
            "false_alarms_per_day": float(false_alarms / observed_days_total) if observed_days_total > 0 else np.nan,
            "false_alarm_episodes": int(false_alarm_episodes),
            "episode_false_alarms": int(false_alarm_episodes),
            "episode_far": float(false_alarm_episodes / observed_days_total) if observed_days_total > 0 else np.nan,
            "first_alert_far": float(false_alarm_episodes / observed_days_total)
            if observed_days_total > 0
            else np.nan,
            "observed_days_total": float(observed_days_total),
            "avg_starts_per_interval": 0.0,
            "alerts_inside_intervals": int(alerts_inside_intervals),
            "alerts_per_detected_interval": 0.0,
            "duplicate_starts_inside_interval": int(duplicate_starts_inside_interval),
            "duplicate_starts_per_detected_interval": 0.0,
            "suppressed_rearms_count": int(
                max(start_count - false_alarm_episodes, 0)
            ),
            "delay_mae_hours": np.nan,
            "median_abs_delay_hours": np.nan,
            "p90_abs_delay_hours": np.nan,
            "first_alert_delay_median_hours": np.nan,
            "first_alert_delay_p90_hours": np.nan,
            "median_delay_ratio": np.nan,
            "p90_delay_ratio": np.nan,
        }
        return summary, interval_df

    detected = interval_df[interval_df["hit"] == 1].copy()
    abs_delays = detected["abs_delay_hours"].dropna().astype(float)
    delay_ratios = detected["delay_ratio"].dropna().astype(float)
    interval_count = int(len(interval_df))
    assessed_mask = _bool_series(interval_df.get("assessed", pd.Series(True, index=interval_df.index)), default=True)
    assessed_interval_count = int(assessed_mask.sum())
    not_assessed_interval_count = int(interval_count - assessed_interval_count)
    hit_count = int(interval_df["hit"].sum())
    summary = {
        "interval_count": interval_count,
        "hit_count": hit_count,
        "hit_rate": float(hit_count / interval_count) if interval_count else 0.0,
        "assessed_interval_count": assessed_interval_count,
        "not_assessed_interval_count": not_assessed_interval_count,
        "coverage_rate": float(assessed_interval_count / interval_count) if interval_count else 0.0,
        "hit_rate_on_assessed": (
            float(hit_count / assessed_interval_count) if assessed_interval_count else 0.0
        ),
        "start_count": start_count,
        "false_alarms": int(false_alarms),
        "false_alarms_per_day": float(false_alarms / observed_days_total) if observed_days_total > 0 else np.nan,
        "false_alarm_episodes": int(false_alarm_episodes),
        "episode_false_alarms": int(false_alarm_episodes),
        "episode_far": float(false_alarm_episodes / observed_days_total) if observed_days_total > 0 else np.nan,
        "first_alert_far": float(false_alarm_episodes / observed_days_total)
        if observed_days_total > 0
        else np.nan,
        "observed_days_total": float(observed_days_total),
        "avg_starts_per_interval": float(start_count / interval_count) if interval_count else 0.0,
        "alerts_inside_intervals": int(alerts_inside_intervals),
        "alerts_per_detected_interval": float(alerts_inside_intervals / hit_count) if hit_count else 0.0,
        "duplicate_starts_inside_interval": int(duplicate_starts_inside_interval),
        "duplicate_starts_per_detected_interval": float(duplicate_starts_inside_interval / hit_count)
        if hit_count
        else 0.0,
        "suppressed_rearms_count": int(
            max(start_count - hit_count - false_alarm_episodes, 0)
        ),
        "delay_mae_hours": float(np.mean(abs_delays)) if len(abs_delays) else np.nan,
        "median_abs_delay_hours": float(np.median(abs_delays)) if len(abs_delays) else np.nan,
        "p90_abs_delay_hours": float(np.quantile(abs_delays, 0.90)) if len(abs_delays) else np.nan,
        "first_alert_delay_median_hours": float(np.median(abs_delays)) if len(abs_delays) else np.nan,
        "first_alert_delay_p90_hours": float(np.quantile(abs_delays, 0.90)) if len(abs_delays) else np.nan,
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


def write_interval_frame(path: str | Path, df: pd.DataFrame) -> None:
    write_table(df, path)
