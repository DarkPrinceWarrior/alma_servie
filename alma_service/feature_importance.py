"""
Interval-based feature importance for anomaly reports.

The report combines three signals:
- physical evidence: how strongly a raw channel changes around a labelled
  anomaly interval relative to a local pre-anomaly baseline;
- model alignment: how well channel deviations align with the detector score
  and the detected onset;
- stability: how consistently the evidence appears across intervals.

This replaces the old channel-ablation report. It does not retrain detectors
and does not use legacy pct_drop summaries.
"""
from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alma_service.anomaly_specs import DetectionSpec, get_detection_spec
from alma_service.benchmark_metrics import load_intervals as load_intervals_df
from alma_service.detection_artifacts import (
    DEFAULT_DETECTOR,
    benchmark_summary_path,
    load_json,
    normalize_detector_key,
    predicted_starts_path,
    results_path,
    scores_path,
)
from alma_service.paths import DB_DIR, REPORTS_DIR, ensure_parent
from alma_service.tabular_io import read_table

FI_SUMMARY_VERSION = 2
PRESSURE_COL = "Давление на приеме насоса кгс/см²"
FREQ_COL = "Выходная частота"
META_COLUMNS = {"well_id", "timestamp", "split", "interval_idx", "anomaly_type"}

DETECTOR_LABELS = {
    "paano_shared": "PaAno Shared Encoder",
    "paano_global": "PaAno Global Encoder",
}

DISPLAY_NAMES = {
    "negermet": "Негерметичность НКТ",
    "pritok": "Изменение притока",
    "salt": "Солеотложение",
}

COLOR_THEMES = {
    "negermet": {"accent": "#c0392b", "soft_bg": "#fdf2f2"},
    "pritok": {"accent": "#e67e22", "soft_bg": "#fef5ed"},
    "salt": {"accent": "#0f766e", "soft_bg": "#f0fdfa"},
}

ANOMALY_WINDOWS = {
    "negermet": {
        "baseline_max_hours": 72.0,
        "baseline_min_points": 24,
        "onset_min_hours": 1.0,
        "onset_max_hours": 12.0,
    },
    "pritok": {
        "baseline_max_hours": 24.0 * 30.0,
        "baseline_min_points": 72,
        "onset_min_hours": 12.0,
        "onset_max_hours": 24.0 * 3.0,
    },
    "salt": {
        "baseline_max_hours": 24.0 * 30.0,
        "baseline_min_points": 72,
        "onset_min_hours": 12.0,
        "onset_max_hours": 24.0 * 5.0,
    },
}


@dataclass(frozen=True)
class IntervalChannelEvidence:
    interval_idx: int
    start: pd.Timestamp
    end: pd.Timestamp
    baseline_points: int
    anomaly_points: int
    baseline_median: float
    anomaly_median: float
    onset_median: float
    baseline_mad: float
    level_delta: float
    level_delta_pct: float | None
    robust_shift: float
    onset_robust_shift: float
    slope_robust_shift: float
    physical_score: float
    model_score: float
    detection_score: float
    final_score: float
    direction: str


def _resolve_detector(spec: DetectionSpec, detector: str | None) -> str:
    if detector:
        return normalize_detector_key(detector)
    benchmark = load_json(benchmark_summary_path(spec))
    selected = benchmark.get("selected_default_detector")
    if selected:
        try:
            return normalize_detector_key(selected)
        except ValueError:
            pass
    return DEFAULT_DETECTOR


def _load_source(spec: DetectionSpec, source_path: str | None, detector_key: str | None = None) -> Path:
    if source_path is not None:
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")
        return src
    if detector_key == "paano_global":
        from alma_service.global_normality import configured_anomaly_source_path

        src = configured_anomaly_source_path(spec.anomaly_key)
        if not src.exists():
            raise FileNotFoundError(f"Global detector source file not found: {src}")
        return src
    for name in spec.dataset.source_candidates:
        src = DB_DIR / name
        if src.exists():
            return src
    raise FileNotFoundError(f"No source dataset found for {spec.anomaly_key}")


def _load_timeseries(spec: DetectionSpec, source_path: str | None, detector_key: str | None = None) -> pd.DataFrame:
    df = read_table(
        _load_source(spec, source_path, detector_key=detector_key),
        dtypes={"well_id": str},
        parse_dates=["timestamp"],
        low_memory=False,
    )
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    return df.dropna(subset=["timestamp"]).sort_values(["well_id", "timestamp"]).reset_index(drop=True)


def _load_intervals(spec: DetectionSpec) -> pd.DataFrame:
    if not spec.dataset.intervals_path.exists():
        raise RuntimeError(f"Intervals file not found: {spec.dataset.intervals_path}")
    intervals = load_intervals_df(spec.dataset.intervals_path)
    intervals["well_id"] = intervals["well_id"].astype(str).str.strip().str.lower()
    return intervals.sort_values(["well_id", "start_date", "interval_idx"]).reset_index(drop=True)


def _load_scores(spec: DetectionSpec, detector_key: str) -> pd.DataFrame:
    path = scores_path(spec, detector_key)
    if not path.exists():
        raise RuntimeError(f"Scores file not found: {path}. Run detection first.")
    df = read_table(path, dtypes={"well_id": str}, parse_dates=["timestamp"])
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    return df.dropna(subset=["timestamp"]).sort_values(["well_id", "timestamp"]).reset_index(drop=True)


def _load_detection_results(spec: DetectionSpec, detector_key: str) -> pd.DataFrame:
    path = results_path(spec, detector_key)
    if not path.exists():
        return pd.DataFrame()
    df = read_table(
        path,
        dtypes={"well_id": str},
        parse_dates=["actual_start", "actual_end", "detected_time", "data_start", "data_end"],
    )
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    if "interval_idx" not in df.columns:
        df["interval_idx"] = df.groupby("well_id").cumcount() + 1
    return df.sort_values(["well_id", "interval_idx"]).reset_index(drop=True)


def _load_predictions(spec: DetectionSpec, detector_key: str) -> pd.DataFrame:
    path = predicted_starts_path(spec, detector_key)
    if not path.exists():
        return pd.DataFrame(columns=["well_id", "detected_time", "split"])
    df = read_table(path, dtypes={"well_id": str}, parse_dates=["detected_time"])
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    return df.dropna(subset=["detected_time"]).sort_values(["well_id", "detected_time"]).reset_index(drop=True)


def _score_column(scores_df: pd.DataFrame) -> str:
    if "score" in scores_df.columns:
        return "score"
    if "paano_score" in scores_df.columns:
        return "paano_score"
    for column in scores_df.columns:
        if column.endswith("_score"):
            return column
    raise RuntimeError("No score column found in scores parquet.")


def _numeric_channels(df: pd.DataFrame) -> list[str]:
    channels: list[str] = []
    for column in df.columns:
        if column in META_COLUMNS:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        if series.notna().sum() >= 10:
            channels.append(column)
    return channels


def _window_config(anomaly_key: str) -> dict[str, float]:
    return ANOMALY_WINDOWS.get(anomaly_key, ANOMALY_WINDOWS["salt"])


def _infer_step_hours(timestamps: pd.Series) -> float:
    ts = pd.to_datetime(timestamps).sort_values()
    diffs = ts.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1.0
    return max(float(diffs.median()) / 3600.0, 1.0 / 3600.0)


def _robust_mad(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad <= 1e-12:
        std = float(np.std(arr))
        return std if std > 1e-12 else 1.0
    return mad


def _safe_pct(delta: float, baseline: float) -> float | None:
    if not np.isfinite(delta) or not np.isfinite(baseline) or abs(baseline) < 1e-9:
        return None
    return 100.0 * delta / abs(baseline)


def _clip_score(value: float, scale: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(abs(value) * scale, 0.0, 100.0))


def _channel_group(channel: str) -> str:
    lower = channel.lower()
    if "давление" in lower:
        return "pressure"
    if "температур" in lower:
        return "temperature"
    if "вибрац" in lower:
        return "vibration"
    if "напряж" in lower or "ua" in lower or "ub" in lower or "uc" in lower:
        return "voltage"
    if "ток" in lower:
        return "current"
    if "мощность" in lower or "cos" in lower or "загруз" in lower:
        return "power_load"
    if "частот" in lower:
        return "frequency"
    if "дисбаланс" in lower:
        return "imbalance"
    return "other"


def _direction(delta: float) -> str:
    if not np.isfinite(delta) or abs(delta) < 1e-9:
        return "flat"
    return "up" if delta > 0 else "down"


def _adaptive_baseline(
    well_df: pd.DataFrame,
    start: pd.Timestamp,
    cfg: dict[str, float],
    step_hours: float,
) -> pd.DataFrame:
    max_hours = float(cfg["baseline_max_hours"])
    min_points = int(cfg["baseline_min_points"])
    before = well_df[well_df["timestamp"] < start].copy()
    if before.empty:
        return before
    candidate = before[before["timestamp"] >= start - pd.Timedelta(hours=max_hours)].copy()
    if len(candidate) >= min_points:
        return candidate
    needed_hours = max(max_hours, min_points * step_hours)
    return before[before["timestamp"] >= start - pd.Timedelta(hours=needed_hours)].copy()


def _onset_window(
    well_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: dict[str, float],
) -> pd.DataFrame:
    interval_hours = max((end - start).total_seconds() / 3600.0, 0.0)
    onset_hours = max(float(cfg["onset_min_hours"]), min(float(cfg["onset_max_hours"]), interval_hours * 0.25))
    onset_end = min(end, start + pd.Timedelta(hours=onset_hours))
    return well_df[(well_df["timestamp"] >= start) & (well_df["timestamp"] <= onset_end)].copy()


def _score_alignment(
    channel_values: pd.Series,
    score_values: pd.Series,
    baseline_median: float,
    baseline_mad: float,
) -> float:
    values = pd.to_numeric(channel_values, errors="coerce")
    scores = pd.to_numeric(score_values, errors="coerce")
    scale = 1.4826 * baseline_mad if np.isfinite(baseline_mad) and baseline_mad > 1e-12 else 1.0
    deviation = ((values - baseline_median).abs() / scale).replace([np.inf, -np.inf], np.nan)
    frame = pd.DataFrame({"dev": deviation, "score": scores}).dropna()
    if len(frame) < 8 or frame["dev"].nunique() < 2 or frame["score"].nunique() < 2:
        return 0.0
    corr = frame["dev"].corr(frame["score"], method="spearman")
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(abs(corr) * 100.0, 0.0, 100.0))


def _detection_alignment(
    well_df: pd.DataFrame,
    channel: str,
    detected_time: pd.Timestamp | None,
    baseline_median: float,
    baseline_mad: float,
    step_hours: float,
) -> float:
    if detected_time is None or pd.isna(detected_time):
        return 0.0
    half_window_hours = max(step_hours * 12.0, 2.0)
    view = well_df[
        (well_df["timestamp"] >= detected_time - pd.Timedelta(hours=half_window_hours))
        & (well_df["timestamp"] <= detected_time + pd.Timedelta(hours=half_window_hours))
    ]
    if view.empty:
        return 0.0
    values = pd.to_numeric(view[channel], errors="coerce").dropna()
    if values.empty:
        return 0.0
    scale = 1.4826 * baseline_mad if np.isfinite(baseline_mad) and baseline_mad > 1e-12 else 1.0
    robust_shift = abs(float(values.median()) - baseline_median) / scale
    return _clip_score(robust_shift, 22.0)


def _interval_evidence(
    anomaly_key: str,
    well_df: pd.DataFrame,
    score_df: pd.DataFrame,
    interval_row: pd.Series,
    channel: str,
    detected_time: pd.Timestamp | None,
) -> IntervalChannelEvidence | None:
    start = pd.Timestamp(interval_row["start_date"])
    end = pd.Timestamp(interval_row["end_date"])
    if pd.isna(start) or pd.isna(end) or end <= start:
        return None

    cfg = _window_config(anomaly_key)
    step_hours = _infer_step_hours(well_df["timestamp"])
    baseline_df = _adaptive_baseline(well_df, start, cfg, step_hours)
    anomaly_df = well_df[(well_df["timestamp"] >= start) & (well_df["timestamp"] <= end)].copy()
    onset_df = _onset_window(well_df, start, end, cfg)
    if baseline_df.empty or anomaly_df.empty:
        return None

    baseline = pd.to_numeric(baseline_df[channel], errors="coerce").dropna()
    anomaly = pd.to_numeric(anomaly_df[channel], errors="coerce").dropna()
    onset = pd.to_numeric(onset_df[channel], errors="coerce").dropna()
    if len(baseline) < 8 or len(anomaly) < 4:
        return None

    baseline_median = float(baseline.median())
    anomaly_median = float(anomaly.median())
    onset_median = float(onset.median()) if len(onset) else anomaly_median
    baseline_mad = _robust_mad(baseline)
    scale = 1.4826 * baseline_mad if np.isfinite(baseline_mad) and baseline_mad > 1e-12 else 1.0

    level_delta = anomaly_median - baseline_median
    onset_delta = onset_median - baseline_median
    robust_shift = level_delta / scale
    onset_robust_shift = onset_delta / scale

    first_part = anomaly.iloc[: max(4, min(len(anomaly), len(anomaly) // 4))]
    last_part = anomaly.iloc[-max(4, min(len(anomaly), len(anomaly) // 4)) :]
    slope_delta = float(last_part.median()) - float(first_part.median())
    slope_robust_shift = slope_delta / scale

    score_view = score_df[(score_df["timestamp"] >= start) & (score_df["timestamp"] <= end)].copy()
    merged = pd.merge_asof(
        anomaly_df[["timestamp", channel]].sort_values("timestamp"),
        score_view[["timestamp", "score"]].sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(hours=max(step_hours * 1.5, 0.01)),
    )

    physical_score = float(
        np.clip(
            0.45 * _clip_score(robust_shift, 20.0)
            + 0.40 * _clip_score(onset_robust_shift, 22.0)
            + 0.15 * _clip_score(slope_robust_shift, 18.0),
            0.0,
            100.0,
        )
    )
    model_score = _score_alignment(
        merged[channel],
        merged["score"],
        baseline_median=baseline_median,
        baseline_mad=baseline_mad,
    )
    detection_score = _detection_alignment(
        well_df=well_df,
        channel=channel,
        detected_time=detected_time,
        baseline_median=baseline_median,
        baseline_mad=baseline_mad,
        step_hours=step_hours,
    )
    final_score = float(np.clip(0.55 * physical_score + 0.30 * model_score + 0.15 * detection_score, 0.0, 100.0))

    return IntervalChannelEvidence(
        interval_idx=int(interval_row.get("interval_idx", 1)),
        start=start,
        end=end,
        baseline_points=int(len(baseline)),
        anomaly_points=int(len(anomaly)),
        baseline_median=baseline_median,
        anomaly_median=anomaly_median,
        onset_median=onset_median,
        baseline_mad=float(baseline_mad),
        level_delta=float(level_delta),
        level_delta_pct=_safe_pct(float(level_delta), baseline_median),
        robust_shift=float(robust_shift),
        onset_robust_shift=float(onset_robust_shift),
        slope_robust_shift=float(slope_robust_shift),
        physical_score=physical_score,
        model_score=model_score,
        detection_score=detection_score,
        final_score=final_score,
        direction=_direction(level_delta),
    )


def _aggregate_channel(
    channel: str,
    evidences: list[IntervalChannelEvidence],
    redundant_with: str | None,
) -> dict[str, Any]:
    if not evidences:
        raise ValueError("Cannot aggregate empty evidence list.")
    physical_scores = np.array([e.physical_score for e in evidences], dtype=float)
    model_scores = np.array([e.model_score for e in evidences], dtype=float)
    detection_scores = np.array([e.detection_score for e in evidences], dtype=float)
    final_scores = np.array([e.final_score for e in evidences], dtype=float)
    directions = [e.direction for e in evidences if e.direction != "flat"]
    dominant_direction = max(set(directions), key=directions.count) if directions else "flat"
    stable_fraction = float(np.mean(final_scores >= 20.0))
    direction_fraction = float(directions.count(dominant_direction) / len(directions)) if directions else 0.0
    stability_score = 100.0 * stable_fraction * (0.5 + 0.5 * direction_fraction)
    physical_score = float(np.nanmean(physical_scores))
    model_score = float(np.nanmean(model_scores))
    detection_score = float(np.nanmean(detection_scores))
    final_score = float(np.clip(0.50 * np.nanmean(final_scores) + 0.25 * stability_score + 0.25 * max(final_scores), 0.0, 100.0))

    if physical_score >= 35.0 and model_score >= 25.0:
        verdict = "core_signal"
    elif physical_score >= 35.0 and model_score < 20.0:
        verdict = "physical_missed_signal"
    elif model_score >= 35.0 and physical_score < 20.0:
        verdict = "model_surrogate"
    elif final_score < 12.0:
        verdict = "noise"
    elif redundant_with:
        verdict = "redundant"
    else:
        verdict = "supporting_signal"

    return {
        "channel": channel,
        "group": _channel_group(channel),
        "final_score": final_score,
        "physical_score": physical_score,
        "model_score": model_score,
        "detection_score": detection_score,
        "stability_score": float(np.clip(stability_score, 0.0, 100.0)),
        "direction": dominant_direction,
        "verdict": verdict,
        "redundant_with": redundant_with,
        "interval_count": len(evidences),
        "level_delta_median": float(np.nanmedian([e.level_delta for e in evidences])),
        "level_delta_pct_median": _nanmedian_optional([e.level_delta_pct for e in evidences]),
        "robust_shift_median": float(np.nanmedian([e.robust_shift for e in evidences])),
        "onset_robust_shift_median": float(np.nanmedian([e.onset_robust_shift for e in evidences])),
        "intervals": [
            {
                "interval_idx": e.interval_idx,
                "start": e.start.isoformat(),
                "end": e.end.isoformat(),
                "baseline_points": e.baseline_points,
                "anomaly_points": e.anomaly_points,
                "baseline_median": e.baseline_median,
                "anomaly_median": e.anomaly_median,
                "onset_median": e.onset_median,
                "baseline_mad": e.baseline_mad,
                "level_delta": e.level_delta,
                "level_delta_pct": e.level_delta_pct,
                "robust_shift": e.robust_shift,
                "onset_robust_shift": e.onset_robust_shift,
                "slope_robust_shift": e.slope_robust_shift,
                "physical_score": e.physical_score,
                "model_score": e.model_score,
                "detection_score": e.detection_score,
                "final_score": e.final_score,
                "direction": e.direction,
            }
            for e in evidences
        ],
    }


def _nanmedian_optional(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return None
    return float(np.nanmedian(finite))


def _redundancy_map(well_df: pd.DataFrame, channels: list[str], raw_scores: dict[str, float]) -> dict[str, str | None]:
    ranked = sorted(channels, key=lambda ch: raw_scores.get(ch, 0.0), reverse=True)
    redundant: dict[str, str | None] = {ch: None for ch in channels}
    for i, channel in enumerate(ranked):
        if raw_scores.get(channel, 0.0) < 12.0:
            continue
        values = pd.to_numeric(well_df[channel], errors="coerce")
        for better in ranked[:i]:
            if raw_scores.get(better, 0.0) < raw_scores.get(channel, 0.0):
                other = pd.to_numeric(well_df[better], errors="coerce")
                frame = pd.DataFrame({"a": values, "b": other}).dropna()
                if len(frame) < 20 or frame["a"].nunique() < 2 or frame["b"].nunique() < 2:
                    continue
                corr = frame["a"].corr(frame["b"], method="spearman")
                if np.isfinite(corr) and abs(float(corr)) >= 0.92:
                    redundant[channel] = better
                    break
    return redundant


def _detected_time_for_interval(
    results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    well_id: str,
    interval_row: pd.Series,
) -> pd.Timestamp | None:
    interval_idx = int(interval_row.get("interval_idx", 1))
    if not results_df.empty:
        match = results_df[
            (results_df["well_id"] == well_id)
            & (pd.to_numeric(results_df["interval_idx"], errors="coerce").fillna(-1).astype(int) == interval_idx)
        ]
        if not match.empty and pd.notna(match.iloc[0].get("detected_time")):
            return pd.Timestamp(match.iloc[0]["detected_time"])
    if predictions_df.empty:
        return None
    start = pd.Timestamp(interval_row["start_date"])
    end = pd.Timestamp(interval_row["end_date"])
    well_preds = predictions_df[
        (predictions_df["well_id"] == well_id)
        & (predictions_df["detected_time"] >= start - pd.Timedelta(hours=2))
        & (predictions_df["detected_time"] <= end)
    ]
    if well_preds.empty:
        return None
    return pd.Timestamp(well_preds.iloc[0]["detected_time"])


def _analyze_well(
    anomaly_key: str,
    well_id: str,
    well_df: pd.DataFrame,
    well_scores: pd.DataFrame,
    well_intervals: pd.DataFrame,
    results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    verbose: bool,
) -> dict[str, Any] | None:
    if well_df.empty or well_intervals.empty or well_scores.empty:
        return None

    channels = _numeric_channels(well_df)
    if not channels:
        return None

    channel_evidence: dict[str, list[IntervalChannelEvidence]] = {ch: [] for ch in channels}
    for _, interval_row in well_intervals.iterrows():
        detected_time = _detected_time_for_interval(results_df, predictions_df, well_id, interval_row)
        for channel in channels:
            evidence = _interval_evidence(
                anomaly_key=anomaly_key,
                well_df=well_df,
                score_df=well_scores,
                interval_row=interval_row,
                channel=channel,
                detected_time=detected_time,
            )
            if evidence is not None:
                channel_evidence[channel].append(evidence)

    provisional_scores = {
        channel: float(np.nanmean([e.final_score for e in evidences])) if evidences else 0.0
        for channel, evidences in channel_evidence.items()
    }
    redundant = _redundancy_map(well_df, channels, provisional_scores)
    channels_payload = {
        channel: _aggregate_channel(channel, evidences, redundant.get(channel))
        for channel, evidences in channel_evidence.items()
        if evidences
    }
    if not channels_payload:
        return None

    sorted_channels = sorted(channels_payload.values(), key=lambda item: item["final_score"], reverse=True)
    if verbose:
        top = ", ".join(f"{item['channel']}={item['final_score']:.1f}" for item in sorted_channels[:3])
        print(f"  {well_id}: {len(sorted_channels)} каналов, top: {top}")

    return {
        "well_id": well_id,
        "interval_count": int(len(well_intervals)),
        "channels": {item["channel"]: item for item in sorted_channels},
        "top_channels": [item["channel"] for item in sorted_channels[:8]],
    }


def _fig_to_b64(fig: plt.Figure, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _make_bar_b64(well_payload: dict[str, Any], accent: str, top_n: int = 18) -> str:
    channels = list(well_payload["channels"].values())[:top_n]
    labels = [item["channel"] for item in channels]
    final_scores = [item["final_score"] for item in channels]
    physical_scores = [item["physical_score"] for item in channels]
    model_scores = [item["model_score"] for item in channels]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(labels))))
    ax.barh(y - 0.22, final_scores, height=0.22, color=accent, label="Итог")
    ax.barh(y, physical_scores, height=0.22, color="#2563eb", label="Физика")
    ax.barh(y + 0.22, model_scores, height=0.22, color="#64748b", label="Связь со score")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Важность, 0-100")
    ax.set_title(f"Скважина {well_payload['well_id']} - важность каналов", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _make_timeseries_b64(
    well_payload: dict[str, Any],
    well_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    accent: str,
    top_n: int = 5,
) -> str:
    top_channels = [
        ch for ch in well_payload["top_channels"]
        if ch in well_df.columns
    ][:top_n]
    n_panels = 1 + len(top_channels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.7 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    ts_score = pd.to_datetime(scores_df["timestamp"])
    axes[0].fill_between(ts_score, 0, scores_df["score"].astype(float), color=accent, alpha=0.18)
    axes[0].plot(ts_score, scores_df["score"].astype(float), color=accent, linewidth=0.7)
    axes[0].set_ylabel("score", fontsize=8)
    axes[0].set_title(f"Скважина {well_payload['well_id']} - score и ключевые каналы", fontsize=12, fontweight="bold")

    def draw_marks(ax, labels: bool = False) -> None:
        ymin, ymax = ax.get_ylim()
        for idx, (_, row) in enumerate(intervals_df.iterrows()):
            start = pd.Timestamp(row["start_date"])
            end = pd.Timestamp(row["end_date"])
            ax.axvspan(start, end, color="red", alpha=0.10, zorder=0)
            ax.axvline(start, color="#16a34a", linewidth=1.1)
            ax.axvline(end, color="#dc2626", linewidth=1.1, linestyle="--")
            if labels and idx == 0:
                ax.annotate(
                    f"Факт начало\n{start.strftime('%Y-%m-%d %H:%M')}",
                    xy=(start, ymax),
                    xytext=(5, -5),
                    textcoords="offset points",
                    fontsize=6.5,
                    color="#16a34a",
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#16a34a", alpha=0.85),
                )
        for _, pred in predictions_df.iterrows():
            if pd.notna(pred.get("detected_time")):
                ax.axvline(pd.Timestamp(pred["detected_time"]), color="#7c3aed", linewidth=1.2, linestyle="-.")

    draw_marks(axes[0], labels=True)
    axes[0].grid(True, alpha=0.25)
    legend_handles = [
        mpatches.Patch(facecolor="red", alpha=0.10, edgecolor="none", label="Зона аномалии"),
        plt.Line2D([0], [0], color="#16a34a", linewidth=1.1, label="Начало"),
        plt.Line2D([0], [0], color="#dc2626", linewidth=1.1, linestyle="--", label="Конец"),
        plt.Line2D([0], [0], color="#7c3aed", linewidth=1.2, linestyle="-.", label="Детекция"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.9)

    ts = pd.to_datetime(well_df["timestamp"])
    for idx, channel in enumerate(top_channels, start=1):
        ax = axes[idx]
        ax.plot(ts, pd.to_numeric(well_df[channel], errors="coerce"), color="#2563eb", linewidth=0.6)
        item = well_payload["channels"][channel]
        ax.set_ylabel(
            f"{channel}\nитог {item['final_score']:.0f}, физ {item['physical_score']:.0f}",
            fontsize=7,
        )
        draw_marks(ax)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Время")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_b64(fig, dpi=110)


def _verdict_ru(verdict: str) -> str:
    return {
        "core_signal": "ключевой сигнал",
        "physical_missed_signal": "физический сигнал, модель недоиспользует",
        "model_surrogate": "модельный суррогат",
        "redundant": "дублируется коррелированным каналом",
        "noise": "шум / слабый вклад",
        "supporting_signal": "поддерживающий сигнал",
    }.get(verdict, verdict)


def _format_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:+.1f}%"


def _build_html(
    spec: DetectionSpec,
    detector_key: str,
    summary: dict[str, Any],
    data_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> str:
    theme = COLOR_THEMES.get(spec.anomaly_key, COLOR_THEMES["salt"])
    accent = theme["accent"]
    soft_bg = theme["soft_bg"]
    display = DISPLAY_NAMES.get(spec.anomaly_key, spec.display_name)
    title = f"Анализ важности признаков: {display}"

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; color: #172033; background: #fafafa; }}
    h1 {{ color: #172033; border-bottom: 3px solid {accent}; padding-bottom: 10px; }}
    h2 {{ margin-top: 36px; color: #172033; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0 28px; background: #fff; }}
    th, td {{ border: 1px solid #d0d5dd; padding: 8px 10px; text-align: left; font-size: 0.9em; }}
    th {{ background: {accent}; color: #fff; }}
    tr:nth-child(even) {{ background: {soft_bg}; }}
    .method-box, .summary-box, .plot-container {{ background: #fff; border: 1px solid #d0d5dd; border-radius: 8px; padding: 16px 20px; margin: 18px 0; }}
    .method-box {{ background: {soft_bg}; border-left: 5px solid {accent}; line-height: 1.65; }}
    .plot-container img {{ max-width: 100%; height: auto; }}
    .score {{ font-weight: 700; color: {accent}; }}
    .muted {{ color: #667085; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="muted">Детектор: <b>{escape(DETECTOR_LABELS.get(detector_key, detector_key))}</b>. Метод: interval-based physical + model alignment importance, JSON v{FI_SUMMARY_VERSION}.</p>
  <div class="method-box">
    <h3>Как читать отчёт</h3>
    <p><b>Физика</b> показывает, насколько канал реально изменился в размеченном интервале относительно локального baseline перед аномалией.</p>
    <p><b>Связь со score</b> показывает, насколько отклонения канала совпадают с ростом detector score и моментом детекции.</p>
    <p><b>Итог</b> объединяет физику, связь со score и стабильность по интервалам. Это не legacy ablation и не pct_drop.</p>
    <p>Вердикт отделяет физически важные каналы от модельных суррогатов, шума и коррелированных дублей.</p>
  </div>
"""

    wells = summary["wells"]
    for well_id, well_payload in wells.items():
        well_df = data_df[data_df["well_id"] == well_id].copy()
        well_scores = scores_df[scores_df["well_id"] == well_id].copy()
        well_intervals = intervals_df[intervals_df["well_id"] == well_id].copy()
        well_predictions = predictions_df[predictions_df["well_id"] == well_id].copy()

        html += f"<h2>Скважина {escape(well_id)}</h2>\n"
        top_items = list(well_payload["channels"].values())[:3]
        top_text = ", ".join(
            f"<b>{escape(item['channel'])}</b> (<span class='score'>{item['final_score']:.0f}</span>, {_verdict_ru(item['verdict'])})"
            for item in top_items
        )
        html += f"<div class='summary-box'>Ключевые каналы: {top_text}</div>\n"

        html += f"<div class='plot-container'><img src='data:image/png;base64,{_make_bar_b64(well_payload, accent)}' alt='Важность {escape(well_id)}'></div>\n"
        html += f"<div class='plot-container'><img src='data:image/png;base64,{_make_timeseries_b64(well_payload, well_df, well_scores, well_intervals, well_predictions, accent)}' alt='Каналы {escape(well_id)}'></div>\n"

        html += """
  <table>
    <tr>
      <th>#</th>
      <th>Канал</th>
      <th>Группа</th>
      <th>Итог</th>
      <th>Физика</th>
      <th>Связь со score</th>
      <th>Стабильность</th>
      <th>Направление</th>
      <th>Медианная дельта</th>
      <th>Вердикт</th>
    </tr>
"""
        for rank, item in enumerate(well_payload["channels"].values(), start=1):
            redundant = f" ({escape(item['redundant_with'])})" if item.get("redundant_with") else ""
            html += f"""
    <tr>
      <td>{rank}</td>
      <td>{escape(item['channel'])}</td>
      <td>{escape(item['group'])}</td>
      <td class="score">{item['final_score']:.1f}</td>
      <td>{item['physical_score']:.1f}</td>
      <td>{item['model_score']:.1f}</td>
      <td>{item['stability_score']:.1f}</td>
      <td>{escape(item['direction'])}</td>
      <td>{_format_pct(item.get('level_delta_pct_median'))}</td>
      <td>{escape(_verdict_ru(item['verdict']) + redundant)}</td>
    </tr>
"""
        html += "  </table>\n"

    html += "</body>\n</html>\n"
    return html


def _summary_for_backend(summary: dict[str, Any]) -> dict[str, Any]:
    return summary


def generate_feature_importance_report(
    anomaly_key: str,
    detector: str | None = None,
    output_path: str | None = None,
    source_path: str | None = None,
    verbose: bool = True,
) -> Path:
    spec = get_detection_spec(anomaly_key)
    detector_key = _resolve_detector(spec, detector)
    print(f"Аномалия: {spec.display_name}")
    print(f"Детектор: {DETECTOR_LABELS.get(detector_key, detector_key)} ({detector_key})")

    data_df = _load_timeseries(spec, source_path, detector_key=detector_key)
    intervals_df = _load_intervals(spec)
    scores_df = _load_scores(spec, detector_key)
    score_col = _score_column(scores_df)
    if score_col != "score":
        scores_df = scores_df.rename(columns={score_col: "score"})
    results_df = _load_detection_results(spec, detector_key)
    predictions_df = _load_predictions(spec, detector_key)

    wells: dict[str, Any] = {}
    for well_id in sorted(intervals_df["well_id"].unique()):
        well_payload = _analyze_well(
            anomaly_key=spec.anomaly_key,
            well_id=str(well_id),
            well_df=data_df[data_df["well_id"] == well_id].copy(),
            well_scores=scores_df[scores_df["well_id"] == well_id].copy(),
            well_intervals=intervals_df[intervals_df["well_id"] == well_id].copy(),
            results_df=results_df,
            predictions_df=predictions_df,
            verbose=verbose,
        )
        if well_payload is not None:
            wells[str(well_id)] = well_payload

    if not wells:
        raise RuntimeError("Ни одна скважина не прошла анализ важности признаков")

    summary = {
        "version": FI_SUMMARY_VERSION,
        "method": "interval_physical_model_grouped_importance",
        "anomaly": spec.anomaly_key,
        "detector": detector_key,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "score_column": "score",
        "weights": {
            "interval_final": {"physical": 0.55, "model": 0.30, "detection": 0.15},
            "channel_final": {"mean_interval": 0.50, "stability": 0.25, "max_interval": 0.25},
        },
        "wells": wells,
    }

    fi_json_path = DB_DIR / f"{spec.dataset.output_prefix}_{detector_key}_fi_summary.json"
    fi_json_path.write_text(json.dumps(_summary_for_backend(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Feature importance summary saved: {fi_json_path}")

    html = _build_html(
        spec=spec,
        detector_key=detector_key,
        summary=summary,
        data_df=data_df,
        scores_df=scores_df,
        intervals_df=intervals_df,
        predictions_df=predictions_df,
    )

    if output_path is None:
        subdir = REPORTS_DIR / spec.anomaly_key
        subdir.mkdir(parents=True, exist_ok=True)
        output_path = str(subdir / f"{spec.dataset.output_prefix}_{detector_key}_feature_importance.html")
    out = ensure_parent(Path(output_path))
    out.write_text(html, encoding="utf-8")
    print(f"Отчёт сохранён: {out}")
    return out
