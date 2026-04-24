import json
from datetime import datetime
from pathlib import Path

import polars as pl

from back.api.reports.constants import ALL_DETECTORS, BEST_DETECTOR_BY_ANOMALY
from back.api.reports.schemas import (
    AnomalyInterval,
    AnomalyReportAvailability,
    DetectorAvailability,
    FeatureImportanceItem,
    FeatureImportanceResponse,
    PredictedOnset,
    PredictedStart,
    ScorePoint,
    ScoreSeries,
    TelemetryChannel,
    TimePoint,
    WellSeriesResponse,
)
from back.api.wells.schemas import AnomalyType
from back.services.parquet import read_parquet_cached
from back.services.paths import (
    anomaly_database_parquet_path,
    feature_importance_html_path,
    fi_summary_json_path,
    html_report_path,
    intervals_parquet_path,
    predicted_starts_parquet_path,
    scores_parquet_path,
)

META_COLS = {"timestamp", "well_id"}


def _downsample_stride(n: int, limit: int) -> int:
    if limit <= 0 or n <= limit:
        return 1
    return max(1, (n + limit - 1) // limit)


def load_scores(
    data_root: Path,
    anomaly: str,
    detector: str,
    *,
    well_id: str | None,
    frm: datetime | None,
    to: datetime | None,
    limit: int,
) -> ScoreSeries | None:
    path = scores_parquet_path(data_root, anomaly, detector)
    if not path.exists():
        return None

    df = read_parquet_cached(path)
    if well_id is not None:
        df = df.filter(pl.col("well_id") == well_id)
    if frm is not None:
        df = df.filter(pl.col("timestamp") >= frm)
    if to is not None:
        df = df.filter(pl.col("timestamp") <= to)

    if df.is_empty():
        return ScoreSeries(
            well_id=well_id or "",
            anomaly=anomaly,
            detector=detector,
            n_points=0,
            n_downsampled=0,
            points=[],
        )

    df = df.sort("timestamp")
    n_points = len(df)
    stride = _downsample_stride(n_points, limit)
    downsampled = df.gather_every(stride)
    n_downsampled = len(downsampled)

    points = [
        ScorePoint(t=row["timestamp"], score=row["score"], split=row["split"])
        for row in downsampled.iter_rows(named=True)
    ]

    return ScoreSeries(
        well_id=well_id or "",
        anomaly=anomaly,
        detector=detector,
        n_points=n_points,
        n_downsampled=n_downsampled,
        points=points,
    )


def check_availability(data_root: Path, anomaly: AnomalyType) -> AnomalyReportAvailability:
    detectors: list[DetectorAvailability] = []
    for det in ALL_DETECTORS:
        detectors.append(
            DetectorAvailability(
                detector=det,
                has_report=html_report_path(data_root, anomaly, det).exists(),
                has_feature_importance=feature_importance_html_path(
                    data_root, anomaly, det
                ).exists(),
            )
        )

    preferred = BEST_DETECTOR_BY_ANOMALY.get(anomaly)
    best: str | None = None
    if preferred is not None and any(d.detector == preferred and d.has_report for d in detectors):
        best = preferred
    else:
        for d in detectors:
            if d.has_report:
                best = d.detector
                break

    return AnomalyReportAvailability(
        anomaly=anomaly,
        detectors=detectors,
        best_detector=best,
        has_any_report=best is not None,
    )


def _points(df: pl.DataFrame, t_col: str, v_col: str) -> list[TimePoint]:
    if v_col not in df.columns:
        return []
    sub = df.select([t_col, v_col]).drop_nulls()
    return [TimePoint(t=row[0], v=float(row[1])) for row in sub.iter_rows()]


def load_well_series(
    data_root: Path,
    anomaly: AnomalyType,
    detector: str,
    *,
    well_id: str,
    limit: int = 2000,
) -> WellSeriesResponse | None:
    scores_path = scores_parquet_path(data_root, anomaly, detector)
    if not scores_path.exists():
        return None

    scores = read_parquet_cached(scores_path).filter(pl.col("well_id") == well_id).sort("timestamp")
    if scores.is_empty():
        return WellSeriesResponse(
            well_id=well_id,
            anomaly=anomaly,
            detector=detector,
            n_points_raw=0,
            n_points_downsampled=0,
            time_start=None,
            time_end=None,
            score=[],
            paano_short=[],
            paano_long=[],
            telemetry=[],
            intervals=[],
            predicted_starts=[],
        )

    n_raw = len(scores)
    stride = _downsample_stride(n_raw, limit)
    scores_ds = scores.gather_every(stride)

    time_start = scores_ds["timestamp"][0]
    time_end = scores_ds["timestamp"][-1]

    score_series = _points(scores_ds, "timestamp", "score")
    paano_short = _points(scores_ds, "timestamp", "paano_short")
    paano_long = _points(scores_ds, "timestamp", "paano_long")

    telemetry: list[TelemetryChannel] = []
    tel_path = anomaly_database_parquet_path(data_root, anomaly)
    if tel_path.exists():
        tel = read_parquet_cached(tel_path).filter(pl.col("well_id") == well_id).sort("timestamp")
        if not tel.is_empty():
            tel = tel.filter(
                (pl.col("timestamp") >= time_start) & (pl.col("timestamp") <= time_end)
            )
        if not tel.is_empty():
            tel_ds = tel.gather_every(_downsample_stride(len(tel), limit))
            for col in tel.columns:
                if col in META_COLS:
                    continue
                dtype = tel_ds.schema[col]
                if dtype.is_numeric():
                    telemetry.append(
                        TelemetryChannel(name=col, points=_points(tel_ds, "timestamp", col))
                    )

    intervals: list[AnomalyInterval] = []
    iv_path = intervals_parquet_path(data_root, anomaly)
    if iv_path.exists():
        iv = read_parquet_cached(iv_path).filter(pl.col("well_id") == well_id).sort("interval_idx")
        for row in iv.iter_rows(named=True):
            intervals.append(
                AnomalyInterval(
                    start=row["start_date"],
                    end=row["end_date"],
                    interval_idx=row["interval_idx"],
                    split=row["split"],
                )
            )

    predicted_starts: list[PredictedOnset] = []
    ps_path = predicted_starts_parquet_path(data_root, anomaly, detector)
    if ps_path.exists():
        ps = read_parquet_cached(ps_path).filter(pl.col("well_id") == well_id).sort("detected_time")
        for row in ps.iter_rows(named=True):
            predicted_starts.append(PredictedOnset(t=row["detected_time"], split=row["split"]))

    return WellSeriesResponse(
        well_id=well_id,
        anomaly=anomaly,
        detector=detector,
        n_points_raw=n_raw,
        n_points_downsampled=len(scores_ds),
        time_start=time_start,
        time_end=time_end,
        score=score_series,
        paano_short=paano_short,
        paano_long=paano_long,
        telemetry=telemetry,
        intervals=intervals,
        predicted_starts=predicted_starts,
    )


def load_feature_importance(
    data_root: Path,
    anomaly: AnomalyType,
    detector: str,
    *,
    well_id: str,
) -> FeatureImportanceResponse | None:
    path = fi_summary_json_path(data_root, anomaly, detector)
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    well_data = data.get(well_id)
    if not isinstance(well_data, dict):
        return FeatureImportanceResponse(
            well_id=well_id, anomaly=anomaly, detector=detector, items=[]
        )

    items = [
        FeatureImportanceItem(feature=str(k), importance=float(v))
        for k, v in well_data.items()
        if isinstance(v, (int, float))
    ]
    items.sort(key=lambda x: abs(x.importance), reverse=True)
    return FeatureImportanceResponse(
        well_id=well_id, anomaly=anomaly, detector=detector, items=items
    )


def load_predicted_starts(
    data_root: Path,
    anomaly: str,
    detector: str,
    *,
    well_id: str | None,
    split: str | None,
) -> list[PredictedStart] | None:
    path = predicted_starts_parquet_path(data_root, anomaly, detector)
    if not path.exists():
        return None

    df = read_parquet_cached(path)
    if well_id is not None:
        df = df.filter(pl.col("well_id") == well_id)
    if split is not None:
        df = df.filter(pl.col("split") == split)

    return [
        PredictedStart(
            well_id=row["well_id"],
            detected_time=row["detected_time"],
            split=row["split"],
        )
        for row in df.sort("detected_time").iter_rows(named=True)
    ]
