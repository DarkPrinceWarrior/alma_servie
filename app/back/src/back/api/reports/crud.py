from datetime import datetime
from pathlib import Path

import polars as pl

from back.api.reports.constants import ALL_DETECTORS, BEST_DETECTOR_BY_ANOMALY
from back.api.reports.schemas import (
    AnomalyReportAvailability,
    DetectorAvailability,
    PredictedStart,
    ScorePoint,
    ScoreSeries,
)
from back.api.wells.schemas import AnomalyType
from back.services.parquet import read_parquet_cached
from back.services.paths import (
    feature_importance_html_path,
    html_report_path,
    predicted_starts_parquet_path,
    scores_parquet_path,
)


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
