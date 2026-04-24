from datetime import datetime
from pathlib import Path

import polars as pl

from back.api.reports.schemas import PredictedStart, ScorePoint, ScoreSeries
from back.services.parquet import read_parquet_cached
from back.services.paths import predicted_starts_parquet_path, scores_parquet_path


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
