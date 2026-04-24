from pathlib import Path

import polars as pl

from back.api.wells.schemas import (
    AnomalyType,
    WellDetail,
    WellInterval,
    WellSummary,
)
from back.services.parquet import intervals_path, read_parquet_cached


def _load(data_root: Path, anomaly: AnomalyType) -> pl.DataFrame:
    path = intervals_path(data_root, anomaly)
    if not path.exists():
        return pl.DataFrame(
            schema={
                "well_id": pl.String,
                "start_date": pl.Datetime,
                "end_date": pl.Datetime,
                "data_start": pl.Datetime,
                "data_end": pl.Datetime,
                "split": pl.String,
                "interval_idx": pl.Int64,
            }
        )
    return read_parquet_cached(path)


def _summary_from_row(row: dict, anomaly: AnomalyType) -> WellSummary:
    return WellSummary(
        well_id=row["well_id"],
        anomaly=anomaly,
        split=row["split"],
        n_intervals=row["n_intervals"],
        data_start=row["data_start"],
        data_end=row["data_end"],
    )


def list_wells(data_root: Path, anomaly: AnomalyType) -> list[WellSummary]:
    df = _load(data_root, anomaly)
    if df.is_empty():
        return []
    grouped = (
        df.group_by("well_id")
        .agg(
            pl.col("split").first().alias("split"),
            pl.len().alias("n_intervals"),
            pl.col("data_start").min().alias("data_start"),
            pl.col("data_end").max().alias("data_end"),
        )
        .sort("well_id")
    )
    return [_summary_from_row(row, anomaly) for row in grouped.iter_rows(named=True)]


def get_well(data_root: Path, anomaly: AnomalyType, well_id: str) -> WellDetail | None:
    df = _load(data_root, anomaly).filter(pl.col("well_id") == well_id)
    if df.is_empty():
        return None

    intervals = [
        WellInterval(
            interval_idx=row["interval_idx"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            data_start=row["data_start"],
            data_end=row["data_end"],
            split=row["split"],
        )
        for row in df.sort("interval_idx").iter_rows(named=True)
    ]
    summary = WellSummary(
        well_id=well_id,
        anomaly=anomaly,
        split=intervals[0].split,
        n_intervals=len(intervals),
        data_start=min(i.data_start for i in intervals),
        data_end=max(i.data_end for i in intervals),
    )
    return WellDetail(**summary.model_dump(), intervals=intervals)
