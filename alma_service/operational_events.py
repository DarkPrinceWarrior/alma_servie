from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_EVENT_COLUMNS = (
    "well_id",
    "event_type",
    "start_date",
    "end_date",
    "pre_window_hours",
    "source",
)
START_COLUMN_ALIASES = (
    "start_date",
    "start_time",
    "event_time",
    "timestamp",
    "date",
    "Дата начала",
    "Время начала",
    "Дата",
)
END_COLUMN_ALIASES = (
    "end_date",
    "end_time",
    "finish_time",
    "Дата окончания",
    "Время окончания",
)
TYPE_COLUMN_ALIASES = (
    "event_type",
    "type",
    "event",
    "comment",
    "Тип события",
    "Комментарий",
)
WELL_COLUMN_ALIASES = ("well_id", "well", "Скважина", "Номер скважины")
SOURCE_COLUMN_ALIASES = ("source", "Источник")


@dataclass(frozen=True)
class EventZoneMasks:
    pre_anomaly_mask: np.ndarray
    event_mask: np.ndarray


def _first_existing_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower_map = {str(column).casefold(): str(column) for column in frame.columns}
    for alias in aliases:
        column = lower_map.get(alias.casefold())
        if column is not None:
            return column
    return None


def _optional_series(frame: pd.DataFrame, column: str | None, default: Any = "") -> pd.Series:
    if column is None:
        return pd.Series(default, index=frame.index)
    return frame[column]


def normalize_operational_events(events: pd.DataFrame | None) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame(columns=CANONICAL_EVENT_COLUMNS)

    frame = events.copy()
    start_col = _first_existing_column(frame, START_COLUMN_ALIASES)
    if start_col is None:
        raise ValueError("Operational events require a start_date/start_time/timestamp column.")

    end_col = _first_existing_column(frame, END_COLUMN_ALIASES)
    type_col = _first_existing_column(frame, TYPE_COLUMN_ALIASES)
    well_col = _first_existing_column(frame, WELL_COLUMN_ALIASES)
    source_col = _first_existing_column(frame, SOURCE_COLUMN_ALIASES)
    pre_window_col = "pre_window_hours" if "pre_window_hours" in frame.columns else None

    normalized = pd.DataFrame(
        {
            "well_id": _optional_series(frame, well_col, ""),
            "event_type": _optional_series(frame, type_col, "operational_event"),
            "start_date": pd.to_datetime(frame[start_col], errors="coerce"),
            "end_date": pd.to_datetime(_optional_series(frame, end_col, pd.NaT), errors="coerce"),
            "pre_window_hours": pd.to_numeric(
                _optional_series(frame, pre_window_col, np.nan),
                errors="coerce",
            ),
            "source": _optional_series(frame, source_col, ""),
        }
    )
    normalized["end_date"] = normalized["end_date"].fillna(normalized["start_date"])
    return normalized.dropna(subset=["start_date"]).reset_index(drop=True)


def events_for_well(events: pd.DataFrame | None, well_id: str) -> pd.DataFrame:
    normalized = normalize_operational_events(events)
    if normalized.empty:
        return normalized
    key = str(well_id).strip().casefold()
    well_values = normalized["well_id"].astype(str).str.strip().str.casefold()
    return normalized[(well_values == "") | (well_values == key)].reset_index(drop=True)


def build_event_zone_masks(
    *,
    timestamps: np.ndarray,
    events: pd.DataFrame | None,
    default_pre_window_hours: float = 24.0,
) -> EventZoneMasks:
    n_points = int(len(timestamps))
    pre_mask = np.zeros(n_points, dtype=bool)
    event_mask = np.zeros(n_points, dtype=bool)
    normalized = normalize_operational_events(events)
    if n_points == 0 or normalized.empty:
        return EventZoneMasks(pre_anomaly_mask=pre_mask, event_mask=event_mask)

    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    for row in normalized.itertuples(index=False):
        start = pd.Timestamp(row.start_date)
        end = pd.Timestamp(row.end_date)
        if pd.isna(start):
            continue
        if pd.isna(end) or end < start:
            end = start
        pre_window_hours = float(row.pre_window_hours)
        if not np.isfinite(pre_window_hours) or pre_window_hours < 0:
            pre_window_hours = float(default_pre_window_hours)
        pre_start = start - pd.Timedelta(hours=pre_window_hours)
        pre_mask |= np.asarray((ts >= pre_start) & (ts < start), dtype=bool)
        event_mask |= np.asarray((ts >= start) & (ts <= end), dtype=bool)

    return EventZoneMasks(pre_anomaly_mask=pre_mask, event_mask=event_mask)
