from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alma_service.telemetry_status import (
    EVENT_BAD_DATA,
    EVENT_LABELLED_ANOMALY,
    EVENT_PRE_ANOMALY,
    EVENT_REGIME,
)

START_ANOMALY_CANDIDATE = "anomaly_candidate"
START_BAD_DATA = "bad_data"
START_REGIME_EVENT = "regime_event"
START_PRE_ANOMALY_ZONE = "pre_anomaly_zone"
START_LABELLED_ANOMALY = "labelled_anomaly"

ACTIONABLE_START_CLASSES = {
    START_ANOMALY_CANDIDATE,
    START_PRE_ANOMALY_ZONE,
    START_LABELLED_ANOMALY,
}


@dataclass(frozen=True)
class IncidentBuildResult:
    starts: pd.DataFrame
    incidents: pd.DataFrame


def _start_class(event_class: object) -> str:
    value = str(event_class or "").strip().lower()
    if value == EVENT_BAD_DATA:
        return START_BAD_DATA
    if value == EVENT_REGIME:
        return START_REGIME_EVENT
    if value == EVENT_PRE_ANOMALY:
        return START_PRE_ANOMALY_ZONE
    if value == EVENT_LABELLED_ANOMALY:
        return START_LABELLED_ANOMALY
    return START_ANOMALY_CANDIDATE


def annotate_predicted_starts(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        result = predictions.copy()
        for column in ("start_class", "actionable_alert", "suppression_reason"):
            if column not in result.columns:
                result[column] = pd.Series(dtype=object)
        return result

    result = predictions.copy()
    if "event_class" not in result.columns:
        result["event_class"] = ""
    result["start_class"] = result["event_class"].map(_start_class)
    if "zone_status" in result.columns:
        zone = result["zone_status"].fillna("").astype(str).str.lower()
        result.loc[zone == EVENT_PRE_ANOMALY, "start_class"] = START_PRE_ANOMALY_ZONE
        result.loc[zone == EVENT_LABELLED_ANOMALY, "start_class"] = START_LABELLED_ANOMALY
    result["actionable_alert"] = result["start_class"].isin(ACTIONABLE_START_CLASSES)
    result["suppression_reason"] = ""
    result.loc[result["start_class"] == START_BAD_DATA, "suppression_reason"] = "bad_data"
    result.loc[result["start_class"] == START_REGIME_EVENT, "suppression_reason"] = "regime_event"
    return result


def filter_actionable_starts(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "actionable_alert" not in predictions.columns:
        return predictions
    return predictions[predictions["actionable_alert"].fillna(True).astype(bool)].copy()


def build_incidents(
    predictions: pd.DataFrame,
    *,
    merge_window_hours: float,
) -> IncidentBuildResult:
    starts = annotate_predicted_starts(predictions)
    if starts.empty:
        return IncidentBuildResult(starts=starts, incidents=pd.DataFrame())

    starts = starts.copy()
    starts["detected_time"] = pd.to_datetime(starts["detected_time"])
    starts["incident_id"] = ""
    starts["incident_state"] = "suppressed"
    starts["incident_close_time"] = pd.NaT

    incidents: list[dict[str, object]] = []
    actionable = starts[starts["actionable_alert"].fillna(True).astype(bool)].copy()
    if actionable.empty:
        return IncidentBuildResult(starts=starts, incidents=pd.DataFrame())

    merge_delta = pd.Timedelta(hours=max(float(merge_window_hours), 0.0))
    group_columns = [column for column in ("anomaly", "detector", "well_id") if column in actionable.columns]
    if "well_id" not in group_columns:
        group_columns = ["well_id"] if "well_id" in actionable.columns else []

    grouped = actionable.sort_values(["well_id", "detected_time"]).groupby(group_columns, dropna=False) if group_columns else [((), actionable)]
    for group_key, group in grouped:
        incident_index = 0
        current_rows: list[int] = []
        current_start: pd.Timestamp | None = None
        last_time: pd.Timestamp | None = None

        def flush() -> None:
            nonlocal incident_index, current_rows, current_start, last_time
            if not current_rows or current_start is None or last_time is None:
                return
            first_row = starts.loc[current_rows[0]]
            anomaly = str(first_row.get("anomaly", ""))
            detector = str(first_row.get("detector", ""))
            well_id = str(first_row.get("well_id", ""))
            incident_id = f"{well_id}:{anomaly}:{detector}:{incident_index:04d}"
            close_time = last_time + merge_delta
            for pos, row_idx in enumerate(current_rows):
                starts.at[row_idx, "incident_id"] = incident_id
                starts.at[row_idx, "incident_state"] = "open" if pos == 0 else "continue"
                starts.at[row_idx, "incident_close_time"] = close_time
            incidents.append(
                {
                    "incident_id": incident_id,
                    "well_id": well_id,
                    "anomaly": anomaly,
                    "detector": detector,
                    "opened_at": current_start,
                    "last_start_at": last_time,
                    "closed_at": close_time,
                    "lifecycle_status": "closed",
                    "start_count": len(current_rows),
                    "merge_window_hours": float(merge_window_hours),
                }
            )
            incident_index += 1
            current_rows = []
            current_start = None
            last_time = None

        for row_idx, row in group.iterrows():
            detected_time = pd.Timestamp(row["detected_time"])
            if last_time is not None and detected_time - last_time > merge_delta:
                flush()
            if not current_rows:
                current_start = detected_time
            current_rows.append(int(row_idx))
            last_time = detected_time
        flush()

    incident_df = pd.DataFrame(incidents)
    if not incident_df.empty:
        incident_df = incident_df.sort_values(["well_id", "opened_at"]).reset_index(drop=True)
    return IncidentBuildResult(starts=starts, incidents=incident_df)
