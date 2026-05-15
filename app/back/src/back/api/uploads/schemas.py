from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from back.api.wells.schemas import AnomalyType


class UploadScorePoint(BaseModel):
    t: str
    score: float


class UploadTimePoint(BaseModel):
    t: str
    v: float


class UploadChannel(BaseModel):
    name: str
    points: list[UploadTimePoint]


class UploadAnomalyResult(BaseModel):
    anomaly: AnomalyType
    status: str  # "succeeded" | "failed" | "pending"
    well_id: str | None = None
    detector: str | None = None
    n_points: int | None = None
    n_detected: int | None = None
    detected_starts: list[str] = []
    score_min: float | None = None
    score_median: float | None = None
    score_max: float | None = None
    time_start: str | None = None
    time_end: str | None = None
    score_series: list[UploadScorePoint] = []
    telemetry: list[UploadChannel] = []
    error: str | None = None


class UploadResultBundle(BaseModel):
    run_id: str
    well_id: str
    status: str  # overall DetectionRun status
    n_done: int
    n_total: int
    results: list[UploadAnomalyResult]


class UploadListItem(BaseModel):
    run_id: str
    well_id: str
    status: str
    created_at: datetime
    n_done: int
    n_total: int
    n_detected_total: int


class UploadList(BaseModel):
    items: list[UploadListItem]


class BulkDeleteRequest(BaseModel):
    run_ids: list[str]


class BulkDeleteResponse(BaseModel):
    deleted: int
