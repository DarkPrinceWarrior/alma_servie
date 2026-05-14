from __future__ import annotations

from pydantic import BaseModel

from back.api.wells.schemas import AnomalyType


class UploadScorePoint(BaseModel):
    t: str
    score: float


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
    error: str | None = None


class UploadResultBundle(BaseModel):
    run_id: str
    well_id: str
    status: str  # overall DetectionRun status
    n_done: int
    n_total: int
    results: list[UploadAnomalyResult]
