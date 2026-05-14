from __future__ import annotations

from pydantic import BaseModel

from back.api.wells.schemas import AnomalyType


class UploadScorePoint(BaseModel):
    t: str
    score: float


class UploadResult(BaseModel):
    run_id: str
    anomaly: AnomalyType
    well_id: str
    detector: str
    status: str
    n_points: int
    n_detected: int
    detected_starts: list[str]
    score_min: float | None
    score_median: float | None
    score_max: float | None
    time_start: str | None
    time_end: str | None
    score_series: list[UploadScorePoint]
