from datetime import datetime

from pydantic import BaseModel


class ScorePoint(BaseModel):
    t: datetime
    score: float
    split: str


class ScoreSeries(BaseModel):
    well_id: str
    anomaly: str
    detector: str
    n_points: int
    n_downsampled: int
    points: list[ScorePoint]


class PredictedStart(BaseModel):
    well_id: str
    detected_time: datetime
    split: str


class DetectorAvailability(BaseModel):
    detector: str
    has_report: bool
    has_feature_importance: bool


class AnomalyReportAvailability(BaseModel):
    anomaly: str
    detectors: list[DetectorAvailability]
    best_detector: str | None
    has_any_report: bool


class TimePoint(BaseModel):
    t: datetime
    v: float


class TelemetryChannel(BaseModel):
    name: str
    points: list[TimePoint]


class AnomalyInterval(BaseModel):
    start: datetime
    end: datetime
    interval_idx: int
    split: str


class PredictedOnset(BaseModel):
    t: datetime
    split: str


class IntervalResult(BaseModel):
    interval_idx: int
    actual_start: datetime
    actual_end: datetime
    detected_time: datetime | None
    delay_hours: float | None
    status: str
    split: str
    data_start: datetime | None
    data_end: datetime | None


class WellSeriesResponse(BaseModel):
    well_id: str
    anomaly: str
    detector: str
    n_points_raw: int
    n_points_downsampled: int
    time_start: datetime | None
    time_end: datetime | None
    score: list[TimePoint]
    paano_short: list[TimePoint]
    paano_long: list[TimePoint]
    telemetry: list[TelemetryChannel]
    intervals: list[AnomalyInterval]
    predicted_starts: list[PredictedOnset]
    results: list[IntervalResult]


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    well_id: str
    anomaly: str
    detector: str
    items: list[FeatureImportanceItem]
