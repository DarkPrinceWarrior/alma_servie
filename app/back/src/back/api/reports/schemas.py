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
