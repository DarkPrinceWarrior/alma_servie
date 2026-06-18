from datetime import datetime
from typing import Literal

from pydantic import BaseModel

AnomalyType = Literal["negermet", "pritok"]
SplitType = Literal["train", "test"]


class WellInterval(BaseModel):
    interval_idx: int
    start_date: datetime
    end_date: datetime
    data_start: datetime
    data_end: datetime
    split: SplitType


class WellSummary(BaseModel):
    well_id: str
    anomaly: AnomalyType
    split: SplitType
    n_intervals: int
    data_start: datetime
    data_end: datetime


class WellDetail(WellSummary):
    intervals: list[WellInterval]
