from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from back.api.wells.schemas import AnomalyType

DetectorType = Literal["paano_shared"]
StatusType = Literal["pending", "running", "succeeded", "failed", "cancelled"]


class DetectionRunCreate(BaseModel):
    anomaly: AnomalyType
    detector: DetectorType


class DetectionRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    anomaly: str
    detector: str
    status: str
    command: str
    exit_code: int | None
    stdout_tail: str | None
    summary_json: dict[str, Any] | None
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


class DetectionRunList(BaseModel):
    items: list[DetectionRunRead]
    total: int
