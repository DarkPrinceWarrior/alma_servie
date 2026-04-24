from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from back.api.detections.schemas import DetectorType
from back.api.wells.schemas import AnomalyType
from back.models.detection_run import DetectionRun

ACTIVE_STATUSES = ("pending", "running")


def command_for(anomaly: str, detector: str) -> str:
    return f"python scripts/detection/detect_{anomaly}.py --detector {detector}"


async def find_active(
    db: AsyncSession, anomaly: AnomalyType, detector: DetectorType
) -> DetectionRun | None:
    result = await db.execute(
        select(DetectionRun)
        .where(DetectionRun.anomaly == anomaly)
        .where(DetectionRun.detector == detector)
        .where(DetectionRun.status.in_(ACTIVE_STATUSES))
        .order_by(DetectionRun.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def create_run(
    db: AsyncSession, anomaly: AnomalyType, detector: DetectorType
) -> DetectionRun:
    run = DetectionRun(
        anomaly=anomaly,
        detector=detector,
        status="pending",
        command=command_for(anomaly, detector),
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return run


async def get_run(db: AsyncSession, run_id: UUID) -> DetectionRun | None:
    result = await db.execute(select(DetectionRun).where(DetectionRun.id == run_id))
    return result.scalar_one_or_none()


async def list_runs(
    db: AsyncSession,
    *,
    anomaly: str | None = None,
    detector: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[Sequence[DetectionRun], int]:
    from sqlalchemy import func

    base = select(DetectionRun)
    if anomaly:
        base = base.where(DetectionRun.anomaly == anomaly)
    if detector:
        base = base.where(DetectionRun.detector == detector)
    if status:
        base = base.where(DetectionRun.status == status)

    count_result = await db.execute(select(func.count()).select_from(base.subquery()))
    total = count_result.scalar_one()

    items_result = await db.execute(
        base.order_by(DetectionRun.created_at.desc()).limit(limit).offset(offset)
    )
    return items_result.scalars().all(), int(total)
