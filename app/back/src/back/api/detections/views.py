from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse

from back.api.deps import DbDep
from back.api.detections import crud, service
from back.api.detections.schemas import (
    DetectionRunCreate,
    DetectionRunList,
    DetectionRunRead,
)
from back.core.config import settings
from back.rbac.guards import require_permission

router = APIRouter(tags=["Detections"])


@router.post(
    "/detections",
    response_model=DetectionRunRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_permission("detection:run"))],
)
async def launch_detection(payload: DetectionRunCreate, db: DbDep) -> DetectionRunRead:
    active = await crud.find_active(db, payload.anomaly, payload.detector)
    if active is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "active_run_exists",
                "run_id": str(active.id),
                "status": active.status,
            },
        )
    run = await crud.create_run(db, payload.anomaly, payload.detector)
    if settings.detection_mock:
        service.schedule(run.id, payload.anomaly, payload.detector, run.command)
    return DetectionRunRead.model_validate(run)


@router.get("/detections", response_model=DetectionRunList)
async def list_detections(
    db: DbDep,
    anomaly: Annotated[str | None, Query()] = None,
    detector: Annotated[str | None, Query()] = None,
    run_status: Annotated[str | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DetectionRunList:
    items, total = await crud.list_runs(
        db,
        anomaly=anomaly,
        detector=detector,
        status=run_status,
        limit=limit,
        offset=offset,
    )
    return DetectionRunList(
        items=[DetectionRunRead.model_validate(x) for x in items],
        total=total,
    )


@router.get("/detections/{run_id}", response_model=DetectionRunRead)
async def get_detection(run_id: UUID, db: DbDep) -> DetectionRunRead:
    run = await crud.get_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Detection run '{run_id}' not found")
    return DetectionRunRead.model_validate(run)


@router.get("/detections/{run_id}/report", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
async def get_detection_report(run_id: UUID, db: DbDep) -> RedirectResponse:
    run = await crud.get_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Detection run '{run_id}' not found")
    return RedirectResponse(
        url=f"/api/reports/{run.anomaly}/{run.detector}/html",
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
    )
