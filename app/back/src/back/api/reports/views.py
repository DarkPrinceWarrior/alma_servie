from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from back.api.deps import DataRootDep
from back.api.detections.schemas import DetectorType
from back.api.reports import crud
from back.api.reports.schemas import PredictedStart, ScoreSeries
from back.api.wells.schemas import AnomalyType
from back.services.paths import html_report_path

router = APIRouter(tags=["Reports"])


@router.get("/reports/{anomaly}/{detector}/html")
async def get_report_html(
    anomaly: AnomalyType, detector: DetectorType, data_root: DataRootDep
) -> FileResponse:
    path = html_report_path(data_root, anomaly, detector)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Report for '{anomaly}/{detector}' not found")
    return FileResponse(
        path=path,
        media_type="text/html; charset=utf-8",
        filename=path.name,
        headers={"Content-Security-Policy": "frame-ancestors *"},
    )


@router.get("/reports/{anomaly}/{detector}/scores", response_model=ScoreSeries)
async def get_scores(
    anomaly: AnomalyType,
    detector: DetectorType,
    data_root: DataRootDep,
    well_id: Annotated[str | None, Query(description="Filter by well_id")] = None,
    frm: Annotated[datetime | None, Query(alias="from", description="Inclusive start")] = None,
    to: Annotated[datetime | None, Query(description="Inclusive end")] = None,
    limit: Annotated[
        int, Query(ge=1, le=20000, description="Max points after downsampling")
    ] = 2000,
) -> ScoreSeries:
    series = crud.load_scores(
        data_root, anomaly, detector, well_id=well_id, frm=frm, to=to, limit=limit
    )
    if series is None:
        raise HTTPException(
            status_code=404,
            detail=f"Scores parquet for '{anomaly}/{detector}' not found",
        )
    return series


@router.get("/reports/{anomaly}/{detector}/starts", response_model=list[PredictedStart])
async def get_predicted_starts(
    anomaly: AnomalyType,
    detector: DetectorType,
    data_root: DataRootDep,
    well_id: Annotated[str | None, Query()] = None,
    split: Annotated[str | None, Query()] = None,
) -> list[PredictedStart]:
    starts = crud.load_predicted_starts(data_root, anomaly, detector, well_id=well_id, split=split)
    if starts is None:
        raise HTTPException(
            status_code=404,
            detail=f"Predicted starts parquet for '{anomaly}/{detector}' not found",
        )
    return starts
