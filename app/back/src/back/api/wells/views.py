from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from back.api.deps import DataRootDep
from back.api.wells import crud
from back.api.wells.schemas import (
    AnomalyType,
    WellDetail,
    WellInterval,
    WellSummary,
)

router = APIRouter(tags=["Wells"])

AnomalyQuery = Annotated[AnomalyType, Query(description="Anomaly class")]


@router.get("/wells", response_model=list[WellSummary])
async def list_wells(data_root: DataRootDep, anomaly: AnomalyQuery) -> list[WellSummary]:
    return crud.list_wells(data_root, anomaly)


@router.get("/wells/{well_id}", response_model=WellDetail)
async def get_well(well_id: str, data_root: DataRootDep, anomaly: AnomalyQuery) -> WellDetail:
    well = crud.get_well(data_root, anomaly, well_id)
    if well is None:
        raise HTTPException(
            status_code=404, detail=f"Well '{well_id}' not found for anomaly '{anomaly}'"
        )
    return well


@router.get("/wells/{well_id}/intervals", response_model=list[WellInterval])
async def get_well_intervals(
    well_id: str, data_root: DataRootDep, anomaly: AnomalyQuery
) -> list[WellInterval]:
    well = crud.get_well(data_root, anomaly, well_id)
    if well is None:
        raise HTTPException(
            status_code=404, detail=f"Well '{well_id}' not found for anomaly '{anomaly}'"
        )
    return well.intervals
