from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from back.api.deps import DbDep
from back.api.health.schemas import DbHealthResponse, HealthResponse
from back.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, operation_id="healthcheck")
async def healthcheck() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get("/health/db", response_model=DbHealthResponse, operation_id="healthcheck_db")
async def healthcheck_db(db: DbDep) -> DbHealthResponse:
    try:
        await db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc

    return DbHealthResponse(
        status="ok",
        database="connected",
    )
