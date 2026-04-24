from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from back.api.detections.views import router as detections_router
from back.api.health.views import router as health_router
from back.api.wells.views import router as wells_router
from back.core.config import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(health_router, prefix="/api")
app.include_router(wells_router, prefix="/api")
app.include_router(detections_router, prefix="/api")


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running"}


def main() -> None:
    uvicorn.run("back.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
