from __future__ import annotations

import json
import re
import uuid
from typing import Annotated

import polars as pl
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from back.api.deps import DbDep
from back.api.detections import crud as det_crud
from back.api.detections.schemas import DetectionRunRead
from back.api.uploads.schemas import UploadResult, UploadScorePoint
from back.core.config import settings
from back.models.detection_run import DetectionRun
from back.rbac.guards import require_permission

router = APIRouter(tags=["Uploads"])

_ALLOWED_ANOMALIES = {"negermet", "pritok", "salt"}
_WELL_ID_RE = re.compile(r"^[\w\-.]{1,64}$", re.UNICODE)
_MAX_SERIES_POINTS = 2000


@router.post(
    "/uploads",
    response_model=DetectionRunRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_permission("detection:run"))],
)
async def create_upload(
    db: DbDep,
    file: Annotated[UploadFile, File()],
    anomaly: Annotated[str, Form()],
    well_id: Annotated[str, Form()],
) -> DetectionRunRead:
    if anomaly not in _ALLOWED_ANOMALIES:
        raise HTTPException(status_code=422, detail=f"Неизвестный класс аномалии '{anomaly}'")
    well_id = well_id.strip()
    if not _WELL_ID_RE.match(well_id):
        raise HTTPException(
            status_code=422,
            detail="well_id: только буквы/цифры/-/. (до 64 символов)",
        )
    if not (file.filename or "").lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=422, detail="Ожидается файл .xlsx")

    run_id = uuid.uuid4()
    excel_dir = settings.uploads_root / "excel"
    excel_dir.mkdir(parents=True, exist_ok=True)
    excel_path = excel_dir / f"{run_id}.xlsx"
    excel_path.write_bytes(await file.read())

    out_dir = settings.uploads_root / "results" / str(run_id)
    command = (
        "python scripts/detection/detect_uploaded_well.py "
        f"--anomaly {anomaly} --excel {excel_path} "
        f"--well-id {well_id} --output-dir {out_dir}"
    )
    run = DetectionRun(
        id=run_id,
        anomaly=anomaly,
        detector="paano_shared",
        status="pending",
        command=command,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return DetectionRunRead.model_validate(run)


@router.get("/uploads/{run_id}", response_model=DetectionRunRead)
async def get_upload(run_id: uuid.UUID, db: DbDep) -> DetectionRunRead:
    run = await det_crud.get_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Прогон загрузки '{run_id}' не найден")
    return DetectionRunRead.model_validate(run)


@router.get("/uploads/{run_id}/result", response_model=UploadResult)
async def get_upload_result(run_id: uuid.UUID, db: DbDep) -> UploadResult:
    run = await det_crud.get_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Прогон загрузки '{run_id}' не найден")

    out_dir = settings.uploads_root / "results" / str(run_id)
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Результат ещё не готов (статус прогона: {run.status})",
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    series: list[UploadScorePoint] = []
    scores_path = out_dir / "scores.parquet"
    if scores_path.exists():
        sdf = pl.read_parquet(scores_path)
        if sdf.height and {"timestamp", "score"}.issubset(sdf.columns):
            sdf = sdf.sort("timestamp")
            stride = max(1, sdf.height // _MAX_SERIES_POINTS)
            sdf = sdf.gather_every(stride)
            series = [
                UploadScorePoint(t=str(t), score=float(s))
                for t, s in zip(
                    sdf["timestamp"].to_list(), sdf["score"].to_list(), strict=True
                )
            ]

    return UploadResult(
        run_id=str(run_id),
        anomaly=summary["anomaly"],
        well_id=summary["well_id"],
        detector=summary["detector"],
        status=run.status,
        n_points=summary["n_points"],
        n_detected=summary["n_detected"],
        detected_starts=summary["detected_starts"],
        score_min=summary.get("score_min"),
        score_median=summary.get("score_median"),
        score_max=summary.get("score_max"),
        time_start=summary.get("time_start"),
        time_end=summary.get("time_end"),
        score_series=series,
    )
