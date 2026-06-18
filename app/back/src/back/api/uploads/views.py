from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

import polars as pl
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select

from back.api.deps import DbDep
from back.api.detections import crud as det_crud
from back.api.detections.schemas import DetectionRunRead
from back.api.uploads.schemas import (
    BulkDeleteRequest,
    BulkDeleteResponse,
    UploadAnomalyResult,
    UploadChannel,
    UploadList,
    UploadListItem,
    UploadResultBundle,
    UploadScorePoint,
    UploadTimePoint,
)
from back.core.config import settings
from back.models.detection_run import DetectionRun
from back.rbac.guards import require_permission

router = APIRouter(tags=["Uploads"])

_ANOMALIES = ("negermet", "pritok")
_WELL_ID_RE = re.compile(r"^[\w\-.]{1,64}$", re.UNICODE)
_MAX_SERIES_POINTS = 2000
_MAX_TELEMETRY_POINTS = 1200
_TELEMETRY_META_COLS = {"timestamp", "well_id"}


@router.post(
    "/uploads",
    response_model=DetectionRunRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_permission("detection:run"))],
)
async def create_upload(
    db: DbDep,
    file: Annotated[UploadFile, File()],
    well_id: Annotated[str, Form()],
) -> DetectionRunRead:
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
        f"--excel {excel_path} --well-id {well_id} --output-dir {out_dir} "
        "--detector paano_global --anomalies negermet,pritok "
        "--use-population-memory-bank"
    )
    run = DetectionRun(
        id=run_id,
        anomaly="multi",
        detector="paano_global",
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


def _read_score_series(scores_path: Path) -> list[UploadScorePoint]:
    if not scores_path.exists():
        return []
    sdf = pl.read_parquet(scores_path)
    if not sdf.height or not {"timestamp", "score"}.issubset(sdf.columns):
        return []
    sdf = sdf.sort("timestamp")
    stride = max(1, sdf.height // _MAX_SERIES_POINTS)
    sdf = sdf.gather_every(stride)
    return [
        UploadScorePoint(t=str(t), score=float(s))
        for t, s in zip(sdf["timestamp"].to_list(), sdf["score"].to_list(), strict=True)
    ]


def _parse_bounds(
    time_start: str | None, time_end: str | None
) -> tuple[datetime, datetime] | None:
    if not time_start or not time_end:
        return None
    try:
        return datetime.fromisoformat(time_start), datetime.fromisoformat(time_end)
    except ValueError:
        return None


def _read_telemetry(
    source_path: Path, time_start: str | None, time_end: str | None
) -> list[UploadChannel]:
    if not source_path.exists():
        return []
    df = pl.read_parquet(source_path)
    if not df.height or "timestamp" not in df.columns:
        return []
    df = df.sort("timestamp")
    bounds = _parse_bounds(time_start, time_end)
    if bounds is not None:
        lo, hi = bounds
        df = df.filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") <= hi))
    if not df.height:
        return []
    stride = max(1, df.height // _MAX_TELEMETRY_POINTS)
    df = df.gather_every(stride)
    ts = [str(t) for t in df["timestamp"].to_list()]
    channels: list[UploadChannel] = []
    for col in df.columns:
        if col in _TELEMETRY_META_COLS or not df.schema[col].is_numeric():
            continue
        points = [
            UploadTimePoint(t=t, v=float(v))
            for t, v in zip(ts, df[col].to_list(), strict=True)
            if v is not None
        ]
        if points:
            channels.append(UploadChannel(name=col, points=points))
    return channels


def _read_anomaly_result(anomaly: str, anomaly_dir: Path) -> UploadAnomalyResult:
    summary_path = anomaly_dir / "summary.json"
    error_path = anomaly_dir / "error.json"

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return UploadAnomalyResult(
            anomaly=anomaly,
            status="succeeded",
            well_id=summary.get("well_id"),
            detector=summary.get("detector"),
            n_points=summary.get("n_points"),
            n_detected=summary.get("n_detected"),
            detected_starts=summary.get("detected_starts", []),
            score_min=summary.get("score_min"),
            score_median=summary.get("score_median"),
            score_max=summary.get("score_max"),
            time_start=summary.get("time_start"),
            time_end=summary.get("time_end"),
            score_series=_read_score_series(anomaly_dir / "scores.parquet"),
            telemetry=_read_telemetry(
                anomaly_dir / "source.parquet",
                summary.get("time_start"),
                summary.get("time_end"),
            ),
        )

    if error_path.exists():
        err = json.loads(error_path.read_text(encoding="utf-8"))
        return UploadAnomalyResult(
            anomaly=anomaly,
            status="failed",
            well_id=err.get("well_id"),
            error=err.get("error"),
        )

    return UploadAnomalyResult(anomaly=anomaly, status="pending")


@router.get("/uploads/{run_id}/result", response_model=UploadResultBundle)
async def get_upload_result(run_id: uuid.UUID, db: DbDep) -> UploadResultBundle:
    run = await det_crud.get_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Прогон загрузки '{run_id}' не найден")

    out_dir = settings.uploads_root / "results" / str(run_id)
    results = [_read_anomaly_result(a, out_dir / a) for a in _ANOMALIES]
    n_done = sum(1 for r in results if r.status != "pending")

    return UploadResultBundle(
        run_id=str(run_id),
        well_id=_well_id_from_results(results) or _well_id_from_command(run.command),
        status=run.status,
        n_done=n_done,
        n_total=len(_ANOMALIES),
        results=results,
    )


def _well_id_from_results(results: list[UploadAnomalyResult]) -> str:
    return next((r.well_id for r in results if r.well_id), "")


def _well_id_from_command(command: str) -> str:
    match = re.search(r"--well-id\s+(\S+)", command)
    return match.group(1) if match else ""


def _summarize_run_dir(out_dir: Path) -> tuple[int, int]:
    n_done = 0
    n_detected = 0
    for anomaly in _ANOMALIES:
        sub = out_dir / anomaly
        if (sub / "summary.json").exists():
            n_done += 1
            try:
                summary = json.loads((sub / "summary.json").read_text(encoding="utf-8"))
                n_detected += int(summary.get("n_detected") or 0)
            except (json.JSONDecodeError, OSError):
                pass
        elif (sub / "error.json").exists():
            n_done += 1
    return n_done, n_detected


@router.get(
    "/uploads",
    response_model=UploadList,
    dependencies=[Depends(require_permission("detection:run"))],
)
async def list_uploads(db: DbDep) -> UploadList:
    result = await db.execute(
        select(DetectionRun)
        .where(DetectionRun.anomaly == "multi")
        .order_by(DetectionRun.created_at.desc())
        .limit(200)
    )
    runs = result.scalars().all()
    items: list[UploadListItem] = []
    for run in runs:
        out_dir = settings.uploads_root / "results" / str(run.id)
        n_done, n_detected = _summarize_run_dir(out_dir)
        items.append(
            UploadListItem(
                run_id=str(run.id),
                well_id=_well_id_from_command(run.command),
                status=run.status,
                created_at=run.created_at,
                n_done=n_done,
                n_total=len(_ANOMALIES),
                n_detected_total=n_detected,
            )
        )
    return UploadList(items=items)


def _delete_run_files(run_id: uuid.UUID) -> None:
    excel_path = settings.uploads_root / "excel" / f"{run_id}.xlsx"
    excel_path.unlink(missing_ok=True)
    out_dir = settings.uploads_root / "results" / str(run_id)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)


@router.delete(
    "/uploads/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_permission("detection:run"))],
)
async def delete_upload(run_id: uuid.UUID, db: DbDep) -> None:
    run = await db.scalar(
        select(DetectionRun).where(
            DetectionRun.id == run_id, DetectionRun.anomaly == "multi"
        )
    )
    if run is None:
        raise HTTPException(status_code=404, detail=f"Загрузка '{run_id}' не найдена")
    _delete_run_files(run_id)
    await db.delete(run)
    await db.commit()


@router.post(
    "/uploads/bulk-delete",
    response_model=BulkDeleteResponse,
    dependencies=[Depends(require_permission("detection:run"))],
)
async def bulk_delete_uploads(req: BulkDeleteRequest, db: DbDep) -> BulkDeleteResponse:
    if not req.run_ids:
        return BulkDeleteResponse(deleted=0)
    try:
        uuids = [uuid.UUID(rid) for rid in req.run_ids]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Некорректный run_id: {exc}") from exc
    result = await db.execute(
        select(DetectionRun).where(
            DetectionRun.id.in_(uuids), DetectionRun.anomaly == "multi"
        )
    )
    runs = list(result.scalars().all())
    for run in runs:
        _delete_run_files(run.id)
        await db.delete(run)
    await db.commit()
    return BulkDeleteResponse(deleted=len(runs))
