from __future__ import annotations

import asyncio
import shlex
from datetime import UTC, datetime
from uuid import UUID

from loguru import logger
from sqlalchemy import select

from back.core.config import settings
from back.db.database import AsyncSessionLocal
from back.models.detection_run import DetectionRun

_STDOUT_TAIL_LIMIT = 8000


async def _mark_started(run_id: UUID) -> None:
    async with AsyncSessionLocal() as db:
        run = await db.scalar(select(DetectionRun).where(DetectionRun.id == run_id))
        if run is None:
            return
        run.status = "running"
        run.started_at = datetime.now(UTC)
        await db.commit()


async def _mark_finished(
    run_id: UUID,
    *,
    status: str,
    exit_code: int | None,
    stdout_tail: str | None,
    summary_json: dict | None,
    error_message: str | None,
) -> None:
    async with AsyncSessionLocal() as db:
        run = await db.scalar(select(DetectionRun).where(DetectionRun.id == run_id))
        if run is None:
            return
        run.status = status
        run.exit_code = exit_code
        run.stdout_tail = stdout_tail
        run.summary_json = summary_json
        run.error_message = error_message
        run.finished_at = datetime.now(UTC)
        await db.commit()


async def _run_mock(run_id: UUID, anomaly: str, detector: str) -> None:
    await _mark_started(run_id)
    await asyncio.sleep(settings.detection_mock_duration_seconds)
    summary = {
        "mock": True,
        "anomaly": anomaly,
        "detector": detector,
        "n_wells": 0,
        "n_onsets": 0,
    }
    await _mark_finished(
        run_id,
        status="succeeded",
        exit_code=0,
        stdout_tail=f"[mock] detection for {anomaly}/{detector} completed\n",
        summary_json=summary,
        error_message=None,
    )


async def _run_subprocess(run_id: UUID, command: str) -> None:
    await _mark_started(run_id)
    logger.info(f"[detection {run_id}] launching: {command}")

    try:
        proc = await asyncio.create_subprocess_exec(
            *shlex.split(command),
            cwd=str(settings.research_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout_bytes, _ = await proc.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        tail = stdout[-_STDOUT_TAIL_LIMIT:]
        status = "succeeded" if proc.returncode == 0 else "failed"
        await _mark_finished(
            run_id,
            status=status,
            exit_code=proc.returncode,
            stdout_tail=tail,
            summary_json=None,
            error_message=None if status == "succeeded" else "non-zero exit code",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"[detection {run_id}] crashed")
        await _mark_finished(
            run_id,
            status="failed",
            exit_code=None,
            stdout_tail=None,
            summary_json=None,
            error_message=f"{type(exc).__name__}: {exc}",
        )


async def execute(run_id: UUID, anomaly: str, detector: str, command: str) -> None:
    if settings.detection_mock:
        await _run_mock(run_id, anomaly, detector)
    else:
        await _run_subprocess(run_id, command)


def schedule(run_id: UUID, anomaly: str, detector: str, command: str) -> asyncio.Task:
    return asyncio.create_task(execute(run_id, anomaly, detector, command))
