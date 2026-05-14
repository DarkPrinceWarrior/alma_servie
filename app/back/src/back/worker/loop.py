from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from loguru import logger
from sqlalchemy import select

from back.core.config import settings
from back.db.database import AsyncSessionLocal
from back.models.detection_run import DetectionRun

POLL_INTERVAL_SECONDS = 2.0
STDOUT_TAIL_LIMIT = 8000


async def claim_pending() -> DetectionRun | None:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(DetectionRun)
            .where(DetectionRun.status == "pending")
            .order_by(DetectionRun.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        run = result.scalar_one_or_none()
        if run is None:
            return None
        run.status = "running"
        run.started_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(run)
        return run


async def finalize(
    run_id: UUID,
    *,
    status: str,
    exit_code: int | None,
    stdout_tail: str | None,
    summary_json: dict[str, Any] | None,
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


async def execute(run: DetectionRun) -> None:
    logger.info(f"[worker] launching run={run.id} cmd={run.command!r}")
    try:
        # Run research scripts from the rw data mount so alma_service resolves
        # PROJECT_ROOT to /data (real models/db/artifacts), not the baked /build copy.
        run_root = str(settings.data_root)
        env = {**os.environ, "PYTHONPATH": run_root}
        proc = await asyncio.create_subprocess_exec(
            *shlex.split(run.command),
            cwd=run_root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout_bytes, _ = await proc.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        tail = stdout[-STDOUT_TAIL_LIMIT:]
        status = "succeeded" if proc.returncode == 0 else "failed"
        await finalize(
            run.id,
            status=status,
            exit_code=proc.returncode,
            stdout_tail=tail,
            summary_json=None,
            error_message=None if status == "succeeded" else "non-zero exit code",
        )
        logger.info(f"[worker] run={run.id} finished with status={status} exit={proc.returncode}")
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"[worker] run={run.id} crashed")
        await finalize(
            run.id,
            status="failed",
            exit_code=None,
            stdout_tail=None,
            summary_json=None,
            error_message=f"{type(exc).__name__}: {exc}",
        )


async def loop(stop: asyncio.Event) -> None:
    logger.info(f"[worker] started; poll_interval={POLL_INTERVAL_SECONDS}s")
    while not stop.is_set():
        try:
            run = await claim_pending()
        except Exception:  # noqa: BLE001
            logger.exception("[worker] claim_pending failed; sleeping")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            continue

        if run is None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=POLL_INTERVAL_SECONDS)
            continue

        await execute(run)
    logger.info("[worker] shutdown")


def _install_signal_handlers(stop: asyncio.Event) -> None:
    loop_ = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop_.add_signal_handler(sig, stop.set)


async def main() -> None:
    stop = asyncio.Event()
    _install_signal_handlers(stop)
    await loop(stop)


if __name__ == "__main__":
    asyncio.run(main())
