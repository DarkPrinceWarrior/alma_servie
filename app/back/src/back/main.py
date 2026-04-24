import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from back.api.auth.views import router as auth_router
from back.api.detections.views import router as detections_router
from back.api.health.views import router as health_router
from back.api.reports.views import router as reports_router
from back.api.users.views import router as users_router
from back.api.wells.views import router as wells_router
from back.core.config import settings
from back.db.database import AsyncSessionLocal
from back.rbac.roles import BASE_ROLES, RoleCode


async def _ensure_base_roles() -> None:
    from back.models.role import Role

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Role.code))
        existing = {code for (code,) in result.all()}
        for code, name in BASE_ROLES:
            if code not in existing:
                db.add(Role(code=code, name=name))
        await db.commit()


async def _seed_admin_from_env() -> None:
    email = settings.admin_email
    password = settings.admin_password
    if not email or not password:
        return

    from back.core.security import get_password_hash
    from back.models.role import Role
    from back.models.user import User

    async with AsyncSessionLocal() as db, db.begin():
        result = await db.execute(select(Role).where(Role.code == RoleCode.ADMIN.value))
        admin_role = result.scalar_one_or_none()
        if admin_role is None:
            logger.warning("[seed] ADMIN role missing; skipping admin seed")
            return

        result = await db.execute(
            select(User).options(selectinload(User.roles)).where(User.email == email)
        )
        user = result.scalar_one_or_none()

        if user is None:
            user = User(email=email, password_hash=get_password_hash(password), is_active=True)
            user.roles = [admin_role]
            db.add(user)
        else:
            if settings.admin_reset_password:
                user.password_hash = get_password_hash(password)
            if admin_role not in user.roles:
                user.roles.append(admin_role)

    logger.info(f"[seed] admin ensured: {email}")


async def _cleanup_expired_tokens_loop() -> None:
    from back.api.auth.crud import cleanup_expired_tokens

    while True:
        try:
            async with AsyncSessionLocal() as db:
                await cleanup_expired_tokens(db)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[token_cleanup] error: {exc}")
        await asyncio.sleep(86_400)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await _ensure_base_roles()
    await _seed_admin_from_env()
    cleanup_task = asyncio.create_task(_cleanup_expired_tokens_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(health_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(users_router, prefix="/api")
app.include_router(wells_router, prefix="/api")
app.include_router(detections_router, prefix="/api")
app.include_router(reports_router, prefix="/api")


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running"}


def main() -> None:
    uvicorn.run("back.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
