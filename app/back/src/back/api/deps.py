import uuid
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from back.core.config import settings
from back.core.security import decode_access_token
from back.db.database import get_db
from back.models.user import User
from back.rbac.roles import RoleCode

DbDep = Annotated[AsyncSession, Depends(get_db)]


def get_data_root() -> Path:
    return settings.data_root


DataRootDep = Annotated[Path, Depends(get_data_root)]

_bearer_scheme = HTTPBearer(auto_error=True)


async def get_current_user(
    db: DbDep,
    creds: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_scheme)],
) -> User:
    from back.api.auth.crud import is_jti_blacklisted

    token = creds.credentials
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    jti = payload.get("jti")
    if jti and await is_jti_blacklisted(db, jti):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload"
        )

    try:
        uid = uuid.UUID(str(sub))
        result = await db.execute(
            select(User).options(selectinload(User.roles)).where(User.id == uid)
        )
    except (ValueError, TypeError):
        result = await db.execute(
            select(User).options(selectinload(User.roles)).where(User.email == str(sub))
        )

    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User inactive or not found",
        )

    return user


UserDep = Annotated[User, Depends(get_current_user)]


def _has_role(user: User, code: str) -> bool:
    return any(getattr(r, "code", None) == code for r in getattr(user, "roles", []))


async def require_admin(current_user: UserDep) -> User:
    if not _has_role(current_user, RoleCode.ADMIN.value):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admins only")
    return current_user


AdminUserDep = Annotated[User, Depends(require_admin)]
