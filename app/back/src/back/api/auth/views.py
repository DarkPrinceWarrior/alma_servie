from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordRequestForm
from sqlalchemy import select

from back.api.auth.crud import (
    blacklist_access_token,
    create_refresh_token_record,
    get_refresh_token_by_hash,
    revoke_all_user_refresh_tokens,
    revoke_refresh_token,
)
from back.api.auth.schemas import TokenResponse
from back.api.deps import DbDep, UserDep
from back.core.config import settings
from back.core.security import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    hash_token,
    verify_password,
)
from back.models.user import User

router = APIRouter(prefix="/auth", tags=["Auth"])

_bearer = HTTPBearer(auto_error=True)
_RT_COOKIE = "refresh_token"
_RT_PATH = "/api/auth/refresh"


def _rt_max_age() -> int:
    return settings.refresh_token_expire_days * 24 * 60 * 60


def _set_refresh_cookie(response: Response, raw_token: str) -> None:
    response.set_cookie(
        key=_RT_COOKIE,
        value=raw_token,
        httponly=True,
        secure=settings.refresh_cookie_secure,
        samesite="lax",
        max_age=_rt_max_age(),
        path=_RT_PATH,
    )


def _delete_refresh_cookie(response: Response) -> None:
    response.delete_cookie(key=_RT_COOKIE, path=_RT_PATH)


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    response: Response,
    db: DbDep,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> TokenResponse:
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="user_not_found")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="account_inactive")
    if not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_password")

    access_token, _jti = create_access_token(subject=str(user.id), extra={"email": user.email})
    raw_refresh, refresh_hash, expires_at = create_refresh_token(str(user.id))

    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent", "")[:256]

    await create_refresh_token_record(
        db,
        user_id=user.id,
        token_hash=refresh_hash,
        jti=_jti,
        expires_at=expires_at,
        ip_address=ip,
        user_agent=ua,
    )
    await db.commit()

    _set_refresh_cookie(response, raw_refresh)

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(
    response: Response,
    request: Request,
    db: DbDep,
) -> TokenResponse:
    raw_refresh = request.cookies.get(_RT_COOKIE)
    if not raw_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="refresh_token_missing"
        )

    payload = decode_refresh_token(raw_refresh)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="refresh_token_invalid"
        )

    record = await get_refresh_token_by_hash(db, hash_token(raw_refresh))
    if record is None or record.is_expired:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="refresh_token_expired"
        )

    if record.is_revoked:
        await revoke_all_user_refresh_tokens(db, record.user_id)
        await db.commit()
        _delete_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="refresh_token_reused_sessions_revoked",
        )

    await revoke_refresh_token(db, record)

    sub = payload["sub"]
    access_token, _jti = create_access_token(subject=sub)
    new_raw_refresh, refresh_hash, expires_at = create_refresh_token(sub)

    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent", "")[:256]

    await create_refresh_token_record(
        db,
        user_id=record.user_id,
        token_hash=refresh_hash,
        jti=_jti,
        expires_at=expires_at,
        ip_address=ip,
        user_agent=ua,
    )
    await db.commit()

    _set_refresh_cookie(response, new_raw_refresh)

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/logout")
async def logout(
    response: Response,
    request: Request,
    db: DbDep,
    current_user: UserDep,
    creds: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
) -> dict[str, str]:
    del current_user

    at_payload = decode_access_token(creds.credentials)
    if at_payload:
        jti = at_payload.get("jti")
        exp = at_payload.get("exp")
        if jti and exp:
            if isinstance(exp, int | float):
                expires_at = datetime.fromtimestamp(exp, tz=UTC)
            else:
                expires_at = exp if exp.tzinfo else exp.replace(tzinfo=UTC)
            await blacklist_access_token(db, jti=jti, expires_at=expires_at)

    raw_refresh = request.cookies.get(_RT_COOKIE)
    if raw_refresh:
        record = await get_refresh_token_by_hash(db, hash_token(raw_refresh))
        if record and not record.is_revoked:
            await revoke_refresh_token(db, record)

    await db.commit()
    _delete_refresh_cookie(response)
    return {"detail": "logged_out"}
