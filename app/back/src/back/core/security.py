from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from back.core.config import settings

ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_access_token(
    *,
    subject: str,
    extra: dict[str, Any] | None = None,
    expires_delta: timedelta | None = None,
) -> tuple[str, str]:
    jti = str(uuid.uuid4())
    to_encode: dict[str, Any] = {
        "sub": subject,
        "jti": jti,
        "typ": "access",
        "iat": datetime.now(UTC),
    }
    if extra:
        to_encode.update(extra)
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode["exp"] = expire
    token = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return token, jti


def create_refresh_token(user_id: str) -> tuple[str, str, datetime]:
    jti = str(uuid.uuid4())
    now = datetime.now(UTC)
    expires_at = now + timedelta(days=settings.refresh_token_expire_days)
    payload: dict[str, Any] = {
        "sub": user_id,
        "jti": jti,
        "typ": "refresh",
        "iat": now,
        "exp": expires_at,
    }
    raw_token = jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)
    return raw_token, hash_token(raw_token), expires_at


def decode_access_token(token: str) -> dict[str, Any] | None:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        if payload.get("typ") != "access":
            return None
        return payload
    except JWTError:
        return None


def decode_refresh_token(token: str) -> dict[str, Any] | None:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        if payload.get("typ") != "refresh":
            return None
        return payload
    except JWTError:
        return None
