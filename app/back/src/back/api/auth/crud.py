from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from back.models.refresh_token import RefreshToken
from back.models.token_blacklist import TokenBlacklist


async def create_refresh_token_record(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    token_hash: str,
    jti: str,
    expires_at: datetime,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> RefreshToken:
    record = RefreshToken(
        user_id=user_id,
        token_hash=token_hash,
        jti=jti,
        expires_at=expires_at,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(record)
    await db.flush()
    return record


async def get_refresh_token_by_hash(db: AsyncSession, token_hash: str) -> RefreshToken | None:
    result = await db.execute(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
    return result.scalar_one_or_none()


async def revoke_refresh_token(db: AsyncSession, record: RefreshToken) -> None:
    record.revoked_at = datetime.now(UTC)
    db.add(record)
    await db.flush()


async def revoke_all_user_refresh_tokens(db: AsyncSession, user_id: uuid.UUID) -> None:
    now = datetime.now(UTC)
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=now)
        .execution_options(synchronize_session="fetch")
    )
    await db.flush()


async def blacklist_access_token(db: AsyncSession, *, jti: str, expires_at: datetime) -> None:
    entry = TokenBlacklist(jti=jti, expires_at=expires_at)
    db.add(entry)
    await db.flush()


async def is_jti_blacklisted(db: AsyncSession, jti: str) -> bool:
    result = await db.execute(select(TokenBlacklist).where(TokenBlacklist.jti == jti))
    return result.scalar_one_or_none() is not None


async def cleanup_expired_tokens(db: AsyncSession) -> None:
    now = datetime.now(UTC)
    await db.execute(delete(TokenBlacklist).where(TokenBlacklist.expires_at < now))
    await db.execute(delete(RefreshToken).where(RefreshToken.expires_at < now))
    await db.commit()
