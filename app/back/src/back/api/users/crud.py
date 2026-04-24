from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from back.core.security import get_password_hash
from back.models.role import Role
from back.models.user import User


async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> User | None:
    result = await db.execute(
        select(User).options(selectinload(User.roles)).where(User.id == user_id)
    )
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(
        select(User).options(selectinload(User.roles)).where(User.email == email)
    )
    return result.scalar_one_or_none()


async def list_users(
    db: AsyncSession,
    *,
    role_filter: str | None = None,
    search: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> Sequence[User]:
    stmt = select(User).options(selectinload(User.roles))

    if role_filter:
        stmt = stmt.join(User.roles).where(Role.code == role_filter).distinct()

    if search:
        stmt = stmt.where(func.lower(User.email).contains(search.lower()))

    stmt = stmt.order_by(User.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(stmt)
    return result.unique().scalars().all()


async def create_user(
    db: AsyncSession,
    *,
    email: str,
    password: str,
    is_active: bool = True,
    role_codes: Sequence[str] = (),
) -> User:
    user = User(
        email=email,
        password_hash=get_password_hash(password),
        is_active=is_active,
    )
    db.add(user)
    await db.flush()

    if role_codes:
        result = await db.execute(select(Role).where(Role.code.in_(role_codes)))
        user.roles = list(result.scalars().all())

    await db.commit()
    await db.refresh(user, attribute_names=["roles"])
    return user


async def set_user_password(db: AsyncSession, user: User, new_password: str) -> User:
    user.password_hash = get_password_hash(new_password)
    user.is_active = True
    db.add(user)
    await db.commit()
    return user
