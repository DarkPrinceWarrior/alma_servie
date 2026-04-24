from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from back.api.auth.crud import revoke_all_user_refresh_tokens
from back.api.deps import AdminUserDep, DbDep, UserDep
from back.api.users.crud import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    list_users,
    set_user_password,
)
from back.api.users.schemas import (
    AdminChangePasswordRequest,
    PasswordActionResponse,
    UserCreateRequest,
    UserRead,
)

router = APIRouter(tags=["Users"])


@router.get("/users/me", response_model=UserRead)
async def get_me(current_user: UserDep) -> UserRead:
    return UserRead.from_orm_user(current_user)


@router.get("/users", response_model=list[UserRead])
async def list_users_endpoint(
    db: DbDep,
    current_user: AdminUserDep,
    role: Annotated[str | None, Query(description="Role code filter")] = None,
    search: Annotated[str | None, Query(description="Search by email")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[UserRead]:
    del current_user
    users = await list_users(db, role_filter=role, search=search, limit=limit, offset=offset)
    return [UserRead.from_orm_user(u) for u in users]


@router.post("/users", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user_endpoint(
    db: DbDep,
    current_user: AdminUserDep,
    payload: UserCreateRequest,
) -> UserRead:
    del current_user
    if await get_user_by_email(db, payload.email) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Пользователь с таким email уже существует",
        )
    user = await create_user(
        db,
        email=payload.email,
        password=payload.password,
        is_active=payload.is_active,
        role_codes=[code.value for code in payload.role_codes],
    )
    return UserRead.from_orm_user(user)


@router.get("/users/{user_id}", response_model=UserRead)
async def get_user_endpoint(
    user_id: UUID,
    db: DbDep,
    current_user: AdminUserDep,
) -> UserRead:
    del current_user
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserRead.from_orm_user(user)


@router.patch("/users/{user_id}/password", response_model=PasswordActionResponse)
async def admin_set_user_password(
    user_id: UUID,
    body: AdminChangePasswordRequest,
    db: DbDep,
    current_user: AdminUserDep,
) -> PasswordActionResponse:
    del current_user
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    await set_user_password(db, user, body.new_password)
    await revoke_all_user_refresh_tokens(db, user.id)
    await db.commit()
    return PasswordActionResponse(message="Password updated")
