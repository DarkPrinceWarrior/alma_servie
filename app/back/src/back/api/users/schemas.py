from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from back.rbac.roles import RoleCode


class UserRead(BaseModel):
    id: uuid.UUID
    email: str
    is_active: bool
    roles: list[RoleCode]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_orm_user(cls, user) -> UserRead:
        return cls(
            id=user.id,
            email=user.email,
            is_active=user.is_active,
            roles=[RoleCode(r.code) for r in getattr(user, "roles", []) if r.code],
            created_at=user.created_at,
            updated_at=user.updated_at,
        )


class UserCreateRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    is_active: bool = True
    role_codes: list[RoleCode] = Field(default_factory=lambda: [RoleCode.USER])


class AdminChangePasswordRequest(BaseModel):
    new_password: str = Field(min_length=8, max_length=128)


class PasswordActionResponse(BaseModel):
    message: str
