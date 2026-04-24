from sqlalchemy import Column, ForeignKey, Integer, Table, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from back.db.base import Base

user_roles = Table(
    "user_roles",
    Base.metadata,
    Column(
        "user_id",
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "role_id",
        Integer,
        ForeignKey("roles.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    UniqueConstraint("user_id", "role_id", name="uq_user_role"),
)
