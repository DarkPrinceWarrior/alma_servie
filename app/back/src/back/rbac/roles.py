from enum import StrEnum


class RoleCode(StrEnum):
    USER = "user"
    ADMIN = "admin"


ROLE_IMPLICATIONS: dict[RoleCode, set[RoleCode]] = {
    RoleCode.USER: {RoleCode.USER},
    RoleCode.ADMIN: {RoleCode.ADMIN, RoleCode.USER},
}


BASE_ROLES: list[tuple[str, str]] = [
    (RoleCode.USER.value, "Пользователь"),
    (RoleCode.ADMIN.value, "Администратор"),
]
