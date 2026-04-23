from back.rbac.roles import RoleCode

PERMISSIONS: dict[str, set[str]] = {
    "wells:read": {RoleCode.USER.value, RoleCode.ADMIN.value},
    "detection:run": {RoleCode.USER.value, RoleCode.ADMIN.value},
    "detection:admin": {RoleCode.ADMIN.value},
    "users:read": {RoleCode.ADMIN.value},
    "users:admin": {RoleCode.ADMIN.value},
}
