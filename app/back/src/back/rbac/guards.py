from fastapi import Depends, HTTPException, status

from back.models.user import User
from back.rbac.permissions import PERMISSIONS
from back.rbac.roles import ROLE_IMPLICATIONS, RoleCode


def _effective_role_codes(user: User) -> set[str]:
    assigned = {getattr(role, "code", None) for role in getattr(user, "roles", [])}
    assigned.discard(None)

    implied: set[str] = set()
    for code in assigned:
        try:
            role = RoleCode(code)
        except ValueError:
            implied.add(code)
            continue
        for implication in ROLE_IMPLICATIONS.get(role, {role}):
            implied.add(implication.value)
    return implied


def has_role(user: User, code: str) -> bool:
    return code in _effective_role_codes(user)


def require_permission(permission: str):
    from back.api.deps import get_current_user

    async def _dep(current_user: User = Depends(get_current_user)) -> User:  # noqa: B008
        allowed = PERMISSIONS.get(permission, set())
        user_roles = _effective_role_codes(current_user)
        if not (allowed & user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required",
            )
        return current_user

    return _dep
