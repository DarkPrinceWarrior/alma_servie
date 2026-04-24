from httpx import ASGITransport, AsyncClient

from back.core.security import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    hash_token,
    verify_password,
)
from back.main import app


def test_password_hash_roundtrip() -> None:
    h = get_password_hash("secret-pass-123")
    assert h != "secret-pass-123"
    assert verify_password("secret-pass-123", h) is True
    assert verify_password("wrong", h) is False


def test_access_token_roundtrip() -> None:
    token, jti = create_access_token(subject="user-1")
    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "user-1"
    assert payload["jti"] == jti
    assert payload["typ"] == "access"


def test_decode_access_rejects_refresh_typ() -> None:
    from back.core.security import create_refresh_token

    raw, _, _ = create_refresh_token("user-1")
    assert decode_access_token(raw) is None


def test_hash_token_is_sha256_hex() -> None:
    h = hash_token("abc")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


async def test_users_me_requires_auth() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/users/me")
    assert r.status_code in (401, 403)


async def test_users_me_with_override(auth_user) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get(
            "/api/users/me",
            headers={"Authorization": "Bearer dummy"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["email"] == "fake@test"
    assert "admin" in body["roles"]
