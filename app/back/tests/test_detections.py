from httpx import ASGITransport, AsyncClient

from back.api.detections.crud import command_for
from back.main import app


def test_command_for_format() -> None:
    assert (
        command_for("negermet", "pca_spe")
        == "python scripts/detection/detect_negermet.py --detector pca_spe"
    )
    assert (
        command_for("salt", "paano_shared")
        == "python scripts/detection/detect_salt.py --detector paano_shared"
    )


async def test_post_requires_auth() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post(
            "/api/detections",
            json={"anomaly": "negermet", "detector": "pca_spe"},
        )
    assert r.status_code in (401, 403)


async def test_post_validates_anomaly(auth_user) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post(
            "/api/detections", json={"anomaly": "bogus", "detector": "pca_spe"}
        )
    assert r.status_code == 422


async def test_post_validates_detector(auth_user) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post(
            "/api/detections",
            json={"anomaly": "negermet", "detector": "bogus_det"},
        )
    assert r.status_code == 422
