from httpx import ASGITransport, AsyncClient

from back.api.reports.crud import _downsample_stride
from back.main import app


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def test_downsample_stride() -> None:
    assert _downsample_stride(100, 2000) == 1
    assert _downsample_stride(2000, 2000) == 1
    assert _downsample_stride(6000, 2000) == 3
    assert _downsample_stride(5001, 2000) == 3
    assert _downsample_stride(0, 2000) == 1


async def test_get_html_ok(data_root) -> None:
    async with await _client() as c:
        r = await c.get("/api/reports/negermet/paano_shared/html")

    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Report negermet/paano_shared" in r.text
    assert r.headers.get("content-security-policy") == "frame-ancestors *"


async def test_get_html_404(data_root) -> None:
    async with await _client() as c:
        r = await c.get("/api/reports/pritok/paano_shared/html")

    assert r.status_code == 404


async def test_get_scores_downsamples(data_root) -> None:
    async with await _client() as c:
        r = await c.get(
            "/api/reports/negermet/paano_shared/scores",
            params={"limit": 500},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["n_points"] == 3000
    assert body["n_downsampled"] <= 500
    assert len(body["points"]) == body["n_downsampled"]


async def test_get_scores_filter_well(data_root) -> None:
    async with await _client() as c:
        r = await c.get(
            "/api/reports/negermet/paano_shared/scores",
            params={"well_id": "W-200", "limit": 5000},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["n_points"] == 1500
    for pt in body["points"]:
        assert pt["split"] == "test"


async def test_get_predicted_starts(data_root) -> None:
    async with await _client() as c:
        r = await c.get("/api/reports/negermet/paano_shared/starts")

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 3
    assert body[0]["detected_time"] < body[1]["detected_time"]


async def test_predicted_starts_split_filter(data_root) -> None:
    async with await _client() as c:
        r = await c.get(
            "/api/reports/negermet/paano_shared/starts",
            params={"split": "test"},
        )

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["well_id"] == "W-200"


async def test_scores_404_when_missing(data_root) -> None:
    async with await _client() as c:
        r = await c.get("/api/reports/salt/paano_shared/scores")

    assert r.status_code == 404
