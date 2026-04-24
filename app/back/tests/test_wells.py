from httpx import ASGITransport, AsyncClient

from back.main import app


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_list_wells_returns_three(data_root) -> None:
    async with await _client() as c:
        r = await c.get("/api/wells", params={"anomaly": "negermet"})

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 3
    well_ids = [w["well_id"] for w in body]
    assert well_ids == ["W-100", "W-200", "W-300"]

    w100 = next(w for w in body if w["well_id"] == "W-100")
    assert w100["n_intervals"] == 2
    assert w100["split"] == "train"
    assert w100["anomaly"] == "negermet"


async def test_get_well_returns_intervals(data_root) -> None:
    async with await _client() as c:
        r = await c.get("/api/wells/W-100", params={"anomaly": "negermet"})

    assert r.status_code == 200
    body = r.json()
    assert body["well_id"] == "W-100"
    assert body["n_intervals"] == 2
    assert len(body["intervals"]) == 2
    assert body["intervals"][0]["interval_idx"] == 1
    assert body["intervals"][1]["interval_idx"] == 2


async def test_get_well_404(data_root) -> None:
    async with await _client() as c:
        r = await c.get("/api/wells/UNKNOWN", params={"anomaly": "negermet"})

    assert r.status_code == 404
    assert "UNKNOWN" in r.json()["detail"]


async def test_well_intervals_endpoint(data_root) -> None:
    async with await _client() as c:
        r = await c.get("/api/wells/W-200/intervals", params={"anomaly": "pritok"})

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["split"] == "test"


async def test_invalid_anomaly_returns_422(data_root) -> None:
    async with await _client() as c:
        r = await c.get("/api/wells", params={"anomaly": "nonexistent"})

    assert r.status_code == 422
