from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from back.api.deps import get_current_user, get_data_root
from back.main import app
from back.services.parquet import _read_parquet_cached
from tests.fixtures.intervals import write_intervals_parquet
from tests.fixtures.reports import (
    write_html_report,
    write_predicted_starts,
    write_scores_parquet,
)


@pytest.fixture
def data_root(tmp_path: Path) -> Iterator[Path]:
    for anomaly in ("negermet", "pritok", "salt"):
        write_intervals_parquet(tmp_path, anomaly)

    write_scores_parquet(tmp_path, "negermet", "paano_shared")
    write_predicted_starts(tmp_path, "negermet", "paano_shared")
    write_html_report(tmp_path, "negermet", "paano_shared")

    app.dependency_overrides[get_data_root] = lambda: tmp_path
    _read_parquet_cached.cache_clear()
    try:
        yield tmp_path
    finally:
        app.dependency_overrides.pop(get_data_root, None)
        _read_parquet_cached.cache_clear()


@pytest.fixture
def auth_user() -> Iterator[SimpleNamespace]:
    now = datetime.now(UTC)
    fake = SimpleNamespace(
        id=uuid4(),
        email="fake@test",
        is_active=True,
        roles=[SimpleNamespace(code="admin"), SimpleNamespace(code="user")],
        created_at=now,
        updated_at=now,
    )
    app.dependency_overrides[get_current_user] = lambda: fake
    try:
        yield fake
    finally:
        app.dependency_overrides.pop(get_current_user, None)
