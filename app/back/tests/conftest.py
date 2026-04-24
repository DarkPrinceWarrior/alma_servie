from collections.abc import Iterator
from pathlib import Path

import pytest

from back.api.deps import get_data_root
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

    write_scores_parquet(tmp_path, "negermet", "pca_spe")
    write_predicted_starts(tmp_path, "negermet", "pca_spe")
    write_html_report(tmp_path, "negermet", "pca_spe")

    app.dependency_overrides[get_data_root] = lambda: tmp_path
    _read_parquet_cached.cache_clear()
    try:
        yield tmp_path
    finally:
        app.dependency_overrides.pop(get_data_root, None)
        _read_parquet_cached.cache_clear()
