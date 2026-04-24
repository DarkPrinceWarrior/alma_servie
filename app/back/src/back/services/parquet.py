from functools import lru_cache
from pathlib import Path

import polars as pl


def intervals_path(data_root: Path, anomaly: str) -> Path:
    return data_root / "db" / f"{anomaly}_intervals.parquet"


@lru_cache(maxsize=32)
def _read_parquet_cached(path_str: str, mtime_ns: int) -> pl.DataFrame:
    return pl.read_parquet(path_str)


def read_parquet_cached(path: Path) -> pl.DataFrame:
    return _read_parquet_cached(str(path), path.stat().st_mtime_ns)
