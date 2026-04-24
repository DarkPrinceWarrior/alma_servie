from datetime import datetime
from pathlib import Path

import polars as pl


def write_intervals_parquet(data_root: Path, anomaly: str) -> Path:
    db_dir = data_root / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    path = db_dir / f"{anomaly}_intervals.parquet"

    df = pl.DataFrame(
        {
            "well_id": ["W-100", "W-100", "W-200", "W-300"],
            "start_date": [
                datetime(2015, 3, 6, 8, 0),
                datetime(2015, 6, 1, 12, 0),
                datetime(2016, 4, 2, 9, 30),
                datetime(2017, 1, 10, 0, 0),
            ],
            "end_date": [
                datetime(2015, 3, 6, 20, 0),
                datetime(2015, 6, 2, 6, 0),
                datetime(2016, 4, 2, 18, 0),
                datetime(2017, 1, 10, 23, 0),
            ],
            "data_start": [
                datetime(2015, 3, 5, 0, 0),
                datetime(2015, 5, 30, 0, 0),
                datetime(2016, 4, 1, 0, 0),
                datetime(2017, 1, 8, 0, 0),
            ],
            "data_end": [
                datetime(2015, 3, 7, 23, 0),
                datetime(2015, 6, 3, 23, 0),
                datetime(2016, 4, 3, 23, 0),
                datetime(2017, 1, 12, 23, 0),
            ],
            "split": ["train", "train", "test", "train"],
            "interval_idx": [1, 2, 1, 1],
        }
    )
    df.write_parquet(path)
    return path
