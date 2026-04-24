from datetime import datetime, timedelta
from pathlib import Path

import polars as pl


def write_scores_parquet(data_root: Path, anomaly: str, detector: str, n: int = 3000) -> Path:
    db_dir = data_root / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    path = db_dir / f"{anomaly}_{detector}_scores.parquet"

    base = datetime(2025, 6, 20, 0, 0, 0)
    rows = [
        {
            "well_id": "W-100" if i < n // 2 else "W-200",
            "timestamp": base + timedelta(minutes=2 * i),
            "split": "train" if i < n // 2 else "test",
            "score": float(i % 50) / 10.0,
        }
        for i in range(n)
    ]
    pl.DataFrame(rows).write_parquet(path)
    return path


def write_predicted_starts(data_root: Path, anomaly: str, detector: str) -> Path:
    db_dir = data_root / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    path = db_dir / f"{anomaly}_{detector}_predicted_starts.parquet"

    pl.DataFrame(
        {
            "well_id": ["W-100", "W-100", "W-200"],
            "detected_time": [
                datetime(2025, 6, 20, 5, 30),
                datetime(2025, 6, 20, 8, 45),
                datetime(2025, 6, 21, 12, 10),
            ],
            "split": ["train", "train", "test"],
        }
    ).write_parquet(path)
    return path


def write_html_report(data_root: Path, anomaly: str, detector: str) -> Path:
    folder = data_root / "artifacts" / "reports" / anomaly
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{anomaly}_{detector}_report.html"
    path.write_text(
        f"<!doctype html><html><body><h1>Report {anomaly}/{detector}</h1></body></html>",
        encoding="utf-8",
    )
    return path
