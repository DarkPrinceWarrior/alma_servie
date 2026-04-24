from pathlib import Path


def html_report_path(data_root: Path, anomaly: str, detector: str) -> Path:
    return data_root / "artifacts" / "reports" / anomaly / f"{anomaly}_{detector}_report.html"


def feature_importance_html_path(data_root: Path, anomaly: str, detector: str) -> Path:
    return (
        data_root
        / "artifacts"
        / "reports"
        / anomaly
        / f"{anomaly}_{detector}_feature_importance.html"
    )


def scores_parquet_path(data_root: Path, anomaly: str, detector: str) -> Path:
    return data_root / "db" / f"{anomaly}_{detector}_scores.parquet"


def predicted_starts_parquet_path(data_root: Path, anomaly: str, detector: str) -> Path:
    return data_root / "db" / f"{anomaly}_{detector}_predicted_starts.parquet"


def anomaly_database_parquet_path(data_root: Path, anomaly: str, freq: str = "2min") -> Path:
    return data_root / "db" / f"{anomaly}_anomaly_database_{freq}.parquet"


def fi_summary_json_path(data_root: Path, anomaly: str, detector: str) -> Path:
    return data_root / "db" / f"{anomaly}_{detector}_fi_summary.json"


def intervals_parquet_path(data_root: Path, anomaly: str) -> Path:
    return data_root / "db" / f"{anomaly}_intervals.parquet"
