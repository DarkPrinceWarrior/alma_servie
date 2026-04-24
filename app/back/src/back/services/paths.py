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
