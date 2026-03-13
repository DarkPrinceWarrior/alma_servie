from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alma_service.anomaly_specs import DetectionSpec
from alma_service.paths import DB_DIR, MODELS_DIR, REPORTS_DIR, RESULTS_DIR

DETECTOR_KEYS = (
    "paano_feat",
    "pca_spe",
    "lof",
    "iforest",
    "fused",
)
LOCAL_DETECTOR_KEYS = (
    "paano_feat",
    "pca_spe",
    "lof",
    "iforest",
    "fused",
)
DEFAULT_DETECTOR = "pca_spe"


def normalize_detector_key(detector: str | None) -> str:
    key = str(detector or DEFAULT_DETECTOR).strip().lower()
    if key not in DETECTOR_KEYS:
        raise ValueError(f"Unknown detector: {detector}")
    return key


def detector_stem(spec: DetectionSpec, detector: str) -> str:
    return f"{spec.dataset.output_prefix}_{normalize_detector_key(detector)}"


def results_path(spec: DetectionSpec, detector: str) -> Path:
    return RESULTS_DIR / f"{detector_stem(spec, detector)}_results.parquet"


def scores_path(spec: DetectionSpec, detector: str) -> Path:
    return DB_DIR / f"{detector_stem(spec, detector)}_scores.parquet"


def predicted_starts_path(spec: DetectionSpec, detector: str) -> Path:
    return DB_DIR / f"{detector_stem(spec, detector)}_predicted_starts.parquet"


def config_path(spec: DetectionSpec, detector: str) -> Path:
    return DB_DIR / f"{detector_stem(spec, detector)}_config.json"


def tuning_path(spec: DetectionSpec, detector: str) -> Path:
    return DB_DIR / f"{detector_stem(spec, detector)}_tuning.json"


def summary_path(spec: DetectionSpec, detector: str) -> Path:
    return results_path(spec, detector).with_suffix(".summary.json")


def report_path(spec: DetectionSpec, detector: str) -> Path:
    return REPORTS_DIR / f"{detector_stem(spec, detector)}_report.html"


def model_path(spec: DetectionSpec, detector: str) -> Path:
    return MODELS_DIR / f"{detector_stem(spec, detector)}.pt"


def benchmark_summary_path(spec: DetectionSpec) -> Path:
    return RESULTS_DIR / f"{spec.dataset.output_prefix}_benchmark_summary.json"


def legacy_summary_path(spec: DetectionSpec) -> Path:
    return spec.results_path.with_suffix(".summary.json")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
