from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REFERENCE_DATA_DIR = DATA_DIR / "reference"

NEGREMET_RAW_DIR = RAW_DATA_DIR / "negermet"
PRITOK_RAW_DIR = RAW_DATA_DIR / "pritok"
SALT_RAW_DIR = RAW_DATA_DIR / "salt"

SUMMARY_INFO_PATH = REFERENCE_DATA_DIR / "Сводная информация.xlsx"
MODEL_PARAMS_PATH = REFERENCE_DATA_DIR / "Параметры для модели.xlsx"

DB_DIR = PROJECT_ROOT / "db"
RAW_CACHE_DIR = DB_DIR / "raw_cache"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

BEST_TRAINED_ENCODER_PATH = MODELS_DIR / "best_trained_encoder.pth"
TRAINED_ENCODER_PATH = MODELS_DIR / "trained_encoder.pth"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
