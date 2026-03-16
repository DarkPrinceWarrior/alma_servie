from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alma_service.dataset_config import NEGREMET_WELL_FILES, PRITOK_WELL_FILES, SALT_WELL_FILES
from alma_service.paths import DB_DIR, RESULTS_DIR


@dataclass(frozen=True)
class DatasetSpec:
    anomaly_key: str
    display_name: str
    well_files: dict[str, Path]
    output_prefix: str
    default_freq: str
    summary_match: str
    source_candidates: tuple[str, ...]

    @property
    def intervals_path(self) -> Path:
        return DB_DIR / f"{self.output_prefix}_intervals.parquet"

    @property
    def default_source_path(self) -> Path:
        freq_label = self.default_freq.replace(" ", "")
        return DB_DIR / f"{self.output_prefix}_anomaly_database_{freq_label}.parquet"


@dataclass(frozen=True)
class DetectionSpec:
    anomaly_key: str
    display_name: str
    dataset: DatasetSpec
    results_path: Path
    scores_path: Path
    predicted_starts_path: Path
    config_path: Path
    tuning_path: Path
    feature_importance_path: Path


DATASET_SPECS: dict[str, DatasetSpec] = {
    "negermet": DatasetSpec(
        anomaly_key="negermet",
        display_name="Негерметичность НКТ",
        well_files=NEGREMET_WELL_FILES,
        output_prefix="negermet",
        default_freq="2min",
        summary_match="негерметичность",
        source_candidates=(
            "negermet_anomaly_database_2min.parquet",
            "negermet_anomaly_database_15s.parquet",
        ),
    ),
    "pritok": DatasetSpec(
        anomaly_key="pritok",
        display_name="Изменение притока",
        well_files=PRITOK_WELL_FILES,
        output_prefix="pritok",
        default_freq="10min",
        summary_match="приток",
        source_candidates=(
            "pritok_anomaly_database_10min.parquet",
            "pritok_anomaly_database_2min.parquet",
            "pritok_anomaly_database_15s.parquet",
        ),
    ),
    "salt": DatasetSpec(
        anomaly_key="salt",
        display_name="Солеотложение",
        well_files=SALT_WELL_FILES,
        output_prefix="salt",
        default_freq="15min",
        summary_match="соли",
        source_candidates=(
            "salt_anomaly_database_15min.parquet",
            "salt_anomaly_database_2min.parquet",
            "salt_anomaly_database_15s.parquet",
        ),
    ),
}


DETECTION_SPECS: dict[str, DetectionSpec] = {
    key: DetectionSpec(
        anomaly_key=key,
        display_name=spec.display_name,
        dataset=spec,
        results_path=RESULTS_DIR / f"{spec.output_prefix}_paano_results.parquet",
        scores_path=DB_DIR / f"{spec.output_prefix}_paano_scores.parquet",
        predicted_starts_path=DB_DIR / f"{spec.output_prefix}_paano_predicted_starts.parquet",
        config_path=DB_DIR / f"{spec.output_prefix}_paano_config.json",
        tuning_path=DB_DIR / f"{spec.output_prefix}_paano_tuning.json",
        feature_importance_path=RESULTS_DIR.parent / "reports" / f"{spec.output_prefix}_paano_feature_importance.html",
    )
    for key, spec in DATASET_SPECS.items()
}


def get_dataset_spec(anomaly_key: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[anomaly_key]
    except KeyError as exc:
        raise KeyError(f"Unknown anomaly spec: {anomaly_key}") from exc


def get_detection_spec(anomaly_key: str) -> DetectionSpec:
    try:
        return DETECTION_SPECS[anomaly_key]
    except KeyError as exc:
        raise KeyError(f"Unknown detection spec: {anomaly_key}") from exc
