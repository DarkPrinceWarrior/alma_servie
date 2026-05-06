from back.api.detections.schemas import DetectorType
from back.api.wells.schemas import AnomalyType

ALL_DETECTORS: tuple[DetectorType, ...] = (
    "pca_spe",
    "paano_feat",
    "paano_shared",
    "ensemble",
)

BEST_DETECTOR_BY_ANOMALY: dict[AnomalyType, DetectorType] = {
    "negermet": "paano_shared",
    "pritok": "paano_shared",
    "salt": "paano_shared",
}
