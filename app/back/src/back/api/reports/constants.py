from back.api.detections.schemas import DetectorType
from back.api.wells.schemas import AnomalyType

ALL_DETECTORS: tuple[DetectorType, ...] = ("paano_global",)

BEST_DETECTOR_BY_ANOMALY: dict[AnomalyType, DetectorType] = {
    "negermet": "paano_global",
    "pritok": "paano_global",
}
