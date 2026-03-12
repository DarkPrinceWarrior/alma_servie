from __future__ import annotations

import unittest

import numpy as np
import torch

from alma_service.generic_detectors import DetectorScoreOutput, FusedDetector, PCASPEDetector


class _DummyDetector:
    def __init__(self, values: np.ndarray, name: str) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.name = name

    def fit_reference(self, X_ref, mask_ref=None):
        return self

    def score_stream(self, X_all, mask_all=None) -> DetectorScoreOutput:
        return DetectorScoreOutput(
            primary=self.values.copy(),
            components={f"{self.name}_raw": self.values.copy()},
            detail={"name": self.name},
        )


class GenericDetectorTests(unittest.TestCase):
    def test_pca_spe_scores_anomalous_tail_higher(self) -> None:
        rng = np.random.default_rng(7)
        ref = rng.normal(0.0, 0.2, size=(80, 4)).astype(np.float32)
        normal = rng.normal(0.0, 0.2, size=(20, 4)).astype(np.float32)
        anomaly = rng.normal(2.5, 0.2, size=(10, 4)).astype(np.float32)
        series = np.vstack([ref, normal, anomaly]).astype(np.float32)

        detector = PCASPEDetector()
        detector.fit_reference(ref)
        output = detector.score_stream(series)

        self.assertGreater(float(output.primary[-1]), float(np.median(output.primary[:80])))

    def test_fused_detector_combines_weighted_component_scores(self) -> None:
        base = np.array([0.0] * 10 + [5.0, 6.0], dtype=np.float32)
        fused = FusedDetector(device=torch.device("cpu"), verbose=False)
        fused.detectors_ = {
            "paano_feat": _DummyDetector(base, "paano"),
            "pca_spe": _DummyDetector(base * 0.5, "pca"),
            "lof": _DummyDetector(base * 0.25, "lof"),
            "iforest": _DummyDetector(base * 0.25, "iforest"),
        }
        fused.reference_mask_ = np.array([True] * 10 + [False, False], dtype=bool)

        output = fused.score_stream(np.zeros((12, 2), dtype=np.float32))

        self.assertIn("paano_feat_score", output.components)
        self.assertIn("pca_spe_score", output.components)
        self.assertGreater(float(output.primary[-1]), float(output.primary[0]))


if __name__ == "__main__":
    unittest.main()
