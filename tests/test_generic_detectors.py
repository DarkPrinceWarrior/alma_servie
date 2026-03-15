from __future__ import annotations

import unittest

import numpy as np
import torch

from alma_service.generic_detectors import DetectorScoreOutput, PCASPEDetector


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


if __name__ == "__main__":
    unittest.main()
