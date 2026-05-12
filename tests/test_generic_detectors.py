from __future__ import annotations

import numpy as np
import unittest

from alma_service.generic_detectors import DetectorScoreOutput, _ensure_2d_float32


class GenericDetectorTests(unittest.TestCase):
    def test_score_output_keeps_primary_and_components(self) -> None:
        primary = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        output = DetectorScoreOutput(
            primary=primary,
            components={"paano_score": primary.copy()},
            detail={"detector": "paano_shared"},
        )

        self.assertEqual(output.detail["detector"], "paano_shared")
        self.assertTrue(np.array_equal(output.primary, output.components["paano_score"]))

    def test_ensure_2d_float32_replaces_non_finite_values(self) -> None:
        arr = _ensure_2d_float32(np.array([[1.0, np.nan], [np.inf, -np.inf]]))

        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.isfinite(arr).all())


if __name__ == "__main__":
    unittest.main()
