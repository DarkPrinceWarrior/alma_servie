from __future__ import annotations

import numpy as np
import unittest

from alma_service.generic_detectors import (
    DetectorScoreOutput,
    PAANO_INPUT_PADDING_EDGE_HOLD,
    PAANO_INPUT_PADDING_ENV,
    PAANO_INPUT_PADDING_NONE,
    _edge_hold_pad_prefix,
    _ensure_2d_float32,
    _input_contract_from_padding,
    _paano_input_padding_mode,
    _trim_prefix_padding,
)


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

    def test_edge_hold_padding_prepends_first_row(self) -> None:
        arr = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)

        padded, pad_count = _edge_hold_pad_prefix(arr, target_len=5)

        self.assertEqual(pad_count, 2)
        self.assertEqual(padded.shape, (5, 2))
        self.assertTrue(np.array_equal(padded[:2], np.array([[1.0, 10.0], [1.0, 10.0]], dtype=np.float32)))
        self.assertTrue(np.array_equal(padded[2:], arr))

    def test_trim_prefix_padding_restores_original_length(self) -> None:
        scores = np.array([9.0, 8.0, 1.0, 2.0, 3.0], dtype=np.float32)

        trimmed = _trim_prefix_padding(scores, pad_count=2, original_len=3)

        self.assertTrue(np.array_equal(trimmed, np.array([1.0, 2.0, 3.0], dtype=np.float32)))

    def test_input_contract_marks_only_actual_padding(self) -> None:
        self.assertEqual(_input_contract_from_padding({"enabled": False}), "real_window")
        self.assertEqual(
            _input_contract_from_padding({
                "enabled": True,
                "short": {"well_prefix_points": 0, "reference_prefix_points": 0},
                "long": {"well_prefix_points": 0, "reference_prefix_points": 0},
            }),
            "real_window",
        )
        self.assertEqual(
            _input_contract_from_padding({
                "enabled": True,
                "short": {"well_prefix_points": 0, "reference_prefix_points": 1},
                "long": {"well_prefix_points": 0, "reference_prefix_points": 0},
            }),
            "edge_hold_padded",
        )

    def test_explicit_input_padding_mode_does_not_depend_on_environment(self) -> None:
        import os

        old_value = os.environ.get(PAANO_INPUT_PADDING_ENV)
        try:
            os.environ[PAANO_INPUT_PADDING_ENV] = PAANO_INPUT_PADDING_NONE

            self.assertEqual(_paano_input_padding_mode(PAANO_INPUT_PADDING_EDGE_HOLD), PAANO_INPUT_PADDING_EDGE_HOLD)
        finally:
            if old_value is None:
                os.environ.pop(PAANO_INPUT_PADDING_ENV, None)
            else:
                os.environ[PAANO_INPUT_PADDING_ENV] = old_value


if __name__ == "__main__":
    unittest.main()
