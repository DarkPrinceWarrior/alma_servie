from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alma_service.engineered_features import _build_instability_mask


class EngineeredFeatureMaskTests(unittest.TestCase):
    def test_instability_mask_flags_step_change(self) -> None:
        timestamps = pd.date_range("2025-01-01", periods=80, freq="2min")
        freq = np.full(len(timestamps), 50.0, dtype=np.float32)
        power = np.full(len(timestamps), 25.0, dtype=np.float32)
        freq[40:] += 20.0
        raw_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "Выходная частота": freq,
                "Полная выходная мощность": power,
            }
        )
        mask, anchors = _build_instability_mask(
            raw_df=raw_df[["Выходная частота", "Полная выходная мощность"]],
            filled_matrix=raw_df[["Выходная частота", "Полная выходная мощность"]].to_numpy(dtype=np.float32),
            base_columns=["Выходная частота", "Полная выходная мощность"],
            step_seconds=120.0,
            profile={
                "name": "test",
                "back_minutes": 10,
                "forward_minutes": 30,
                "step_sigma": 6.0,
                "flatline_minutes": 60,
                "missing_run_length": 3,
            },
        )

        self.assertIn("Выходная частота", anchors)
        self.assertTrue((~mask[35:55]).any())

    def test_onset_mask_can_ignore_step_change_events(self) -> None:
        timestamps = pd.date_range("2025-01-01", periods=80, freq="2min")
        freq = np.full(len(timestamps), 50.0, dtype=np.float32)
        power = np.full(len(timestamps), 25.0, dtype=np.float32)
        freq[40:] += 20.0
        raw_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "Выходная частота": freq,
                "Полная выходная мощность": power,
            }
        )
        mask, _ = _build_instability_mask(
            raw_df=raw_df[["Выходная частота", "Полная выходная мощность"]],
            filled_matrix=raw_df[["Выходная частота", "Полная выходная мощность"]].to_numpy(dtype=np.float32),
            base_columns=["Выходная частота", "Полная выходная мощность"],
            step_seconds=120.0,
            profile={
                "name": "test",
                "back_minutes": 2,
                "forward_minutes": 5,
                "step_sigma": 6.0,
                "flatline_minutes": 60,
                "missing_run_length": 3,
            },
            include_step_events=False,
            include_start_stop_events=False,
        )

        self.assertTrue(mask[40:45].all())


if __name__ == "__main__":
    unittest.main()
