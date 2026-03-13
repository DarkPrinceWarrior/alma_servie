from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alma_service.onset_detection import CausalThresholds, detect_causal_onsets


class OnsetDetectionStatefulTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = CausalThresholds(
            score_threshold=1.0,
            ema_z_threshold=1.0,
            cusum_threshold=999.0,
            drift=0.0,
            baseline_median=0.0,
            baseline_mad=1.0,
            quantile=0.99,
        )

    def test_consolidates_single_episode_through_short_dip(self) -> None:
        timestamps = pd.date_range("2026-01-01 00:00:00", periods=16, freq="1min").to_numpy()
        scores = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                2.0,
                2.0,
                0.7,
                0.7,
                2.0,
                2.0,
                2.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        diagnostics = {
            "ema_z": scores.copy(),
            "cusum": np.zeros_like(scores),
        }

        starts = detect_causal_onsets(
            scores=scores,
            timestamps=timestamps,
            diagnostics=diagnostics,
            thresholds=self.thresholds,
            reference_end_idx=2,
            min_run_points=2,
            cooldown_hours=0.0,
            gate_mode="score_ema",
            rearm_window_minutes=5.0,
            hysteresis_scale=0.60,
        )

        self.assertEqual([pd.Timestamp("2026-01-01 00:05:00")], starts)

    def test_rearms_after_long_calm_and_emits_second_start(self) -> None:
        timestamps = pd.date_range("2026-01-01 00:00:00", periods=24, freq="1min").to_numpy()
        scores = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                2.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                2.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        diagnostics = {
            "ema_z": scores.copy(),
            "cusum": np.zeros_like(scores),
        }

        starts = detect_causal_onsets(
            scores=scores,
            timestamps=timestamps,
            diagnostics=diagnostics,
            thresholds=self.thresholds,
            reference_end_idx=2,
            min_run_points=2,
            cooldown_hours=0.0,
            gate_mode="score_ema",
            rearm_window_minutes=5.0,
            hysteresis_scale=0.60,
        )

        self.assertEqual(
            [
                pd.Timestamp("2026-01-01 00:04:00"),
                pd.Timestamp("2026-01-01 00:14:00"),
            ],
            starts,
        )

    def test_rearmed_episode_can_emit_inside_cooldown_window(self) -> None:
        timestamps = pd.date_range("2026-01-01 00:00:00", periods=24, freq="1min").to_numpy()
        scores = np.array(
            [
                0.0,
                0.0,
                2.0,
                2.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                2.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        diagnostics = {
            "ema_z": scores.copy(),
            "cusum": np.zeros_like(scores),
        }

        starts = detect_causal_onsets(
            scores=scores,
            timestamps=timestamps,
            diagnostics=diagnostics,
            thresholds=self.thresholds,
            reference_end_idx=1,
            min_run_points=2,
            cooldown_hours=24.0,
            gate_mode="score_ema",
            rearm_window_minutes=5.0,
            hysteresis_scale=0.60,
        )

        self.assertEqual(
            [
                pd.Timestamp("2026-01-01 00:02:00"),
                pd.Timestamp("2026-01-01 00:11:00"),
            ],
            starts,
        )


if __name__ == "__main__":
    unittest.main()
