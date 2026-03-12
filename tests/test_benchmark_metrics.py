from __future__ import annotations

import unittest

import pandas as pd

from alma_service.benchmark_metrics import evaluate_predictions


class BenchmarkMetricsTests(unittest.TestCase):
    def test_delay_ratio_clips_early_detection_to_zero(self) -> None:
        intervals = pd.DataFrame(
            [
                {
                    "well_id": "w1",
                    "interval_idx": 1,
                    "start_date": pd.Timestamp("2025-01-01 10:00:00"),
                    "end_date": pd.Timestamp("2025-01-01 22:00:00"),
                    "data_start": pd.Timestamp("2025-01-01 00:00:00"),
                    "data_end": pd.Timestamp("2025-01-02 00:00:00"),
                    "split": "test",
                }
            ]
        )
        predictions = pd.DataFrame(
            [{"well_id": "w1", "detected_time": pd.Timestamp("2025-01-01 09:00:00"), "split": "test"}]
        )
        summary, interval_df = evaluate_predictions(intervals, predictions, prestart_hours=2.0)

        self.assertEqual(summary["hit_count"], 1)
        self.assertAlmostEqual(float(interval_df.iloc[0]["delay_ratio"]), 0.0, places=6)
        self.assertAlmostEqual(float(interval_df.iloc[0]["delay_hours"]), -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
