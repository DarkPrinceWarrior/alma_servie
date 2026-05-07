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

    def test_duplicate_alert_metrics_count_repeated_starts_inside_interval(self) -> None:
        intervals = pd.DataFrame(
            [
                {
                    "well_id": "w1",
                    "interval_idx": 1,
                    "start_date": pd.Timestamp("2025-01-01 10:00:00"),
                    "end_date": pd.Timestamp("2025-01-02 10:00:00"),
                    "data_start": pd.Timestamp("2025-01-01 00:00:00"),
                    "data_end": pd.Timestamp("2025-01-04 00:00:00"),
                    "split": "test",
                }
            ]
        )
        predictions = pd.DataFrame(
            [
                {"well_id": "w1", "detected_time": pd.Timestamp("2025-01-01 10:15:00")},
                {"well_id": "w1", "detected_time": pd.Timestamp("2025-01-01 12:00:00")},
                {"well_id": "w1", "detected_time": pd.Timestamp("2025-01-01 18:00:00")},
                {"well_id": "w1", "detected_time": pd.Timestamp("2025-01-03 00:00:00")},
                {"well_id": "w1", "detected_time": pd.Timestamp("2025-01-03 00:30:00")},
            ]
        )
        scores = pd.DataFrame(
            {
                "well_id": ["w1"] * 10,
                "timestamp": pd.date_range("2025-01-01", periods=10, freq="15min"),
            }
        )

        summary, interval_df = evaluate_predictions(
            intervals,
            predictions,
            scores=scores,
            prestart_hours=2.0,
        )

        self.assertEqual(summary["hit_count"], 1)
        self.assertEqual(summary["start_count"], 5)
        self.assertEqual(summary["false_alarms"], 2)
        self.assertEqual(summary["false_alarm_episodes"], 1)
        self.assertEqual(summary["duplicate_starts_inside_interval"], 2)
        self.assertEqual(summary["alerts_inside_intervals"], 3)
        self.assertAlmostEqual(summary["alerts_per_detected_interval"], 3.0)
        self.assertEqual(summary["suppressed_rearms_count"], 3)
        self.assertEqual(int(interval_df.iloc[0]["alert_count_inside_interval"]), 3)
        self.assertEqual(int(interval_df.iloc[0]["duplicate_starts_inside_interval"]), 2)


if __name__ == "__main__":
    unittest.main()
