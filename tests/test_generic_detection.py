from __future__ import annotations

import unittest

from alma_service.generic_detection import _operational_score_key


class GenericDetectionObjectiveTests(unittest.TestCase):
    def test_operational_score_key_prefers_lower_false_alarm_over_small_delay_gain(self) -> None:
        spamy = {
            "hit_count": 2,
            "p90_delay_ratio": 0.02,
            "false_alarms_per_day": 1.10,
            "avg_starts_per_interval": 45.0,
            "p90_abs_delay_hours": 4.0,
        }
        calmer = {
            "hit_count": 2,
            "p90_delay_ratio": 0.04,
            "false_alarms_per_day": 0.18,
            "avg_starts_per_interval": 4.0,
            "p90_abs_delay_hours": 6.0,
        }

        self.assertGreater(
            _operational_score_key("pritok", "pca_spe", calmer),
            _operational_score_key("pritok", "pca_spe", spamy),
        )

    def test_operational_score_key_strongly_prefers_lower_start_count(self) -> None:
        noisy = {
            "hit_count": 8,
            "p90_delay_ratio": 0.02,
            "false_alarms_per_day": 0.20,
            "avg_starts_per_interval": 42.0,
            "p90_abs_delay_hours": 2.0,
        }
        calmer = {
            "hit_count": 8,
            "p90_delay_ratio": 0.08,
            "false_alarms_per_day": 0.24,
            "avg_starts_per_interval": 7.0,
            "p90_abs_delay_hours": 6.0,
        }

        self.assertGreater(
            _operational_score_key("salt", "pca_spe", calmer),
            _operational_score_key("salt", "pca_spe", noisy),
        )

    def test_operational_score_key_penalizes_extreme_delay_before_small_far_gain(self) -> None:
        too_late = {
            "hit_count": 9,
            "p90_delay_ratio": 0.50,
            "false_alarms_per_day": 0.14,
            "avg_starts_per_interval": 19.0,
            "p90_abs_delay_hours": 370.0,
        }
        on_time = {
            "hit_count": 9,
            "p90_delay_ratio": 0.14,
            "false_alarms_per_day": 0.18,
            "avg_starts_per_interval": 23.0,
            "p90_abs_delay_hours": 70.0,
        }

        self.assertGreater(
            _operational_score_key("salt", "pca_spe", on_time),
            _operational_score_key("salt", "pca_spe", too_late),
        )


if __name__ == "__main__":
    unittest.main()
