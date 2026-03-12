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
            _operational_score_key("pritok", "fused", spamy),
        )


if __name__ == "__main__":
    unittest.main()
