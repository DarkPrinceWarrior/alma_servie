from __future__ import annotations

import unittest

from alma_service.generic_detection import _operational_score_key, _robust_tuning_score_key


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
            _operational_score_key("pritok", "paano_shared", calmer),
            _operational_score_key("pritok", "paano_shared", spamy),
        )

    def test_salt_operational_score_key_prefers_early_onset_with_feasible_noise(self) -> None:
        earlier = {
            "hit_count": 8,
            "p90_delay_ratio": 0.02,
            "false_alarms_per_day": 0.20,
            "avg_starts_per_interval": 9.0,
            "p90_abs_delay_hours": 2.0,
        }
        cleaner_but_later = {
            "hit_count": 8,
            "p90_delay_ratio": 0.08,
            "false_alarms_per_day": 0.24,
            "avg_starts_per_interval": 7.0,
            "p90_abs_delay_hours": 6.0,
        }

        self.assertGreater(
            _operational_score_key("salt", "paano_shared", earlier),
            _operational_score_key("salt", "paano_shared", cleaner_but_later),
        )

    def test_salt_operational_score_key_rejects_extreme_repeated_starts(self) -> None:
        noisy = {
            "hit_count": 8,
            "p90_delay_ratio": 0.02,
            "false_alarms_per_day": 0.20,
            "avg_starts_per_interval": 42.0,
            "p90_abs_delay_hours": 2.0,
        }
        controlled = {
            "hit_count": 8,
            "p90_delay_ratio": 0.02,
            "false_alarms_per_day": 0.24,
            "avg_starts_per_interval": 7.0,
            "p90_abs_delay_hours": 2.0,
        }

        self.assertGreater(
            _operational_score_key("salt", "paano_shared", controlled),
            _operational_score_key("salt", "paano_shared", noisy),
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
            _operational_score_key("salt", "paano_shared", on_time),
            _operational_score_key("salt", "paano_shared", too_late),
        )

    def test_robust_tuning_score_key_penalizes_single_well_failure(self) -> None:
        aggregate = {
            "hit_count": 6,
            "interval_count": 6,
            "p90_delay_ratio": 0.05,
            "false_alarms_per_day": 0.02,
            "avg_starts_per_interval": 3.0,
            "p90_abs_delay_hours": 3.0,
            "median_abs_delay_hours": 1.0,
        }
        balanced_wells = {
            "w1": {
                "hit_count": 1,
                "interval_count": 1,
                "hit_rate": 1.0,
                "p90_delay_ratio": 0.05,
                "false_alarms_per_day": 0.02,
                "avg_starts_per_interval": 3.0,
            },
            "w2": {
                "hit_count": 1,
                "interval_count": 1,
                "hit_rate": 1.0,
                "p90_delay_ratio": 0.06,
                "false_alarms_per_day": 0.02,
                "avg_starts_per_interval": 3.0,
            },
        }
        failed_well = {
            "w1": balanced_wells["w1"],
            "w2": {
                "hit_count": 0,
                "interval_count": 1,
                "hit_rate": 0.0,
                "p90_delay_ratio": 1.0,
                "false_alarms_per_day": 0.02,
                "avg_starts_per_interval": 3.0,
            },
        }

        self.assertGreater(
            _robust_tuning_score_key("salt", "paano_shared", aggregate, balanced_wells),
            _robust_tuning_score_key("salt", "paano_shared", aggregate, failed_well),
        )


if __name__ == "__main__":
    unittest.main()
