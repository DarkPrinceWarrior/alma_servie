from __future__ import annotations

import unittest

import pandas as pd

from alma_service.prediction_postprocess import (
    START_ANOMALY_CANDIDATE,
    START_BAD_DATA,
    START_LABELLED_ANOMALY,
    START_PRE_ANOMALY_ZONE,
    START_EARLY_WARNING,
    START_REGIME_EVENT,
    build_incidents,
    filter_actionable_starts,
)


class TestPredictionPostprocess(unittest.TestCase):
    def test_classifies_and_filters_non_actionable_starts(self) -> None:
        predictions = pd.DataFrame(
            {
                "well_id": ["1", "1", "1"],
                "detected_time": pd.to_datetime(
                    ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"]
                ),
                "event_class": ["bad_data", "regime_event", "normal_context"],
            }
        )

        result = build_incidents(predictions, merge_window_hours=6.0).starts
        self.assertEqual(
            list(result["start_class"]),
            [START_BAD_DATA, START_REGIME_EVENT, START_ANOMALY_CANDIDATE],
        )
        self.assertEqual(int(result["actionable_alert"].sum()), 1)
        self.assertEqual(len(filter_actionable_starts(result)), 1)

    def test_labelled_zones_override_regime_or_bad_data_context(self) -> None:
        predictions = pd.DataFrame(
            {
                "well_id": ["1", "1"],
                "detected_time": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
                "event_class": ["regime_event", "bad_data"],
                "zone_status": ["labelled_anomaly", "pre_anomaly_zone"],
            }
        )

        result = build_incidents(predictions, merge_window_hours=6.0).starts

        self.assertEqual(
            list(result["start_class"]),
            [START_LABELLED_ANOMALY, START_PRE_ANOMALY_ZONE],
        )
        self.assertEqual(int(result["actionable_alert"].sum()), 2)

    def test_early_warnings_respect_bad_data_and_regime_flags(self) -> None:
        predictions = pd.DataFrame(
            {
                "well_id": ["1", "1", "1"],
                "detected_time": pd.to_datetime(
                    ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"]
                ),
                "event_class": ["early_warning", "early_warning", "early_warning"],
                "is_bad_data": [True, False, False],
                "is_regime_event": [False, True, False],
            }
        )

        result = build_incidents(predictions, merge_window_hours=6.0).starts

        self.assertEqual(
            list(result["start_class"]),
            [START_BAD_DATA, START_REGIME_EVENT, START_EARLY_WARNING],
        )
        self.assertEqual(int(result["actionable_alert"].sum()), 1)

    def test_trend_protected_survives_bad_data_suppression(self) -> None:
        # Трендовый (fusion) старт притока, попавший на точку sensor_stuck/bad_data
        # от замороженного вспомогательного датчика, НЕ подавляется; обычный bad_data — да.
        predictions = pd.DataFrame(
            {
                "well_id": ["1", "1"],
                "detected_time": pd.to_datetime(["2026-01-09 00:00", "2026-01-10 00:00"]),
                "event_class": ["bad_data", "bad_data"],
                "is_bad_data": [True, True],
                "trend_protected": [True, False],
            }
        )

        result = build_incidents(predictions, merge_window_hours=6.0).starts

        self.assertEqual(
            list(result["start_class"]),
            [START_ANOMALY_CANDIDATE, START_BAD_DATA],
        )
        self.assertEqual(int(result["actionable_alert"].sum()), 1)

    def test_merges_repeated_actionable_starts_into_incident(self) -> None:
        predictions = pd.DataFrame(
            {
                "well_id": ["1", "1", "1"],
                "anomaly": ["salt", "salt", "salt"],
                "detector": ["paano_shared", "paano_shared", "paano_shared"],
                "detected_time": pd.to_datetime(
                    ["2026-01-01 00:00", "2026-01-01 04:00", "2026-01-02 00:30"]
                ),
                "event_class": ["normal_context", "normal_context", "normal_context"],
            }
        )

        result = build_incidents(predictions, merge_window_hours=6.0)
        self.assertEqual(len(result.incidents), 2)
        self.assertEqual(list(result.starts["incident_state"]), ["open", "continue", "open"])
        self.assertEqual(result.incidents["start_count"].tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
