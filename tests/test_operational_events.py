from __future__ import annotations

import unittest

import pandas as pd

from alma_service.operational_events import (
    build_event_zone_masks,
    events_for_well,
    normalize_operational_events,
)


class TestOperationalEvents(unittest.TestCase):
    def test_normalizes_common_event_columns(self) -> None:
        events = pd.DataFrame(
            {
                "Скважина": ["101"],
                "Дата начала": ["2026-01-02 12:00"],
                "Дата окончания": ["2026-01-02 13:00"],
                "Комментарий": ["ремонт"],
                "pre_window_hours": [6],
            }
        )

        normalized = normalize_operational_events(events)

        self.assertEqual(normalized.loc[0, "well_id"], "101")
        self.assertEqual(normalized.loc[0, "event_type"], "ремонт")
        self.assertEqual(normalized.loc[0, "pre_window_hours"], 6)

    def test_filters_events_for_well_and_keeps_global_events(self) -> None:
        events = pd.DataFrame(
            {
                "well_id": ["101", "102", ""],
                "start_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            }
        )

        filtered = events_for_well(events, "101")

        self.assertEqual(filtered["well_id"].tolist(), ["101", ""])

    def test_builds_pre_anomaly_and_event_masks(self) -> None:
        timestamps = pd.date_range("2026-01-02 00:00", periods=48, freq="1h")
        events = pd.DataFrame(
            {
                "start_date": [pd.Timestamp("2026-01-02 12:00")],
                "end_date": [pd.Timestamp("2026-01-02 14:00")],
                "pre_window_hours": [3],
            }
        )

        masks = build_event_zone_masks(timestamps=timestamps.to_numpy(), events=events)

        self.assertTrue(bool(masks.pre_anomaly_mask[9]))
        self.assertFalse(bool(masks.pre_anomaly_mask[8]))
        self.assertTrue(bool(masks.event_mask[12]))
        self.assertTrue(bool(masks.event_mask[14]))
        self.assertFalse(bool(masks.event_mask[15]))


if __name__ == "__main__":
    unittest.main()
