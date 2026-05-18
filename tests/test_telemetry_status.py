from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alma_service.telemetry_status import (
    EVENT_BAD_DATA,
    EVENT_PRE_ANOMALY,
    EVENT_REGIME,
    QUALITY_GAP,
    REGIME_FREQUENCY_CHANGE,
    build_telemetry_status,
)


class TestTelemetryStatus(unittest.TestCase):
    def test_gap_is_bad_data(self) -> None:
        timestamps = list(pd.date_range("2026-01-01", periods=20, freq="2min"))
        timestamps[10] = timestamps[9] + pd.Timedelta(hours=2)
        raw = np.full((20, 2), 50.0, dtype=np.float32)

        status = build_telemetry_status(
            timestamps=np.asarray(timestamps, dtype="datetime64[ns]"),
            raw_columns=["Выходная частота", "Ток на фазе А"],
            raw_matrix=raw,
        ).frame

        self.assertIn(QUALITY_GAP, set(status["quality_status"]))
        self.assertIn(EVENT_BAD_DATA, set(status["event_class"]))

    def test_frequency_change_is_regime_event(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=80, freq="10min")
        freq = np.full(len(timestamps), 45.0, dtype=np.float32)
        current = np.full(len(timestamps), 30.0, dtype=np.float32)
        freq[40:] = 50.0
        raw = np.column_stack([freq, current]).astype(np.float32)

        status = build_telemetry_status(
            timestamps=timestamps.to_numpy(),
            raw_columns=["Выходная частота", "Ток на фазе А"],
            raw_matrix=raw,
        ).frame

        self.assertIn(REGIME_FREQUENCY_CHANGE, set(status["regime_status"]))
        self.assertIn(EVENT_REGIME, set(status["event_class"]))

    def test_pre_anomaly_zone_is_marked(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=400, freq="10min")
        raw = np.column_stack(
            [
                np.full(len(timestamps), 50.0, dtype=np.float32),
                np.linspace(20.0, 21.0, len(timestamps), dtype=np.float32),
            ]
        )
        intervals = pd.DataFrame(
            {
                "start_date": [timestamps[300]],
                "end_date": [timestamps[340]],
            }
        )

        status = build_telemetry_status(
            timestamps=timestamps.to_numpy(),
            raw_columns=["Выходная частота", "Ток на фазе А"],
            raw_matrix=raw,
            anomaly_key="pritok",
            anomaly_intervals=intervals,
            patch_size=96,
        ).frame

        self.assertIn(EVENT_PRE_ANOMALY, set(status["zone_status"]))
        self.assertIn(EVENT_PRE_ANOMALY, set(status["event_class"]))

    def test_operational_event_marks_pre_anomaly_zone(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=72, freq="1h")
        raw = np.column_stack(
            [
                np.full(len(timestamps), 50.0, dtype=np.float32),
                np.full(len(timestamps), 25.0, dtype=np.float32),
            ]
        )
        events = pd.DataFrame(
            {
                "start_date": [pd.Timestamp("2026-01-02 12:00")],
                "end_date": [pd.Timestamp("2026-01-02 13:00")],
                "pre_window_hours": [6],
            }
        )

        status = build_telemetry_status(
            timestamps=timestamps.to_numpy(),
            raw_columns=["Выходная частота", "Ток на фазе А"],
            raw_matrix=raw,
            operational_events=events,
        ).frame

        self.assertEqual(status.loc[30, "zone_status"], EVENT_PRE_ANOMALY)
        self.assertEqual(status.loc[30, "event_class"], EVENT_PRE_ANOMALY)
        self.assertEqual(status.loc[36, "zone_status"], "operational_event")
        self.assertEqual(status.loc[36, "event_class"], EVENT_REGIME)


if __name__ == "__main__":
    unittest.main()
