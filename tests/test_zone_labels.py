from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alma_service.zone_labels import (
    Zone,
    label_zones,
    make_clean_normal_mask,
    make_onset_allowed_from_zones,
    make_training_exclusion_mask,
)


class ZoneLabelBasicTests(unittest.TestCase):
    def _make_timestamps(self, n: int, freq: str = "2min") -> np.ndarray:
        return pd.date_range("2026-01-01", periods=n, freq=freq).to_numpy()

    def test_no_intervals_all_clean(self) -> None:
        ts = self._make_timestamps(200)
        labels = label_zones(ts, pd.DataFrame(), patch_size=96, anomaly_key="salt")
        self.assertTrue((labels == Zone.CLEAN_NORMAL).all())

    def test_none_intervals_all_clean(self) -> None:
        ts = self._make_timestamps(200)
        labels = label_zones(ts, None, patch_size=96, anomaly_key="salt")
        self.assertTrue((labels == Zone.CLEAN_NORMAL).all())

    def test_single_anomaly_marks_four_zones(self) -> None:
        ts = self._make_timestamps(500, freq="15s")
        intervals = pd.DataFrame(
            [{"start_date": pd.Timestamp("2026-01-01 00:30:00"), "end_date": pd.Timestamp("2026-01-01 00:45:00")}]
        )
        labels = label_zones(ts, intervals, patch_size=48, anomaly_key="negermet")
        unique_zones = set(labels.tolist())
        self.assertIn(Zone.CLEAN_NORMAL, unique_zones)
        self.assertIn(Zone.PRE_ANOMALY_BUFFER, unique_zones)
        self.assertIn(Zone.ANOMALY, unique_zones)
        self.assertIn(Zone.POST_ANOMALY_RECOVERY, unique_zones)

    def test_anomaly_zone_does_not_overlap_clean(self) -> None:
        ts = self._make_timestamps(500, freq="15s")
        intervals = pd.DataFrame(
            [{"start_date": pd.Timestamp("2026-01-01 00:30:00"), "end_date": pd.Timestamp("2026-01-01 00:45:00")}]
        )
        labels = label_zones(ts, intervals, patch_size=48, anomaly_key="negermet")
        anomaly_mask = labels == Zone.ANOMALY
        clean_mask = labels == Zone.CLEAN_NORMAL
        self.assertFalse((anomaly_mask & clean_mask).any())

    def test_buffer_at_least_patch_size(self) -> None:
        ts = self._make_timestamps(1000, freq="15s")
        intervals = pd.DataFrame(
            [{"start_date": pd.Timestamp("2026-01-01 01:00:00"), "end_date": pd.Timestamp("2026-01-01 01:15:00")}]
        )
        patch_size = 96
        labels = label_zones(ts, intervals, patch_size=patch_size, anomaly_key="negermet")
        anom_indices = np.flatnonzero(labels == Zone.ANOMALY)
        first_anom = anom_indices[0]
        pre_buffer = np.flatnonzero(labels[:first_anom] == Zone.PRE_ANOMALY_BUFFER)
        self.assertGreaterEqual(len(pre_buffer), patch_size)

    def test_two_anomalies_dont_corrupt_each_other(self) -> None:
        ts = self._make_timestamps(2000, freq="15s")
        intervals = pd.DataFrame([
            {"start_date": pd.Timestamp("2026-01-01 01:00:00"), "end_date": pd.Timestamp("2026-01-01 01:10:00")},
            {"start_date": pd.Timestamp("2026-01-01 04:00:00"), "end_date": pd.Timestamp("2026-01-01 04:10:00")},
        ])
        labels = label_zones(ts, intervals, patch_size=48, anomaly_key="negermet")
        anomaly_count = (labels == Zone.ANOMALY).sum()
        self.assertGreater(anomaly_count, 0)
        # Both anomalies should be marked — check we have anomaly points near both timestamps
        ts_pd = pd.DatetimeIndex(ts)
        anom1_mask = (ts_pd >= pd.Timestamp("2026-01-01 01:00:00")) & (ts_pd <= pd.Timestamp("2026-01-01 01:10:00"))
        anom2_mask = (ts_pd >= pd.Timestamp("2026-01-01 04:00:00")) & (ts_pd <= pd.Timestamp("2026-01-01 04:10:00"))
        self.assertTrue((labels[anom1_mask] == Zone.ANOMALY).all())
        self.assertTrue((labels[anom2_mask] == Zone.ANOMALY).all())


class ZoneMaskTests(unittest.TestCase):
    def _make_timestamps(self, n: int, freq: str = "2min") -> np.ndarray:
        return pd.date_range("2026-01-01", periods=n, freq=freq).to_numpy()

    def test_clean_normal_mask(self) -> None:
        labels = np.array([0, 0, 1, 2, 2, 3, 0, 0], dtype=np.int8)
        mask = make_clean_normal_mask(labels)
        expected = np.array([True, True, False, False, False, False, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_training_exclusion_mask(self) -> None:
        labels = np.array([0, 0, 1, 2, 2, 3, 0, 0], dtype=np.int8)
        mask = make_training_exclusion_mask(labels)
        expected = np.array([False, False, True, True, True, True, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_onset_allowed_suppresses_buffers_and_reference(self) -> None:
        labels = np.array([0, 0, 0, 0, 1, 2, 2, 3, 3, 0, 0], dtype=np.int8)
        ref_end = 3
        mask = make_onset_allowed_from_zones(labels, reference_end_idx=ref_end)
        # First 3 (reference): False
        # idx 3: clean_normal after ref → True
        # idx 4: pre_buffer → False
        # idx 5-6: anomaly → True (onset detection is allowed during anomaly)
        # idx 7-8: post_recovery → False
        # idx 9-10: clean_normal after ref → True
        expected = np.array([False, False, False, True, False, True, True, False, False, True, True])
        np.testing.assert_array_equal(mask, expected)


if __name__ == "__main__":
    unittest.main()
