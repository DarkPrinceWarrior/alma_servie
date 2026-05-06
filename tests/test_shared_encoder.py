from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from alma_service.zone_labels import Zone, label_zones, make_clean_normal_mask


class MockPreparedWellData:
    """Minimal mock of PreparedWellData for testing shared encoder pool."""

    def __init__(
        self,
        well_id: str,
        split: str,
        feature_columns: list[str],
        feature_matrix: np.ndarray,
        reference_mask: np.ndarray,
    ) -> None:
        self.well_id = well_id
        self.split = split
        self.feature_columns = feature_columns
        self.feature_matrix = feature_matrix
        self.reference_mask = reference_mask


class SharedEncoderPoolTests(unittest.TestCase):
    def test_collect_pool_intersects_channels(self) -> None:
        from alma_service.shared_encoder import collect_shared_train_pool

        cols_a = ["ch_x", "ch_y", "ch_z"]
        cols_b = ["ch_y", "ch_z", "ch_w"]
        n = 100
        well_a = MockPreparedWellData(
            well_id="w1",
            split="train",
            feature_columns=cols_a,
            feature_matrix=np.random.randn(n, 3).astype(np.float32),
            reference_mask=np.ones(n, dtype=bool),
        )
        well_b = MockPreparedWellData(
            well_id="w2",
            split="train",
            feature_columns=cols_b,
            feature_matrix=np.random.randn(n, 3).astype(np.float32),
            reference_mask=np.ones(n, dtype=bool),
        )
        pool, shared_ch, ids = collect_shared_train_pool({"w1": well_a, "w2": well_b})  # type: ignore
        self.assertEqual(sorted(shared_ch), ["ch_y", "ch_z"])
        self.assertEqual(pool.shape[1], 2)
        self.assertEqual(pool.shape[0], 200)  # 100 from each well
        self.assertEqual(sorted(ids), ["w1", "w2"])

    def test_collect_pool_empty_intersection_raises(self) -> None:
        from alma_service.shared_encoder import collect_shared_train_pool

        n = 50
        well_a = MockPreparedWellData(
            well_id="w1",
            split="train",
            feature_columns=["ch_a"],
            feature_matrix=np.random.randn(n, 1).astype(np.float32),
            reference_mask=np.ones(n, dtype=bool),
        )
        well_b = MockPreparedWellData(
            well_id="w2",
            split="train",
            feature_columns=["ch_b"],
            feature_matrix=np.random.randn(n, 1).astype(np.float32),
            reference_mask=np.ones(n, dtype=bool),
        )
        with self.assertRaises(ValueError):
            collect_shared_train_pool({"w1": well_a, "w2": well_b})  # type: ignore

    def test_collect_pool_only_train_split(self) -> None:
        from alma_service.shared_encoder import collect_shared_train_pool

        cols = ["ch_y", "ch_z"]
        n = 100
        well_train = MockPreparedWellData(
            well_id="w1",
            split="train",
            feature_columns=cols,
            feature_matrix=np.random.randn(n, 2).astype(np.float32),
            reference_mask=np.ones(n, dtype=bool),
        )
        well_test = MockPreparedWellData(
            well_id="w2",
            split="test",
            feature_columns=cols,
            feature_matrix=np.random.randn(n, 2).astype(np.float32),
            reference_mask=np.ones(n, dtype=bool),
        )
        pool, _, ids = collect_shared_train_pool({"w1": well_train, "w2": well_test})  # type: ignore
        self.assertEqual(ids, ["w1"])
        self.assertEqual(pool.shape[0], 100)

    def test_collect_pool_reference_mask_filtering(self) -> None:
        from alma_service.shared_encoder import collect_shared_train_pool

        cols = ["ch_a", "ch_b"]
        n = 200
        ref_mask = np.zeros(n, dtype=bool)
        ref_mask[:50] = True  # only first 50 are reference
        well = MockPreparedWellData(
            well_id="w1",
            split="train",
            feature_columns=cols,
            feature_matrix=np.random.randn(n, 2).astype(np.float32),
            reference_mask=ref_mask,
        )
        pool, _, _ = collect_shared_train_pool({"w1": well})  # type: ignore
        self.assertEqual(pool.shape[0], 50)


class SelectSharedColumnsTests(unittest.TestCase):
    def test_projection(self) -> None:
        from alma_service.shared_encoder import select_shared_columns

        feature_columns = ["ch_a", "ch_b", "ch_c", "ch_d"]
        n = 10
        matrix = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
        shared_channels = ["ch_b", "ch_d"]
        result = select_shared_columns(feature_columns, matrix, shared_channels)
        self.assertEqual(result.shape, (n, 2))
        np.testing.assert_array_equal(result[:, 0], matrix[:, 1])  # ch_b
        np.testing.assert_array_equal(result[:, 1], matrix[:, 3])  # ch_d

    def test_projection_fills_missing_channels_with_neutral_zero(self) -> None:
        from alma_service.shared_encoder import select_shared_columns

        feature_columns = ["ch_a", "ch_c"]
        n = 10
        matrix = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
        shared_channels = ["ch_a", "ch_b", "ch_c"]
        result = select_shared_columns(feature_columns, matrix, shared_channels)
        self.assertEqual(result.shape, (n, 3))
        np.testing.assert_array_equal(result[:, 0], matrix[:, 0])
        np.testing.assert_array_equal(result[:, 1], np.zeros(n, dtype=np.float32))
        np.testing.assert_array_equal(result[:, 2], matrix[:, 1])


if __name__ == "__main__":
    unittest.main()
