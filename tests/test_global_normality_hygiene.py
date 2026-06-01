from __future__ import annotations

import unittest
from dataclasses import dataclass, field, replace  # noqa: F401  (replace used via module under test)
from typing import Any

import numpy as np
import pandas as pd

from alma_service.global_normality import _apply_norm_pool_hygiene


@dataclass
class MockPreparedWellData:
    well_id: str
    split: str
    timestamps: np.ndarray
    reference_mask: np.ndarray
    detail: dict[str, Any] = field(default_factory=dict)


def _make_prepared(well_id: str, days: int, freq_minutes: int = 5) -> MockPreparedWellData:
    points = days * 24 * 60 // freq_minutes
    timestamps = pd.date_range("2026-01-01", periods=points, freq=f"{freq_minutes}min").to_numpy()
    return MockPreparedWellData(
        well_id=well_id,
        split="train",
        timestamps=timestamps,
        reference_mask=np.ones(points, dtype=bool),
    )


class NormPoolHygieneTests(unittest.TestCase):
    def test_no_rules_returns_pool_unchanged(self) -> None:
        pool = {"pritok:610": _make_prepared("610", days=10)}
        cleaned, audit = _apply_norm_pool_hygiene(pool, (), verbose=False)
        self.assertIs(cleaned["pritok:610"], pool["pritok:610"])
        self.assertFalse(audit["enabled"])

    def test_exclude_removes_well_from_pool(self) -> None:
        pool = {
            "negermet:5271г": _make_prepared("5271г", days=10),
            "pritok:713": _make_prepared("713", days=10),
        }
        rules = ({"well_id": "5271г", "source": "negermet", "action": "exclude", "reason": "тест"},)
        cleaned, audit = _apply_norm_pool_hygiene(pool, rules, verbose=False)
        self.assertNotIn("negermet:5271г", cleaned)
        self.assertIn("pritok:713", cleaned)
        self.assertEqual(len(audit["applied"]), 1)
        self.assertEqual(audit["applied"][0]["action"], "exclude")

    def test_exclude_respects_source_filter(self) -> None:
        pool = {
            "negermet:5271г": _make_prepared("5271г", days=10),
            "norm_work:5271г": _make_prepared("5271г", days=10),
        }
        rules = ({"well_id": "5271г", "source": "negermet", "action": "exclude", "reason": "тест"},)
        cleaned, _ = _apply_norm_pool_hygiene(pool, rules, verbose=False)
        self.assertNotIn("negermet:5271г", cleaned)
        self.assertIn("norm_work:5271г", cleaned)

    def test_trim_reference_tail_removes_last_days(self) -> None:
        prepared = _make_prepared("610", days=50)
        pool = {"pritok:610": prepared}
        rules = (
            {
                "well_id": "610",
                "source": "pritok",
                "action": "trim_reference_tail_days",
                "days": 21,
                "reason": "тест",
            },
        )
        cleaned, audit = _apply_norm_pool_hygiene(pool, rules, verbose=False)
        self.assertIn("pritok:610", cleaned)
        new_mask = cleaned["pritok:610"].reference_mask
        timestamps = pd.to_datetime(prepared.timestamps)
        kept_last = timestamps[np.flatnonzero(new_mask)[-1]]
        original_last = timestamps[-1]
        trimmed_days = (original_last - kept_last).total_seconds() / 86400.0
        self.assertGreaterEqual(trimmed_days, 21.0 - 0.01)
        # исходная маска не изменена (копия)
        self.assertTrue(prepared.reference_mask.all())
        self.assertEqual(audit["applied"][0]["action"], "trim_reference_tail_days")
        self.assertGreater(audit["applied"][0]["reference_rows_removed"], 0)

    def test_trim_drops_well_when_nothing_left(self) -> None:
        prepared = _make_prepared("902", days=5)
        pool = {"pritok:902": prepared}
        rules = (
            {
                "well_id": "902",
                "action": "trim_reference_tail_days",
                "days": 10,
                "reason": "тест: обрезка длиннее префикса",
            },
        )
        cleaned, _ = _apply_norm_pool_hygiene(pool, rules, verbose=False)
        self.assertNotIn("pritok:902", cleaned)

    def test_well_id_matching_is_case_insensitive(self) -> None:
        pool = {"pritok:1442Л": _make_prepared("1442Л", days=10)}
        rules = ({"well_id": "1442л", "action": "exclude", "reason": "тест"},)
        cleaned, _ = _apply_norm_pool_hygiene(pool, rules, verbose=False)
        self.assertNotIn("pritok:1442Л", cleaned)

    def test_unknown_action_raises(self) -> None:
        pool = {"pritok:610": _make_prepared("610", days=10)}
        rules = ({"well_id": "610", "action": "unknown_action", "reason": "тест"},)
        with self.assertRaises(ValueError):
            _apply_norm_pool_hygiene(pool, rules, verbose=False)


if __name__ == "__main__":
    unittest.main()
