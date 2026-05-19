from __future__ import annotations

import numpy as np
import pandas as pd

from alma_service.domain_rule_diagnostics import interval_domain_rule_diagnostics
from alma_service.engineered_features import PreparedWellData


def _prepared_well(pressure: np.ndarray, frequency: np.ndarray) -> PreparedWellData:
    timestamps = pd.date_range("2026-01-01", periods=len(pressure), freq="5min").to_numpy()
    raw_matrix = np.column_stack([pressure, frequency]).astype(float)
    return PreparedWellData(
        well_id="test",
        split="train",
        timestamps=timestamps,
        raw_columns=["Давление на приеме насоса кгс/см²", "Выходная частота"],
        feature_columns=[],
        raw_matrix=raw_matrix,
        feature_matrix=np.empty((len(pressure), 0), dtype=float),
        reference_end_idx=len(pressure) // 2,
        reference_mask=np.ones(len(pressure), dtype=bool),
        stability_mask=np.ones(len(pressure), dtype=bool),
        onset_allowed_mask=np.ones(len(pressure), dtype=bool),
        detail={},
    )


def test_interval_domain_rule_diagnostics_detects_local_pressure_reversal() -> None:
    n = 12 * 24 * 4
    start_idx = n // 2
    pre = np.linspace(40.0, 30.0, start_idx, dtype=float)
    post = np.linspace(30.0, 36.0, n - start_idx, dtype=float)
    prepared = _prepared_well(
        pressure=np.concatenate([pre, post]),
        frequency=np.full(n, 50.0, dtype=float),
    )

    actual_start = pd.Timestamp(prepared.timestamps[start_idx])
    actual_end = actual_start + pd.Timedelta(days=7)
    result = interval_domain_rule_diagnostics(
        prepared,
        "salt",
        actual_start,
        actual_end,
    )

    assert result["domain_rule_status"] == "ok"
    assert result["pressure_pre_slope_direction"] == "down"
    assert result["pressure_post_slope_direction"] == "up"
    assert result["pressure_slope_change_per_day"] > 0
    assert result["trend_reversal_score"] > 99
    assert result["frequency_stability_score"] == 100.0


def test_interval_domain_rule_diagnostics_reports_missing_pressure() -> None:
    pressure = np.linspace(1.0, 2.0, 100, dtype=float)
    prepared = _prepared_well(pressure, np.full(100, 50.0, dtype=float))
    prepared.raw_columns[0] = "Неизвестный параметр"

    result = interval_domain_rule_diagnostics(
        prepared,
        "pritok",
        pd.Timestamp(prepared.timestamps[50]),
        pd.Timestamp(prepared.timestamps[80]),
    )

    assert result["domain_rule_status"] == "missing_pressure_channel"
