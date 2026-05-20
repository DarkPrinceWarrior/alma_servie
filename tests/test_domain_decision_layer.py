from __future__ import annotations

import numpy as np
import pandas as pd

from alma_service.domain_decision_layer import (
    DOMAIN_NEGERMET_CANDIDATE,
    DOMAIN_PRITOK_CANDIDATE,
    DOMAIN_REJECTED_FREQUENCY_TRANSITION,
    DOMAIN_SALT_CANDIDATE,
    assess_domain_start,
)
from alma_service.engineered_features import FREQ_COL, PRESSURE_COL, PreparedWellData


def _prepared(pressure: np.ndarray, frequency: np.ndarray) -> PreparedWellData:
    timestamps = pd.date_range("2026-01-01", periods=len(pressure), freq="5min").to_numpy()
    raw_matrix = np.column_stack([pressure.astype(float), frequency.astype(float)])
    return PreparedWellData(
        well_id="w1",
        split="train",
        timestamps=timestamps,
        raw_columns=[PRESSURE_COL, FREQ_COL],
        feature_columns=[PRESSURE_COL, FREQ_COL],
        raw_matrix=raw_matrix,
        feature_matrix=raw_matrix,
        reference_end_idx=len(pressure),
        reference_mask=np.ones(len(pressure), dtype=bool),
        stability_mask=np.ones(len(pressure), dtype=bool),
        onset_allowed_mask=np.ones(len(pressure), dtype=bool),
        detail={},
    )


def _row(detected_time: str) -> pd.Series:
    return pd.Series({
        "well_id": "w1",
        "detected_time": pd.Timestamp(detected_time),
        "start_class": "anomaly_candidate",
        "is_bad_data": False,
        "is_regime_event": False,
    })


def test_pritok_candidate_requires_stable_frequency() -> None:
    pressure = np.r_[np.full(288, 100.0), np.linspace(100.0, 104.0, 288)]
    frequency = np.full(576, 50.0)
    result = assess_domain_start(
        anomaly_key="pritok",
        prepared=_prepared(pressure, frequency),
        start_row=_row("2026-01-02"),
    )
    assert result["domain_verdict"] == DOMAIN_PRITOK_CANDIDATE
    assert result["domain_action"] == "accept"


def test_pritok_frequency_transition_rejects_pressure_trend() -> None:
    pressure = np.r_[np.full(288, 100.0), np.linspace(100.0, 104.0, 288)]
    frequency = np.r_[np.full(288, 50.0), np.full(288, 55.0)]
    result = assess_domain_start(
        anomaly_key="pritok",
        prepared=_prepared(pressure, frequency),
        start_row=_row("2026-01-02"),
    )
    assert result["domain_verdict"] == DOMAIN_REJECTED_FREQUENCY_TRANSITION
    assert result["domain_action"] == "reject"


def test_salt_uses_slope_reversal_up_not_global_median() -> None:
    pressure = np.r_[np.linspace(120.0, 100.0, 864), np.linspace(96.0, 102.0, 864)]
    frequency = np.full(len(pressure), 50.0)
    result = assess_domain_start(
        anomaly_key="salt",
        prepared=_prepared(pressure, frequency),
        start_row=_row("2026-01-04"),
    )
    assert result["domain_verdict"] == DOMAIN_SALT_CANDIDATE
    assert result["domain_action"] == "accept"
    assert result["domain_pressure_post_vs_pre_pct"] < 0
    assert result["domain_pressure_slope_change_per_day"] > 0


def test_negermet_candidate_on_pressure_step_up() -> None:
    pressure = np.r_[np.full(24, 100.0), np.full(24, 130.0)]
    frequency = np.full(len(pressure), 50.0)
    result = assess_domain_start(
        anomaly_key="negermet",
        prepared=_prepared(pressure, frequency),
        start_row=_row("2026-01-01 02:00:00"),
    )
    assert result["domain_verdict"] == DOMAIN_NEGERMET_CANDIDATE
    assert result["domain_action"] == "accept"


def test_negermet_does_not_reject_supported_step_by_regime_flag_only() -> None:
    pressure = np.r_[np.full(24, 100.0), np.full(24, 130.0)]
    frequency = np.full(len(pressure), 50.0)
    row = _row("2026-01-01 02:00:00")
    row["start_class"] = "regime_event"
    row["is_regime_event"] = True
    result = assess_domain_start(
        anomaly_key="negermet",
        prepared=_prepared(pressure, frequency),
        start_row=row,
    )
    assert result["domain_verdict"] == DOMAIN_NEGERMET_CANDIDATE
    assert result["domain_action"] == "accept"
    assert "despite_regime_context" in result["domain_reason"]


def test_negermet_weak_candidate_without_physical_response_stays_uncertain() -> None:
    pressure = np.r_[np.full(24, 100.0), np.full(24, 100.2)]
    frequency = np.full(len(pressure), 50.0)
    result = assess_domain_start(
        anomaly_key="negermet",
        prepared=_prepared(pressure, frequency),
        start_row=_row("2026-01-01 02:00:00"),
    )
    assert result["domain_verdict"] == "uncertain"
    assert result["domain_action"] == "uncertain"
