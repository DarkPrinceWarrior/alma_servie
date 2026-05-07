from __future__ import annotations

import numpy as np
import pandas as pd

from alma_service.engineered_features import PreparedWellData
from alma_service.pressure_trend import PRESSURE_COL
from alma_service.salt_trend import build_salt_deposition_branch, fuse_model_with_salt_trend


def _prepared_salt(raw_matrix: np.ndarray, reference_points: int) -> PreparedWellData:
    n = len(raw_matrix)
    reference_mask = np.zeros(n, dtype=bool)
    reference_mask[:reference_points] = True
    feature_columns = [
        "soft::pressure_freq_ratio::raw",
        "soft::power_freq_ratio::raw",
        "soft::vibration_vector::raw",
    ]
    pressure = raw_matrix[:, 0]
    freq = raw_matrix[:, 1]
    power = raw_matrix[:, 2]
    vibration = raw_matrix[:, 3]
    feature_matrix = np.column_stack(
        [
            pressure / (np.abs(freq) + 1e-3),
            power / (np.abs(freq) + 1e-3),
            vibration,
        ]
    ).astype(np.float32)
    return PreparedWellData(
        well_id="test",
        split="train",
        timestamps=pd.date_range("2026-01-01", periods=n, freq="15min").to_numpy(),
        raw_columns=[
            PRESSURE_COL,
            "Выходная частота",
            "Полная выходная мощность",
            "Вибрация ХY",
            "Температура масла двигателя",
        ],
        feature_columns=feature_columns,
        raw_matrix=raw_matrix.astype(np.float32),
        feature_matrix=feature_matrix,
        reference_end_idx=reference_points,
        reference_mask=reference_mask,
        stability_mask=np.ones(n, dtype=bool),
        onset_allowed_mask=np.ones(n, dtype=bool),
        detail={},
    )


def test_salt_trend_scores_sustained_multigroup_drift() -> None:
    rng = np.random.default_rng(456)
    normal_n = 360
    drift_n = 240
    pressure = np.concatenate(
        [
            40.0 + rng.normal(0.0, 0.03, normal_n),
            np.linspace(40.2, 44.0, drift_n) + rng.normal(0.0, 0.03, drift_n),
        ]
    )
    freq = np.concatenate(
        [
            190.0 + rng.normal(0.0, 0.02, normal_n),
            np.linspace(190.0, 194.0, drift_n) + rng.normal(0.0, 0.02, drift_n),
        ]
    )
    power = np.concatenate(
        [
            48.0 + rng.normal(0.0, 0.05, normal_n),
            np.linspace(48.0, 44.0, drift_n) + rng.normal(0.0, 0.05, drift_n),
        ]
    )
    vibration = np.concatenate(
        [
            0.20 + rng.normal(0.0, 0.005, normal_n),
            np.linspace(0.20, 0.35, drift_n) + rng.normal(0.0, 0.005, drift_n),
        ]
    )
    temp = np.concatenate(
        [
            85.0 + rng.normal(0.0, 0.02, normal_n),
            np.linspace(85.0, 87.0, drift_n) + rng.normal(0.0, 0.02, drift_n),
        ]
    )
    prepared = _prepared_salt(
        np.column_stack([pressure, freq, power, vibration, temp]),
        reference_points=320,
    )

    out = build_salt_deposition_branch(prepared)

    ref_score = out.score[prepared.reference_mask]
    drift_score = out.score[430:]
    assert out.detail["salt_trend_enabled"] is True
    assert float(np.nanmedian(drift_score)) > float(np.nanquantile(ref_score, 0.95))
    assert float(np.nanmax(drift_score)) > 1.0
    assert float(np.nanmedian(out.components["salt_group_agreement"][430:])) > 0.0
    assert "salt_distribution_shift_score" in out.components
    assert "salt_deposition_conformal_tail_score" in out.components
    shift_ref = out.components["salt_distribution_shift_score"][prepared.reference_mask]
    shift_drift = out.components["salt_distribution_shift_score"][430:]
    assert float(np.nanmedian(shift_drift)) > float(np.nanquantile(shift_ref, 0.95))


def test_salt_trend_fusion_keeps_model_score_by_default() -> None:
    raw = np.column_stack(
        [
            np.concatenate([np.full(260, 40.0), np.linspace(40.0, 43.0, 140)]),
            np.concatenate([np.full(260, 190.0), np.linspace(190.0, 193.0, 140)]),
            np.concatenate([np.full(260, 48.0), np.linspace(48.0, 45.0, 140)]),
            np.concatenate([np.full(260, 0.20), np.linspace(0.20, 0.30, 140)]),
            np.concatenate([np.full(260, 85.0), np.linspace(85.0, 86.0, 140)]),
        ]
    )
    prepared = _prepared_salt(raw, reference_points=220)
    salt = build_salt_deposition_branch(prepared)
    model = np.zeros(len(raw), dtype=np.float32)
    model[-10:] = 3.0

    fused, components, detail = fuse_model_with_salt_trend(
        model_score=model,
        salt_output=salt,
        reference_mask=prepared.reference_mask,
    )

    assert detail["fusion"] == "paano_plus_tuned_salt_deposition_residual"
    assert "paano_tail_score" in components
    assert "salt_deposition_score" in components
    assert "salt_deposition_tail_score" in components
    assert "salt_deposition_calibrated_fusion_score" in components
    assert "salt_distribution_shift_tail_score" in components
    assert detail["tail_calibration"] == "per_well_reference_conformal_rank"
    assert float(fused[-1]) == float(model[-1])
