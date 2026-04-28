from __future__ import annotations

import numpy as np
import pandas as pd

from alma_service.engineered_features import PreparedWellData
from alma_service.pressure_trend import (
    PRESSURE_COL,
    build_pressure_trend_branch,
    fuse_model_with_pressure_trend,
)


def _prepared_pressure(values: np.ndarray, reference_points: int) -> PreparedWellData:
    n = len(values)
    reference_mask = np.zeros(n, dtype=bool)
    reference_mask[:reference_points] = True
    return PreparedWellData(
        well_id="test",
        split="train",
        timestamps=pd.date_range("2026-01-01", periods=n, freq="10min").to_numpy(),
        raw_columns=[PRESSURE_COL],
        feature_columns=["x"],
        raw_matrix=values.reshape(-1, 1).astype(np.float32),
        feature_matrix=np.zeros((n, 1), dtype=np.float32),
        reference_end_idx=reference_points,
        reference_mask=reference_mask,
        stability_mask=np.ones(n, dtype=bool),
        onset_allowed_mask=np.ones(n, dtype=bool),
        detail={},
    )


def test_pressure_trend_scores_sustained_shift_above_reference() -> None:
    rng = np.random.default_rng(123)
    normal = 50.0 + rng.normal(0.0, 0.04, size=240)
    shifted = np.linspace(50.1, 54.0, 180) + rng.normal(0.0, 0.04, size=180)
    prepared = _prepared_pressure(np.concatenate([normal, shifted]), reference_points=220)

    out = build_pressure_trend_branch(prepared)

    ref_score = out.score[prepared.reference_mask]
    anomaly_score = out.score[260:]
    assert out.detail["pressure_trend_enabled"] is True
    assert float(np.nanmedian(anomaly_score)) > float(np.nanquantile(ref_score, 0.95))
    assert float(np.nanmax(anomaly_score)) > 90.0
    assert np.nanmedian(out.components["pressure_trend_direction"][260:]) > 0.0


def test_pressure_trend_fusion_keeps_model_or_pressure_evidence() -> None:
    values = np.concatenate([np.full(220, 50.0), np.linspace(50.0, 53.0, 120)])
    prepared = _prepared_pressure(values, reference_points=200)
    pressure = build_pressure_trend_branch(prepared)
    model = np.zeros(len(values), dtype=np.float32)
    model[-10:] = 10.0

    fused, components, detail = fuse_model_with_pressure_trend(
        model_score=model,
        pressure_output=pressure,
        reference_mask=prepared.reference_mask,
    )

    assert detail["fusion"] == "paano_plus_tuned_pressure_trend"
    assert "paano_tail_score" in components
    assert "pressure_trend_score" in components
    assert float(fused[-1]) == float(model[-1])
