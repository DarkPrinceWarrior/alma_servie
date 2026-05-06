from __future__ import annotations

import numpy as np
import pandas as pd

from alma_service.engineered_features import PreparedWellData
from alma_service.negermet_signature import (
    LOAD_COLUMNS,
    build_negermet_signature_branch,
    fuse_model_with_negermet_signature,
)
from alma_service.pressure_trend import PRESSURE_COL


def _prepared_negermet(raw_matrix: np.ndarray, reference_points: int) -> PreparedWellData:
    n = len(raw_matrix)
    reference_mask = np.zeros(n, dtype=bool)
    reference_mask[:reference_points] = True
    return PreparedWellData(
        well_id="test",
        split="train",
        timestamps=pd.date_range("2026-01-01", periods=n, freq="2min").to_numpy(),
        raw_columns=[PRESSURE_COL, LOAD_COLUMNS[0], "Температура масла двигателя"],
        feature_columns=["x"],
        raw_matrix=raw_matrix.astype(np.float32),
        feature_matrix=np.zeros((n, 1), dtype=np.float32),
        reference_end_idx=reference_points,
        reference_mask=reference_mask,
        stability_mask=np.ones(n, dtype=bool),
        onset_allowed_mask=np.ones(n, dtype=bool),
        detail={},
    )


def test_negermet_signature_scores_short_step_above_reference() -> None:
    rng = np.random.default_rng(321)
    normal_pressure = 80.0 + rng.normal(0.0, 0.05, size=240)
    normal_current = 35.0 + rng.normal(0.0, 0.03, size=240)
    normal_temp = 70.0 + rng.normal(0.0, 0.02, size=240)
    shifted_pressure = 86.0 + rng.normal(0.0, 0.05, size=120)
    shifted_current = 41.0 + rng.normal(0.0, 0.03, size=120)
    shifted_temp = 72.0 + rng.normal(0.0, 0.02, size=120)
    raw = np.column_stack(
        [
            np.concatenate([normal_pressure, shifted_pressure]),
            np.concatenate([normal_current, shifted_current]),
            np.concatenate([normal_temp, shifted_temp]),
        ]
    )
    prepared = _prepared_negermet(raw, reference_points=220)

    out = build_negermet_signature_branch(prepared)

    ref_score = out.score[prepared.reference_mask]
    post_score = out.score[260:]
    assert out.detail["negermet_signature_enabled"] is True
    assert float(np.nanmedian(post_score)) > float(np.nanquantile(ref_score, 0.95))
    assert float(np.nanmax(post_score)) > 1.0
    assert np.nanmedian(out.components["negermet_pressure_direction"][260:]) > 0.0


def test_negermet_signature_fusion_keeps_model_score_by_default() -> None:
    raw = np.column_stack(
        [
            np.concatenate([np.full(220, 80.0), np.full(80, 86.0)]),
            np.concatenate([np.full(220, 35.0), np.full(80, 42.0)]),
            np.concatenate([np.full(220, 70.0), np.full(80, 71.5)]),
        ]
    )
    prepared = _prepared_negermet(raw, reference_points=200)
    signature = build_negermet_signature_branch(prepared)
    model = np.zeros(len(raw), dtype=np.float32)
    model[-10:] = 5.0

    fused, components, detail = fuse_model_with_negermet_signature(
        model_score=model,
        signature_output=signature,
        reference_mask=prepared.reference_mask,
    )

    assert detail["fusion"] == "paano_plus_tuned_negermet_signature"
    assert "paano_tail_score" in components
    assert "negermet_signature_score" in components
    assert float(fused[-1]) == float(model[-1])
