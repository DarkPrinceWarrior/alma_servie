from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alma_service.onset_detection import choose_reference_end_index
from alma_service.well_features import get_well_feature_columns


@dataclass
class WellMatrix:
    data: np.ndarray
    timestamps: np.ndarray
    feature_columns: list[str]
    reference_end_idx: int
    trim_start_idx: int


def forward_fill_causal(data: np.ndarray) -> np.ndarray:
    out = np.asarray(data, dtype=np.float32).copy()
    if out.size == 0:
        return out

    for col_idx in range(out.shape[1]):
        col = out[:, col_idx]
        mask = np.isnan(col)
        if mask.all():
            continue
        idx = np.where(~mask, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = col[idx]
        out[:, col_idx] = filled
    return out


def _select_feature_columns(
    wd: pd.DataFrame,
    reference_end_idx: int,
    patch_size: int,
    min_reference_coverage: float,
    min_total_coverage: float,
) -> list[str]:
    candidates = get_well_feature_columns(wd)
    chosen: list[str] = []
    relaxed: list[str] = []

    for column in candidates:
        series = wd[column]
        total_cov = float(series.notna().mean())
        ref_slice = series.iloc[:reference_end_idx]
        ref_cov = float(ref_slice.notna().mean()) if len(ref_slice) else 0.0
        first_valid = series.first_valid_index()
        if first_valid is None:
            continue
        enough_history = int(first_valid) <= max(reference_end_idx - patch_size, 0)
        if total_cov >= min_total_coverage and ref_cov >= min_reference_coverage and enough_history:
            chosen.append(column)
        elif total_cov >= min_total_coverage and ref_cov >= max(min_reference_coverage - 0.2, 0.4):
            relaxed.append(column)

    return chosen or relaxed


def prepare_blind_well_matrix(
    well_df: pd.DataFrame,
    patch_size: int,
    reference_min_ratio: float,
    reference_max_ratio: float,
    reference_min_days: float,
    min_reference_coverage: float,
    min_total_coverage: float,
) -> WellMatrix | None:
    wd = well_df.sort_values("timestamp").reset_index(drop=True)
    timestamps = wd["timestamp"].to_numpy()
    if len(timestamps) < patch_size * 4:
        return None

    provisional_ref_end_idx = choose_reference_end_index(
        timestamps=timestamps,
        patch_size=patch_size,
        min_ratio=reference_min_ratio,
        max_ratio=reference_max_ratio,
        min_days=reference_min_days,
    )
    feature_columns = _select_feature_columns(
        wd=wd,
        reference_end_idx=provisional_ref_end_idx,
        patch_size=patch_size,
        min_reference_coverage=min_reference_coverage,
        min_total_coverage=min_total_coverage,
    )
    if not feature_columns:
        return None

    first_valid_positions = [int(wd[column].first_valid_index()) for column in feature_columns]
    trim_start_idx = max(first_valid_positions)
    wd = wd.iloc[trim_start_idx:].reset_index(drop=True)
    timestamps = wd["timestamp"].to_numpy()
    if len(timestamps) < patch_size * 4:
        return None

    reference_end_idx = choose_reference_end_index(
        timestamps=timestamps,
        patch_size=patch_size,
        min_ratio=reference_min_ratio,
        max_ratio=reference_max_ratio,
        min_days=reference_min_days,
    )
    feature_columns = _select_feature_columns(
        wd=wd,
        reference_end_idx=reference_end_idx,
        patch_size=patch_size,
        min_reference_coverage=min_reference_coverage,
        min_total_coverage=min_total_coverage,
    )
    if not feature_columns:
        return None

    first_valid_positions = [int(wd[column].first_valid_index()) for column in feature_columns]
    second_trim = max(first_valid_positions)
    if second_trim > 0:
        trim_start_idx += second_trim
        wd = wd.iloc[second_trim:].reset_index(drop=True)
        timestamps = wd["timestamp"].to_numpy()
        if len(timestamps) < patch_size * 4:
            return None
        reference_end_idx = choose_reference_end_index(
            timestamps=timestamps,
            patch_size=patch_size,
            min_ratio=reference_min_ratio,
            max_ratio=reference_max_ratio,
            min_days=reference_min_days,
        )

    data = wd[feature_columns].to_numpy(dtype=np.float32)
    data = forward_fill_causal(data)

    if np.isnan(data).any():
        good_columns = []
        for idx, column in enumerate(feature_columns):
            if not np.isnan(data[:, idx]).any():
                good_columns.append(column)
        if not good_columns:
            return None
        data = wd[good_columns].to_numpy(dtype=np.float32)
        data = forward_fill_causal(data)
        feature_columns = good_columns

    if data.shape[1] == 0:
        return None

    return WellMatrix(
        data=data,
        timestamps=timestamps,
        feature_columns=feature_columns,
        reference_end_idx=reference_end_idx,
        trim_start_idx=trim_start_idx,
    )
