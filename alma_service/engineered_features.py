from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from alma_service.onset_detection import choose_reference_end_index, infer_step_seconds
from alma_service.well_features import get_well_feature_columns
from alma_service.well_preprocess import forward_fill_causal
from alma_service.zone_labels import label_zones, make_clean_normal_mask, make_onset_allowed_from_zones

PRESSURE_COL = "Давление на приеме насоса кгс/см²"
FREQ_COL = "Выходная частота"
POWER_COL = "Полная выходная мощность"
OUTPUT_CURRENT_COL = "Выходной ток ПЧ"
PHASE_CURRENT_COLS = ("Ток на фазе А", "Ток на фазе В", "Ток на фазе С")
PHASE_VOLTAGE_COLS = ("Фазное напряжение Ua", "Фазное напряжение Ub", "Фазное напряжение Uc")
VIBRATION_COLS = ("Вибрация Х", "Вибрация Y", "Вибрация Z")

MASK_PRIORITY_COLUMNS = (
    FREQ_COL,
    POWER_COL,
    OUTPUT_CURRENT_COL,
    PRESSURE_COL,
)
FULL_WINDOW_MINUTES = (10, 60, 240)
COMPACT_WINDOW_MINUTES = (10, 60)
TIGHT_WINDOW_MINUTES = (10,)
FULL_SLOPE_WINDOWS_MINUTES = (10, 60)
COMPACT_SLOPE_WINDOWS_MINUTES = (10,)
MASK_PROFILES = (
    {
        "name": "default",
        "back_minutes": 10,
        "forward_minutes": 30,
        "step_sigma": 6.0,
        "flatline_minutes": 60,
        "missing_run_length": 3,
    },
    {
        "name": "relaxed",
        "back_minutes": 5,
        "forward_minutes": 15,
        "step_sigma": 8.0,
        "flatline_minutes": 90,
        "missing_run_length": 5,
    },
    {
        "name": "minimal",
        "back_minutes": 2,
        "forward_minutes": 5,
        "step_sigma": 10.0,
        "flatline_minutes": 120,
        "missing_run_length": 8,
    },
)
from alma_service.paano_defaults import PATCH_SIZES
ANOMALY_PATCH_SIZE = {key: sizes[1] for key, sizes in PATCH_SIZES.items()}
MASK_TARGETS = {
    "negermet": {"target_masked_fraction": 0.65, "hard_max_fraction": 0.85},
    "pritok": {"target_masked_fraction": 0.75, "hard_max_fraction": 0.90},
    "salt": {"target_masked_fraction": 0.75, "hard_max_fraction": 0.90},
}
EPS = 1e-6


@dataclass
class PreparedWellData:
    well_id: str
    split: str
    timestamps: np.ndarray
    raw_columns: list[str]
    feature_columns: list[str]
    raw_matrix: np.ndarray
    feature_matrix: np.ndarray
    reference_end_idx: int
    reference_mask: np.ndarray
    stability_mask: np.ndarray
    onset_allowed_mask: np.ndarray
    detail: dict[str, object]


def _window_steps(step_seconds: float, minutes: int) -> int:
    return max(int(np.ceil((minutes * 60.0) / max(step_seconds, 1.0))), 2)


def _robust_mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def _expand_event_mask(events: np.ndarray, back_steps: int, forward_steps: int) -> np.ndarray:
    mask = np.asarray(events, dtype=bool)
    if not mask.any():
        return mask
    marks = np.zeros(len(mask) + 1, dtype=np.int32)
    for idx in np.flatnonzero(mask):
        start = max(idx - back_steps, 0)
        end = min(idx + forward_steps + 1, len(mask))
        marks[start] += 1
        marks[end] -= 1
    return np.cumsum(marks[:-1]) > 0


def _stale_run_length(values: np.ndarray, eps: float = EPS) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    out = np.zeros(len(x), dtype=np.float32)
    for idx in range(1, len(x)):
        if np.isfinite(x[idx]) and np.isfinite(x[idx - 1]) and abs(float(x[idx] - x[idx - 1])) <= eps:
            out[idx] = out[idx - 1] + 1.0
    return out


def _rolling_zscore(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    frame = pl.DataFrame({"x": np.asarray(values, dtype=np.float32)})
    stats = frame.with_columns(
        [
            pl.col("x").rolling_mean(window_size=window, min_samples=1).alias("mean"),
            pl.col("x").rolling_std(window_size=window, min_samples=2, ddof=0).fill_null(0.0).alias("std"),
        ]
    )
    mean = stats["mean"].to_numpy().astype(np.float32)
    std = stats["std"].to_numpy().astype(np.float32)
    z = np.divide(
        np.asarray(values, dtype=np.float32) - mean,
        std,
        out=np.zeros_like(std, dtype=np.float32),
        where=np.abs(std) > EPS,
    )
    return z.astype(np.float32), std


def _slope(values: np.ndarray, window: int) -> np.ndarray:
    frame = pl.DataFrame({"x": np.asarray(values, dtype=np.float32)})
    slope = frame.with_columns(
        ((pl.col("x") - pl.col("x").shift(window)) / float(window)).fill_null(0.0).alias("slope")
    )["slope"]
    return slope.to_numpy().astype(np.float32)


def _choose_anchor_columns(raw_df: pd.DataFrame, candidates: list[str]) -> list[str]:
    anchors = [column for column in MASK_PRIORITY_COLUMNS if column in candidates]
    if anchors:
        return anchors
    return candidates[: min(len(candidates), 4)]


def _build_instability_mask(
    raw_df: pd.DataFrame,
    filled_matrix: np.ndarray,
    base_columns: list[str],
    step_seconds: float,
    profile: dict[str, float | int | str],
    *,
    include_step_events: bool = True,
    include_missing_events: bool = True,
    include_flatline_events: bool = True,
    include_start_stop_events: bool = True,
) -> tuple[np.ndarray, list[str]]:
    if len(base_columns) == 0:
        return np.ones(len(raw_df), dtype=bool), []

    anchors = _choose_anchor_columns(raw_df, base_columns)
    event_mask = np.zeros(len(raw_df), dtype=bool)
    back_steps = _window_steps(step_seconds, int(profile["back_minutes"]))
    forward_steps = _window_steps(step_seconds, int(profile["forward_minutes"]))
    flatline_steps = max(_window_steps(step_seconds, int(profile["flatline_minutes"])), 12)
    missing_run_length = int(profile["missing_run_length"])
    step_sigma = float(profile["step_sigma"])

    for column in anchors:
        idx = base_columns.index(column)
        raw_series = pd.to_numeric(raw_df[column], errors="coerce")
        filled = filled_matrix[:, idx]
        delta = np.diff(filled, prepend=filled[0])
        abs_delta = np.abs(delta[1:])
        delta_mad = _robust_mad(abs_delta)
        if delta_mad > 0.0:
            step_thr = step_sigma * delta_mad
        else:
            positive_delta = abs_delta[abs_delta > EPS]
            step_thr = max(float(np.quantile(positive_delta, 0.90)) * 0.5, EPS) if len(positive_delta) else np.inf
        if include_step_events and np.isfinite(step_thr):
            event_mask |= np.abs(delta) > step_thr

        missing = raw_series.isna().to_numpy()
        if include_missing_events and missing.any():
            missing_run = (
                pd.Series(missing.astype(np.int8))
                .rolling(window=missing_run_length, min_periods=1)
                .sum()
                .to_numpy()
            )
            event_mask |= missing_run >= missing_run_length

        if column in (FREQ_COL, POWER_COL, OUTPUT_CURRENT_COL):
            abs_values = np.abs(filled[np.isfinite(filled)])
            positive = abs_values[abs_values > EPS]
            if len(positive):
                low_thr = max(float(np.quantile(positive, 0.10)) * 0.2, EPS)
                stale = _stale_run_length(filled)
                if include_flatline_events:
                    event_mask |= (stale >= flatline_steps) & (np.abs(filled) <= low_thr)
                prev = np.r_[filled[0], filled[:-1]]
                start_stop = ((np.abs(prev) > low_thr) & (np.abs(filled) <= low_thr)) | (
                    (np.abs(prev) <= low_thr) & (np.abs(filled) > low_thr)
                )
                if include_start_stop_events:
                    event_mask |= start_stop

    unstable = _expand_event_mask(event_mask, back_steps=back_steps, forward_steps=forward_steps)
    return ~unstable, anchors


def _add_base_feature_family(
    feature_dict: dict[str, np.ndarray],
    name: str,
    values: np.ndarray,
    missing_flag: np.ndarray,
    step_seconds: float,
    window_minutes: tuple[int, ...],
    slope_window_minutes: tuple[int, ...],
    include_stale: bool,
) -> None:
    values = np.asarray(values, dtype=np.float32)
    feature_dict[f"{name}::raw"] = values
    feature_dict[f"{name}::diff_1"] = np.diff(values, prepend=values[0]).astype(np.float32)
    feature_dict[f"{name}::missing"] = missing_flag.astype(np.float32)
    if include_stale:
        feature_dict[f"{name}::stale_run"] = _stale_run_length(values)

    for minutes in window_minutes:
        steps = _window_steps(step_seconds, minutes)
        z, std = _rolling_zscore(values, window=steps)
        feature_dict[f"{name}::z_{minutes}m"] = z
        feature_dict[f"{name}::std_{minutes}m"] = std

    for minutes in slope_window_minutes:
        steps = _window_steps(step_seconds, minutes)
        feature_dict[f"{name}::slope_{minutes}m"] = _slope(values, window=steps)


def _feature_mode(reference_points: int, masked_fraction: float) -> tuple[str, tuple[int, ...], tuple[int, ...], bool]:
    if reference_points < 1024 or masked_fraction >= 0.85:
        return "tight", TIGHT_WINDOW_MINUTES, COMPACT_SLOPE_WINDOWS_MINUTES, False
    if reference_points < 2500 or masked_fraction >= 0.70:
        return "compact", COMPACT_WINDOW_MINUTES, COMPACT_SLOPE_WINDOWS_MINUTES, True
    return "full", FULL_WINDOW_MINUTES, FULL_SLOPE_WINDOWS_MINUTES, True


def _build_soft_sensor_signals(base_columns: list[str], raw_matrix: np.ndarray) -> dict[str, np.ndarray]:
    lookup = {column: raw_matrix[:, idx] for idx, column in enumerate(base_columns)}
    derived: dict[str, np.ndarray] = {}

    if PRESSURE_COL in lookup and FREQ_COL in lookup:
        pressure = lookup[PRESSURE_COL]
        freq = lookup[FREQ_COL]
        derived["soft::pressure_freq_ratio"] = pressure / (np.abs(freq) + 1e-3)
        derived["soft::pressure_freq_gap"] = pressure - freq

    if POWER_COL in lookup and FREQ_COL in lookup:
        power = lookup[POWER_COL]
        freq = lookup[FREQ_COL]
        derived["soft::power_freq_ratio"] = power / (np.abs(freq) + 1e-3)
        derived["soft::power_freq_gap"] = power - freq

    if all(column in lookup for column in PHASE_CURRENT_COLS):
        currents = np.stack([lookup[column] for column in PHASE_CURRENT_COLS], axis=1)
        derived["soft::current_unbalance"] = np.std(currents, axis=1) / (np.mean(np.abs(currents), axis=1) + 1e-3)

    if all(column in lookup for column in PHASE_VOLTAGE_COLS):
        voltages = np.stack([lookup[column] for column in PHASE_VOLTAGE_COLS], axis=1)
        derived["soft::voltage_unbalance"] = np.std(voltages, axis=1) / (np.mean(np.abs(voltages), axis=1) + 1e-3)

    vib_cols = [column for column in VIBRATION_COLS if column in lookup]
    if vib_cols:
        vib = np.stack([lookup[column] for column in vib_cols], axis=1)
        derived["soft::vibration_vector"] = np.sqrt(np.sum(np.square(vib), axis=1))

    return derived


def _select_base_columns(
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
        series = pd.to_numeric(wd[column], errors="coerce")
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


def prepare_engineered_well(
    anomaly_key: str,
    well_id: str,
    split: str,
    well_df: pd.DataFrame,
    patch_size: int,
    reference_min_ratio: float,
    reference_max_ratio: float,
    reference_min_days: float,
    min_reference_coverage: float,
    min_total_coverage: float,
    anomaly_intervals: pd.DataFrame | None = None,
) -> PreparedWellData | None:
    patch_size = int(ANOMALY_PATCH_SIZE.get(anomaly_key, patch_size))
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
    base_columns = _select_base_columns(
        wd=wd,
        reference_end_idx=provisional_ref_end_idx,
        patch_size=patch_size,
        min_reference_coverage=min_reference_coverage,
        min_total_coverage=min_total_coverage,
    )
    if not base_columns:
        return None

    first_valid_positions = [int(pd.to_numeric(wd[column], errors="coerce").first_valid_index()) for column in base_columns]
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
    base_columns = _select_base_columns(
        wd=wd,
        reference_end_idx=reference_end_idx,
        patch_size=patch_size,
        min_reference_coverage=min_reference_coverage,
        min_total_coverage=min_total_coverage,
    )
    if not base_columns:
        return None

    first_valid_positions = [int(pd.to_numeric(wd[column], errors="coerce").first_valid_index()) for column in base_columns]
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

    raw_df = wd[base_columns].apply(pd.to_numeric, errors="coerce")
    raw_matrix = raw_df.to_numpy(dtype=np.float32)
    raw_matrix = forward_fill_causal(raw_matrix)
    if np.isnan(raw_matrix).any():
        keep_columns = [column for idx, column in enumerate(base_columns) if not np.isnan(raw_matrix[:, idx]).any()]
        if not keep_columns:
            return None
        base_columns = keep_columns
        raw_df = wd[base_columns].apply(pd.to_numeric, errors="coerce")
        raw_matrix = forward_fill_causal(raw_df.to_numpy(dtype=np.float32))

    step_seconds = infer_step_seconds(timestamps)
    min_ref_points = max(patch_size * 2, 64)
    mask_target = MASK_TARGETS.get(anomaly_key, MASK_TARGETS["salt"])
    chosen_profile_name = "default"
    chosen_masked_fraction = 0.0
    anchor_columns: list[str] = []
    stability_mask = np.ones(len(wd), dtype=bool)
    reference_mask = np.zeros(len(wd), dtype=bool)
    best_candidate: tuple[np.ndarray, np.ndarray, str, list[str], float, int] | None = None

    for profile in MASK_PROFILES:
        candidate_mask, anchors = _build_instability_mask(
            raw_df=raw_df,
            filled_matrix=raw_matrix,
            base_columns=base_columns,
            step_seconds=step_seconds,
            profile=profile,
        )
        candidate_reference_mask = np.zeros(len(wd), dtype=bool)
        candidate_reference_mask[:reference_end_idx] = True
        candidate_reference_mask &= candidate_mask
        candidate_ref_points = int(candidate_reference_mask.sum())
        candidate_masked_fraction = float((~candidate_mask).mean()) if len(candidate_mask) else 0.0
        current_rank = (
            candidate_ref_points,
            -candidate_masked_fraction,
        )
        if best_candidate is None or current_rank > (best_candidate[5], -best_candidate[4]):
            best_candidate = (
                candidate_mask,
                candidate_reference_mask,
                str(profile["name"]),
                anchors,
                candidate_masked_fraction,
                candidate_ref_points,
            )
        if (
            candidate_ref_points >= min_ref_points
            and candidate_masked_fraction <= float(mask_target["target_masked_fraction"])
        ):
            stability_mask = candidate_mask
            reference_mask = candidate_reference_mask
            anchor_columns = anchors
            chosen_profile_name = str(profile["name"])
            chosen_masked_fraction = candidate_masked_fraction
            break
    else:
        if best_candidate is None:
            return None
        stability_mask, reference_mask, chosen_profile_name, anchor_columns, chosen_masked_fraction, _ = best_candidate

    if int(reference_mask.sum()) < min_ref_points:
        reference_mask = np.zeros(len(wd), dtype=bool)
        reference_mask[:reference_end_idx] = True
    if int(reference_mask.sum()) < min_ref_points:
        return None

    # --- 4-zone data curation ---
    zone_labels_arr = None
    if anomaly_intervals is not None and not anomaly_intervals.empty:
        zone_labels_arr = label_zones(
            timestamps=timestamps,
            anomaly_intervals=anomaly_intervals,
            patch_size=patch_size,
            anomaly_key=anomaly_key,
        )
        clean_mask = make_clean_normal_mask(zone_labels_arr)
        # Tighten reference_mask: only clean_normal within reference window
        reference_mask = reference_mask & clean_mask
        if int(reference_mask.sum()) < min_ref_points:
            # Fallback: use original reference_mask without zone filtering
            reference_mask = np.zeros(len(wd), dtype=bool)
            reference_mask[:reference_end_idx] = True
            reference_mask &= stability_mask if stability_mask is not None else True

    if chosen_masked_fraction > float(mask_target["hard_max_fraction"]):
        onset_allowed_mask = np.ones(len(wd), dtype=bool)
        onset_allowed_mask[:reference_end_idx] = False
    elif anomaly_key == "negermet":
        onset_allowed_mask, _ = _build_instability_mask(
            raw_df=raw_df,
            filled_matrix=raw_matrix,
            base_columns=base_columns,
            step_seconds=step_seconds,
            profile={
                **next(
                    profile for profile in MASK_PROFILES if str(profile["name"]) == chosen_profile_name
                ),
                "back_minutes": 2,
                "forward_minutes": 5,
            },
            include_step_events=False,
            include_start_stop_events=False,
        )
        onset_allowed_mask[:reference_end_idx] = False
    else:
        onset_allowed_mask = stability_mask.copy()

    # Refine onset_allowed_mask with zone-aware suppression
    if zone_labels_arr is not None:
        zone_onset = make_onset_allowed_from_zones(zone_labels_arr, reference_end_idx)
        onset_allowed_mask = onset_allowed_mask & zone_onset

    feature_mode, feature_windows, slope_windows, include_stale = _feature_mode(
        reference_points=int(reference_mask.sum()),
        masked_fraction=chosen_masked_fraction,
    )

    feature_dict: dict[str, np.ndarray] = {}
    missing_matrix = raw_df.isna().to_numpy(dtype=np.float32)
    for idx, column in enumerate(base_columns):
        _add_base_feature_family(
            feature_dict=feature_dict,
            name=column,
            values=raw_matrix[:, idx],
            missing_flag=missing_matrix[:, idx],
            step_seconds=step_seconds,
            window_minutes=feature_windows,
            slope_window_minutes=slope_windows,
            include_stale=include_stale,
        )

    if anomaly_key == "salt":
        for name, values in _build_soft_sensor_signals(base_columns, raw_matrix).items():
            _add_base_feature_family(
                feature_dict=feature_dict,
                name=name,
                values=np.asarray(values, dtype=np.float32),
                missing_flag=np.zeros(len(values), dtype=np.float32),
                step_seconds=step_seconds,
                window_minutes=feature_windows,
                slope_window_minutes=slope_windows,
                include_stale=include_stale,
            )

    feature_df = pd.DataFrame(feature_dict)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_matrix = feature_df.to_numpy(dtype=np.float32)

    ref_values = feature_matrix[reference_mask]
    feature_mean = np.mean(ref_values, axis=0, keepdims=True).astype(np.float32)
    feature_std = np.std(ref_values, axis=0, keepdims=True).astype(np.float32)
    feature_std = np.where(feature_std < EPS, 1.0, feature_std)
    feature_matrix = (feature_matrix - feature_mean) / feature_std
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    detail = {
        "points": int(len(wd)),
        "raw_channels": int(len(base_columns)),
        "feature_count": int(feature_matrix.shape[1]),
        "reference_end_idx": int(reference_end_idx),
        "reference_points": int(reference_mask.sum()),
        "trim_start_idx": int(trim_start_idx),
        "masked_fraction": chosen_masked_fraction,
        "anchor_columns": anchor_columns,
        "mask_profile": chosen_profile_name,
        "feature_mode": feature_mode,
        "patch_size": patch_size,
        "zone_aware": zone_labels_arr is not None,
    }
    return PreparedWellData(
        well_id=well_id,
        split=split,
        timestamps=timestamps,
        raw_columns=base_columns,
        feature_columns=list(feature_df.columns),
        raw_matrix=raw_matrix.astype(np.float32),
        feature_matrix=feature_matrix.astype(np.float32),
        reference_end_idx=int(reference_end_idx),
        reference_mask=reference_mask,
        stability_mask=stability_mask,
        onset_allowed_mask=onset_allowed_mask,
        detail=detail,
    )
