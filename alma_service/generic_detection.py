from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
try:
    import optuna
except ImportError:  # pragma: no cover - optional fallback for older envs
    optuna = None

from alma_service.anomaly_specs import DetectionSpec, get_detection_spec
from alma_service.benchmark_metrics import (
    evaluate_predictions,
    load_intervals as load_intervals_df,
    predicted_from_mapping,
    select_interval_detection,
    summarize_splits,
)
from alma_service.detection_artifacts import (
    DEFAULT_DETECTOR,
    DETECTOR_KEYS,
    benchmark_summary_path,
    config_path,
    load_json,
    normalize_detector_key,
    predicted_starts_path,
    report_path,
    results_path,
    scores_path,
    summary_path,
    tuning_path,
)
from alma_service.engineered_features import PreparedWellData, prepare_engineered_well
from alma_service.generic_detectors import (
    DetectorScoreOutput,
    SharedPaAnoDetector,
    set_seed,
)
from alma_service.onset_detection import (
    calibrate_causal_thresholds_from_reference_mask,
    detect_causal_onsets_masked,
)
from alma_service.paano_defaults import (
    LONG_PATCH,
    PATCH_SIZES,
    SHORT_PATCH,
    MIN_REFERENCE_COVERAGE,
    MIN_TOTAL_COVERAGE,
    PRESTART_TOLERANCE_HOURS,
    REFERENCE_MAX_RATIO,
    REFERENCE_MIN_DAYS,
    REFERENCE_MIN_RATIO,
)
from alma_service.paths import DB_DIR, ensure_dir, ensure_parent
from alma_service.tabular_io import read_table, write_table

BASE_ONSET_CONFIG = {
    "target_far_per_day": 0.50,
    "min_run_points": 3,
    "cooldown_hours": 12.0,
    "rearm_window_minutes": 60.0,
    "ema_alpha": 0.08,
    "gate_mode": "score_ema",
    "hysteresis_scale": 0.60,
    "bypass_cooldown_after_clear": True,
}

BASE_ONSET_TUNE_GRID = {
    "target_far_per_day": [0.10, 0.25, 0.50],
    "min_run_points": [3, 4, 6],
    "cooldown_hours": [8.0, 12.0, 24.0],
    "rearm_window_minutes": [30.0, 60.0, 120.0],
    "ema_alpha": [0.04, 0.08, 0.12],
    "gate_mode": ["score_ema", "relaxed", "strict"],
}

ANOMALY_ONSET_PROFILES = {
    "negermet": {},
    "pritok": {
        "defaults": {
            "bypass_cooldown_after_clear": False,
        },
        "grid": {
            "min_run_points": [3, 4],
            "cooldown_hours": [12.0, 24.0, 72.0, 120.0, 168.0],
            "rearm_window_minutes": [30.0, 60.0, 120.0, 240.0],
        },
    },
    "salt": {},
}

SALT_SHARED_ONSET_TUNE_GRID = {
    "target_far_per_day": [0.10, 0.25, 0.50],
    "min_run_points": [3, 4, 6],
    "cooldown_hours": [8.0, 12.0, 24.0, 48.0, 72.0],
    "rearm_window_minutes": [120.0, 240.0, 480.0, 720.0],
    "ema_alpha": [0.04, 0.08, 0.12],
    "gate_mode": ["relaxed"],
    "bypass_cooldown_after_clear": [True, False],
}

PRESSURE_TREND_WEIGHT_GRID = [0.0, 0.0025, 0.005]
NEGERMET_SIGNATURE_WEIGHT_GRID = [0.0, 0.0025, 0.005]
PHYSICAL_BRANCH_WEIGHT_GRIDS = {
    "pritok": PRESSURE_TREND_WEIGHT_GRID,
    "negermet": NEGERMET_SIGNATURE_WEIGHT_GRID,
    "salt": [0.0, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02],
}
SALT_TREND_WEIGHT_GRID = PHYSICAL_BRANCH_WEIGHT_GRIDS["salt"]
RETUNE_MODE_FAST = "fast"
RETUNE_MODE_QUALITY = "quality"
RETUNE_MODE_GRID = "grid"
RETUNE_MODES = {RETUNE_MODE_FAST, RETUNE_MODE_QUALITY, RETUNE_MODE_GRID}
ANOMALY_RUNTIME_CONFIG = {
    "negermet": {
        "prepare_patch_size": PATCH_SIZES["negermet"][1],
        "paano_patch_short": PATCH_SIZES["negermet"][0],
        "paano_patch_long": PATCH_SIZES["negermet"][1],
        "max_far_per_day": 0.25,
        "max_starts_per_interval": 2.0,
        "max_p90_delay_ratio": 0.25,
    },
    "pritok": {
        "prepare_patch_size": PATCH_SIZES["pritok"][1],
        "paano_patch_short": PATCH_SIZES["pritok"][0],
        "paano_patch_long": PATCH_SIZES["pritok"][1],
        "max_far_per_day": 0.25,
        "max_starts_per_interval": 7.0,
        "max_p90_delay_ratio": 0.45,
    },
    "salt": {
        "prepare_patch_size": PATCH_SIZES["salt"][1],
        "paano_patch_short": PATCH_SIZES["salt"][0],
        "paano_patch_long": PATCH_SIZES["salt"][1],
        "max_far_per_day": 0.40,
        "max_starts_per_interval": 10.0,
        "max_p90_delay_ratio": 0.20,
    },
}
CUDA_REQUIRED_DETECTORS = {"paano_shared"}


def _resolve_torch_device(detector_key: str, verbose: bool = True) -> torch.device:
    if detector_key in CUDA_REQUIRED_DETECTORS and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA GPU is required for detector '{detector_key}', "
            "but torch.cuda.is_available() is False. "
            "Check the active venv, CUDA drivers, and PyTorch CUDA build."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            memory_gb = props.total_memory / 1024**3
            print(f"  Device: cuda ({props.name}, {memory_gb:.1f} GB)")
        else:
            print("  Device: cpu")
    return device


@dataclass
class PreparedDetectorRun:
    prepared: PreparedWellData
    score_output: DetectorScoreOutput


def _runtime_config(anomaly_key: str) -> dict[str, float | int]:
    return ANOMALY_RUNTIME_CONFIG.get(anomaly_key, ANOMALY_RUNTIME_CONFIG["salt"])


def _default_onset_config(anomaly_key: str, detector_key: str) -> dict[str, Any]:
    cfg = BASE_ONSET_CONFIG.copy()
    profile = ANOMALY_ONSET_PROFILES.get(anomaly_key, {})
    cfg.update(profile.get("defaults", {}))
    if detector_key == "paano_shared":
        cfg["fusion_weight_short"] = 0.60
    return cfg


def _onset_tune_grid(anomaly_key: str) -> dict[str, list[Any]]:
    grid = {key: list(values) for key, values in BASE_ONSET_TUNE_GRID.items()}
    profile = ANOMALY_ONSET_PROFILES.get(anomaly_key, {})
    for key, values in profile.get("grid", {}).items():
        grid[key] = list(values)
    return grid


def _safe_metric(value: Any, large: float = 1e9) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return large
    if not np.isfinite(numeric):
        return large
    return numeric


def _operational_score_key(anomaly_key: str, detector_key: str, summary: dict[str, Any]) -> tuple[float, ...]:
    runtime_cfg = _runtime_config(anomaly_key)
    max_far = float(runtime_cfg["max_far_per_day"])
    max_starts = float(runtime_cfg["max_starts_per_interval"])
    max_delay_ratio = float(runtime_cfg["max_p90_delay_ratio"])
    far = _safe_metric(summary.get("false_alarms_per_day"), large=1e9)
    starts = _safe_metric(summary.get("avg_starts_per_interval"), large=1e9)
    p90_ratio = _safe_metric(summary.get("p90_delay_ratio"), large=1e9)
    p90_abs_delay = _safe_metric(summary.get("p90_abs_delay_hours"), large=1e9)
    median_abs_delay = _safe_metric(summary.get("median_abs_delay_hours"), large=1e9)
    hit_count = float(summary.get("hit_count", 0))
    feasible_far = int(far <= max_far)
    feasible_starts = int(starts <= max_starts)
    feasible_delay = int(p90_ratio <= max_delay_ratio)
    far_over = max(far - max_far, 0.0)
    starts_over = max(starts - max_starts, 0.0)
    delay_over = max(p90_ratio - max_delay_ratio, 0.0)
    priority = 1 if detector_key == DEFAULT_DETECTOR else 0
    if anomaly_key == "salt":
        return (
            hit_count,
            float(feasible_far),
            float(feasible_delay),
            -delay_over,
            -p90_ratio,
            -p90_abs_delay,
            -median_abs_delay,
            float(feasible_starts),
            -starts_over,
            -starts,
            -far,
            float(priority),
        )
    if anomaly_key == "negermet":
        return (
            hit_count,
            float(feasible_starts),
            float(feasible_far),
            float(feasible_delay),
            -starts_over,
            -far_over,
            -delay_over,
            -p90_ratio,
            -p90_abs_delay,
            -median_abs_delay,
            -starts,
            -far,
            float(priority),
        )
    return (
        float(feasible_starts),
        float(feasible_far),
        float(feasible_delay),
        -starts_over,
        -far_over,
        -delay_over,
        hit_count,
        -p90_ratio,
        -p90_abs_delay,
        -median_abs_delay,
        -starts,
        -far,
        float(priority),
    )


def load_anomaly_data(spec: DetectionSpec, source_path: str | None = None) -> pd.DataFrame:
    if source_path is not None:
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")
    else:
        candidates = [DB_DIR / name for name in spec.dataset.source_candidates]
        src = next((path for path in candidates if path.exists()), None)
        if src is None:
            raise FileNotFoundError(f"No source dataset found for {spec.anomaly_key}")

    print(f"Loading {spec.anomaly_key} data from: {src}")
    df = read_table(
        src,
        dtypes={"well_id": str},
        parse_dates=["timestamp"],
        low_memory=False,
    )
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    return df.dropna(subset=["timestamp"]).sort_values(["well_id", "timestamp"]).reset_index(drop=True)


def load_intervals(spec: DetectionSpec, required: bool = True) -> pd.DataFrame:
    src = spec.dataset.intervals_path
    if not src.exists():
        if required:
            raise FileNotFoundError(f"Intervals file not found: {src}")
        return pd.DataFrame(
            columns=["well_id", "start_date", "end_date", "data_start", "data_end", "split", "interval_idx"]
        )
    return load_intervals_df(src)


def _prepare_all_wells(
    spec: DetectionSpec,
    df: pd.DataFrame,
    intervals: pd.DataFrame,
    verbose: bool,
    zone_aware: bool = False,
) -> dict[str, PreparedWellData]:
    from alma_service.dataset_config import split_for_well

    runtime_cfg = _runtime_config(spec.anomaly_key)
    split_map = (
        intervals[["well_id", "split"]]
        .drop_duplicates("well_id")
        .set_index("well_id")["split"]
        .to_dict()
    )
    prepared_runs: dict[str, PreparedWellData] = {}
    for well_id in sorted(df["well_id"].unique()):
        well_df = df[df["well_id"] == well_id].copy()

        # Truncate at end of first anomaly interval
        well_intervals = intervals[intervals["well_id"] == well_id].sort_values("start_date")
        if not well_intervals.empty:
            first_end = pd.Timestamp(well_intervals.iloc[0]["end_date"])
            if pd.notna(first_end):
                well_df = well_df[well_df["timestamp"] <= first_end]
                if verbose:
                    print(f"  Truncated {well_id} at first anomaly end: {first_end}")
        prepared = prepare_engineered_well(
            anomaly_key=spec.anomaly_key,
            well_id=well_id,
            split=split_map.get(well_id, split_for_well(spec.anomaly_key, well_id)),
            well_df=well_df,
            patch_size=int(runtime_cfg["prepare_patch_size"]),
            reference_min_ratio=REFERENCE_MIN_RATIO,
            reference_max_ratio=REFERENCE_MAX_RATIO,
            reference_min_days=REFERENCE_MIN_DAYS,
            min_reference_coverage=MIN_REFERENCE_COVERAGE,
            min_total_coverage=MIN_TOTAL_COVERAGE,
            anomaly_intervals=(
                intervals[intervals["well_id"] == well_id] if zone_aware else None
            ),
        )
        if prepared is None:
            if verbose:
                print(f"  Skip {well_id}: not enough usable data after preprocessing")
            continue
        prepared_runs[well_id] = prepared
        if verbose:
            print(
                f"  Prepared {well_id}: split={prepared.split}, points={prepared.detail['points']}, "
                f"raw={prepared.detail['raw_channels']}, features={prepared.detail['feature_count']}, "
                f"ref={prepared.detail['reference_points']}, masked={prepared.detail['masked_fraction']:.1%}"
            )
    return prepared_runs


def _merge_physical_output(
    base: DetectorScoreOutput,
    fused: np.ndarray,
    components: dict[str, np.ndarray],
    detail: dict[str, Any],
) -> DetectorScoreOutput:
    return DetectorScoreOutput(
        primary=np.asarray(fused, dtype=np.float32),
        components={**base.components, **components},
        detail={**base.detail, **detail},
    )


def _build_local_runs(
    anomaly_key: str,
    detector_key: str,
    prepared_runs: dict[str, PreparedWellData],
    device: torch.device,
    verbose: bool,
    shared_state: Any = None,
) -> dict[str, PreparedDetectorRun]:
    if detector_key != "paano_shared":
        raise ValueError(f"Unsupported production detector: {detector_key}")
    if shared_state is None:
        raise ValueError("paano_shared requires a trained or loaded shared encoder state.")

    from alma_service.shared_encoder import select_shared_columns

    out: dict[str, PreparedDetectorRun] = {}
    for well_id, prepared in prepared_runs.items():
        X_proj = select_shared_columns(
            prepared.feature_columns,
            prepared.feature_matrix,
            shared_state.shared_channels,
        )
        detector = SharedPaAnoDetector(
            shared_state=shared_state,
            device=device,
            verbose=verbose,
        )
        detector.fit_reference(X_proj[prepared.reference_mask], mask_ref=prepared.reference_mask)
        score_output = detector.score_stream(X_proj, mask_all=prepared.stability_mask)
        if anomaly_key == "pritok":
            from alma_service.pressure_trend import (
                build_pressure_trend_branch,
                fuse_model_with_pressure_trend,
            )

            pressure_output = build_pressure_trend_branch(prepared)
            fused, fusion_components, fusion_detail = fuse_model_with_pressure_trend(
                model_score=score_output.primary,
                pressure_output=pressure_output,
                reference_mask=prepared.reference_mask,
            )
            score_output = _merge_physical_output(score_output, fused, fusion_components, fusion_detail)
        elif anomaly_key == "negermet":
            from alma_service.negermet_signature import (
                build_negermet_signature_branch,
                fuse_model_with_negermet_signature,
            )

            signature_output = build_negermet_signature_branch(prepared)
            fused, fusion_components, fusion_detail = fuse_model_with_negermet_signature(
                model_score=score_output.primary,
                signature_output=signature_output,
                reference_mask=prepared.reference_mask,
            )
            score_output = _merge_physical_output(score_output, fused, fusion_components, fusion_detail)
        elif anomaly_key == "salt":
            from alma_service.salt_trend import (
                build_salt_deposition_branch,
                fuse_model_with_salt_trend,
            )

            salt_output = build_salt_deposition_branch(prepared)
            fused, fusion_components, fusion_detail = fuse_model_with_salt_trend(
                model_score=score_output.primary,
                salt_output=salt_output,
                reference_mask=prepared.reference_mask,
            )
            score_output = _merge_physical_output(score_output, fused, fusion_components, fusion_detail)
        out[well_id] = PreparedDetectorRun(prepared=prepared, score_output=score_output)
    return out


def _score_for_config(run: PreparedDetectorRun, detector_key: str, cfg: dict[str, Any]) -> np.ndarray:
    if detector_key != "paano_shared":
        raise ValueError(f"Unsupported production detector: {detector_key}")

    model_score = run.score_output.components.get("paano_score")
    if model_score is None:
        return run.score_output.primary.astype(np.float32)

    physical_components = (
        ("pressure_trend_weight", "pressure_trend_score"),
        ("negermet_signature_weight", "negermet_signature_score"),
        ("salt_trend_weight", "salt_deposition_calibrated_fusion_score"),
    )
    score = np.asarray(model_score, dtype=np.float32).copy()
    for weight_key, component_key in physical_components:
        component = run.score_output.components.get(component_key)
        weight = float(cfg.get(weight_key, 0.0))
        if component is not None and weight > 0.0:
            score = score + weight * np.asarray(component, dtype=np.float32)
    return score.astype(np.float32)


def _detect_starts_for_run(
    detector_key: str,
    run: PreparedDetectorRun,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, Any, list[pd.Timestamp]]:
    cfg = {**BASE_ONSET_CONFIG, **cfg}
    score = _score_for_config(run, detector_key, cfg)
    thresholds, diagnostics = calibrate_causal_thresholds_from_reference_mask(
        scores=score,
        timestamps=run.prepared.timestamps,
        reference_mask=run.prepared.reference_mask,
        target_far_per_day=float(cfg["target_far_per_day"]),
        min_run_points=int(cfg["min_run_points"]),
        ema_alpha=float(cfg["ema_alpha"]),
    )
    starts = detect_causal_onsets_masked(
        scores=score,
        timestamps=run.prepared.timestamps,
        diagnostics=diagnostics,
        thresholds=thresholds,
        reference_mask=run.prepared.reference_mask,
        onset_mask=run.prepared.onset_allowed_mask,
        min_run_points=int(cfg["min_run_points"]),
        cooldown_hours=float(cfg["cooldown_hours"]),
        rearm_window_minutes=float(cfg["rearm_window_minutes"]),
        gate_mode=str(cfg["gate_mode"]),
        hysteresis_scale=float(cfg["hysteresis_scale"]),
        bypass_cooldown_after_clear=bool(cfg.get("bypass_cooldown_after_clear", True)),
    )
    return score, thresholds, starts


def _candidate_configs(anomaly_key: str, detector_key: str) -> list[dict[str, Any]]:
    if detector_key != "paano_shared":
        raise ValueError(f"Unsupported production detector: {detector_key}")

    candidates: list[dict[str, Any]] = []
    pressure_weight_grid = (
        PRESSURE_TREND_WEIGHT_GRID
        if anomaly_key == "pritok"
        else [None]
    )
    negermet_signature_weight_grid = (
        NEGERMET_SIGNATURE_WEIGHT_GRID
        if anomaly_key == "negermet"
        else [None]
    )
    salt_trend_weight_grid = (
        SALT_TREND_WEIGHT_GRID
        if anomaly_key == "salt"
        else [None]
    )
    grid = (
        {key: list(values) for key, values in SALT_SHARED_ONSET_TUNE_GRID.items()}
        if anomaly_key == "salt"
        else _onset_tune_grid(anomaly_key)
    )
    default_cfg = _default_onset_config(anomaly_key, detector_key)
    bypass_grid = grid.get(
        "bypass_cooldown_after_clear",
        [bool(default_cfg.get("bypass_cooldown_after_clear", True))],
    )
    for target_far_per_day in grid["target_far_per_day"]:
        for min_run_points in grid["min_run_points"]:
            for cooldown_hours in grid["cooldown_hours"]:
                for rearm_window_minutes in grid["rearm_window_minutes"]:
                    for ema_alpha in grid["ema_alpha"]:
                        for gate_mode in grid["gate_mode"]:
                            for bypass_cooldown_after_clear in bypass_grid:
                                for pressure_trend_weight in pressure_weight_grid:
                                    for negermet_signature_weight in negermet_signature_weight_grid:
                                        for salt_trend_weight in salt_trend_weight_grid:
                                            cfg = default_cfg.copy()
                                            cfg.update(
                                                {
                                                    "target_far_per_day": float(target_far_per_day),
                                                    "min_run_points": int(min_run_points),
                                                    "cooldown_hours": float(cooldown_hours),
                                                    "rearm_window_minutes": float(rearm_window_minutes),
                                                    "ema_alpha": float(ema_alpha),
                                                    "gate_mode": str(gate_mode),
                                                    "bypass_cooldown_after_clear": bool(
                                                        bypass_cooldown_after_clear
                                                    ),
                                                }
                                            )
                                            if pressure_trend_weight is not None:
                                                cfg["pressure_trend_weight"] = float(pressure_trend_weight)
                                            if negermet_signature_weight is not None:
                                                cfg["negermet_signature_weight"] = float(
                                                    negermet_signature_weight
                                                )
                                            if salt_trend_weight is not None:
                                                cfg["salt_trend_weight"] = float(salt_trend_weight)
                                            candidates.append(cfg)
    return candidates


def _optuna_objective_value(anomaly_key: str, detector_key: str, summary: dict[str, Any]) -> float:
    runtime_cfg = _runtime_config(anomaly_key)
    max_far = float(runtime_cfg["max_far_per_day"])
    max_starts = float(runtime_cfg["max_starts_per_interval"])
    max_delay_ratio = float(runtime_cfg["max_p90_delay_ratio"])
    far = _safe_metric(summary.get("false_alarms_per_day"), large=1e6)
    starts = _safe_metric(summary.get("avg_starts_per_interval"), large=1e6)
    p90_ratio = _safe_metric(summary.get("p90_delay_ratio"), large=1e6)
    p90_abs_delay = _safe_metric(summary.get("p90_abs_delay_hours"), large=1e6)
    hits = float(summary.get("hit_count", 0))
    feasible_far = 1.0 if far <= max_far else 0.0
    feasible_starts = 1.0 if starts <= max_starts else 0.0
    feasible_delay = 1.0 if p90_ratio <= max_delay_ratio else 0.0
    far_over = max(far - max_far, 0.0)
    starts_over = max(starts - max_starts, 0.0)
    delay_over = max(p90_ratio - max_delay_ratio, 0.0)
    return (
        hits * 1_000_000_000.0
        + feasible_delay * 10_000_000.0
        + feasible_starts * 1_000_000.0
        + feasible_far * 100_000.0
        - delay_over * 1_000_000.0
        - starts_over * 100_000.0
        - far_over * 10_000.0
        - p90_ratio * 10_000.0
        - starts * 100.0
        - p90_abs_delay
        - far * 100.0
    )


def _normalize_well_id(value: Any) -> str:
    return str(value).strip().lower()


def _build_per_well_tuning_intervals(train_intervals: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if train_intervals.empty:
        return {}
    normalized = train_intervals["well_id"].map(_normalize_well_id)
    groups: dict[str, pd.DataFrame] = {}
    for well_id in sorted(normalized.unique()):
        groups[str(well_id)] = train_intervals.loc[normalized == well_id].copy()
    return groups


def _config_cache_key(cfg: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, (np.floating, float)):
            return round(float(value), 10)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        return value

    return tuple(sorted((str(key), _normalize_value(value)) for key, value in cfg.items()))


def _per_well_tuning_summaries_from_mapping(
    per_well_intervals: dict[str, pd.DataFrame],
    predicted: dict[str, list[pd.Timestamp]],
) -> dict[str, dict[str, Any]]:
    predictions_by_well = {_normalize_well_id(well_id): starts for well_id, starts in predicted.items()}
    per_well: dict[str, dict[str, Any]] = {}
    for well_id, well_intervals in per_well_intervals.items():
        starts = predictions_by_well.get(well_id, [])
        well_predictions = predicted_from_mapping({well_id: starts})
        per_well[well_id], _ = evaluate_predictions(
            well_intervals,
            well_predictions,
            scores=None,
            prestart_hours=PRESTART_TOLERANCE_HOURS,
        )
    return per_well


def _per_well_tuning_summaries(
    train_intervals: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    per_well_intervals = _build_per_well_tuning_intervals(train_intervals)
    pred_groups: dict[str, list[pd.Timestamp]] = {}
    if not pred_df.empty:
        for well_id, group in pred_df.groupby(pred_df["well_id"].map(_normalize_well_id), sort=True):
            pred_groups[str(well_id)] = [pd.Timestamp(ts) for ts in group["detected_time"].tolist()]
    return _per_well_tuning_summaries_from_mapping(per_well_intervals, pred_groups)


def _detect_starts_for_config(
    detector_key: str,
    train_runs: dict[str, PreparedDetectorRun],
    cfg: dict[str, Any],
    starts_cache: dict[tuple[str, tuple[tuple[str, Any], ...]], list[pd.Timestamp]] | None = None,
) -> dict[str, list[pd.Timestamp]]:
    cfg_key = _config_cache_key({**BASE_ONSET_CONFIG, **cfg})
    predicted: dict[str, list[pd.Timestamp]] = {}
    for well_id, run in train_runs.items():
        cache_key = (str(well_id), cfg_key)
        if starts_cache is not None and cache_key in starts_cache:
            predicted[well_id] = starts_cache[cache_key]
            continue
        _, _, starts = _detect_starts_for_run(detector_key, run, cfg)
        predicted[well_id] = starts
        if starts_cache is not None:
            starts_cache[cache_key] = starts
    return predicted


def _retune_mode() -> str:
    mode = os.environ.get("ALMA_RETUNE_MODE", RETUNE_MODE_FAST).strip().lower()
    if mode not in RETUNE_MODES:
        return RETUNE_MODE_FAST
    return mode


def _optuna_seed() -> int:
    raw_value = os.environ.get("ALMA_OPTUNA_SEED", "2027").strip()
    try:
        return int(raw_value)
    except ValueError:
        return 2027


def _optuna_trial_count(default_trials: int, mode: str) -> int:
    env_name = "ALMA_OPTUNA_QUALITY_TRIALS" if mode == RETUNE_MODE_QUALITY else "ALMA_OPTUNA_N_TRIALS"
    raw_value = os.environ.get(env_name, "").strip()
    if not raw_value:
        return default_trials
    try:
        n_trials = int(raw_value)
    except ValueError:
        return default_trials
    return max(n_trials, 1)


def _tuning_n_jobs(anomaly_key: str, detector_key: str, mode: str) -> int:
    if mode == RETUNE_MODE_QUALITY:
        return 1
    raw_value = os.environ.get("ALMA_OPTUNA_N_JOBS", "1").strip()
    try:
        n_jobs = int(raw_value)
    except ValueError:
        n_jobs = 1
    if n_jobs < 1:
        return 1
    if anomaly_key == "salt":
        return min(n_jobs, 16)
    return min(n_jobs, 8)


def _well_balance_stats(per_well_summaries: dict[str, dict[str, Any]]) -> dict[str, float]:
    if not per_well_summaries:
        return {
            "well_count": 0.0,
            "full_hit_well_count": 0.0,
            "full_hit_well_rate": 0.0,
            "worst_hit_rate": 0.0,
            "median_hit_rate": 0.0,
            "worst_p90_delay_ratio": 1e9,
            "median_p90_delay_ratio": 1e9,
            "worst_false_alarms_per_day": 1e9,
            "worst_avg_starts_per_interval": 1e9,
        }

    hit_rates: list[float] = []
    p90_ratios: list[float] = []
    fars: list[float] = []
    starts: list[float] = []
    full_hit_count = 0
    for summary in per_well_summaries.values():
        interval_count = int(summary.get("interval_count", 0))
        hit_count = int(summary.get("hit_count", 0))
        if interval_count > 0 and hit_count >= interval_count:
            full_hit_count += 1
        hit_rates.append(float(hit_count / interval_count) if interval_count else 0.0)
        p90_ratios.append(_safe_metric(summary.get("p90_delay_ratio"), large=1e9))
        fars.append(_safe_metric(summary.get("false_alarms_per_day"), large=1e9))
        starts.append(_safe_metric(summary.get("avg_starts_per_interval"), large=1e9))

    well_count = len(per_well_summaries)
    return {
        "well_count": float(well_count),
        "full_hit_well_count": float(full_hit_count),
        "full_hit_well_rate": float(full_hit_count / well_count) if well_count else 0.0,
        "worst_hit_rate": float(min(hit_rates)) if hit_rates else 0.0,
        "median_hit_rate": float(np.median(hit_rates)) if hit_rates else 0.0,
        "worst_p90_delay_ratio": float(max(p90_ratios)) if p90_ratios else 1e9,
        "median_p90_delay_ratio": float(np.median(p90_ratios)) if p90_ratios else 1e9,
        "worst_false_alarms_per_day": float(max(fars)) if fars else 1e9,
        "worst_avg_starts_per_interval": float(max(starts)) if starts else 1e9,
    }


def _robust_tuning_score_key(
    anomaly_key: str,
    detector_key: str,
    summary: dict[str, Any],
    per_well_summaries: dict[str, dict[str, Any]],
) -> tuple[float, ...]:
    base_key = _operational_score_key(anomaly_key, detector_key, summary)
    stats = _well_balance_stats(per_well_summaries)
    if stats["well_count"] <= 1:
        return base_key
    return (
        base_key[0],
        stats["full_hit_well_count"],
        stats["full_hit_well_rate"],
        stats["worst_hit_rate"],
        stats["median_hit_rate"],
        -stats["worst_p90_delay_ratio"],
        -stats["median_p90_delay_ratio"],
        -stats["worst_false_alarms_per_day"],
        -stats["worst_avg_starts_per_interval"],
        *base_key[1:],
    )


def _robust_optuna_objective_value(
    anomaly_key: str,
    detector_key: str,
    summary: dict[str, Any],
    per_well_summaries: dict[str, dict[str, Any]],
) -> float:
    value = _optuna_objective_value(anomaly_key, detector_key, summary)
    stats = _well_balance_stats(per_well_summaries)
    if stats["well_count"] <= 1:
        return value
    return (
        value
        + stats["full_hit_well_count"] * 10_000_000.0
        + stats["worst_hit_rate"] * 1_000_000.0
        + stats["median_hit_rate"] * 100_000.0
        - stats["worst_p90_delay_ratio"] * 10_000.0
        - stats["worst_avg_starts_per_interval"] * 100.0
        - stats["worst_false_alarms_per_day"] * 100.0
    )


def _suggest_optuna_config(trial: Any, anomaly_key: str, detector_key: str) -> dict[str, Any]:
    if detector_key != "paano_shared":
        raise ValueError(f"Unsupported production detector: {detector_key}")

    grid = (
        SALT_SHARED_ONSET_TUNE_GRID
        if anomaly_key == "salt"
        else _onset_tune_grid(anomaly_key)
    )
    cfg = _default_onset_config(anomaly_key, detector_key)
    cfg.update(
        {
            "target_far_per_day": float(
                trial.suggest_categorical("target_far_per_day", grid["target_far_per_day"])
            ),
            "min_run_points": int(
                trial.suggest_categorical("min_run_points", grid["min_run_points"])
            ),
            "cooldown_hours": float(
                trial.suggest_categorical("cooldown_hours", grid["cooldown_hours"])
            ),
            "rearm_window_minutes": float(
                trial.suggest_categorical("rearm_window_minutes", grid["rearm_window_minutes"])
            ),
            "ema_alpha": float(trial.suggest_categorical("ema_alpha", grid["ema_alpha"])),
            "gate_mode": str(trial.suggest_categorical("gate_mode", grid["gate_mode"])),
        }
    )
    if "bypass_cooldown_after_clear" in grid:
        cfg["bypass_cooldown_after_clear"] = bool(
            trial.suggest_categorical(
                "bypass_cooldown_after_clear",
                grid["bypass_cooldown_after_clear"],
            )
        )
    if anomaly_key == "pritok":
        cfg["pressure_trend_weight"] = float(
            trial.suggest_categorical("pressure_trend_weight", PRESSURE_TREND_WEIGHT_GRID)
        )
    if anomaly_key == "negermet":
        cfg["negermet_signature_weight"] = float(
            trial.suggest_categorical(
                "negermet_signature_weight",
                NEGERMET_SIGNATURE_WEIGHT_GRID,
            )
        )
    if anomaly_key == "salt":
        cfg["salt_trend_weight"] = float(
            trial.suggest_categorical("salt_trend_weight", SALT_TREND_WEIGHT_GRID)
        )
    return cfg


def _tune_config_with_optuna(
    anomaly_key: str,
    detector_key: str,
    train_runs: dict[str, PreparedDetectorRun],
    train_intervals: pd.DataFrame,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if optuna is None:  # pragma: no cover - runtime fallback
        return _tune_config_with_grid(anomaly_key, detector_key, train_runs, train_intervals, verbose)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mode = _retune_mode()
    seed = _optuna_seed()
    sampler_kwargs: dict[str, Any] = {"seed": seed}
    if mode == RETUNE_MODE_QUALITY:
        sampler_kwargs.update({"multivariate": True, "group": True, "n_startup_trials": 0})
    sampler = optuna.samplers.TPESampler(**sampler_kwargs)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    per_well_intervals = _build_per_well_tuning_intervals(train_intervals)
    starts_cache: dict[tuple[str, tuple[tuple[str, Any], ...]], list[pd.Timestamp]] = {}
    objective_calls = 0
    n_trials = 36
    if anomaly_key == "pritok":
        n_trials = 72
        for seed_cfg, seeded_safe in (
            ({
                "target_far_per_day": 0.25,
                "min_run_points": 3,
                "cooldown_hours": 72.0,
                "rearm_window_minutes": 60.0,
                "ema_alpha": 0.12,
                "gate_mode": "strict",
                "pressure_trend_weight": 0.0025,
            }, True),
            ({
                "target_far_per_day": 0.50,
                "min_run_points": 3,
                "cooldown_hours": 72.0,
                "rearm_window_minutes": 60.0,
                "ema_alpha": 0.12,
                "gate_mode": "strict",
                "pressure_trend_weight": 0.0025,
            }, True),
            ({
                "target_far_per_day": 0.50,
                "min_run_points": 3,
                "cooldown_hours": 72.0,
                "rearm_window_minutes": 240.0,
                "ema_alpha": 0.12,
                "gate_mode": "strict",
                "pressure_trend_weight": 0.0025,
            }, True),
            ({
                "target_far_per_day": 0.25,
                "min_run_points": 4,
                "cooldown_hours": 72.0,
                "rearm_window_minutes": 30.0,
                "ema_alpha": 0.12,
                "gate_mode": "relaxed",
                "pressure_trend_weight": 0.0025,
            }, True),
            ({
                "target_far_per_day": 0.25,
                "min_run_points": 4,
                "cooldown_hours": 72.0,
                "rearm_window_minutes": 30.0,
                "ema_alpha": 0.12,
                "gate_mode": "score_ema",
                "pressure_trend_weight": 0.0025,
            }, True),
            ({
                "target_far_per_day": 0.25,
                "min_run_points": 4,
                "cooldown_hours": 120.0,
                "rearm_window_minutes": 30.0,
                "ema_alpha": 0.12,
                "gate_mode": "relaxed",
                "pressure_trend_weight": 0.0025,
            }, True),
            ({
                "target_far_per_day": 0.25,
                "min_run_points": 4,
                "cooldown_hours": 168.0,
                "rearm_window_minutes": 30.0,
                "ema_alpha": 0.12,
                "gate_mode": "score_ema",
                "pressure_trend_weight": 0.0025,
            }, True),
            ({
                "target_far_per_day": 0.10,
                "min_run_points": 4,
                "cooldown_hours": 120.0,
                "rearm_window_minutes": 60.0,
                "ema_alpha": 0.08,
                "gate_mode": "relaxed",
                "pressure_trend_weight": 0.0025,
            }, False),
            ({
                "target_far_per_day": 0.10,
                "min_run_points": 4,
                "cooldown_hours": 168.0,
                "rearm_window_minutes": 60.0,
                "ema_alpha": 0.08,
                "gate_mode": "score_ema",
                "pressure_trend_weight": 0.0025,
            }, False),
        ):
            study.enqueue_trial(seed_cfg, user_attrs={"seeded_safe": seeded_safe})
        if mode == RETUNE_MODE_QUALITY:
            n_trials = max(n_trials, 144)
    elif anomaly_key == "negermet":
        n_trials = 48
        for seed_cfg in (
            {
                "target_far_per_day": 0.25,
                "min_run_points": 3,
                "cooldown_hours": 12.0,
                "rearm_window_minutes": 30.0,
                "ema_alpha": 0.08,
                "gate_mode": "score_ema",
                "negermet_signature_weight": 0.0025,
            },
            {
                "target_far_per_day": 0.25,
                "min_run_points": 3,
                "cooldown_hours": 24.0,
                "rearm_window_minutes": 30.0,
                "ema_alpha": 0.12,
                "gate_mode": "strict",
                "negermet_signature_weight": 0.0025,
            },
            {
                "target_far_per_day": 0.10,
                "min_run_points": 4,
                "cooldown_hours": 24.0,
                "rearm_window_minutes": 60.0,
                "ema_alpha": 0.08,
                "gate_mode": "relaxed",
                "negermet_signature_weight": 0.005,
            },
        ):
            study.enqueue_trial(seed_cfg)
    elif anomaly_key == "salt":
        n_trials = 96
        for seed_cfg, seeded_safe in (
            ({
                "target_far_per_day": 0.10,
                "min_run_points": 4,
                "cooldown_hours": 72.0,
                "rearm_window_minutes": 720.0,
                "ema_alpha": 0.04,
                "gate_mode": "relaxed",
                "bypass_cooldown_after_clear": False,
                "salt_trend_weight": 0.01,
            }, True),
            ({
                "target_far_per_day": 0.50,
                "min_run_points": 4,
                "cooldown_hours": 24.0,
                "rearm_window_minutes": 720.0,
                "ema_alpha": 0.04,
                "gate_mode": "relaxed",
                "bypass_cooldown_after_clear": False,
                "salt_trend_weight": 0.02,
            }, True),
            ({
                "target_far_per_day": 0.50,
                "min_run_points": 4,
                "cooldown_hours": 72.0,
                "rearm_window_minutes": 720.0,
                "ema_alpha": 0.08,
                "gate_mode": "relaxed",
                "bypass_cooldown_after_clear": False,
                "salt_trend_weight": 0.02,
            }, True),
            ({
                "target_far_per_day": 0.25,
                "min_run_points": 4,
                "cooldown_hours": 8.0,
                "rearm_window_minutes": 480.0,
                "ema_alpha": 0.04,
                "gate_mode": "relaxed",
                "bypass_cooldown_after_clear": True,
                "salt_trend_weight": 0.01,
            }, False),
            ({
                "target_far_per_day": 0.25,
                "min_run_points": 4,
                "cooldown_hours": 8.0,
                "rearm_window_minutes": 720.0,
                "ema_alpha": 0.04,
                "gate_mode": "relaxed",
                "bypass_cooldown_after_clear": True,
                "salt_trend_weight": 0.01,
            }, False),
            ({
                "target_far_per_day": 0.25,
                "min_run_points": 4,
                "cooldown_hours": 24.0,
                "rearm_window_minutes": 240.0,
                "ema_alpha": 0.04,
                "gate_mode": "relaxed",
                "bypass_cooldown_after_clear": False,
                "salt_trend_weight": 0.01,
            }, False),
            ({
                "target_far_per_day": 0.10,
                "min_run_points": 4,
                "cooldown_hours": 48.0,
                "rearm_window_minutes": 240.0,
                "ema_alpha": 0.04,
                "gate_mode": "relaxed",
                "bypass_cooldown_after_clear": False,
                "salt_trend_weight": 0.01,
            }, False),
            ({
                "target_far_per_day": 0.10,
                "min_run_points": 4,
                "cooldown_hours": 72.0,
                "rearm_window_minutes": 720.0,
                "ema_alpha": 0.04,
                "gate_mode": "relaxed",
                "bypass_cooldown_after_clear": False,
                "salt_trend_weight": 0.01,
            }, True),
        ):
            study.enqueue_trial(seed_cfg, user_attrs={"seeded_safe": seeded_safe})
        if mode == RETUNE_MODE_QUALITY:
            n_trials = max(n_trials, 192)

    n_trials = _optuna_trial_count(n_trials, mode)

    def objective(trial: Any) -> float:
        nonlocal objective_calls
        objective_calls += 1
        cfg = _suggest_optuna_config(trial, anomaly_key, detector_key)
        predicted = _detect_starts_for_config(detector_key, train_runs, cfg, starts_cache)
        pred_df = predicted_from_mapping(predicted)
        summary, _ = evaluate_predictions(
            train_intervals,
            pred_df,
            scores=None,
            prestart_hours=PRESTART_TOLERANCE_HOURS,
        )
        per_well_summaries = _per_well_tuning_summaries_from_mapping(per_well_intervals, predicted)
        score_key = _robust_tuning_score_key(
            anomaly_key,
            detector_key,
            summary,
            per_well_summaries,
        )
        well_balance = _well_balance_stats(per_well_summaries)
        trial.set_user_attr("config", cfg)
        trial.set_user_attr("summary", summary)
        trial.set_user_attr("per_well_summaries", per_well_summaries)
        trial.set_user_attr("well_balance", well_balance)
        trial.set_user_attr("score_key", list(score_key))
        return _robust_optuna_objective_value(
            anomaly_key,
            detector_key,
            summary,
            per_well_summaries,
        )

    n_jobs = _tuning_n_jobs(anomaly_key, detector_key, mode)
    backend = f"optuna_tpe_{mode}"
    if verbose:
        print(f"  Auto-tune backend: {backend}, trials={n_trials}, n_jobs={n_jobs}, seed={seed}")
    tuning_start = time.monotonic()
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)
    tuning_seconds = time.monotonic() - tuning_start

    leaderboard = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        cfg = trial.user_attrs.get("config")
        summary = trial.user_attrs.get("summary")
        per_well_summaries = trial.user_attrs.get("per_well_summaries")
        well_balance = trial.user_attrs.get("well_balance")
        score_key = trial.user_attrs.get("score_key")
        if not cfg or not summary or score_key is None:
            continue
        leaderboard.append(
            {
                "score_key": list(score_key),
                "objective": float(trial.value) if trial.value is not None else None,
                "config": cfg,
                "summary": summary,
                "per_well_summaries": per_well_summaries or {},
                "well_balance": well_balance or {},
                "seeded_safe": bool(trial.user_attrs.get("seeded_safe", False)),
            }
        )

    leaderboard.sort(key=lambda row: tuple(row["score_key"]), reverse=True)
    if not leaderboard:
        return _tune_config_with_grid(anomaly_key, detector_key, train_runs, train_intervals, verbose)

    selected = leaderboard[0]
    selected_reason = "best_score_key"
    if mode == RETUNE_MODE_QUALITY and anomaly_key == "pritok":
        train_interval_count = max(int(train_intervals.shape[0]), 0)
        min_hit_count = max(train_interval_count - 2, 0)
        seeded_safe_rows = [
            row
            for row in leaderboard
            if row.get("seeded_safe")
            and int(row["summary"].get("hit_count", 0)) >= min_hit_count
            and float(row["summary"].get("false_alarms_per_day", float("inf"))) <= 0.09
            and float(row["summary"].get("avg_starts_per_interval", float("inf"))) <= 6.6
            and float(row["summary"].get("p90_delay_ratio", float("inf"))) <= 0.20
        ]
        if seeded_safe_rows:
            selected = sorted(
                seeded_safe_rows,
                key=lambda row: (
                    int(row["summary"].get("hit_count", 0)),
                    -float(row["summary"].get("avg_starts_per_interval", float("inf"))),
                    -float(row["summary"].get("false_alarms_per_day", float("inf"))),
                    -float(row["summary"].get("p90_delay_ratio", float("inf"))),
                ),
                reverse=True,
            )[0]
            selected_reason = "pritok_seeded_quality_guard"
    elif mode == RETUNE_MODE_QUALITY and anomaly_key == "salt":
        seeded_safe_rows = [
            row
            for row in leaderboard
            if row.get("seeded_safe")
            and row["summary"].get("hit_count", 0) == row["summary"].get("interval_count", -1)
            and float(row["summary"].get("false_alarms_per_day", float("inf"))) <= 0.02
            and float(row["summary"].get("p90_delay_ratio", float("inf"))) <= 0.11
        ]
        if seeded_safe_rows:
            selected = sorted(seeded_safe_rows, key=lambda row: tuple(row["score_key"]), reverse=True)[0]
            selected_reason = "seeded_safe_quality_guard"

    best_cfg = dict(selected["config"])
    best_key = tuple(selected["score_key"])
    tuning_summary = {
        "backend": backend,
        "mode": mode,
        "n_trials": n_trials,
        "n_jobs": n_jobs,
        "sampler_seed": seed,
        "sampler_kwargs": sampler_kwargs,
        "elapsed_seconds": float(tuning_seconds),
        "objective_calls": int(objective_calls),
        "starts_cache_entries": int(len(starts_cache)),
        "selected_reason": selected_reason,
        "selected_seeded_safe": bool(selected.get("seeded_safe", False)),
        "selected_config": best_cfg,
        "selected_score_key": list(best_key),
        "best_score_key": list(best_key),
        "top10": leaderboard[:10],
    }
    if verbose and tuning_summary["top10"]:
        print(
            "  Auto-tune finished: "
            f"{tuning_seconds:.1f}s, objective_calls={objective_calls}, "
            f"starts_cache_entries={len(starts_cache)}"
        )
        print("  Auto-tune top configs:")
        for idx, row in enumerate(tuning_summary["top10"][:5], 1):
            summary = row["summary"]
            cfg = row["config"]
            well_balance = row.get("well_balance", {})
            print(
                f"    {idx}. hit={summary['hit_count']}/{summary['interval_count']}, "
                f"p90_delay_ratio={summary['p90_delay_ratio']:.3f}, "
                f"FAR/day={summary['false_alarms_per_day']:.3f}, "
                f"starts/interval={summary['avg_starts_per_interval']:.2f}, "
                f"wells={well_balance.get('full_hit_well_count', 0):.0f}/"
                f"{well_balance.get('well_count', 0):.0f}, "
                f"worst_p90={well_balance.get('worst_p90_delay_ratio', float('nan')):.3f}, "
                f"gate={cfg['gate_mode']}, run={cfg['min_run_points']}, cd={cfg['cooldown_hours']:.0f}, "
                f"rearm={cfg['rearm_window_minutes']:.0f}m, "
                f"ema={cfg['ema_alpha']:.2f}, "
                f"bypass={int(bool(cfg.get('bypass_cooldown_after_clear', True)))}"
                + (
                    f", w={cfg['fusion_weight_short']:.2f}"
                    if "fusion_weight_short" in cfg
                    else ""
                )
                + (
                    f", pw={cfg['pressure_trend_weight']:.4f}"
                    if "pressure_trend_weight" in cfg
                    else ""
                )
                + (
                    f", nw={cfg['negermet_signature_weight']:.4f}"
                    if "negermet_signature_weight" in cfg
                    else ""
                )
                + (
                    f", sw={cfg['salt_trend_weight']:.4f}"
                    if "salt_trend_weight" in cfg
                    else ""
                )
            )
    return best_cfg, tuning_summary


def _tune_config_with_grid(
    anomaly_key: str,
    detector_key: str,
    train_runs: dict[str, PreparedDetectorRun],
    train_intervals: pd.DataFrame,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not train_runs:
        cfg = _default_onset_config(anomaly_key, detector_key)
        return cfg, {"message": "No train runs available"}

    best_cfg: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    leaderboard: list[dict[str, Any]] = []
    per_well_intervals = _build_per_well_tuning_intervals(train_intervals)
    starts_cache: dict[tuple[str, tuple[tuple[str, Any], ...]], list[pd.Timestamp]] = {}
    tuning_start = time.monotonic()

    for cfg in _candidate_configs(anomaly_key, detector_key):
        predicted = _detect_starts_for_config(detector_key, train_runs, cfg, starts_cache)
        pred_df = predicted_from_mapping(predicted)
        summary, _ = evaluate_predictions(
            train_intervals,
            pred_df,
            scores=None,
            prestart_hours=PRESTART_TOLERANCE_HOURS,
        )
        per_well_summaries = _per_well_tuning_summaries_from_mapping(per_well_intervals, predicted)
        key = _robust_tuning_score_key(anomaly_key, detector_key, summary, per_well_summaries)
        leaderboard.append(
            {
                "score_key": list(key),
                "config": cfg,
                "summary": summary,
                "per_well_summaries": per_well_summaries,
                "well_balance": _well_balance_stats(per_well_summaries),
            }
        )
        if best_key is None or key > best_key:
            best_cfg = cfg.copy()
            best_key = key

    leaderboard = sorted(leaderboard, key=lambda row: tuple(row["score_key"]), reverse=True)
    if best_cfg is None:
        best_cfg = _default_onset_config(anomaly_key, detector_key)
    tuning_seconds = time.monotonic() - tuning_start
    tuning_summary = {
        "backend": "grid",
        "elapsed_seconds": float(tuning_seconds),
        "starts_cache_entries": int(len(starts_cache)),
        "best_score_key": list(best_key) if best_key is not None else None,
        "top10": leaderboard[:10],
    }
    if verbose and tuning_summary["top10"]:
        print(
            "  Auto-tune finished: "
            f"{tuning_seconds:.1f}s, candidates={len(leaderboard)}, "
            f"starts_cache_entries={len(starts_cache)}"
        )
        print("  Auto-tune top configs:")
        for idx, row in enumerate(tuning_summary["top10"][:5], 1):
            summary = row["summary"]
            cfg = row["config"]
            well_balance = row.get("well_balance", {})
            print(
                f"    {idx}. hit={summary['hit_count']}/{summary['interval_count']}, "
                f"p90_delay_ratio={summary['p90_delay_ratio']:.3f}, "
                f"FAR/day={summary['false_alarms_per_day']:.3f}, "
                f"starts/interval={summary['avg_starts_per_interval']:.2f}, "
                f"wells={well_balance.get('full_hit_well_count', 0):.0f}/"
                f"{well_balance.get('well_count', 0):.0f}, "
                f"worst_p90={well_balance.get('worst_p90_delay_ratio', float('nan')):.3f}, "
                f"gate={cfg['gate_mode']}, run={cfg['min_run_points']}, cd={cfg['cooldown_hours']:.0f}, "
                f"rearm={cfg['rearm_window_minutes']:.0f}m, "
                f"ema={cfg['ema_alpha']:.2f}, "
                f"bypass={int(bool(cfg.get('bypass_cooldown_after_clear', True)))}"
                + (
                    f", w={cfg['fusion_weight_short']:.2f}"
                    if "fusion_weight_short" in cfg
                    else ""
                )
                + (
                    f", pw={cfg['pressure_trend_weight']:.4f}"
                    if "pressure_trend_weight" in cfg
                    else ""
                )
                + (
                    f", nw={cfg['negermet_signature_weight']:.4f}"
                    if "negermet_signature_weight" in cfg
                    else ""
                )
                + (
                    f", sw={cfg['salt_trend_weight']:.4f}"
                    if "salt_trend_weight" in cfg
                    else ""
                )
            )
    return best_cfg, tuning_summary


def _tune_config(
    anomaly_key: str,
    detector_key: str,
    train_runs: dict[str, PreparedDetectorRun],
    train_intervals: pd.DataFrame,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mode = _retune_mode()
    if mode == RETUNE_MODE_GRID:
        return _tune_config_with_grid(anomaly_key, detector_key, train_runs, train_intervals, verbose)
    if optuna is None:
        return _tune_config_with_grid(anomaly_key, detector_key, train_runs, train_intervals, verbose)
    return _tune_config_with_optuna(anomaly_key, detector_key, train_runs, train_intervals, verbose)


def _load_or_build_config(
    spec: DetectionSpec,
    detector_key: str,
    train_runs: dict[str, PreparedDetectorRun],
    train_intervals: pd.DataFrame,
    retune: bool,
    verbose: bool,
) -> dict[str, Any]:
    cfg_path = config_path(spec, detector_key)
    tune_path = tuning_path(spec, detector_key)
    if cfg_path.exists() and not retune:
        payload = load_json(cfg_path)
        if isinstance(payload, dict) and "config" in payload:
            return {**_default_onset_config(spec.anomaly_key, detector_key), **payload["config"]}
        if payload:
            return {**_default_onset_config(spec.anomaly_key, detector_key), **payload}

    cfg, tuning_summary = _tune_config(
        spec.anomaly_key,
        detector_key,
        train_runs,
        train_intervals,
        verbose=verbose,
    )
    ensure_parent(cfg_path).write_text(
        json.dumps({"detector": detector_key, "config": cfg}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ensure_parent(tune_path).write_text(
        json.dumps(tuning_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if verbose:
        print(f"Saved tuned config to {cfg_path}")
        print(f"Saved tuning summary to {tune_path}")
    return cfg


def _build_score_rows(
    detector_key: str,
    runs: dict[str, PreparedDetectorRun],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[pd.Timestamp]], dict[str, dict[str, Any]]]:
    score_rows: list[dict[str, Any]] = []
    predicted: dict[str, list[pd.Timestamp]] = {}
    detail_map: dict[str, dict[str, Any]] = {}

    for well_id, run in runs.items():
        score, thresholds, starts = _detect_starts_for_run(detector_key, run, cfg)
        predicted[well_id] = starts
        detail_map[well_id] = {
            "prepared": run.prepared.detail,
            "detector": run.score_output.detail,
            "thresholds": asdict(thresholds),
            "n_predicted_starts": len(starts),
        }
        components = dict(run.score_output.components)
        components["score"] = score

        for idx, ts in enumerate(run.prepared.timestamps):
            row = {
                "well_id": well_id,
                "timestamp": ts,
                "split": run.prepared.split,
                "score": float(score[idx]),
                "reference_mask": bool(run.prepared.reference_mask[idx]),
                "stability_mask": bool(run.prepared.stability_mask[idx]),
                "onset_allowed_mask": bool(run.prepared.onset_allowed_mask[idx]),
            }
            for name, values in components.items():
                if len(values) != len(score):
                    continue
                row[name] = float(values[idx])
            score_rows.append(row)

    return score_rows, predicted, detail_map


def _merge_result_details(
    result_df: pd.DataFrame,
    predicted: dict[str, list[pd.Timestamp]],
    detail_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in result_df.iterrows():
        well_id = row["well_id"]
        detail = detail_map.get(well_id, {})
        starts = predicted.get(well_id, [])
        merged = row.to_dict()
        merged["n_predicted_starts"] = len(starts)
        merged["n_channels"] = int(detail.get("prepared", {}).get("raw_channels", 0))
        merged["n_features"] = int(detail.get("prepared", {}).get("feature_count", 0))
        merged["detail"] = json.dumps(detail, ensure_ascii=False)
        rows.append(merged)
    return pd.DataFrame(rows)


def _summary_for_payload(payload: dict[str, Any]) -> dict[str, Any]:
    splits = payload.get("splits", {})
    if isinstance(splits, dict) and "all" in splits:
        return splits["all"]
    return payload


def _choose_default_detector(
    anomaly_key: str,
    detector_summaries: dict[str, dict[str, Any]],
) -> str:
    if DEFAULT_DETECTOR in detector_summaries:
        return DEFAULT_DETECTOR
    return sorted(detector_summaries)[0] if detector_summaries else DEFAULT_DETECTOR


def _update_benchmark_summary(spec: DetectionSpec) -> dict[str, Any]:
    detector_payloads: dict[str, dict[str, Any]] = {}
    for detector_key in DETECTOR_KEYS:
        path = summary_path(spec, detector_key)
        if path.exists():
            detector_payloads[detector_key] = load_json(path)
    selected = _choose_default_detector(spec.anomaly_key, detector_payloads)
    payload = {
        "anomaly": spec.anomaly_key,
        "selected_default_detector": selected,
        "detectors": detector_payloads,
    }
    out_path = ensure_parent(benchmark_summary_path(spec))
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _print_result_table(spec: DetectionSpec, detector_key: str, result_df: pd.DataFrame) -> None:
    print(f"\n=== {spec.display_name} Results [{detector_key}] ===")
    if result_df.empty:
        print("No interval results.")
        return
    print(
        result_df[
            ["well_id", "interval_idx", "split", "detected_time", "actual_start", "actual_end", "status"]
        ].to_string(index=False)
    )


def run_detection(
    anomaly_key: str,
    detector: str = DEFAULT_DETECTOR,
    output_path: str | None = None,
    source_path: str | None = None,
    retune: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    detector_key = normalize_detector_key(detector)
    spec = get_detection_spec(anomaly_key)
    print(f"=== {spec.display_name} Detection [{detector_key}] ===")
    set_seed()
    ensure_dir(DB_DIR)
    output = ensure_parent(Path(output_path) if output_path else results_path(spec, detector_key))
    device = _resolve_torch_device(detector_key, verbose=True)

    df = load_anomaly_data(spec, source_path=source_path)
    intervals = load_intervals(spec, required=True)
    if df.empty or intervals.empty:
        raise RuntimeError("Empty data or intervals for detection.")

    # Keep only first interval per well (data is truncated at first anomaly end)
    intervals = (
        intervals
        .sort_values(["well_id", "start_date", "interval_idx"])
        .groupby("well_id", as_index=False)
        .first()
    )

    prepared_runs = _prepare_all_wells(
        spec, df, intervals, verbose=verbose,
        zone_aware=True,
    )
    if not prepared_runs:
        raise RuntimeError("No wells survived engineered preprocessing.")

    from alma_service.shared_encoder import load_or_train_shared_encoder

    runtime_cfg = _runtime_config(anomaly_key)
    shared_state = load_or_train_shared_encoder(
        prepared_wells=prepared_runs,
        patch_short=int(runtime_cfg.get("paano_patch_short", SHORT_PATCH)),
        patch_long=int(runtime_cfg.get("paano_patch_long", LONG_PATCH)),
        anomaly_key=anomaly_key,
        device=device,
        verbose=verbose,
    )
    if verbose:
        print(f"  Shared encoder ready: {shared_state.detail}")

    detector_runs = _build_local_runs(
        spec.anomaly_key, detector_key, prepared_runs,
        device=device, verbose=verbose,
        shared_state=shared_state,
    )

    train_runs = {well_id: run for well_id, run in detector_runs.items() if run.prepared.split == "train"}
    train_intervals = intervals[intervals["split"].astype(str).str.lower() == "train"].copy()
    cfg = _load_or_build_config(
        spec=spec,
        detector_key=detector_key,
        train_runs=train_runs,
        train_intervals=train_intervals,
        retune=retune,
        verbose=verbose,
    )

    score_rows, predicted, detail_map = _build_score_rows(detector_key, detector_runs, cfg)
    score_df = pd.DataFrame(score_rows)
    pred_df = predicted_from_mapping(predicted)
    if not pred_df.empty:
        split_lookup = {well_id: run.prepared.split for well_id, run in detector_runs.items()}
        pred_df["split"] = pred_df["well_id"].map(split_lookup).fillna("train")

    split_summaries, split_frames = summarize_splits(
        intervals=intervals,
        predictions=pred_df,
        scores=score_df,
        prestart_hours=PRESTART_TOLERANCE_HOURS,
    )
    result_df = split_frames.get("all", pd.DataFrame())
    result_df = _merge_result_details(result_df, predicted=predicted, detail_map=detail_map)
    _print_result_table(spec, detector_key, result_df)

    write_table(result_df, output)
    print(f"\nResults saved to {output}")

    score_output_path = scores_path(spec, detector_key)
    ensure_parent(score_output_path)
    write_table(score_df, score_output_path)
    print(f"Per-point scores saved to {score_output_path}")

    pred_output_path = predicted_starts_path(spec, detector_key)
    ensure_parent(pred_output_path)
    write_table(pred_df, pred_output_path)
    print(f"Predicted starts saved to {pred_output_path}")

    summary_payload = {
        "anomaly": spec.anomaly_key,
        "detector": detector_key,
        "config": cfg,
        "prestart_hours": PRESTART_TOLERANCE_HOURS,
        "splits": split_summaries,
        "artifacts": {
            "results_path": str(output),
            "scores_path": str(score_output_path),
            "predicted_starts_path": str(pred_output_path),
            "report_path": str(report_path(spec, detector_key)),
        },
    }
    summary_output_path = ensure_parent(summary_path(spec, detector_key))
    summary_output_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary saved to {summary_output_path}")

    benchmark_payload = _update_benchmark_summary(spec)
    print(
        f"Benchmark summary updated: {benchmark_summary_path(spec)} "
        f"(default={benchmark_payload['selected_default_detector']})"
    )
    return result_df


def run_single_well(
    anomaly_key: str,
    well_id: str,
    detector: str = DEFAULT_DETECTOR,
    source_path: str | None = None,
    retune: bool = False,
) -> None:
    detector_key = normalize_detector_key(detector)
    spec = get_detection_spec(anomaly_key)
    df = load_anomaly_data(spec, source_path=source_path)
    intervals = load_intervals(spec, required=False)
    well_id = well_id.strip().lower()
    well_df = df[df["well_id"] == well_id]
    if well_df.empty:
        raise ValueError(f"No data for well {well_id}")
    from alma_service.dataset_config import split_for_well

    split = split_for_well(anomaly_key, well_id)
    if not intervals.empty:
        well_intervals = intervals[intervals["well_id"] == well_id]
        if not well_intervals.empty and "split" in well_intervals.columns:
            split = str(well_intervals["split"].iloc[0]).strip().lower()

    prepared = prepare_engineered_well(
        anomaly_key=spec.anomaly_key,
        well_id=well_id,
        split=split,
        well_df=well_df,
        patch_size=int(_runtime_config(spec.anomaly_key)["prepare_patch_size"]),
        reference_min_ratio=REFERENCE_MIN_RATIO,
        reference_max_ratio=REFERENCE_MAX_RATIO,
        reference_min_days=REFERENCE_MIN_DAYS,
        min_reference_coverage=MIN_REFERENCE_COVERAGE,
        min_total_coverage=MIN_TOTAL_COVERAGE,
    )
    if prepared is None:
        print("No usable data after engineered preprocessing.")
        return

    device = _resolve_torch_device(detector_key, verbose=True)
    from alma_service.shared_encoder import load_shared_encoder_state

    shared_state = load_shared_encoder_state(spec.anomaly_key, device=device, verbose=True)
    detector_runs = _build_local_runs(
        spec.anomaly_key,
        detector_key,
        {well_id: prepared},
        device=device,
        verbose=True,
        shared_state=shared_state,
    )
    run = detector_runs[well_id]

    cfg_payload = load_json(config_path(spec, detector_key))
    cfg = {**_default_onset_config(spec.anomaly_key, detector_key), **(cfg_payload.get("config", cfg_payload) if cfg_payload else {})}

    score, thresholds, starts = _detect_starts_for_run(detector_key, run, cfg)
    print(f"Prepared detail: {json.dumps(run.prepared.detail, ensure_ascii=False, indent=2)}")
    print(f"Detector detail: {json.dumps(run.score_output.detail, ensure_ascii=False, indent=2)}")
    print(f"Thresholds: {json.dumps(asdict(thresholds), ensure_ascii=False, indent=2)}")
    print(f"Detected starts: {[pd.Timestamp(ts) for ts in starts]}")
    print(f"Score summary: min={float(np.min(score)):.4f}, median={float(np.median(score)):.4f}, max={float(np.max(score)):.4f}")

    if not intervals.empty:
        well_intervals = intervals[intervals["well_id"] == well_id].sort_values(["start_date", "interval_idx"])
        for _, row in well_intervals.iterrows():
            det = select_interval_detection(starts, row["start_date"], row["end_date"], PRESTART_TOLERANCE_HOURS)
            print(
                f"  interval={int(row['interval_idx'])} start={row['start_date']} end={row['end_date']} "
                f"detected={det}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified engineered-feature anomaly detection pipeline.")
    parser.add_argument("anomaly", choices=["negermet", "pritok", "salt"])
    parser.add_argument("--detector", choices=sorted(DETECTOR_KEYS), default=DEFAULT_DETECTOR)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--well", type=str, default=None)
    parser.add_argument("--retune", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.well:
        run_single_well(
            anomaly_key=args.anomaly,
            well_id=args.well,
            detector=args.detector,
            source_path=args.source,
            retune=args.retune,
        )
        return
    run_detection(
        anomaly_key=args.anomaly,
        detector=args.detector,
        output_path=args.output,
        source_path=args.source,
        retune=args.retune,
        verbose=True,
    )


if __name__ == "__main__":
    main()
