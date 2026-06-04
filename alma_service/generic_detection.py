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
    incidents_path,
    load_json,
    normalize_detector_key,
    precursor_path,
    predicted_starts_path,
    report_path,
    results_path,
    scores_path,
    summary_path,
    tuning_path,
)
from alma_service.engineered_features import PreparedWellData, prepare_engineered_well
from alma_service.engineered_features import (
    REFERENCE_POLICIES,
    REFERENCE_POLICY_NORMAL_WINDOWS,
)
from alma_service.generic_detectors import (
    DetectorScoreOutput,
    SharedPaAnoDetector,
    set_seed,
)
from alma_service.onset_detection import (
    CausalThresholds,
    calibrate_causal_thresholds_from_reference_mask,
    detect_causal_onsets_masked,
)
from alma_service.precursor_logreg import (
    PrecursorLogregModel,
    extract_features as precursor_extract_features,
    load_model as load_precursor_model,
    save_model as save_precursor_model,
    score_proba as precursor_score_proba,
    train_lowo as train_precursor_lowo,
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
from alma_service.paths import DB_DIR, MODELS_DIR, ensure_dir, ensure_parent
from alma_service.prediction_postprocess import build_incidents, filter_actionable_starts
from alma_service.tabular_io import read_table, write_table
from alma_service.telemetry_status import build_telemetry_status

STATUS_COLUMNS = [
    "quality_status",
    "regime_status",
    "zone_status",
    "event_class",
    "is_bad_data",
    "is_regime_event",
    "is_pre_anomaly_zone",
    "is_labelled_anomaly",
]
INVALID_SCORE_REASONS = {
    "not_enough_points",
    "unlabeled_no_reference",
    "population_reference_unavailable",
}


def _incident_merge_window_hours(cfg: dict[str, Any]) -> float:
    cooldown_hours = float(cfg.get("cooldown_hours", 0.0))
    rearm_hours = float(cfg.get("rearm_window_minutes", 0.0)) / 60.0
    return max(cooldown_hours * 2.0, rearm_hours * 2.0, 1.0)

BASE_ONSET_CONFIG = {
    "target_far_per_day": 0.50,
    "min_run_points": 3,
    "cooldown_hours": 12.0,
    "rearm_window_minutes": 60.0,
    "ema_alpha": 0.08,
    "gate_mode": "score_ema",
    "hysteresis_scale": 0.60,
    "bypass_cooldown_after_clear": True,
    "early_warning_threshold_scale": 0.0,
    "early_warning_dedupe_hours": 2.0,
    "early_warning_cooldown_hours": 3.0,
    "early_warning_rearm_minutes": 30.0,
    "early_warning_logreg_threshold": 0.0,
    "early_warning_logreg_min_run_points": 3,
    "early_warning_warmup_hours": 72.0,
}

ANOMALY_PRECURSOR_DEFAULT_THRESHOLD = {
    "negermet": 0.0093,
    "pritok": 0.815,
    "salt": 0.539,
}

PRECURSOR_TRAIN_PRESTART_HOURS = 24.0

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
PAANO_PRODUCTION_DETECTORS = {"paano_shared", "paano_global"}
CUDA_REQUIRED_DETECTORS = PAANO_PRODUCTION_DETECTORS


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


def _is_paano_production_detector(detector_key: str) -> bool:
    return detector_key in PAANO_PRODUCTION_DETECTORS


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
    if _is_paano_production_detector(detector_key):
        cfg["fusion_weight_short"] = 0.60
    default_thr = ANOMALY_PRECURSOR_DEFAULT_THRESHOLD.get(anomaly_key)
    if default_thr is not None:
        cfg["early_warning_logreg_threshold"] = float(default_thr)
    # fusion-критерий притока (наклон давления + скор нейросети) — дефолт для класса
    # приток после проверки 03.06.2026 (5/5 на тестовых, 0 ложных на 39 контрольных).
    # Отключается ALMA_PRESSURE_TREND_FUSION=0. Для негермет/соль не применяется.
    if anomaly_key == "pritok" and os.getenv("ALMA_PRESSURE_TREND_FUSION", "1").strip().lower() in ("1", "true", "yes"):
        cfg["pressure_trend_fusion"] = {
            "enabled": True,
            "score_threshold": float(os.getenv("ALMA_PTF_SCORE_THRESHOLD", "0.0040")),
            "slope_threshold_pct_per_day": float(os.getenv("ALMA_PTF_SLOPE", "0.15")),
            "window_days": float(os.getenv("ALMA_PTF_WINDOW_DAYS", "10.0")),
            "freq_jump_buffer_days": float(os.getenv("ALMA_PTF_FREQ_BUFFER_DAYS", "2.0")),
            "dedupe_hours": 24.0,
        }
        # Отметка детекции — начало тренда (ретроспективно); early_warning, стоящий сильно
        # ДО начала тренда — нейрошум, его отсекаем (окно-предвестник перед onset сохраняем).
        cfg["early_warning_trend_gate"] = True
        cfg["early_warning_precursor_days"] = float(os.getenv("ALMA_PTF_EW_PRECURSOR_DAYS", "0.0"))
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
    reference_policy: str = REFERENCE_POLICY_NORMAL_WINDOWS,
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
            reference_policy=reference_policy,
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
    population_reference: np.ndarray | None = None,
) -> dict[str, PreparedDetectorRun]:
    if not _is_paano_production_detector(detector_key):
        raise ValueError(f"Unsupported production detector: {detector_key}")
    if shared_state is None:
        raise ValueError(f"{detector_key} requires a trained or loaded shared encoder state.")

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
        if population_reference is not None:
            if verbose:
                print(f"  Using population memory bank: {population_reference.shape} (vs local {int(prepared.reference_mask.sum())})")
            detector.fit_reference(population_reference)
        else:
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


def _build_global_runs(
    prepared_runs: dict[str, PreparedWellData],
    device: torch.device,
    verbose: bool,
    shared_state: Any,
    intervals: pd.DataFrame | None = None,
    population_pool: dict[str, Any] | None = None,
) -> dict[str, PreparedDetectorRun]:
    if shared_state is None:
        raise ValueError("paano_global requires a trained or loaded global encoder state.")

    from alma_service.global_normality import build_global_core_runs

    labelled_wells: set[str] | None = None
    if intervals is not None and not intervals.empty:
        labelled_wells = {
            str(well_id).strip().lower()
            for well_id in intervals["well_id"].dropna().tolist()
        }

    return build_global_core_runs(
        prepared_runs,
        shared_state,
        device,
        verbose=verbose,
        labelled_wells=labelled_wells,
        population_pool=population_pool,
    )


def _score_for_config(run: PreparedDetectorRun, detector_key: str, cfg: dict[str, Any]) -> np.ndarray:
    if not _is_paano_production_detector(detector_key):
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


def _early_starts_from_precursor(
    precursor_model: PrecursorLogregModel,
    timestamps: np.ndarray,
    score: np.ndarray,
    run: PreparedDetectorRun,
    onset_allowed_mask: np.ndarray,
    threshold: float,
    min_run_points: int,
    cooldown_hours: float,
    critical_starts: list[pd.Timestamp],
    dedupe_hours: float,
    labelled_mask: np.ndarray | None = None,
    warmup_hours: float = 0.0,
) -> list[pd.Timestamp]:
    if threshold <= 0.0 or len(score) == 0:
        return []
    features = precursor_extract_features(
        timestamps,
        np.asarray(score, dtype=np.float64),
        run.prepared.raw_columns,
        run.prepared.raw_matrix,
        ema_alpha=precursor_model.ema_alpha,
        rolling_window_minutes=precursor_model.rolling_window_minutes,
    )
    proba = precursor_score_proba(precursor_model, features)
    cooldown_ns = int(cooldown_hours * 3600 * 1e9)
    last_fire_ns = -10**18
    run_count = 0
    fires: list[pd.Timestamp] = []
    valid_mask = np.ones(len(timestamps), dtype=bool)
    if labelled_mask is not None and len(labelled_mask) == len(valid_mask):
        valid_mask = valid_mask & ~np.asarray(labelled_mask, dtype=bool)
    if warmup_hours > 0.0 and len(timestamps) > 0:
        ts_arr = pd.to_datetime(np.asarray(timestamps))
        warmup_end = ts_arr[0] + pd.Timedelta(hours=float(warmup_hours))
        valid_mask = valid_mask & np.asarray(ts_arr >= warmup_end, dtype=bool)
    for i, t in enumerate(timestamps):
        if not valid_mask[i] or proba[i] < threshold:
            run_count = 0
            continue
        run_count += 1
        if run_count < min_run_points:
            continue
        t_ns = pd.Timestamp(t).value
        if t_ns - last_fire_ns < cooldown_ns:
            continue
        fires.append(pd.Timestamp(t))
        last_fire_ns = t_ns
        run_count = 0
    if not fires:
        return []
    dedupe_ns = int(dedupe_hours * 3600 * 1e9)
    crit_ns = [pd.Timestamp(t).value for t in critical_starts]
    if not crit_ns:
        return fires
    return [
        t for t in fires
        if all(abs(pd.Timestamp(t).value - c) > dedupe_ns for c in crit_ns)
    ]


def _detect_starts_for_run(
    detector_key: str,
    run: PreparedDetectorRun,
    cfg: dict[str, Any],
    *,
    precursor_model: PrecursorLogregModel | None = None,
    labelled_mask: np.ndarray | None = None,
    trend_sink: dict[str, Any] | None = None,
) -> tuple[np.ndarray, Any, list[pd.Timestamp], list[pd.Timestamp]]:
    cfg = {**BASE_ONSET_CONFIG, **cfg}
    score = _score_for_config(run, detector_key, cfg)
    score_unavailable_reason = _score_unavailable_reason(run)
    thresholds, diagnostics = calibrate_causal_thresholds_from_reference_mask(
        scores=score,
        timestamps=run.prepared.timestamps,
        reference_mask=run.prepared.reference_mask,
        target_far_per_day=float(cfg["target_far_per_day"]),
        min_run_points=int(cfg["min_run_points"]),
        ema_alpha=float(cfg["ema_alpha"]),
    )
    if score_unavailable_reason is not None:
        return score, thresholds, [], []
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
    starts, trend_onsets = _augment_with_pressure_trend_fusion(starts, run, score, cfg)
    starts = _gate_pre_trend_starts(starts, trend_onsets, cfg)
    if trend_sink is not None:
        trend_sink["onsets"] = list(trend_onsets)
    early_starts: list[pd.Timestamp] = []
    early_threshold_logreg = float(cfg.get("early_warning_logreg_threshold", 0.0))
    if precursor_model is not None and early_threshold_logreg > 0.0:
        early_starts = _early_starts_from_precursor(
            precursor_model,
            run.prepared.timestamps,
            score,
            run,
            run.prepared.onset_allowed_mask,
            threshold=early_threshold_logreg,
            min_run_points=int(cfg.get("early_warning_logreg_min_run_points", 3)),
            cooldown_hours=float(cfg.get("early_warning_cooldown_hours", 3.0)),
            critical_starts=starts,
            dedupe_hours=float(cfg.get("early_warning_dedupe_hours", 2.0)),
            labelled_mask=labelled_mask,
            warmup_hours=float(cfg.get("early_warning_warmup_hours", 0.0)),
        )
        return score, thresholds, starts, _gate_early_warning(early_starts, trend_onsets, cfg)
    early_scale = float(cfg.get("early_warning_threshold_scale", 0.0))
    if early_scale > 0.0 and 0.0 < early_scale < 1.0:
        early_thresholds = CausalThresholds(
            score_threshold=thresholds.score_threshold * early_scale,
            ema_z_threshold=thresholds.ema_z_threshold * early_scale,
            cusum_threshold=thresholds.cusum_threshold * early_scale,
            drift=thresholds.drift,
            baseline_median=thresholds.baseline_median,
            baseline_mad=thresholds.baseline_mad,
            quantile=thresholds.quantile,
        )
        early_cooldown = float(cfg.get("early_warning_cooldown_hours", cfg["cooldown_hours"]))
        early_rearm = float(cfg.get("early_warning_rearm_minutes", cfg["rearm_window_minutes"]))
        early_all = detect_causal_onsets_masked(
            scores=score,
            timestamps=run.prepared.timestamps,
            diagnostics=diagnostics,
            thresholds=early_thresholds,
            reference_mask=run.prepared.reference_mask,
            onset_mask=run.prepared.onset_allowed_mask,
            min_run_points=int(cfg["min_run_points"]),
            cooldown_hours=early_cooldown,
            rearm_window_minutes=early_rearm,
            gate_mode=str(cfg["gate_mode"]),
            hysteresis_scale=float(cfg["hysteresis_scale"]),
            bypass_cooldown_after_clear=bool(cfg.get("bypass_cooldown_after_clear", True)),
        )
        dedupe_ns = int(float(cfg.get("early_warning_dedupe_hours", 2.0)) * 3600 * 1e9)
        crit_ns = [pd.Timestamp(t).value for t in starts]
        for t in early_all:
            t_ns = pd.Timestamp(t).value
            if all(abs(t_ns - c) > dedupe_ns for c in crit_ns):
                early_starts.append(t)
    return score, thresholds, starts, _gate_early_warning(early_starts, trend_onsets, cfg)


def _augment_with_pressure_trend_fusion(
    starts: list[pd.Timestamp],
    run: PreparedDetectorRun,
    score: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[list[pd.Timestamp], list[Any]]:
    # Совмещённый трендовый детектор притока (наклон давления + скор нейросети).
    # По умолчанию выключен; включается только для класса приток через конфиг —
    # негермет и соль не затрагиваются. Подробности: docs/физика_аномалий…, раздел fusion.
    # Возвращает (объединённые старты, принятые TrendOnset). Отметка эпизода — начало
    # тренда (onset), плюс момент причинного срабатывания (trigger) для отчёта.
    fusion_cfg = cfg.get("pressure_trend_fusion")
    if not fusion_cfg or not bool(fusion_cfg.get("enabled", False)):
        return starts, []
    from alma_service.pressure_trend_onset import (
        PressureTrendFusionConfig,
        detect_from_prepared,
    )

    config = PressureTrendFusionConfig.from_dict(fusion_cfg)
    trend_onsets = detect_from_prepared(run.prepared, score, config)
    if not trend_onsets:
        return starts, []
    # Трендовый онсет добавляем ВСЕГДА (защита от подавления зависит от того, что строка
    # старта существует ровно на onset). Дедуп — только по точному совпадению времени,
    # чтобы не плодить буквальные дубли; ближний нейро-старт остаётся отдельной строкой и
    # сольётся в один инцидент. (Старый 24-час дедуп терял онсет — скв. 46-806.)
    merged = list(starts)
    existing_ns = {pd.Timestamp(t).value for t in merged}
    accepted: list[Any] = list(trend_onsets)
    for onset in trend_onsets:
        t_ns = pd.Timestamp(onset.onset).value
        if t_ns not in existing_ns:
            merged.append(pd.Timestamp(onset.onset))
            existing_ns.add(t_ns)
    merged.sort()
    return merged, accepted


def _gate_early_warning(
    early_starts: list[pd.Timestamp],
    trend_onsets: list[Any],
    cfg: dict[str, Any],
) -> list[pd.Timestamp]:
    # Для притока: убираем early_warning, который стоит сильно ДО начала подтверждённого
    # тренда (нейрошум до спада). Оставляем короткое окно-предвестник перед onset. Если
    # тренда нет (в т.ч. на размеченных контрольных без fusion) — early_warning не трогаем.
    if not bool(cfg.get("early_warning_trend_gate", False)) or not early_starts or not trend_onsets:
        return early_starts
    earliest = min(pd.Timestamp(onset.onset) for onset in trend_onsets)
    cutoff = earliest - pd.Timedelta(days=float(cfg.get("early_warning_precursor_days", 7.0)))
    return [t for t in early_starts if pd.Timestamp(t) >= cutoff]


def _gate_pre_trend_starts(
    starts: list[pd.Timestamp],
    trend_onsets: list[Any],
    cfg: dict[str, Any],
) -> list[pd.Timestamp]:
    # Для притока с подтверждённым трендом: критические старты, стоящие ДО начала тренда —
    # нейрошум не по тренду (напр. 46-806: 06.12/15.12 сразу после reference). Убираем их,
    # сами трендовые онсеты сохраняем. Без тренда (sharp/контроль) — не трогаем.
    if not bool(cfg.get("early_warning_trend_gate", False)) or not starts or not trend_onsets:
        return starts
    earliest = min(pd.Timestamp(onset.onset) for onset in trend_onsets)
    cutoff = earliest - pd.Timedelta(days=float(cfg.get("trend_gate_critical_buffer_days", 0.0)))
    onset_ns = {pd.Timestamp(onset.onset).value for onset in trend_onsets}
    return [t for t in starts if pd.Timestamp(t).value in onset_ns or pd.Timestamp(t) >= cutoff]


def _score_unavailable_reason(run: PreparedDetectorRun) -> str | None:
    reason = str(run.score_output.detail.get("reason", "")).strip()
    if reason in INVALID_SCORE_REASONS:
        return reason
    return None


def _candidate_configs(anomaly_key: str, detector_key: str) -> list[dict[str, Any]]:
    if not _is_paano_production_detector(detector_key):
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
        if isinstance(value, dict):
            # вложенные конфиги (например pressure_trend_fusion) — рекурсивно в хешируемый ключ
            return tuple(sorted((str(k), _normalize_value(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(_normalize_value(v) for v in value)
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
        _, _, starts, _ = _detect_starts_for_run(detector_key, run, cfg)
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


def _lowo_enabled() -> bool:
    """LOWO-CV в Optuna-objective. По умолчанию ВКЛЮЧЕН (с 2026-05-25):
    подтверждённо даёт salt-win ×3.8 на test p90 при том же hit-rate.
    Опционально выключается через ALMA_LOWO_TUNING=0.
    """
    return os.environ.get("ALMA_LOWO_TUNING", "1").strip().lower() in ("1", "true", "yes")


def _lowo_optuna_objective_value(
    anomaly_key: str,
    detector_key: str,
    summary: dict[str, Any],
    per_well_summaries: dict[str, dict[str, Any]],
) -> float:
    """Цель = mean per-fold _optuna_objective_value по train-скв.

    Эквивалент Leave-One-Well-Out CV: т.к. config — гиперпараметр без обучения,
    fold = одна скв. оценивается под этим cfg, объективы усредняются. Делает
    выбор cfg робастным — нельзя «перевесить» одну лёгкую скв., все вносят
    равный вклад.
    """
    if not per_well_summaries:
        return _robust_optuna_objective_value(
            anomaly_key, detector_key, summary, per_well_summaries
        )
    per_fold = [
        _optuna_objective_value(anomaly_key, detector_key, s)
        for s in per_well_summaries.values()
    ]
    return float(np.mean(per_fold))


def _suggest_optuna_config(trial: Any, anomaly_key: str, detector_key: str) -> dict[str, Any]:
    if not _is_paano_production_detector(detector_key):
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
        objective_fn = (
            _lowo_optuna_objective_value if _lowo_enabled() else _robust_optuna_objective_value
        )
        return objective_fn(
            anomaly_key,
            detector_key,
            summary,
            per_well_summaries,
        )

    n_jobs = _tuning_n_jobs(anomaly_key, detector_key, mode)
    backend = f"optuna_tpe_{mode}"
    objective_mode = "lowo_mean_per_well" if _lowo_enabled() else "robust_aggregate"
    if verbose:
        print(
            f"  Auto-tune backend: {backend}, trials={n_trials}, n_jobs={n_jobs}, "
            f"seed={seed}, objective={objective_mode}"
        )
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


def _collect_precursor_training_data(
    detector_key: str,
    train_runs: dict[str, PreparedDetectorRun],
    train_intervals: pd.DataFrame,
    cfg: dict[str, Any],
    anomaly_key: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Build per-well features+target for precursor model.

    target = 1 inside [start-PRESTART_TOLERANCE_HOURS, start) для labelled-интервалов;
    target = 0 для нормальной части (lab=0). Labelled-интервалы исключены из выборки
    через valid mask (там target неопределён — это уже зона аномалии, не precursor)."""
    per_well: dict[str, dict[str, np.ndarray]] = {}
    if train_intervals is None or train_intervals.empty:
        return per_well

    intervals_lookup = train_intervals.copy()
    intervals_lookup["well_id"] = intervals_lookup["well_id"].astype(str).str.strip().str.lower()
    intervals_lookup["start_date"] = pd.to_datetime(intervals_lookup["start_date"])
    intervals_lookup["end_date"] = pd.to_datetime(intervals_lookup["end_date"])

    pre_tol = pd.Timedelta(hours=float(PRECURSOR_TRAIN_PRESTART_HOURS))

    for well_id, run in train_runs.items():
        score = _score_for_config(run, detector_key, cfg)
        if _score_unavailable_reason(run) is not None:
            continue
        timestamps = run.prepared.timestamps
        ts = pd.to_datetime(timestamps)
        n = len(ts)
        if n == 0:
            continue
        pre_mask = np.zeros(n, dtype=bool)
        labelled_mask = np.zeros(n, dtype=bool)
        wid_norm = str(well_id).strip().lower()
        w_intervals = intervals_lookup[intervals_lookup["well_id"] == wid_norm]
        for _, row in w_intervals.iterrows():
            start = row["start_date"]
            end = row["end_date"]
            if pd.isna(start) or pd.isna(end):
                continue
            in_label = (ts >= start) & (ts <= end)
            labelled_mask |= in_label
            in_pre = (ts >= start - pre_tol) & (ts < start)
            pre_mask |= in_pre

        target = pre_mask.astype(int)
        valid = ~labelled_mask
        features = precursor_extract_features(
            timestamps,
            np.asarray(score, dtype=np.float64),
            run.prepared.raw_columns,
            run.prepared.raw_matrix,
        )
        per_well[str(well_id)] = {
            "features": features,
            "target": target,
            "valid": valid,
        }
    return per_well


def _train_or_load_precursor(
    spec: DetectionSpec,
    detector_key: str,
    train_runs: dict[str, PreparedDetectorRun],
    train_intervals: pd.DataFrame,
    cfg: dict[str, Any],
    retune: bool,
    verbose: bool,
) -> PrecursorLogregModel | None:
    path = precursor_path(spec, detector_key)
    if path.exists() and not retune:
        model = load_precursor_model(path)
        if model is not None and verbose:
            print(
                f"Loaded precursor model from {path} "
                f"(pos={model.train_pos}, neg={model.train_neg}, "
                f"oof_auc={model.oof_auc if model.oof_auc is not None else float('nan'):.4f})"
            )
        return model
    if not train_runs or train_intervals is None or train_intervals.empty:
        return None
    try:
        per_well = _collect_precursor_training_data(
            detector_key, train_runs, train_intervals, cfg, spec.anomaly_key
        )
        if not per_well:
            if verbose:
                print("Precursor training skipped: no per-well features.")
            return None
        model = train_precursor_lowo(per_well)
    except Exception as exc:
        if verbose:
            print(f"Precursor training failed: {exc!r}")
        return None
    save_precursor_model(model, path)
    if verbose:
        oof = model.oof_auc if model.oof_auc is not None else float("nan")
        print(
            f"Trained precursor model: pos={model.train_pos}, neg={model.train_neg}, "
            f"n_wells={model.n_wells}, oof_auc={oof:.4f} -> {path}"
        )
    return model


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
    *,
    anomaly_key: str = "",
    intervals: pd.DataFrame | None = None,
    precursor_model: PrecursorLogregModel | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[pd.Timestamp]],
    dict[str, list[pd.Timestamp]],
    dict[str, dict[str, Any]],
    dict[str, list[Any]],
]:
    score_rows: list[dict[str, Any]] = []
    predicted: dict[str, list[pd.Timestamp]] = {}
    early_predicted: dict[str, list[pd.Timestamp]] = {}
    trend_predicted: dict[str, list[Any]] = {}
    detail_map: dict[str, dict[str, Any]] = {}

    for well_id, run in runs.items():
        well_intervals = None
        if intervals is not None and not intervals.empty:
            normalized = intervals["well_id"].astype(str).str.strip().str.lower()
            well_intervals = intervals[normalized == str(well_id).strip().lower()]
        labelled_mask = None
        if well_intervals is not None and not well_intervals.empty:
            ts_arr = pd.to_datetime(np.asarray(run.prepared.timestamps))
            first_start = pd.to_datetime(well_intervals["start_date"]).min()
            if pd.notna(first_start):
                labelled_mask = np.asarray(ts_arr >= first_start, dtype=bool)
        trend_sink: dict[str, Any] = {}
        score, thresholds, starts, early_starts = _detect_starts_for_run(
            detector_key, run, cfg,
            precursor_model=precursor_model,
            labelled_mask=labelled_mask,
            trend_sink=trend_sink,
        )
        predicted[well_id] = starts
        early_predicted[well_id] = early_starts
        trend_predicted[well_id] = trend_sink.get("onsets", [])
        score_unavailable_reason = _score_unavailable_reason(run)
        score_valid = score_unavailable_reason is None
        status_frame = build_telemetry_status(
            timestamps=run.prepared.timestamps,
            raw_columns=run.prepared.raw_columns,
            raw_matrix=run.prepared.raw_matrix,
            anomaly_key=anomaly_key,
            anomaly_intervals=well_intervals,
            patch_size=int(run.prepared.detail.get("patch_size", 96)),
        ).frame
        detail_map[well_id] = {
            "prepared": run.prepared.detail,
            "detector": run.score_output.detail,
            "thresholds": asdict(thresholds),
            "n_predicted_starts": len(starts),
            "n_early_warning_starts": len(early_starts),
            "score_valid": score_valid,
            "score_unavailable_reason": score_unavailable_reason,
            "input_contract": str(run.score_output.detail.get("input_contract", "real_window")),
        }
        components = dict(run.score_output.components)
        components["score"] = score

        # Векторно: статус-колонки и компоненты заранее в numpy (раньше status_frame.iloc[idx]
        # на каждую из ~115k точек давал 12-16с pandas-оверхеда — узкое место профиля).
        n_points = len(run.prepared.timestamps)
        ts_values = np.asarray(run.prepared.timestamps)
        ref_mask = np.asarray(run.prepared.reference_mask, dtype=bool)
        stab_mask = np.asarray(run.prepared.stability_mask, dtype=bool)
        onset_mask = np.asarray(run.prepared.onset_allowed_mask, dtype=bool)
        input_contract = str(run.score_output.detail.get("input_contract", "real_window"))
        reason_value = score_unavailable_reason or ""
        status_arrays = (
            {name: status_frame[name].to_numpy() for name in STATUS_COLUMNS}
            if len(status_frame) == len(score)
            else {}
        )
        component_arrays = {name: values for name, values in components.items() if len(values) == len(score)}

        for idx in range(n_points):
            row = {
                "well_id": well_id,
                "timestamp": ts_values[idx],
                "split": run.prepared.split,
                "score": float(score[idx]),
                "reference_mask": bool(ref_mask[idx]),
                "stability_mask": bool(stab_mask[idx]),
                "onset_allowed_mask": bool(onset_mask[idx]),
                "score_valid": bool(score_valid),
                "score_unavailable_reason": reason_value,
                "input_contract": input_contract,
            }
            for name, arr in status_arrays.items():
                row[name] = arr[idx]
            for name, values in component_arrays.items():
                row[name] = float(values[idx])
            score_rows.append(row)

    return score_rows, predicted, early_predicted, detail_map, trend_predicted


def _tag_trend_starts(pred_df: pd.DataFrame, trend_predicted: dict[str, list[Any]]) -> pd.DataFrame:
    # Помечаем строки-старты, совпадающие с началом тренда (onset), как защищённые
    # (trend_protected) и проставляем момент причинного срабатывания (trend_trigger_time).
    if pred_df.empty or not trend_predicted:
        return pred_df
    result = pred_df.copy()
    if "trend_protected" not in result.columns:
        result["trend_protected"] = False
    if "trend_trigger_time" not in result.columns:
        result["trend_trigger_time"] = pd.NaT
    wid = result["well_id"].astype(str).str.strip().str.lower()
    detected = pd.to_datetime(result["detected_time"])
    for well, onsets in trend_predicted.items():
        wkey = str(well).strip().lower()
        for onset in onsets or []:
            mask = (wid == wkey) & (detected == pd.Timestamp(onset.onset))
            if mask.any():
                result.loc[mask, "trend_protected"] = True
                result.loc[mask, "trend_trigger_time"] = pd.Timestamp(onset.trigger)
    return result


def _predicted_with_early_warning(
    critical: dict[str, list[pd.Timestamp]],
    early: dict[str, list[pd.Timestamp]],
) -> pd.DataFrame:
    base = predicted_from_mapping(critical)
    base["event_class"] = ""
    if early and any(len(v) > 0 for v in early.values()):
        early_df = predicted_from_mapping(early)
        if not early_df.empty:
            early_df["event_class"] = "early_warning"
            base = pd.concat([base, early_df], ignore_index=True)
            base = base.sort_values(["well_id", "detected_time"]).reset_index(drop=True)
    return base


def _attach_predicted_start_status(pred_df: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty or score_df.empty:
        return pred_df
    available = ["well_id", "timestamp", *[name for name in STATUS_COLUMNS if name in score_df.columns]]
    if len(available) <= 2:
        return pred_df
    status_df = score_df[available].copy()
    status_df["well_id"] = status_df["well_id"].astype(str).str.strip().str.lower()
    status_df["timestamp"] = pd.to_datetime(status_df["timestamp"])
    result = pred_df.copy()
    result["well_id"] = result["well_id"].astype(str).str.strip().str.lower()
    result["detected_time"] = pd.to_datetime(result["detected_time"])
    preserved: dict[str, pd.Series] = {}
    for column in STATUS_COLUMNS:
        if column in result.columns:
            preserved[column] = result[column]
            result = result.drop(columns=[column])
    merged = result.merge(
        status_df,
        left_on=["well_id", "detected_time"],
        right_on=["well_id", "timestamp"],
        how="left",
    ).drop(columns=["timestamp"])
    for column, original in preserved.items():
        original = original.reindex(merged.index)
        if column in merged.columns:
            base = merged[column]
            mask_keep = original.fillna("").astype(str).str.strip() != ""
            merged[column] = base.where(~mask_keep, original)
        else:
            merged[column] = original
    return merged


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
    reference_policy: str = REFERENCE_POLICY_NORMAL_WINDOWS,
) -> pd.DataFrame:
    detector_key = normalize_detector_key(detector)
    spec = get_detection_spec(anomaly_key)
    print(f"=== {spec.display_name} Detection [{detector_key}] ===")
    set_seed()
    ensure_dir(DB_DIR)
    output = ensure_parent(Path(output_path) if output_path else results_path(spec, detector_key))
    device = _resolve_torch_device(detector_key, verbose=True)

    if detector_key == "paano_global":
        from alma_service.global_normality import prepare_global_normality_runtime

        runtime = prepare_global_normality_runtime(
            spec.anomaly_key,
            device=device,
            verbose=verbose,
            source_path=source_path,
        )
        prepared_runs = runtime.prepared_runs
        intervals = runtime.intervals
        shared_state = runtime.shared_state
        if verbose:
            print(f"  Global encoder ready: {shared_state.detail}")
        detector_runs = _build_global_runs(
            prepared_runs,
            device=device,
            verbose=verbose,
            shared_state=shared_state,
            intervals=intervals,
            population_pool=runtime.population_pool,
        )
    else:
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
            reference_policy=reference_policy,
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

    precursor_model = _train_or_load_precursor(
        spec=spec,
        detector_key=detector_key,
        train_runs=train_runs,
        train_intervals=train_intervals,
        cfg=cfg,
        retune=retune,
        verbose=verbose,
    )

    score_rows, predicted, early_predicted, detail_map, trend_predicted = _build_score_rows(
        detector_key,
        detector_runs,
        cfg,
        anomaly_key=spec.anomaly_key,
        intervals=intervals,
        precursor_model=precursor_model,
    )
    score_df = pd.DataFrame(score_rows)
    pred_df = _predicted_with_early_warning(predicted, early_predicted)
    pred_df = _attach_predicted_start_status(pred_df, score_df)
    if not pred_df.empty:
        pred_df["anomaly"] = spec.anomaly_key
        pred_df["detector"] = detector_key
        split_lookup = {well_id: run.prepared.split for well_id, run in detector_runs.items()}
        pred_df["split"] = pred_df["well_id"].map(split_lookup).fillna("train")
    pred_df = _tag_trend_starts(pred_df, trend_predicted)
    incident_result = build_incidents(
        pred_df,
        merge_window_hours=_incident_merge_window_hours(cfg),
    )
    pred_df = incident_result.starts
    incident_df = incident_result.incidents
    eval_pred_df = filter_actionable_starts(pred_df)

    split_summaries, split_frames = summarize_splits(
        intervals=intervals,
        predictions=eval_pred_df,
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

    incidents_output_path = incidents_path(spec, detector_key)
    ensure_parent(incidents_output_path)
    write_table(incident_df, incidents_output_path)
    print(f"Incidents saved to {incidents_output_path}")

    summary_payload = {
        "anomaly": spec.anomaly_key,
        "detector": detector_key,
        "config": cfg,
        "prestart_hours": PRESTART_TOLERANCE_HOURS,
        "splits": split_summaries,
        "prediction_postprocess": {
            "raw_starts": int(len(pred_df)),
            "actionable_starts": int(len(eval_pred_df)),
            "suppressed_starts": int(len(pred_df) - len(eval_pred_df)),
            "incidents": int(len(incident_df)),
            "incident_merge_window_hours": _incident_merge_window_hours(cfg),
        },
        "artifacts": {
            "results_path": str(output),
            "scores_path": str(score_output_path),
            "predicted_starts_path": str(pred_output_path),
            "incidents_path": str(incidents_output_path),
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
    save_dir: str | None = None,
    reference_policy: str = REFERENCE_POLICY_NORMAL_WINDOWS,
    normal_reference_fraction: float | None = None,
    use_population_memory_bank: bool = False,
    trusted_local_reference: bool = False,
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
        reference_policy=reference_policy,
        normal_reference_fraction=normal_reference_fraction,
    )
    if prepared is None:
        print("No usable data after engineered preprocessing.")
        return
    if trusted_local_reference:
        selector = dict(prepared.detail.get("normal_window_selector") or {})
        selector["trusted_local_reference"] = True
        selector["trusted_local_reference_source"] = "explicit_single_well_reference"
        prepared.detail["normal_window_selector"] = selector

    device = _resolve_torch_device(detector_key, verbose=True)

    population_reference: np.ndarray | None = None
    if detector_key == "paano_global":
        from alma_service.feature_schema import load_feature_schema, restrict_prepared_to_schema
        from alma_service.global_normality import (
            GLOBAL_ENCODER_KEY,
            build_global_single_runs,
        )
        from alma_service.shared_encoder import load_shared_encoder_state

        schema = load_feature_schema()
        prepared_global, schema_audit = restrict_prepared_to_schema(prepared, schema, strict=False)
        if prepared_global is None:
            raise RuntimeError(
                "Global normality feature schema rejected blind well "
                f"{well_id}: {schema_audit}"
            )
        prepared = prepared_global
        shared_state = load_shared_encoder_state(GLOBAL_ENCODER_KEY, device=device, verbose=True)
        if use_population_memory_bank:
            bank_path = MODELS_DIR / f"population_memory_bank_{detector_key}_{GLOBAL_ENCODER_KEY}.npz"
            if not bank_path.exists():
                raise FileNotFoundError(
                    f"Population memory bank not found: {bank_path}. "
                    f"Build it via scripts/utils/build_population_memory_bank.py."
                )
            bank = np.load(bank_path, allow_pickle=True)
            population_reference = np.asarray(bank["features"], dtype=np.float32)
            print(f"Loaded population memory bank: {bank_path.name} shape={population_reference.shape}")
        detector_runs = build_global_single_runs(
            {well_id: prepared},
            shared_state,
            device,
            verbose=True,
            population_reference=population_reference,
        )
    else:
        from alma_service.shared_encoder import load_shared_encoder_state

        shared_state = load_shared_encoder_state(spec.anomaly_key, device=device, verbose=True)
        if use_population_memory_bank:
            bank_path = MODELS_DIR / f"population_memory_bank_{detector_key}_{spec.anomaly_key}.npz"
            if not bank_path.exists():
                raise FileNotFoundError(
                    f"Population memory bank not found: {bank_path}. "
                    f"Build it via scripts/utils/build_population_memory_bank.py."
                )
            bank = np.load(bank_path, allow_pickle=True)
            population_reference = np.asarray(bank["features"], dtype=np.float32)
            print(f"Loaded population memory bank: {bank_path.name} shape={population_reference.shape}")
        detector_runs = _build_local_runs(
            spec.anomaly_key,
            detector_key,
            {well_id: prepared},
            device=device,
            verbose=True,
            shared_state=shared_state,
            population_reference=population_reference,
        )
    run = detector_runs[well_id]

    cfg_payload = load_json(config_path(spec, detector_key))
    cfg = {**_default_onset_config(spec.anomaly_key, detector_key), **(cfg_payload.get("config", cfg_payload) if cfg_payload else {})}
    precursor_model = load_precursor_model(precursor_path(spec, detector_key))

    if save_dir is not None:
        score_rows, predicted, early_predicted, _detail_map, trend_predicted = _build_score_rows(
            detector_key,
            {well_id: run},
            cfg,
            anomaly_key=spec.anomaly_key,
            intervals=intervals,
            precursor_model=precursor_model,
        )
        score_df = pd.DataFrame(score_rows)
        pred_df = _predicted_with_early_warning(predicted, early_predicted)
        pred_df = _attach_predicted_start_status(pred_df, score_df)
        if not pred_df.empty:
            pred_df["anomaly"] = spec.anomaly_key
            pred_df["detector"] = detector_key
            pred_df["split"] = split
        pred_df = _tag_trend_starts(pred_df, trend_predicted)
        incident_result = build_incidents(
            pred_df,
            merge_window_hours=_incident_merge_window_hours(cfg),
        )
        pred_df = incident_result.starts
        incident_df = incident_result.incidents
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_table(score_df, out_dir / "scores.parquet")
        write_table(pred_df, out_dir / "predicted_starts.parquet")
        write_table(incident_df, out_dir / "incidents.parquet")
        starts_list = [str(pd.Timestamp(ts)) for ts in predicted.get(well_id, [])]
        summary = {
            "anomaly": spec.anomaly_key,
            "well_id": well_id,
            "detector": detector_key,
            "split": split,
            "n_points": int(len(score_df)),
            "n_detected": len(starts_list),
            "n_actionable_detected": int(pred_df["actionable_alert"].sum()) if "actionable_alert" in pred_df else len(starts_list),
            "n_incidents": int(len(incident_df)),
            "detected_starts": starts_list,
            "score_min": float(score_df["score"].min()) if not score_df.empty else None,
            "score_median": float(score_df["score"].median()) if not score_df.empty else None,
            "score_max": float(score_df["score"].max()) if not score_df.empty else None,
            "time_start": str(score_df["timestamp"].min()) if not score_df.empty else None,
            "time_end": str(score_df["timestamp"].max()) if not score_df.empty else None,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Single-well outputs saved to {out_dir}")
        print(f"Detected starts: {starts_list}")
        return

    score, thresholds, starts, early_starts = _detect_starts_for_run(
        detector_key, run, cfg, precursor_model=precursor_model
    )
    print(f"Prepared detail: {json.dumps(run.prepared.detail, ensure_ascii=False, indent=2)}")
    print(f"Detector detail: {json.dumps(run.score_output.detail, ensure_ascii=False, indent=2)}")
    print(f"Thresholds: {json.dumps(asdict(thresholds), ensure_ascii=False, indent=2)}")
    print(f"Detected starts: {[pd.Timestamp(ts) for ts in starts]}")
    if early_starts:
        print(f"Early warnings: {[pd.Timestamp(ts) for ts in early_starts]}")
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
    parser.add_argument("--reference-policy", choices=sorted(REFERENCE_POLICIES), default=REFERENCE_POLICY_NORMAL_WINDOWS)
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
            reference_policy=args.reference_policy,
        )
        return
    run_detection(
        anomaly_key=args.anomaly,
        detector=args.detector,
        output_path=args.output,
        source_path=args.source,
        retune=args.retune,
        verbose=True,
        reference_policy=args.reference_policy,
    )


if __name__ == "__main__":
    main()
