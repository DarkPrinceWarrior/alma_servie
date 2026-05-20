from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from alma_service.anomaly_specs import get_detection_spec
from alma_service.engineered_features import (
    ANOMALY_PATCH_SIZE,
    REFERENCE_POLICY_NORMAL_WINDOWS,
)
from alma_service.feature_schema import (
    DEFAULT_GLOBAL_SCHEMA_PATH,
    FeatureSchema,
    load_feature_schema,
    restrict_prepared_mapping_to_schema,
)
from alma_service.generic_detectors import (
    DetectorScoreOutput,
    PAANO_INPUT_PADDING_EDGE_HOLD,
    PAANO_INPUT_PADDING_ENV,
    PAANO_INPUT_PADDING_MODES,
    SharedPaAnoDetector,
)
from alma_service.paano_defaults import PATCH_SIZES
from alma_service.shared_encoder import (
    collect_shared_train_pool,
    load_or_train_shared_encoder,
    select_shared_columns,
)


ANOMALY_KEYS = ("negermet", "pritok", "salt")
GLOBAL_DETECTOR_KEY = "paano_global"
GLOBAL_ENCODER_KEY = "global_normality"
DEFAULT_GLOBAL_CONFIG_PATH = Path("configs/alma_global_normality_5min.json")
DEFAULT_ANOMALY_FREQ = {
    "negermet": "2min",
    "pritok": "10min",
    "salt": "15min",
}
DEFAULT_NORM_WORK_PROFILE = "pritok"
DEFAULT_BALANCE_ROWS_PER_SOURCE = 48_000
DEFAULT_BALANCE_ROWS_PER_WELL = 8_000
DEFAULT_BALANCE_SOURCE_POLICY = "equal_min"
GLOBAL_MEMORY_BANK_MODE_ENV = "ALMA_GLOBAL_MEMORY_BANK_MODE"
GLOBAL_MEMORY_BANK_LOCAL = "local"
GLOBAL_MEMORY_BANK_POPULATION_FALLBACK = "population_fallback"
GLOBAL_MEMORY_BANK_MODES = {
    GLOBAL_MEMORY_BANK_LOCAL,
    GLOBAL_MEMORY_BANK_POPULATION_FALLBACK,
}


@dataclass(frozen=True)
class GlobalNormalitySettings:
    config_path: Path
    common_source_freq: str
    patch_short: int
    patch_long: int
    global_iters: int
    feature_schema_path: Path
    include_norm_work: bool
    norm_work_profile: str
    reference_policy: str
    balance_enabled: bool
    balance_source_policy: str
    balance_rows_per_source: int
    balance_rows_per_well: int
    balance_seed: int
    prepare_patch_size_overrides: dict[str, int]
    enable_feature_reduction: bool


@dataclass(frozen=True)
class GlobalNormalityRuntime:
    prepared_runs: dict[str, Any]
    intervals: pd.DataFrame
    shared_state: Any
    population_pool: dict[str, Any]
    detail: dict[str, Any]


def load_global_normality_settings(path: str | Path = DEFAULT_GLOBAL_CONFIG_PATH) -> GlobalNormalitySettings:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    common_source_freq = str(payload.get("common_source_freq", "5min"))
    balance = dict(payload.get("balance", {}))
    overrides = {
        str(key): int(value)
        for key, value in dict(payload.get("prepare_patch_size_overrides") or {}).items()
    }
    if not overrides:
        overrides = _prepare_patch_overrides_for_common_freq(common_source_freq)
    return GlobalNormalitySettings(
        config_path=config_path,
        common_source_freq=common_source_freq,
        patch_short=int(payload.get("patch_short", 192)),
        patch_long=int(payload.get("patch_long", 384)),
        global_iters=int(payload.get("global_iters", 200)),
        feature_schema_path=Path(payload.get("feature_schema", DEFAULT_GLOBAL_SCHEMA_PATH)),
        include_norm_work=bool(payload.get("include_norm_work", True)),
        norm_work_profile=str(payload.get("norm_work_profile", DEFAULT_NORM_WORK_PROFILE)),
        reference_policy=str(payload.get("reference_policy", REFERENCE_POLICY_NORMAL_WINDOWS)),
        balance_enabled=bool(balance.get("enabled", True)),
        balance_source_policy=str(balance.get("source_policy", DEFAULT_BALANCE_SOURCE_POLICY)),
        balance_rows_per_source=int(balance.get("rows_per_source", DEFAULT_BALANCE_ROWS_PER_SOURCE)),
        balance_rows_per_well=int(balance.get("rows_per_well", DEFAULT_BALANCE_ROWS_PER_WELL)),
        balance_seed=int(balance.get("seed", 20260518)),
        prepare_patch_size_overrides=overrides,
        enable_feature_reduction=bool(payload.get("enable_feature_reduction", False)),
    )


def configured_anomaly_source_path(
    anomaly_key: str,
    config_path: str | Path = DEFAULT_GLOBAL_CONFIG_PATH,
) -> Path:
    settings = load_global_normality_settings(config_path)
    source = _anomaly_source_path(anomaly_key, settings.common_source_freq)
    if source is None:
        raise ValueError("Global normality config must define common_source_freq.")
    return source


def prepare_global_normality_runtime(
    anomaly_key: str,
    *,
    device: torch.device,
    verbose: bool = False,
    config_path: str | Path = DEFAULT_GLOBAL_CONFIG_PATH,
    source_path: str | None = None,
) -> GlobalNormalityRuntime:
    settings = load_global_normality_settings(config_path)
    if anomaly_key not in ANOMALY_KEYS:
        raise ValueError(f"Unsupported global normality anomaly: {anomaly_key}")

    schema = load_feature_schema(settings.feature_schema_path)
    with _temporary_prepare_patch_sizes(settings.prepare_patch_size_overrides):
        prepared_by_class, intervals_by_class = _prepare_by_class(
            settings.reference_policy,
            common_source_freq=settings.common_source_freq,
            source_overrides={anomaly_key: source_path} if source_path else None,
        )
        prepared_by_class, schema_audit_by_class = _apply_feature_schema_by_class(
            prepared_by_class,
            schema,
            strict=True,
        )
        global_pool = _global_train_pool(prepared_by_class)
        if settings.include_norm_work:
            global_pool = _add_norm_work_pool(
                global_pool,
                reference_policy=settings.reference_policy,
                common_source_freq=settings.common_source_freq,
                norm_work_profile=settings.norm_work_profile,
            )
    global_pool, schema_audit_global_pool = _apply_feature_schema_to_pool(
        global_pool,
        schema,
        strict=True,
    )
    balance_audit: dict[str, Any] = {"enabled": False}
    if settings.balance_enabled:
        global_pool, balance_audit = _apply_balanced_reference_pool(
            global_pool,
            max_rows_per_source=settings.balance_rows_per_source,
            max_rows_per_well=settings.balance_rows_per_well,
            source_policy=settings.balance_source_policy,
            seed=settings.balance_seed,
        )

    pool, shared_channels, train_wells = collect_shared_train_pool(
        global_pool,
        enable_reduction=settings.enable_feature_reduction,
    )
    if verbose:
        print(
            "  Global normality pool: "
            f"{len(pool)} points, {len(shared_channels)} channels, {len(train_wells)} entries"
        )

    shared_state = load_or_train_shared_encoder(
        prepared_wells=global_pool,
        patch_short=settings.patch_short,
        patch_long=settings.patch_long,
        anomaly_key=GLOBAL_ENCODER_KEY,
        device=device,
        verbose=verbose,
        num_iter=settings.global_iters,
        enable_reduction=settings.enable_feature_reduction,
    )
    detail = {
        "detector": GLOBAL_DETECTOR_KEY,
        "encoder_key": GLOBAL_ENCODER_KEY,
        "config_path": str(settings.config_path),
        "common_source_freq": settings.common_source_freq,
        "patch_short": settings.patch_short,
        "patch_long": settings.patch_long,
        "global_iters": settings.global_iters,
        "feature_schema": schema.to_detail(),
        "include_norm_work": settings.include_norm_work,
        "norm_work_profile": settings.norm_work_profile if settings.include_norm_work else None,
        "reference_policy": settings.reference_policy,
        "feature_reduction_enabled": settings.enable_feature_reduction,
        "balance_audit": balance_audit,
        "schema_audit_by_class": schema_audit_by_class,
        "schema_audit_global_pool": schema_audit_global_pool,
        "global_pool_points": int(len(pool)),
        "global_shared_channels": int(len(shared_channels)),
        "global_train_entries": train_wells,
        "global_state_detail": shared_state.detail,
    }
    return GlobalNormalityRuntime(
        prepared_runs=prepared_by_class[anomaly_key],
        intervals=intervals_by_class[anomaly_key],
        shared_state=shared_state,
        population_pool=global_pool,
        detail=detail,
    )


def build_global_core_runs(
    prepared_runs: dict[str, Any],
    shared_state: Any,
    device: torch.device,
    *,
    verbose: bool,
    labelled_wells: set[str] | None = None,
    population_pool: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from alma_service.generic_detection import PreparedDetectorRun

    detector_runs: dict[str, Any] = {}
    memory_bank_mode = _global_memory_bank_mode()
    input_padding_mode = _global_paano_input_padding_mode()
    for well_id, prepared in prepared_runs.items():
        normalized_well_id = _normalize_well_id(well_id)
        x_projected = select_shared_columns(
            prepared.feature_columns,
            prepared.feature_matrix,
            shared_state.shared_channels,
        )
        local_ref = x_projected[np.asarray(prepared.reference_mask, dtype=bool)]
        is_labelled = labelled_wells is None or normalized_well_id in labelled_wells
        use_population = (
            memory_bank_mode == GLOBAL_MEMORY_BANK_POPULATION_FALLBACK
            and ((not is_labelled) or len(local_ref) < shared_state.patch_long * 2)
        )

        population_ref: np.ndarray | None = None
        if use_population and population_pool:
            population_ref = _population_reference_matrix(
                population_pool,
                shared_state.shared_channels,
                exclude_well_id=normalized_well_id,
            )

        if labelled_wells is not None and normalized_well_id not in labelled_wells:
            if population_ref is not None and len(population_ref) > 0:
                if verbose:
                    print(
                        f"    Use population memory bank for blind well {well_id}: "
                        f"population_ref={len(population_ref)}"
                    )
            else:
                detector_runs[well_id] = PreparedDetectorRun(
                    prepared=prepared,
                    score_output=_unavailable_global_output(
                        prepared=prepared,
                        reason="unlabeled_no_reference",
                        input_contract="no_local_reference",
                        detail={
                            "blind_unlabeled": True,
                            "local_reference_used": False,
                            "population_memory_used": False,
                            "memory_bank_source": "none",
                            "memory_bank_mode": memory_bank_mode,
                            "local_reference_points": int(len(local_ref)),
                            "population_reference_points": 0,
                        },
                    ),
                )
                if verbose:
                    print(
                        f"    Skip scoring blind well {well_id}: "
                        "no labelled normal interval and no population memory bank"
                    )
                continue

        if use_population:
            if population_ref is None or len(population_ref) == 0:
                detector_runs[well_id] = PreparedDetectorRun(
                    prepared=prepared,
                    score_output=_unavailable_global_output(
                        prepared=prepared,
                        reason="population_reference_unavailable",
                        input_contract="no_population_reference",
                        detail={
                            "blind_unlabeled": not is_labelled,
                            "local_reference_used": False,
                            "population_memory_used": False,
                            "memory_bank_source": "none",
                            "memory_bank_mode": memory_bank_mode,
                            "local_reference_points": int(len(local_ref)),
                            "population_reference_points": 0,
                        },
                    ),
                )
                if verbose:
                    print(
                        f"    Skip scoring {well_id}: local_ref={len(local_ref)}, "
                        "population memory bank unavailable"
                    )
                continue
            reference = population_ref
            memory_bank_source = "population"
        else:
            reference = local_ref
            memory_bank_source = "local_reference"

        if labelled_wells is not None and normalized_well_id not in labelled_wells:
            if verbose:
                print(
                    f"    Score blind well {well_id} with population memory bank "
                    f"({len(reference)} reference points)"
                )
        elif use_population and verbose:
            print(
                f"    Use population memory bank for {well_id}: "
                f"local_ref={len(local_ref)}, population_ref={len(reference)}"
            )

        detector = SharedPaAnoDetector(
            shared_state=shared_state,
            device=device,
            verbose=verbose,
            input_padding_mode=input_padding_mode,
        )
        detector.fit_reference(reference)
        raw_output = detector.score_stream(x_projected, mask_all=prepared.stability_mask)
        primary = np.asarray(raw_output.primary, dtype=np.float32)
        raw_contract = str(raw_output.detail.get("input_contract", "real_window"))
        score_output = DetectorScoreOutput(
            primary=primary,
            components={
                "global_paano_score": primary,
                **raw_output.components,
            },
            detail={
                **raw_output.detail,
                "global_normality_detector": True,
                "class_fine_tune": False,
                "physical_branches": False,
                "blind_unlabeled": not is_labelled,
                "local_reference_used": memory_bank_source == "local_reference",
                "population_memory_used": memory_bank_source == "population",
                "memory_bank_source": memory_bank_source,
                "memory_bank_mode": memory_bank_mode,
                "local_reference_points": int(len(local_ref)),
                "population_reference_points": int(len(reference)) if memory_bank_source == "population" else 0,
                "window_input_contract": raw_contract,
                "input_contract": _compose_input_contract(raw_contract, memory_bank_source),
            },
        )
        detector_runs[well_id] = PreparedDetectorRun(
            prepared=prepared,
            score_output=score_output,
        )
    return detector_runs


def _unavailable_global_output(
    *,
    prepared: Any,
    reason: str,
    input_contract: str,
    detail: dict[str, Any],
) -> DetectorScoreOutput:
    zeros = np.zeros(len(prepared.timestamps), dtype=np.float32)
    return DetectorScoreOutput(
        primary=zeros,
        components={
            "score": zeros.copy(),
            "paano_score": zeros.copy(),
            "global_paano_score": zeros.copy(),
        },
        detail={
            "reason": reason,
            "input_contract": input_contract,
            "global_normality_detector": True,
            "class_fine_tune": False,
            "physical_branches": False,
            **detail,
        },
    )


def _population_reference_matrix(
    population_pool: dict[str, Any],
    shared_channels: list[str],
    *,
    exclude_well_id: str,
) -> np.ndarray | None:
    parts: list[np.ndarray] = []
    for pool_key, prepared in sorted(population_pool.items()):
        if _normalize_well_id(getattr(prepared, "well_id", "")) == exclude_well_id:
            continue
        projected = select_shared_columns(
            prepared.feature_columns,
            prepared.feature_matrix,
            shared_channels,
        )
        ref = projected[np.asarray(prepared.reference_mask, dtype=bool)]
        if len(ref) > 0:
            parts.append(ref.astype(np.float32))
    if not parts:
        return None
    return np.concatenate(parts, axis=0).astype(np.float32)


def _compose_input_contract(window_contract: str, memory_bank_source: str) -> str:
    prefix = "padded_long" if window_contract == "edge_hold_padded" else "real_long"
    suffix = "population_memory" if memory_bank_source == "population" else "local_memory"
    return f"{prefix}_{suffix}"


def _global_memory_bank_mode() -> str:
    mode = os.getenv(GLOBAL_MEMORY_BANK_MODE_ENV, GLOBAL_MEMORY_BANK_LOCAL).strip().lower()
    if mode not in GLOBAL_MEMORY_BANK_MODES:
        raise ValueError(
            f"Unsupported {GLOBAL_MEMORY_BANK_MODE_ENV}={mode!r}. "
            f"Expected one of: {', '.join(sorted(GLOBAL_MEMORY_BANK_MODES))}."
        )
    return mode


def _global_paano_input_padding_mode() -> str:
    mode = os.getenv(PAANO_INPUT_PADDING_ENV, PAANO_INPUT_PADDING_EDGE_HOLD).strip().lower()
    if mode not in PAANO_INPUT_PADDING_MODES:
        raise ValueError(
            f"Unsupported {PAANO_INPUT_PADDING_ENV}={mode!r}. "
            f"Expected one of: {', '.join(sorted(PAANO_INPUT_PADDING_MODES))}."
        )
    return mode


def _normalize_well_id(value: Any) -> str:
    return str(value).strip().lower()


def _freq_label(freq: str) -> str:
    return str(freq).replace(" ", "")


def _anomaly_source_path(anomaly_key: str, common_source_freq: str | None) -> Path | None:
    if common_source_freq is None:
        return None
    return Path("db") / f"{anomaly_key}_anomaly_database_{_freq_label(common_source_freq)}.parquet"


def _norm_work_source_path(anomaly_key: str, common_source_freq: str | None) -> Path:
    freq = common_source_freq or DEFAULT_ANOMALY_FREQ[anomaly_key]
    return Path("db") / f"norm_work_database_{_freq_label(freq)}.parquet"


def _freq_seconds(freq: str) -> float:
    return float(pd.Timedelta(str(freq)).total_seconds())


def _prepare_patch_overrides_for_common_freq(common_source_freq: str | None) -> dict[str, int]:
    if common_source_freq is None:
        return {}
    common_seconds = _freq_seconds(common_source_freq)
    if common_seconds <= 0:
        raise ValueError(f"Invalid common_source_freq: {common_source_freq}")
    overrides: dict[str, int] = {}
    for anomaly_key, base_freq in DEFAULT_ANOMALY_FREQ.items():
        base_seconds = _freq_seconds(base_freq)
        base_long_patch = int(PATCH_SIZES[anomaly_key][1])
        physical_seconds = base_long_patch * base_seconds
        overrides[anomaly_key] = max(8, int(math.ceil(physical_seconds / common_seconds)))
    return overrides


@contextmanager
def _temporary_prepare_patch_sizes(overrides: dict[str, int]):
    if not overrides:
        yield
        return
    original = {key: ANOMALY_PATCH_SIZE.get(key) for key in overrides}
    ANOMALY_PATCH_SIZE.update({key: int(value) for key, value in overrides.items()})
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                ANOMALY_PATCH_SIZE.pop(key, None)
            else:
                ANOMALY_PATCH_SIZE[key] = int(value)


def _stable_seed_offset(value: str) -> int:
    total = 0
    for idx, char in enumerate(str(value)):
        total += (idx + 1) * ord(char)
    return total % 1_000_000


def _pool_source_from_key(pool_key: str) -> str:
    if pool_key.startswith("norm_work:"):
        return "norm_work"
    return pool_key.split(":", 1)[0]


def _prepare_by_class(
    reference_policy: str,
    *,
    common_source_freq: str | None,
    source_overrides: dict[str, str | None] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    from alma_service.generic_detection import _prepare_all_wells, load_anomaly_data, load_intervals

    prepared_by_class: dict[str, dict[str, Any]] = {}
    intervals_by_class: dict[str, pd.DataFrame] = {}
    for anomaly_key in ANOMALY_KEYS:
        spec = get_detection_spec(anomaly_key)
        source_path = (
            Path(source_overrides[anomaly_key])
            if source_overrides and source_overrides.get(anomaly_key)
            else _anomaly_source_path(anomaly_key, common_source_freq)
        )
        if source_path is not None and not source_path.exists():
            raise FileNotFoundError(f"Common-frequency source not found: {source_path}")
        df = load_anomaly_data(spec, source_path=str(source_path) if source_path is not None else None)
        intervals = _first_intervals(load_intervals(spec, required=True))
        prepared_by_class[anomaly_key] = _prepare_all_wells(
            spec,
            df,
            intervals,
            verbose=False,
            zone_aware=True,
            reference_policy=reference_policy,
        )
        intervals_by_class[anomaly_key] = intervals
    return prepared_by_class, intervals_by_class


def _first_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    return (
        intervals.sort_values(["well_id", "start_date", "interval_idx"])
        .groupby("well_id", as_index=False)
        .first()
    )


def _apply_feature_schema_by_class(
    prepared_by_class: dict[str, dict[str, Any]],
    schema: FeatureSchema,
    *,
    strict: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    filtered_by_class: dict[str, dict[str, Any]] = {}
    audit_by_class: dict[str, dict[str, dict[str, Any]]] = {}
    for anomaly_key, prepared in prepared_by_class.items():
        filtered, audit = restrict_prepared_mapping_to_schema(
            prepared,
            schema,
            strict=strict,
        )
        filtered_by_class[anomaly_key] = filtered
        audit_by_class[anomaly_key] = audit
    return filtered_by_class, audit_by_class


def _global_train_pool(prepared_by_class: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for anomaly_key, prepared in prepared_by_class.items():
        for well_id, item in prepared.items():
            if item.split == "train":
                out[f"{anomaly_key}:{well_id}"] = item
    return out


def _add_norm_work_pool(
    global_pool: dict[str, Any],
    *,
    reference_policy: str,
    common_source_freq: str | None,
    norm_work_profile: str,
) -> dict[str, Any]:
    from alma_service.generic_detection import _prepare_all_wells

    norm_work_profile = str(norm_work_profile).strip()
    if norm_work_profile not in ANOMALY_KEYS:
        raise ValueError(f"Unknown norm_work_profile: {norm_work_profile}")
    source = _norm_work_source_path(norm_work_profile, common_source_freq)
    if not source.exists():
        print(f"  Norm work skipped: missing {source}")
        return global_pool
    spec = get_detection_spec(norm_work_profile)
    df = pd.read_parquet(source)
    if df.empty:
        return global_pool
    intervals = pd.DataFrame({
        "well_id": sorted(df["well_id"].astype(str).unique()),
        "split": "train",
        "start_date": pd.NaT,
        "end_date": pd.NaT,
        "interval_idx": 0,
    })
    prepared = _prepare_all_wells(
        spec,
        df,
        intervals,
        verbose=False,
        zone_aware=False,
        reference_policy=reference_policy,
    )
    for well_id, item in prepared.items():
        reference_mask = np.asarray(item.stability_mask, dtype=bool).copy()
        if not reference_mask.any():
            reference_mask = np.ones(len(item.timestamps), dtype=bool)
        detail = dict(item.detail)
        detail["normal_work_source"] = str(source)
        detail["norm_work_profile"] = norm_work_profile
        detail["reference_policy"] = "full_norm_work_series"
        detail["reference_points"] = int(reference_mask.sum())
        global_pool[f"norm_work:{well_id}"] = replace(
            item,
            split="train",
            reference_mask=reference_mask,
            reference_end_idx=len(item.timestamps),
            onset_allowed_mask=np.zeros(len(item.timestamps), dtype=bool),
            detail=detail,
        )
    return global_pool


def _apply_feature_schema_to_pool(
    global_pool: dict[str, Any],
    schema: FeatureSchema,
    *,
    strict: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    return restrict_prepared_mapping_to_schema(global_pool, schema, strict=strict)


def _balanced_mask_for_positions(
    positions: np.ndarray,
    *,
    max_count: int,
    seed: int,
) -> np.ndarray:
    if max_count <= 0 or len(positions) <= max_count:
        return positions
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(positions, size=int(max_count), replace=False)).astype(int)


def _apply_balanced_reference_pool(
    global_pool: dict[str, Any],
    *,
    max_rows_per_source: int,
    max_rows_per_well: int,
    source_policy: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    per_well: dict[str, dict[str, Any]] = {}
    selected_positions_by_key: dict[str, np.ndarray] = {}
    source_counts_after_well_cap: dict[str, int] = {}

    for idx, (pool_key, prepared) in enumerate(sorted(global_pool.items())):
        source = _pool_source_from_key(pool_key)
        positions = np.flatnonzero(np.asarray(prepared.reference_mask, dtype=bool))
        selected = _balanced_mask_for_positions(
            positions,
            max_count=int(max_rows_per_well),
            seed=int(seed + idx * 10_007),
        )
        selected_positions_by_key[pool_key] = selected
        source_counts_after_well_cap[source] = source_counts_after_well_cap.get(source, 0) + int(len(selected))
        per_well[pool_key] = {
            "source": source,
            "well_id": prepared.well_id,
            "split": prepared.split,
            "reference_rows_before": int(len(positions)),
            "reference_rows_after_well_cap": int(len(selected)),
            "reference_rows_after_source_cap": 0,
        }

    source_policy = str(source_policy).strip().lower()
    if source_policy not in {"equal_min", "cap"}:
        raise ValueError(f"Unknown balance source policy: {source_policy}")

    positive_counts = [count for count in source_counts_after_well_cap.values() if count > 0]
    equal_min_target = min(positive_counts) if positive_counts else 0
    source_rows: dict[str, list[tuple[str, int]]] = {}
    for pool_key, selected in selected_positions_by_key.items():
        source = _pool_source_from_key(pool_key)
        source_rows.setdefault(source, []).extend((pool_key, int(position)) for position in selected)

    selected_by_source: dict[str, set[tuple[str, int]]] = {}
    source_targets: dict[str, int] = {}
    for source, rows in source_rows.items():
        if source_policy == "equal_min":
            target = int(equal_min_target)
            if int(max_rows_per_source) > 0:
                target = min(target, int(max_rows_per_source))
        else:
            target = int(max_rows_per_source) if int(max_rows_per_source) > 0 else len(rows)
        source_targets[source] = target
        if target > 0 and len(rows) > target:
            rng = np.random.default_rng(seed + _stable_seed_offset(source))
            selected_idx = rng.choice(np.arange(len(rows)), size=target, replace=False)
            selected_by_source[source] = {rows[int(i)] for i in selected_idx}
        else:
            selected_by_source[source] = set(rows)

    balanced_pool: dict[str, Any] = {}
    source_counts_final: dict[str, int] = {}
    for pool_key, prepared in sorted(global_pool.items()):
        source = _pool_source_from_key(pool_key)
        selected_pairs = selected_by_source.get(source, set())
        selected_positions = [
            position
            for position in selected_positions_by_key.get(pool_key, np.array([], dtype=int))
            if (pool_key, int(position)) in selected_pairs
        ]
        mask = np.zeros(len(prepared.timestamps), dtype=bool)
        if selected_positions:
            mask[np.asarray(selected_positions, dtype=int)] = True
        if not mask.any():
            continue
        detail = dict(prepared.detail)
        detail["balanced_reference_pool"] = {
            "source": source,
            "reference_rows_before": int(np.asarray(prepared.reference_mask, dtype=bool).sum()),
            "reference_rows_after": int(mask.sum()),
            "max_rows_per_well": int(max_rows_per_well),
            "max_rows_per_source": int(max_rows_per_source),
            "source_policy": source_policy,
            "source_target": int(source_targets.get(source, 0)),
        }
        balanced_pool[pool_key] = replace(prepared, reference_mask=mask, detail=detail)
        source_counts_final[source] = source_counts_final.get(source, 0) + int(mask.sum())
        per_well[pool_key]["reference_rows_after_source_cap"] = int(mask.sum())

    return balanced_pool, {
        "enabled": True,
        "source_policy": source_policy,
        "max_rows_per_source": int(max_rows_per_source),
        "max_rows_per_well": int(max_rows_per_well),
        "seed": int(seed),
        "source_counts_after_well_cap": source_counts_after_well_cap,
        "source_targets": source_targets,
        "source_counts_final": source_counts_final,
        "per_well": per_well,
    }
