from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.anomaly_specs import get_detection_spec
from alma_service.benchmark_metrics import predicted_from_mapping, summarize_splits
from alma_service.detection_artifacts import summary_path
from alma_service.domain_decision_layer import (
    attach_domain_decisions_to_incidents,
    attach_domain_decisions_to_starts,
)
from alma_service.domain_rule_diagnostics import attach_domain_rule_diagnostics
from alma_service.engineered_features import (
    ANOMALY_PATCH_SIZE,
    REFERENCE_POLICIES,
    REFERENCE_POLICY_NORMAL_WINDOWS,
)
from alma_service.feature_schema import (
    DEFAULT_GLOBAL_SCHEMA_PATH,
    FeatureSchema,
    load_feature_schema,
    restrict_prepared_mapping_to_schema,
)
from alma_service.generic_detection import (
    PRESTART_TOLERANCE_HOURS,
    PreparedDetectorRun,
    _attach_predicted_start_status,
    _build_score_rows,
    _incident_merge_window_hours,
    _load_or_build_config,
    _predicted_with_early_warning,
    _prepare_all_wells,
    _resolve_torch_device,
    _tune_config,
    load_anomaly_data,
    load_intervals,
)
from alma_service.generic_detectors import (
    DetectorScoreOutput,
    PAANO_INPUT_PADDING_EDGE_HOLD,
    PAANO_INPUT_PADDING_ENV,
    PAANO_INPUT_PADDING_MODES,
    SharedPaAnoDetector,
)
from alma_service.global_normality import (
    _apply_norm_pool_hygiene,
    load_global_normality_settings,
)
from alma_service.paano_defaults import PATCH_SIZES
from alma_service.prediction_postprocess import build_incidents, filter_actionable_starts
from alma_service.shared_encoder import (
    collect_shared_train_pool,
    select_shared_columns,
    train_shared_encoder,
)


ANOMALY_KEYS = ("negermet", "pritok", "salt")
DEFAULT_ANOMALY_FREQ = {
    "negermet": "2min",
    "pritok": "10min",
    "salt": "15min",
}
DEFAULT_BALANCE_ROWS_PER_SOURCE = 48_000
DEFAULT_BALANCE_ROWS_PER_WELL = 8_000
DEFAULT_BALANCE_SOURCE_POLICY = "equal_min"
DEFAULT_NORM_WORK_PROFILE = "pritok"
BALANCED_REFERENCE_MASK_DETAIL_KEY = "balanced_reference_mask"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _pool_source_from_key(pool_key: str) -> str:
    if pool_key.startswith("norm_work:"):
        return "norm_work"
    return pool_key.split(":", 1)[0]


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


def _balanced_mask_for_positions(
    positions: np.ndarray,
    *,
    max_count: int,
    seed: int,
) -> np.ndarray:
    if max_count <= 0 or len(positions) <= max_count:
        return positions
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(positions, size=int(max_count), replace=False))
    return selected.astype(int)


def _apply_balanced_reference_pool(
    global_pool: dict[str, Any],
    *,
    max_rows_per_source: int,
    max_rows_per_well: int,
    source_policy: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Limit reference rows per well and per source without changing timelines.

    The detector still sees each full well during scoring. Only rows eligible
    for global encoder training are downsampled through ``reference_mask``.
    """
    per_well: dict[str, dict[str, Any]] = {}
    source_counts_after_well_cap: dict[str, int] = {}
    selected_positions_by_key: dict[str, np.ndarray] = {}

    for idx, (pool_key, prepared) in enumerate(sorted(global_pool.items())):
        source = _pool_source_from_key(pool_key)
        positions = np.flatnonzero(np.asarray(prepared.reference_mask, dtype=bool))
        before = int(len(positions))
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
            "reference_rows_before": before,
            "reference_rows_after_well_cap": int(len(selected)),
            "reference_rows_after_source_cap": 0,
        }

    source_policy = str(source_policy).strip().lower()
    if source_policy not in {"equal_min", "cap"}:
        raise ValueError(f"Unknown balance source policy: {source_policy}")

    positive_counts = [count for count in source_counts_after_well_cap.values() if count > 0]
    equal_min_target = min(positive_counts) if positive_counts else 0

    selected_by_source: dict[str, set[tuple[str, int]]] = {}
    source_targets: dict[str, int] = {}
    source_rows: dict[str, list[tuple[str, int]]] = {}
    for pool_key, selected in selected_positions_by_key.items():
        source = _pool_source_from_key(pool_key)
        source_rows.setdefault(source, []).extend((pool_key, int(position)) for position in selected)

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
            selected_pairs = {rows[int(i)] for i in selected_idx}
        else:
            selected_pairs = set(rows)
        selected_by_source[source] = selected_pairs

    balanced_pool: dict[str, Any] = {}
    source_counts_final: dict[str, int] = {}
    for pool_key, prepared in sorted(global_pool.items()):
        source = _pool_source_from_key(pool_key)
        selected_positions = sorted(
            position
            for key, position in selected_by_source.get(source, set())
            if key == pool_key
        )
        balanced_mask = np.zeros(len(prepared.timestamps), dtype=bool)
        if selected_positions:
            balanced_mask[np.asarray(selected_positions, dtype=int)] = True
        detail = dict(prepared.detail)
        detail[BALANCED_REFERENCE_MASK_DETAIL_KEY] = {
            "enabled": True,
            "source": source,
            "max_rows_per_source": int(max_rows_per_source),
            "max_rows_per_well": int(max_rows_per_well),
            "source_policy": source_policy,
            "reference_rows_before": int(per_well[pool_key]["reference_rows_before"]),
            "reference_rows_after_well_cap": int(per_well[pool_key]["reference_rows_after_well_cap"]),
            "reference_rows_after_source_cap": int(balanced_mask.sum()),
            "seed": int(seed),
        }
        per_well[pool_key]["reference_rows_after_source_cap"] = int(balanced_mask.sum())
        source_counts_final[source] = source_counts_final.get(source, 0) + int(balanced_mask.sum())
        balanced_pool[pool_key] = replace(
            prepared,
            reference_mask=balanced_mask,
            detail=detail,
        )

    audit = {
        "enabled": True,
        "seed": int(seed),
        "max_rows_per_source": int(max_rows_per_source),
        "max_rows_per_well": int(max_rows_per_well),
        "source_policy": source_policy,
        "source_targets": {source: int(target) for source, target in sorted(source_targets.items())},
        "sources": {
            source: {
                "rows_after_well_cap": int(source_counts_after_well_cap.get(source, 0)),
                "rows_after_source_cap": int(source_counts_final.get(source, 0)),
                "well_count": int(sum(1 for item in per_well.values() if item["source"] == source)),
            }
            for source in sorted(source_rows)
        },
        "wells": per_well,
        "total_rows_after_source_cap": int(sum(source_counts_final.values())),
    }
    return balanced_pool, audit


def _compact_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "interval_count": summary.get("interval_count"),
        "hit_count": summary.get("hit_count"),
        "hit_rate": summary.get("hit_rate"),
        "false_alarms_per_day": summary.get("false_alarms_per_day"),
        "start_count": summary.get("start_count"),
        "median_abs_delay_hours": summary.get("median_abs_delay_hours"),
        "p90_abs_delay_hours": summary.get("p90_abs_delay_hours"),
        "p90_delay_ratio": summary.get("p90_delay_ratio"),
    }


def _global_paano_input_padding_mode() -> str:
    mode = os.getenv(PAANO_INPUT_PADDING_ENV, PAANO_INPUT_PADDING_EDGE_HOLD).strip().lower()
    if mode not in PAANO_INPUT_PADDING_MODES:
        raise ValueError(
            f"Unsupported {PAANO_INPUT_PADDING_ENV}={mode!r}. "
            f"Expected one of: {', '.join(sorted(PAANO_INPUT_PADDING_MODES))}."
        )
    return mode


def _load_saved_baseline(anomaly_key: str) -> dict[str, Any]:
    path = summary_path(get_detection_spec(anomaly_key), "paano_shared")
    if not path.exists():
        return {"missing": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        split: _compact_metrics(metrics)
        for split, metrics in payload.get("splits", {}).items()
    }


def _first_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    return (
        intervals.sort_values(["well_id", "start_date", "interval_idx"])
        .groupby("well_id", as_index=False)
        .first()
    )


def _prepare_by_class(
    reference_policy: str,
    *,
    common_source_freq: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    prepared_by_class: dict[str, dict[str, Any]] = {}
    intervals_by_class: dict[str, pd.DataFrame] = {}
    for anomaly_key in ANOMALY_KEYS:
        spec = get_detection_spec(anomaly_key)
        source_path = _anomaly_source_path(anomaly_key, common_source_freq)
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
        skipped = [key for key, item in audit.items() if item.get("reason") != "ok"]
        if skipped:
            message = (
                f"Feature schema skipped {len(skipped)} {anomaly_key} wells: "
                + ", ".join(skipped[:10])
            )
            if strict:
                raise RuntimeError(message)
            print("  " + message)
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
    filtered, audit = restrict_prepared_mapping_to_schema(
        global_pool,
        schema,
        strict=strict,
    )
    skipped = [key for key, item in audit.items() if item.get("reason") != "ok"]
    if skipped:
        message = (
            f"Feature schema skipped {len(skipped)} global pool wells: "
            + ", ".join(skipped[:10])
        )
        if strict:
            raise RuntimeError(message)
        print("  " + message)
    return filtered, audit


def _feature_schema_preview(
    output_dir: Path,
    schema: FeatureSchema,
    global_pool_runs: dict[str, Any],
    schema_audit_by_class: dict[str, dict[str, dict[str, Any]]],
    schema_audit_global_pool: dict[str, dict[str, Any]],
    balance_audit: dict[str, Any],
    *,
    common_source_freq: str | None,
    prepare_patch_size_overrides: dict[str, int],
    norm_work_profile: str | None,
    enable_feature_reduction: bool,
) -> None:
    pool, shared_channels, train_wells = collect_shared_train_pool(
        global_pool_runs,
        enable_reduction=enable_feature_reduction,
    )
    payload = {
        "schema": schema.to_detail(),
        "common_source_freq": common_source_freq,
        "prepare_patch_size_overrides": prepare_patch_size_overrides,
        "norm_work_profile": norm_work_profile,
        "global_pool_points_after_reduction": int(len(pool)),
        "global_shared_channels_after_reduction": int(len(shared_channels)),
        "global_train_wells": train_wells,
        "global_train_well_count": len(train_wells),
        "shared_channels": shared_channels,
        "feature_reduction_enabled": bool(enable_feature_reduction),
        "balance_audit": balance_audit,
        "schema_audit_by_class": schema_audit_by_class,
        "schema_audit_global_pool": schema_audit_global_pool,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "global_fixed_feature_schema_preview.json"
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        "Fixed feature schema preview: "
        f"pool={len(pool)}, shared_features={len(shared_channels)}, wells={len(train_wells)}"
    )
    print(f"Wrote {path}")


def _build_global_core_runs(
    prepared_runs: dict[str, Any],
    shared_state: Any,
    device: Any,
    *,
    verbose: bool,
) -> dict[str, PreparedDetectorRun]:
    detector_runs: dict[str, PreparedDetectorRun] = {}
    input_padding_mode = _global_paano_input_padding_mode()
    for well_id, prepared in prepared_runs.items():
        X = select_shared_columns(
            prepared.feature_columns,
            prepared.feature_matrix,
            shared_state.shared_channels,
        )
        detector = SharedPaAnoDetector(
            shared_state=shared_state,
            device=device,
            verbose=verbose,
            input_padding_mode=input_padding_mode,
        )
        detector.fit_reference(X[prepared.reference_mask])
        raw_output = detector.score_stream(X, mask_all=prepared.stability_mask)
        primary = np.asarray(raw_output.primary, dtype=np.float32)
        components = {
            "global_paano_score": primary,
            **raw_output.components,
        }
        score_output = DetectorScoreOutput(
            primary=primary,
            components=components,
            detail={
                **raw_output.detail,
                "global_normality_detector": True,
                "class_fine_tune": False,
                "physical_branches": False,
            },
        )
        detector_runs[well_id] = PreparedDetectorRun(
            prepared=prepared,
            score_output=score_output,
        )
    return detector_runs


def _evaluate_global_runs(
    anomaly_key: str,
    prepared_runs: dict[str, Any],
    intervals: pd.DataFrame,
    device: Any,
    shared_state: Any,
    *,
    retune: bool,
    verbose: bool,
) -> dict[str, Any]:
    spec = get_detection_spec(anomaly_key)
    detector_runs = _build_global_core_runs(
        prepared_runs,
        shared_state,
        device,
        verbose=False,
    )
    train_runs = {
        well_id: run
        for well_id, run in detector_runs.items()
        if run.prepared.split == "train"
    }
    train_intervals = intervals[
        intervals["split"].astype(str).str.lower() == "train"
    ].copy()
    if retune:
        cfg, tuning_summary = _tune_config(
            anomaly_key,
            "paano_shared",
            train_runs,
            train_intervals,
            verbose=verbose,
        )
    else:
        cfg = _load_or_build_config(
            spec=spec,
            detector_key="paano_shared",
            train_runs=train_runs,
            train_intervals=train_intervals,
            retune=False,
            verbose=verbose,
        )
        tuning_summary = {"retune": False}

    score_rows, predicted, early_predicted, detail_map = _build_score_rows(
        "paano_shared",
        detector_runs,
        cfg,
        anomaly_key=anomaly_key,
        intervals=intervals,
    )
    score_df = pd.DataFrame(score_rows)
    pred_df = _predicted_with_early_warning(predicted, early_predicted)
    pred_df = _attach_predicted_start_status(pred_df, score_df)
    if not pred_df.empty:
        pred_df["anomaly"] = anomaly_key
        pred_df["detector"] = "global_normality_paano"
        split_lookup = {well_id: run.prepared.split for well_id, run in detector_runs.items()}
        pred_df["split"] = pred_df["well_id"].map(split_lookup).fillna("train")

    incident_result = build_incidents(
        pred_df,
        merge_window_hours=_incident_merge_window_hours(cfg),
    )
    pred_df = attach_domain_decisions_to_starts(
        incident_result.starts,
        detector_runs,
        anomaly_key=anomaly_key,
    )
    incident_df = attach_domain_decisions_to_incidents(
        incident_result.incidents,
        pred_df,
    )
    eval_pred_df = filter_actionable_starts(pred_df)

    split_summaries, split_frames = summarize_splits(
        intervals=intervals,
        predictions=eval_pred_df,
        scores=score_df,
        prestart_hours=PRESTART_TOLERANCE_HOURS,
    )
    return {
        "config": cfg,
        "tuning_summary": tuning_summary,
        "splits": {
            split: _compact_metrics(metrics)
            for split, metrics in split_summaries.items()
        },
        "prediction_postprocess": {
            "raw_starts": int(len(pred_df)),
            "actionable_starts": int(len(eval_pred_df)),
            "suppressed_starts": int(len(pred_df) - len(eval_pred_df)),
            "incidents": int(len(incident_df)),
            "incident_merge_window_hours": _incident_merge_window_hours(cfg),
        },
        "domain_decision": {
            "starts_by_action": (
                pred_df["domain_action"].value_counts(dropna=False).to_dict()
                if "domain_action" in pred_df.columns
                else {}
            ),
            "starts_by_verdict": (
                pred_df["domain_verdict"].value_counts(dropna=False).to_dict()
                if "domain_verdict" in pred_df.columns
                else {}
            ),
        },
        "detail_map": detail_map,
        "interval_results": attach_domain_rule_diagnostics(
            split_frames.get("all", pd.DataFrame()).to_dict("records"),
            {well_id: run.prepared for well_id, run in detector_runs.items()},
            anomaly_key,
        ),
        "predicted_starts": pred_df.to_dict("records"),
        "incidents": incident_df.to_dict("records"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one global PaAno normality detector across ALMA anomaly families.",
    )
    parser.add_argument("--output-dir", default="artifacts/results/global_normality_detector")
    parser.add_argument("--patch-short", type=int, default=96)
    parser.add_argument("--patch-long", type=int, default=192)
    parser.add_argument("--global-iters", type=int, default=200)
    parser.add_argument("--min-shared-channels", type=int, default=40)
    parser.add_argument("--include-norm-work", action="store_true")
    parser.add_argument("--norm-work-profile", choices=ANOMALY_KEYS, default=DEFAULT_NORM_WORK_PROFILE)
    parser.add_argument(
        "--common-source-freq",
        default=None,
        help="Use db/<anomaly>_anomaly_database_<freq>.parquet and db/norm_work_database_<freq>.parquet.",
    )
    parser.add_argument("--feature-schema", default=str(DEFAULT_GLOBAL_SCHEMA_PATH))
    parser.add_argument("--allow-missing-schema-channels", action="store_true")
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--enable-feature-reduction", action="store_true")
    parser.add_argument("--disable-balanced-pool", action="store_true")
    parser.add_argument(
        "--disable-norm-pool-hygiene",
        action="store_true",
        help="Не применять правила гигиены банка нормы из конфига global normality.",
    )
    parser.add_argument("--balance-rows-per-source", type=int, default=DEFAULT_BALANCE_ROWS_PER_SOURCE)
    parser.add_argument("--balance-rows-per-well", type=int, default=DEFAULT_BALANCE_ROWS_PER_WELL)
    parser.add_argument("--balance-source-policy", choices=["equal_min", "cap"], default=DEFAULT_BALANCE_SOURCE_POLICY)
    parser.add_argument("--balance-seed", type=int, default=20260518)
    parser.add_argument(
        "--reference-policy",
        choices=sorted(REFERENCE_POLICIES),
        default=REFERENCE_POLICY_NORMAL_WINDOWS,
    )
    parser.add_argument("--no-retune", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_schema = load_feature_schema(args.feature_schema)
    schema_strict = not bool(args.allow_missing_schema_channels)
    common_source_freq = str(args.common_source_freq).strip() if args.common_source_freq else None
    prepare_patch_size_overrides = _prepare_patch_overrides_for_common_freq(common_source_freq)
    with _temporary_prepare_patch_sizes(prepare_patch_size_overrides):
        prepared_by_class, intervals_by_class = _prepare_by_class(
            str(args.reference_policy),
            common_source_freq=common_source_freq,
        )
        prepared_by_class, schema_audit_by_class = _apply_feature_schema_by_class(
            prepared_by_class,
            feature_schema,
            strict=schema_strict,
        )
        global_pool_runs = _global_train_pool(prepared_by_class)
        if args.include_norm_work:
            global_pool_runs = _add_norm_work_pool(
                global_pool_runs,
                reference_policy=str(args.reference_policy),
                common_source_freq=common_source_freq,
                norm_work_profile=str(args.norm_work_profile),
            )
    hygiene_audit: dict[str, Any] = {"enabled": False}
    if not bool(args.disable_norm_pool_hygiene):
        hygiene_rules = load_global_normality_settings().norm_pool_hygiene_rules
        global_pool_runs, hygiene_audit = _apply_norm_pool_hygiene(
            global_pool_runs,
            hygiene_rules,
            verbose=True,
        )
    global_pool_runs, schema_audit_global_pool = _apply_feature_schema_to_pool(
        global_pool_runs,
        feature_schema,
        strict=schema_strict,
    )
    balance_audit: dict[str, Any] = {"enabled": False}
    if not bool(args.disable_balanced_pool):
        global_pool_runs, balance_audit = _apply_balanced_reference_pool(
            global_pool_runs,
            max_rows_per_source=int(args.balance_rows_per_source),
            max_rows_per_well=int(args.balance_rows_per_well),
            source_policy=str(args.balance_source_policy),
            seed=int(args.balance_seed),
        )
    if args.schema_only:
        _feature_schema_preview(
            output_dir,
            feature_schema,
            global_pool_runs,
            schema_audit_by_class,
            schema_audit_global_pool,
            balance_audit,
            common_source_freq=common_source_freq,
            prepare_patch_size_overrides=prepare_patch_size_overrides,
            norm_work_profile=str(args.norm_work_profile) if args.include_norm_work else None,
            enable_feature_reduction=bool(args.enable_feature_reduction),
        )
        return
    global_pool, global_channels, global_train_wells = collect_shared_train_pool(
        global_pool_runs,
        enable_reduction=bool(args.enable_feature_reduction),
    )
    if len(global_channels) < int(args.min_shared_channels):
        raise RuntimeError(
            "Global training pool has too few shared channels after feature reduction: "
            f"{len(global_channels)} < {int(args.min_shared_channels)}. "
            "Check channel naming/coverage before mixing additional normal datasets."
        )

    device = _resolve_torch_device("paano_shared", verbose=True)
    global_state = train_shared_encoder(
        prepared_wells=global_pool_runs,
        patch_short=int(args.patch_short),
        patch_long=int(args.patch_long),
        anomaly_key="global_normality",
        device=device,
        verbose=True,
        num_iter=int(args.global_iters),
        enable_reduction=bool(args.enable_feature_reduction),
    )

    payload: dict[str, Any] = {
        "mode": "global_normality_detector",
        "description": (
            "One PaAno shared encoder trained on all train normal/reference "
            "segments from negermet, pritok, and salt. No class fine-tune and "
            "no anomaly-specific physical branch are used for scoring."
        ),
        "patch_short": int(args.patch_short),
        "patch_long": int(args.patch_long),
        "reference_policy": str(args.reference_policy),
        "common_source_freq": common_source_freq,
        "prepare_patch_size_overrides": prepare_patch_size_overrides,
        "global_iters": int(args.global_iters),
        "include_norm_work": bool(args.include_norm_work),
        "norm_work_profile": str(args.norm_work_profile) if args.include_norm_work else None,
        "retune": not args.no_retune,
        "fixed_feature_schema": feature_schema.to_detail(),
        "schema_strict": schema_strict,
        "feature_reduction_enabled": bool(args.enable_feature_reduction),
        "balance_audit": balance_audit,
        "norm_pool_hygiene_audit": hygiene_audit,
        "schema_audit_by_class": schema_audit_by_class,
        "schema_audit_global_pool": schema_audit_global_pool,
        "global_pool_points_after_reduction": int(len(global_pool)),
        "global_shared_channels_after_reduction": int(len(global_channels)),
        "global_train_wells": global_train_wells,
        "global_state_detail": global_state.detail,
        "saved_class_specific_baseline": {
            anomaly_key: _load_saved_baseline(anomaly_key)
            for anomaly_key in ANOMALY_KEYS
        },
        "classes": {},
    }

    rows: list[dict[str, Any]] = []
    for anomaly_key, baseline in payload["saved_class_specific_baseline"].items():
        rows.append({
            "variant": "class_specific_saved",
            "anomaly": anomaly_key,
            **baseline.get("all", {}),
        })

    for anomaly_key in ANOMALY_KEYS:
        print(f"\n=== Global normality detector: {anomaly_key} ===")
        class_payload = _evaluate_global_runs(
            anomaly_key=anomaly_key,
            prepared_runs=prepared_by_class[anomaly_key],
            intervals=intervals_by_class[anomaly_key],
            device=device,
            shared_state=global_state,
            retune=not args.no_retune,
            verbose=True,
        )
        payload["classes"][anomaly_key] = class_payload
        rows.append({
            "variant": "global_normality",
            "anomaly": anomaly_key,
            **class_payload["splits"].get("all", {}),
        })

        interval_path = output_dir / f"global_normality_{anomaly_key}_intervals.csv"
        pd.DataFrame(class_payload["interval_results"]).to_csv(interval_path, index=False)
        print(f"Wrote {interval_path}")

        starts_path = output_dir / f"global_normality_{anomaly_key}_starts.csv"
        pd.DataFrame(class_payload["predicted_starts"]).to_csv(starts_path, index=False)
        print(f"Wrote {starts_path}")

        incidents_path = output_dir / f"global_normality_{anomaly_key}_incidents.csv"
        pd.DataFrame(class_payload["incidents"]).to_csv(incidents_path, index=False)
        print(f"Wrote {incidents_path}")

    payload_path = output_dir / "global_normality_benchmark.json"
    payload_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    summary_path_out = output_dir / "global_normality_benchmark_summary.csv"
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_path_out, index=False)
    print(summary_df.to_string(index=False))
    print(f"Wrote {payload_path}")
    print(f"Wrote {summary_path_out}")


if __name__ == "__main__":
    main()
