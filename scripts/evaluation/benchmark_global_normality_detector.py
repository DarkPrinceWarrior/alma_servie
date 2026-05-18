from __future__ import annotations

import argparse
import json
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
from alma_service.engineered_features import (
    REFERENCE_POLICIES,
    REFERENCE_POLICY_NORMAL_WINDOWS,
)
from alma_service.generic_detection import (
    PRESTART_TOLERANCE_HOURS,
    PreparedDetectorRun,
    _attach_predicted_start_status,
    _build_score_rows,
    _incident_merge_window_hours,
    _load_or_build_config,
    _prepare_all_wells,
    _resolve_torch_device,
    _tune_config,
    load_anomaly_data,
    load_intervals,
)
from alma_service.generic_detectors import DetectorScoreOutput, SharedPaAnoDetector
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


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


def _prepare_by_class(reference_policy: str) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    prepared_by_class: dict[str, dict[str, Any]] = {}
    intervals_by_class: dict[str, pd.DataFrame] = {}
    for anomaly_key in ANOMALY_KEYS:
        spec = get_detection_spec(anomaly_key)
        df = load_anomaly_data(spec)
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
) -> dict[str, Any]:
    for anomaly_key in ANOMALY_KEYS:
        source = Path("db") / f"norm_work_database_{DEFAULT_ANOMALY_FREQ[anomaly_key]}.parquet"
        if not source.exists():
            print(f"  Norm work skipped for {anomaly_key}: missing {source}")
            continue
        spec = get_detection_spec(anomaly_key)
        df = pd.read_parquet(source)
        if df.empty:
            continue
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
            detail["reference_policy"] = "full_norm_work_series"
            detail["reference_points"] = int(reference_mask.sum())
            global_pool[f"norm_work:{anomaly_key}:{well_id}"] = replace(
                item,
                split="train",
                reference_mask=reference_mask,
                reference_end_idx=len(item.timestamps),
                onset_allowed_mask=np.zeros(len(item.timestamps), dtype=bool),
                detail=detail,
            )
    return global_pool


def _build_global_core_runs(
    prepared_runs: dict[str, Any],
    shared_state: Any,
    device: Any,
    *,
    verbose: bool,
) -> dict[str, PreparedDetectorRun]:
    detector_runs: dict[str, PreparedDetectorRun] = {}
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

    score_rows, predicted, detail_map = _build_score_rows(
        "paano_shared",
        detector_runs,
        cfg,
        anomaly_key=anomaly_key,
        intervals=intervals,
    )
    score_df = pd.DataFrame(score_rows)
    pred_df = predicted_from_mapping(predicted)
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
    pred_df = incident_result.starts
    incident_df = incident_result.incidents
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
        "detail_map": detail_map,
        "interval_results": split_frames.get("all", pd.DataFrame()).to_dict("records"),
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

    prepared_by_class, intervals_by_class = _prepare_by_class(str(args.reference_policy))
    global_pool_runs = _global_train_pool(prepared_by_class)
    if args.include_norm_work:
        global_pool_runs = _add_norm_work_pool(
            global_pool_runs,
            reference_policy=str(args.reference_policy),
        )
    global_pool, global_channels, global_train_wells = collect_shared_train_pool(
        global_pool_runs,
        enable_reduction=True,
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
        "global_iters": int(args.global_iters),
        "include_norm_work": bool(args.include_norm_work),
        "retune": not args.no_retune,
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
