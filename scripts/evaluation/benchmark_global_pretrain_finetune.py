from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from alma_service.anomaly_specs import get_detection_spec
from alma_service.benchmark_metrics import predicted_from_mapping, summarize_splits
from alma_service.detection_artifacts import summary_path
from alma_service.generic_detection import (
    PRESTART_TOLERANCE_HOURS,
    _build_local_runs,
    _build_score_rows,
    _load_or_build_config,
    _prepare_all_wells,
    _resolve_torch_device,
    _tune_config,
    load_anomaly_data,
    load_intervals,
)
from alma_service.shared_encoder import (
    collect_shared_train_pool,
    fine_tune_shared_encoder,
    train_shared_encoder,
)


ANOMALY_KEYS = ("negermet", "pritok", "salt")


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


def _prepare_by_class() -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    prepared_by_class: dict[str, dict[str, Any]] = {}
    intervals_by_class: dict[str, pd.DataFrame] = {}
    for anomaly_key in ANOMALY_KEYS:
        spec = get_detection_spec(anomaly_key)
        df = load_anomaly_data(spec)
        intervals = load_intervals(spec, required=True)
        intervals = (
            intervals.sort_values(["well_id", "start_date", "interval_idx"])
            .groupby("well_id", as_index=False)
            .first()
        )
        prepared_by_class[anomaly_key] = _prepare_all_wells(
            spec,
            df,
            intervals,
            verbose=False,
            zone_aware=True,
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


def _evaluate_runs(
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
    detector_runs = _build_local_runs(
        anomaly_key,
        "paano_shared",
        prepared_runs,
        device=device,
        verbose=False,
        shared_state=shared_state,
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
    )
    pred_df = predicted_from_mapping(predicted)
    if not pred_df.empty:
        split_lookup = {
            well_id: run.prepared.split for well_id, run in detector_runs.items()
        }
        pred_df["split"] = pred_df["well_id"].map(split_lookup).fillna("train")
    score_df = pd.DataFrame(score_rows)
    split_summaries, split_frames = summarize_splits(
        intervals=intervals,
        predictions=pred_df,
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
        "detail_map": detail_map,
        "interval_results": split_frames.get("all", pd.DataFrame()).to_dict("records"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark global pretrain -> class fine-tune for paano_shared.",
    )
    parser.add_argument("--output-dir", default="artifacts/results/global_pretrain_finetune")
    parser.add_argument("--patch-short", type=int, default=96)
    parser.add_argument("--patch-long", type=int, default=192)
    parser.add_argument("--fine-tune-iters", type=int, default=200)
    parser.add_argument("--no-retune", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_by_class, intervals_by_class = _prepare_by_class()
    global_pool_runs = _global_train_pool(prepared_by_class)
    global_pool, global_channels, global_train_wells = collect_shared_train_pool(
        global_pool_runs,
        enable_reduction=True,
    )

    device = _resolve_torch_device("paano_shared", verbose=True)
    global_state = train_shared_encoder(
        prepared_wells=global_pool_runs,
        patch_short=args.patch_short,
        patch_long=args.patch_long,
        anomaly_key="global",
        device=device,
        verbose=True,
    )

    payload: dict[str, Any] = {
        "mode": "global_pretrain_class_finetune",
        "patch_short": args.patch_short,
        "patch_long": args.patch_long,
        "fine_tune_iters": args.fine_tune_iters,
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
        print(f"\n=== Fine-tune from global: {anomaly_key} ===")
        fine_tuned_state = fine_tune_shared_encoder(
            pretrained_state=global_state,
            prepared_wells=prepared_by_class[anomaly_key],
            anomaly_key=anomaly_key,
            device=device,
            verbose=True,
            num_iter=args.fine_tune_iters,
        )
        class_payload = _evaluate_runs(
            anomaly_key=anomaly_key,
            prepared_runs=prepared_by_class[anomaly_key],
            intervals=intervals_by_class[anomaly_key],
            device=device,
            shared_state=fine_tuned_state,
            retune=not args.no_retune,
            verbose=True,
        )
        class_payload["fine_tuned_state_detail"] = fine_tuned_state.detail
        payload["classes"][anomaly_key] = class_payload
        rows.append({
            "variant": "global_pretrain_finetune",
            "anomaly": anomaly_key,
            **class_payload["splits"].get("all", {}),
        })

        interval_path = output_dir / f"global_pretrain_finetune_{anomaly_key}_intervals.csv"
        pd.DataFrame(class_payload["interval_results"]).to_csv(interval_path, index=False)
        print(f"Wrote {interval_path}")

    payload_path = output_dir / "global_pretrain_finetune_benchmark.json"
    payload_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    summary_path_out = output_dir / "global_pretrain_finetune_benchmark_summary.csv"
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_path_out, index=False)
    print(summary_df.to_string(index=False))
    print(f"Wrote {payload_path}")
    print(f"Wrote {summary_path_out}")


if __name__ == "__main__":
    main()
