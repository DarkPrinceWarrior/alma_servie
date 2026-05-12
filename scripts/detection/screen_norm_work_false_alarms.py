from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alma_service.anomaly_specs import get_detection_spec
from alma_service.detection_artifacts import config_path, load_json
from alma_service.engineered_features import PreparedWellData, prepare_engineered_well
from alma_service.generic_detection import (
    MIN_REFERENCE_COVERAGE,
    MIN_TOTAL_COVERAGE,
    REFERENCE_MAX_RATIO,
    REFERENCE_MIN_DAYS,
    REFERENCE_MIN_RATIO,
    _build_local_runs,
    _detect_starts_for_run,
    _resolve_torch_device,
    _runtime_config,
)
from alma_service.paths import DB_DIR, RESULTS_DIR, ensure_dir
from alma_service.shared_encoder import load_shared_encoder_state, train_shared_encoder
from alma_service.tabular_io import read_table, write_table


DEFAULT_ANOMALY_FREQ = {
    "negermet": "2min",
    "pritok": "10min",
    "salt": "15min",
}
DEFAULT_ANOMALIES = ("negermet", "pritok", "salt")


def _parse_anomalies(value: str) -> list[str]:
    anomalies = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(anomalies) - set(DEFAULT_ANOMALIES))
    if unknown:
        raise ValueError(f"Unknown anomaly keys: {unknown}")
    return anomalies


def _load_detector_config(anomaly_key: str, detector_key: str) -> dict[str, Any]:
    payload = load_json(config_path(get_detection_spec(anomaly_key), detector_key))
    if isinstance(payload, dict) and "config" in payload:
        return dict(payload["config"])
    if isinstance(payload, dict):
        return dict(payload)
    raise FileNotFoundError(f"Detector config not found for {anomaly_key}/{detector_key}")


def _source_path(freq: str) -> Path:
    return DB_DIR / f"norm_work_database_{freq.replace(' ', '')}.parquet"


def _prepare_norm_wells(
    *,
    anomaly_key: str,
    source_path: Path,
    split: str,
) -> dict[str, PreparedWellData]:
    df = read_table(source_path, parse_dates=["timestamp"])
    runtime_cfg = _runtime_config(anomaly_key)
    prepared: dict[str, PreparedWellData] = {}
    for well_id in sorted(df["well_id"].unique()):
        well_df = df[df["well_id"] == well_id].copy()
        item = prepare_engineered_well(
            anomaly_key=anomaly_key,
            well_id=str(well_id),
            split=split,
            well_df=well_df,
            patch_size=int(runtime_cfg["prepare_patch_size"]),
            reference_min_ratio=REFERENCE_MIN_RATIO,
            reference_max_ratio=REFERENCE_MAX_RATIO,
            reference_min_days=REFERENCE_MIN_DAYS,
            min_reference_coverage=MIN_REFERENCE_COVERAGE,
            min_total_coverage=MIN_TOTAL_COVERAGE,
        )
        if item is not None:
            prepared[str(well_id)] = item
    return prepared


def _build_shared_state(
    *,
    anomaly_key: str,
    detector_key: str,
    encoder_source: str,
    prepared: dict[str, PreparedWellData],
    device: Any,
    verbose: bool,
) -> Any:
    if detector_key != "paano_shared":
        return None
    if encoder_source == "frozen":
        return load_shared_encoder_state(anomaly_key, device=device, verbose=verbose)
    if encoder_source == "norm_work":
        runtime_cfg = _runtime_config(anomaly_key)
        return train_shared_encoder(
            prepared_wells=prepared,
            patch_short=int(runtime_cfg.get("paano_patch_short", 64)),
            patch_long=int(runtime_cfg.get("paano_patch_long", 128)),
            anomaly_key=f"norm_work_{anomaly_key}",
            device=device,
            verbose=verbose,
        )
    raise ValueError(f"Unknown encoder source: {encoder_source}")


def _screen_anomaly(
    *,
    anomaly_key: str,
    detector_key: str,
    encoder_source: str,
    source_path: Path,
    output_dir: Path,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    print(
        f"=== Norm-work false alarm screening: {anomaly_key}/{detector_key}, "
        f"encoder={encoder_source}, source={source_path} ===",
        flush=True,
    )
    started = time.perf_counter()
    device = _resolve_torch_device(detector_key, verbose=True)
    split = "train" if encoder_source == "norm_work" else "screen"
    prepared = _prepare_norm_wells(anomaly_key=anomaly_key, source_path=source_path, split=split)
    if not prepared:
        raise RuntimeError(f"No norm_work wells survived preprocessing for {anomaly_key}")

    shared_state = _build_shared_state(
        anomaly_key=anomaly_key,
        detector_key=detector_key,
        encoder_source=encoder_source,
        prepared=prepared,
        device=device,
        verbose=verbose,
    )
    cfg = _load_detector_config(anomaly_key, detector_key)
    runs = _build_local_runs(
        anomaly_key,
        detector_key,
        prepared,
        device=device,
        verbose=False,
        shared_state=shared_state,
    )

    result_rows: list[dict[str, Any]] = []
    start_rows: list[dict[str, Any]] = []
    for well_id, run in runs.items():
        score, _, starts = _detect_starts_for_run(detector_key, run, cfg)
        starts_list = [pd.Timestamp(ts) for ts in starts]
        for start_idx, ts in enumerate(starts_list, start=1):
            pos = int(np.searchsorted(run.prepared.timestamps, np.datetime64(ts), side="left"))
            pos = min(max(pos, 0), len(score) - 1)
            start_rows.append(
                {
                    "anomaly": anomaly_key,
                    "detector": detector_key,
                    "encoder_source": encoder_source,
                    "well_id": well_id,
                    "start_idx": start_idx,
                    "detected_time": ts,
                    "score_at_start": float(score[pos]),
                }
            )
        result_rows.append(
            {
                "anomaly": anomaly_key,
                "detector": detector_key,
                "encoder_source": encoder_source,
                "well_id": well_id,
                "has_false_alarm": bool(starts_list),
                "false_alarm_count": len(starts_list),
                "first_false_alarm_time": starts_list[0] if starts_list else pd.NaT,
                "last_false_alarm_time": starts_list[-1] if starts_list else pd.NaT,
                "max_score": float(np.nanmax(score)) if len(score) else np.nan,
                "median_score": float(np.nanmedian(score)) if len(score) else np.nan,
                "points": int(len(run.prepared.timestamps)),
                "raw_channels": int(run.prepared.detail.get("raw_channels", 0)),
                "features": int(run.prepared.detail.get("feature_count", 0)),
                "reference_points": int(run.prepared.detail.get("reference_points", 0)),
                "masked_fraction": float(run.prepared.detail.get("masked_fraction", np.nan)),
                "data_start": pd.Timestamp(run.prepared.timestamps[0]) if len(run.prepared.timestamps) else pd.NaT,
                "data_end": pd.Timestamp(run.prepared.timestamps[-1]) if len(run.prepared.timestamps) else pd.NaT,
            }
        )

    results = pd.DataFrame(result_rows)
    starts_df = pd.DataFrame(start_rows)
    stem = f"norm_work_{encoder_source}_{anomaly_key}_{detector_key}"
    write_table(results, output_dir / f"{stem}_results.parquet")
    write_table(starts_df, output_dir / f"{stem}_starts.parquet")

    observed_days = 0.0
    if not results.empty:
        durations = pd.to_datetime(results["data_end"]) - pd.to_datetime(results["data_start"])
        observed_days = float(durations.dt.total_seconds().clip(lower=0).sum() / 86400.0)
    summary = {
        "anomaly": anomaly_key,
        "detector": detector_key,
        "encoder_source": encoder_source,
        "source_path": str(source_path),
        "wells_total": int(len(results)),
        "wells_with_false_alarm": int(results["has_false_alarm"].sum()) if not results.empty else 0,
        "wells_without_false_alarm": int((~results["has_false_alarm"]).sum()) if not results.empty else 0,
        "total_false_alarm_starts": int(results["false_alarm_count"].sum()) if not results.empty else 0,
        "observed_days_total": observed_days,
        "false_alarm_starts_per_day": (
            float(results["false_alarm_count"].sum() / observed_days)
            if observed_days > 0 and not results.empty
            else np.nan
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "config": cfg,
        "shared_state_detail": getattr(shared_state, "detail", None),
    }
    (output_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        f"  {anomaly_key}: wells={summary['wells_total']}, "
        f"with_false_alarm={summary['wells_with_false_alarm']}, "
        f"starts={summary['total_false_alarm_starts']}, "
        f"FAR/day={summary['false_alarm_starts_per_day']:.4f}",
        flush=True,
    )
    return results, starts_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen expert-confirmed normal-work wells for false alarms.")
    parser.add_argument("--anomalies", default=",".join(DEFAULT_ANOMALIES))
    parser.add_argument("--detector", default="paano_shared")
    parser.add_argument("--encoder-source", choices=("frozen", "norm_work"), default="frozen")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "norm_work_false_alarms")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anomalies = _parse_anomalies(args.anomalies)
    output_dir = ensure_dir(args.output_dir)
    all_results: list[pd.DataFrame] = []
    all_starts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for anomaly_key in anomalies:
        source = _source_path(DEFAULT_ANOMALY_FREQ[anomaly_key])
        result_df, starts_df, summary = _screen_anomaly(
            anomaly_key=anomaly_key,
            detector_key=str(args.detector),
            encoder_source=str(args.encoder_source),
            source_path=source,
            output_dir=output_dir,
            verbose=not args.quiet,
        )
        all_results.append(result_df)
        all_starts.append(starts_df)
        summaries.append(summary)

    combined_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    combined_starts = pd.concat(all_starts, ignore_index=True) if all_starts else pd.DataFrame()
    write_table(combined_results, output_dir / f"norm_work_{args.encoder_source}_combined_results.parquet")
    write_table(combined_starts, output_dir / f"norm_work_{args.encoder_source}_combined_starts.parquet")
    (output_dir / f"norm_work_{args.encoder_source}_summary.json").write_text(
        json.dumps({"encoder_source": args.encoder_source, "summaries": summaries}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
