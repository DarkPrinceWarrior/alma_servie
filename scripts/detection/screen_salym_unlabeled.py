from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.anomaly_specs import get_detection_spec
from alma_service.detection_artifacts import (
    config_path,
    load_json,
    normalize_detector_key,
)
from alma_service.engineered_features import prepare_engineered_well
from alma_service.generic_detection import (
    _build_local_runs,
    _default_onset_config,
    _detect_starts_for_run,
    _resolve_torch_device,
    _runtime_config,
)
from alma_service.generic_detectors import set_seed
from alma_service.paano_defaults import (
    MIN_REFERENCE_COVERAGE,
    MIN_TOTAL_COVERAGE,
    REFERENCE_MAX_RATIO,
    REFERENCE_MIN_DAYS,
    REFERENCE_MIN_RATIO,
)
from alma_service.paths import ARTIFACTS_DIR, SALYM_PREPARED_DIR, ensure_dir
from alma_service.salym_raw_pipeline import SELECTED_PARAM_MAP
from alma_service.shared_encoder import load_shared_encoder_state
from alma_service.tabular_io import write_table


DEFAULT_ANOMALY_FREQ = {
    "negermet": "2min",
    "pritok": "10min",
    "salt": "15min",
}


def _parse_anomalies(value: str) -> list[str]:
    out = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"negermet", "pritok", "salt"}
    unknown = sorted(set(out) - allowed)
    if unknown:
        raise ValueError(f"Unknown anomaly keys: {unknown}")
    return out


def _years_cover_all(value: Any) -> bool:
    if isinstance(value, str):
        return all(year in value for year in ("2015", "2016", "2017", "2018"))
    try:
        values = {str(item) for item in value}
    except TypeError:
        return False
    return {"2015", "2016", "2017", "2018"}.issubset(values)


def _eligible_salym_wells(root: Path, *, limit: int | None = None, require_all_years: bool = False) -> list[str]:
    coverage_path = root / "qc" / "well_parameter_coverage.parquet"
    if not coverage_path.exists():
        raise FileNotFoundError(f"Salym coverage file not found: {coverage_path}")

    coverage = pd.read_parquet(coverage_path)
    coverage["has_valid"] = pd.to_numeric(coverage["n_valid_rows"], errors="coerce").fillna(0).gt(0)
    coverage["all_years"] = coverage["years_present"].map(_years_cover_all)
    grouped = coverage.groupby("well_id", as_index=True).agg(
        params=("param_key", "nunique"),
        valid_params=("has_valid", "sum"),
        all_year_params=("all_years", "sum"),
    )
    mask = (grouped["params"] == len(SELECTED_PARAM_MAP)) & (grouped["valid_params"] == len(SELECTED_PARAM_MAP))
    if require_all_years:
        mask &= grouped["all_year_params"] == len(SELECTED_PARAM_MAP)
    eligible = grouped[mask]
    wells = sorted(str(well_id) for well_id in eligible.index)
    return wells[:limit] if limit is not None else wells


def _load_salym_param_series(path: Path, output_name: str) -> pd.Series | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=["timestamp", "value", "quality_flag"])
    frame = frame[frame["quality_flag"].eq("ok")]
    frame = frame.dropna(subset=["timestamp", "value"])
    if frame.empty:
        return None
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "value"])
    if frame.empty:
        return None
    series = (
        frame[["timestamp", "value"]]
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")["value"]
        .sort_index()
    )
    series.name = output_name
    return series


def _build_salym_well_frame(root: Path, well_id: str, freq: str) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    well_dir = root / "raw_wells" / well_id
    series_list: list[pd.Series] = []
    missing_params: list[str] = []
    for param_key, output_name in SELECTED_PARAM_MAP.items():
        series = _load_salym_param_series(well_dir / f"{param_key}.parquet", output_name)
        if series is None or len(series) < 2:
            missing_params.append(param_key)
            continue
        series_list.append(series)

    if len(series_list) != len(SELECTED_PARAM_MAP):
        return None

    starts = [series.index.min() for series in series_list]
    ends = [series.index.max() for series in series_list]
    union_start = min(starts)
    union_end = max(ends)
    grid_start = union_start.ceil(freq)
    grid_end = union_end.floor(freq)
    grid = pd.date_range(grid_start, grid_end, freq=freq)
    if len(grid) < 2:
        return None

    frame = pd.DataFrame({"timestamp": grid})
    for series in series_list:
        combined_idx = series.index.union(grid)
        frame[series.name] = series.reindex(combined_idx).sort_index().ffill().reindex(grid).to_numpy()

    frame.insert(0, "well_id", well_id)
    detail = {
        "grid_points": int(len(grid)),
        "start": str(grid_start),
        "end": str(grid_end),
        "freq": freq,
        "raw_channels": len(series_list),
        "missing_params": missing_params,
    }
    return frame, detail


def _load_detector_config(anomaly_key: str, detector_key: str) -> dict[str, Any]:
    spec = get_detection_spec(anomaly_key)
    payload = load_json(config_path(spec, detector_key))
    stored = payload.get("config", payload) if payload else {}
    cfg = {**_default_onset_config(anomaly_key, detector_key), **stored}
    if detector_key == "paano_feat" and "fusion_weight_short" not in cfg:
        cfg["fusion_weight_short"] = 0.60
    return cfg


def _build_shared_state(anomaly_key: str, detector_key: str, device: Any, verbose: bool) -> Any:
    if detector_key != "paano_shared":
        return None
    try:
        return load_shared_encoder_state(anomaly_key, device=device, verbose=verbose)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}. Build frozen encoders first: "
            f"CUDA_VISIBLE_DEVICES=1 uv run python scripts/detection/build_shared_encoders.py "
            f"--anomalies {anomaly_key}"
        ) from exc


def _screen_one_anomaly(
    *,
    anomaly_key: str,
    detector_key: str,
    salym_root: Path,
    wells: list[str],
    output_dir: Path,
    freq: str,
    checkpoint_every: int,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec = get_detection_spec(anomaly_key)
    print(f"=== Salym screening: {spec.display_name} [{detector_key}], wells={len(wells)}, freq={freq} ===")
    device = _resolve_torch_device(detector_key, verbose=True)
    shared_state = _build_shared_state(anomaly_key, detector_key, device=device, verbose=verbose)
    cfg = _load_detector_config(anomaly_key, detector_key)
    runtime_cfg = _runtime_config(anomaly_key)

    results: list[dict[str, Any]] = []
    starts_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for idx, well_id in enumerate(wells, start=1):
        item_started = time.perf_counter()
        try:
            built = _build_salym_well_frame(salym_root, well_id, freq)
            if built is None:
                raise RuntimeError("Cannot build full 15-channel Salym frame")
            well_df, input_detail = built
            prepared = prepare_engineered_well(
                anomaly_key=anomaly_key,
                well_id=well_id,
                split="screen",
                well_df=well_df,
                patch_size=int(runtime_cfg["prepare_patch_size"]),
                reference_min_ratio=REFERENCE_MIN_RATIO,
                reference_max_ratio=REFERENCE_MAX_RATIO,
                reference_min_days=REFERENCE_MIN_DAYS,
                min_reference_coverage=MIN_REFERENCE_COVERAGE,
                min_total_coverage=MIN_TOTAL_COVERAGE,
            )
            if prepared is None:
                raise RuntimeError("Not enough usable data after engineered preprocessing")
            run = _build_local_runs(
                anomaly_key,
                detector_key,
                {well_id: prepared},
                device=device,
                verbose=False,
                shared_state=shared_state,
            )[well_id]
            score, thresholds, starts = _detect_starts_for_run(detector_key, run, cfg)
            starts_list = [pd.Timestamp(ts) for ts in starts]
            for start_idx, ts in enumerate(starts_list, start=1):
                pos = int(np.searchsorted(prepared.timestamps, np.datetime64(ts), side="left"))
                pos = min(max(pos, 0), len(score) - 1)
                starts_rows.append(
                    {
                        "anomaly": anomaly_key,
                        "detector": detector_key,
                        "well_id": well_id,
                        "start_idx": start_idx,
                        "detected_time": ts,
                        "score_at_start": float(score[pos]),
                    }
                )
            status = "Detected" if starts_list else "Not detected"
            results.append(
                {
                    "anomaly": anomaly_key,
                    "detector": detector_key,
                    "well_id": well_id,
                    "status": status,
                    "has_detection": bool(starts_list),
                    "first_detected_time": starts_list[0] if starts_list else pd.NaT,
                    "last_detected_time": starts_list[-1] if starts_list else pd.NaT,
                    "n_detected_starts": len(starts_list),
                    "max_score": float(np.nanmax(score)) if len(score) else np.nan,
                    "median_score": float(np.nanmedian(score)) if len(score) else np.nan,
                    "n_points": int(len(prepared.timestamps)),
                    "n_raw_channels": int(prepared.detail.get("raw_channels", 0)),
                    "n_features": int(prepared.detail.get("feature_count", 0)),
                    "reference_points": int(prepared.detail.get("reference_points", 0)),
                    "masked_fraction": float(prepared.detail.get("masked_fraction", np.nan)),
                    "input_grid_points": int(input_detail["grid_points"]),
                    "input_start": input_detail["start"],
                    "input_end": input_detail["end"],
                    "elapsed_seconds": round(time.perf_counter() - item_started, 3),
                }
            )
        except Exception as exc:
            failed_rows.append(
                {
                    "anomaly": anomaly_key,
                    "detector": detector_key,
                    "well_id": well_id,
                    "error": str(exc),
                    "elapsed_seconds": round(time.perf_counter() - item_started, 3),
                }
            )
            results.append(
                {
                    "anomaly": anomaly_key,
                    "detector": detector_key,
                    "well_id": well_id,
                    "status": "Skipped",
                    "has_detection": False,
                    "first_detected_time": pd.NaT,
                    "last_detected_time": pd.NaT,
                    "n_detected_starts": 0,
                    "max_score": np.nan,
                    "median_score": np.nan,
                    "n_points": 0,
                    "n_raw_channels": 0,
                    "n_features": 0,
                    "reference_points": 0,
                    "masked_fraction": np.nan,
                    "input_grid_points": 0,
                    "input_start": None,
                    "input_end": None,
                    "elapsed_seconds": round(time.perf_counter() - item_started, 3),
                }
            )

        if idx == 1 or idx % checkpoint_every == 0 or idx == len(wells):
            result_df = pd.DataFrame(results)
            starts_df = pd.DataFrame(starts_rows)
            failed_df = pd.DataFrame(failed_rows)
            _write_anomaly_outputs(output_dir, anomaly_key, result_df, starts_df, failed_df)
            detected = int(result_df["has_detection"].sum()) if not result_df.empty else 0
            skipped = int(result_df["status"].eq("Skipped").sum()) if not result_df.empty else 0
            print(
                f"  [{anomaly_key}] {idx}/{len(wells)} wells, "
                f"detected={detected}, skipped={skipped}, "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    result_df = pd.DataFrame(results)
    starts_df = pd.DataFrame(starts_rows)
    failed_df = pd.DataFrame(failed_rows)
    _write_anomaly_outputs(output_dir, anomaly_key, result_df, starts_df, failed_df)
    summary = {
        "anomaly": anomaly_key,
        "detector": detector_key,
        "freq": freq,
        "wells_total": len(wells),
        "wells_processed": int(len(result_df)),
        "wells_detected": int(result_df["has_detection"].sum()) if not result_df.empty else 0,
        "wells_not_detected": int(result_df["status"].eq("Not detected").sum()) if not result_df.empty else 0,
        "wells_skipped": int(result_df["status"].eq("Skipped").sum()) if not result_df.empty else 0,
        "total_detected_starts": int(result_df["n_detected_starts"].sum()) if not result_df.empty else 0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "config": cfg,
        "shared_state_detail": getattr(shared_state, "detail", None),
    }
    (output_dir / f"salym_{anomaly_key}_{detector_key}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result_df, starts_df, summary


def _write_anomaly_outputs(output_dir: Path, anomaly_key: str, result_df: pd.DataFrame, starts_df: pd.DataFrame, failed_df: pd.DataFrame) -> None:
    ensure_dir(output_dir)
    if not result_df.empty:
        write_table(result_df, output_dir / f"salym_{anomaly_key}_screening_results.parquet")
        result_df.to_csv(output_dir / f"salym_{anomaly_key}_screening_results.csv", index=False)
    if not starts_df.empty:
        write_table(starts_df, output_dir / f"salym_{anomaly_key}_predicted_starts.parquet")
        starts_df.to_csv(output_dir / f"salym_{anomaly_key}_predicted_starts.csv", index=False)
    if not failed_df.empty:
        failed_df.to_csv(output_dir / f"salym_{anomaly_key}_screening_errors.csv", index=False)


def _write_combined_outputs(output_dir: Path, result_frames: list[pd.DataFrame], starts_frames: list[pd.DataFrame], summaries: list[dict[str, Any]]) -> None:
    if result_frames:
        result_df = pd.concat(result_frames, ignore_index=True)
        write_table(result_df, output_dir / "salym_screening_results.parquet")
        result_df.to_csv(output_dir / "salym_screening_results.csv", index=False)
    if starts_frames:
        starts_df = pd.concat(starts_frames, ignore_index=True)
        write_table(starts_df, output_dir / "salym_predicted_starts.parquet")
        starts_df.to_csv(output_dir / "salym_predicted_starts.csv", index=False)
    (output_dir / "salym_screening_summary.json").write_text(
        json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unlabeled Salym screening with ALMA detectors.")
    parser.add_argument("--anomalies", default="negermet,pritok,salt", help="Comma-separated anomaly keys.")
    parser.add_argument("--detector", default="paano_shared")
    parser.add_argument("--salym-root", default=str(SALYM_PREPARED_DIR))
    parser.add_argument("--output-dir", default=str(ARTIFACTS_DIR / "results" / "salym_screening"))
    parser.add_argument("--limit-wells", type=int, default=None)
    parser.add_argument("--require-all-years", action="store_true", help="Require every parameter to have rows in 2015-2018.")
    parser.add_argument("--freq", default=None, help="Override resample frequency for all anomalies.")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    detector_key = normalize_detector_key(args.detector)
    anomalies = _parse_anomalies(args.anomalies)
    salym_root = Path(args.salym_root)
    output_dir = ensure_dir(Path(args.output_dir))
    wells = _eligible_salym_wells(salym_root, limit=args.limit_wells, require_all_years=args.require_all_years)
    print(f"Selected Salym wells: {len(wells)}")
    (output_dir / "salym_screening_wells.json").write_text(
        json.dumps({"wells": wells, "require_all_years": bool(args.require_all_years)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result_frames: list[pd.DataFrame] = []
    starts_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for anomaly_key in anomalies:
        freq = args.freq or DEFAULT_ANOMALY_FREQ[anomaly_key]
        result_df, starts_df, summary = _screen_one_anomaly(
            anomaly_key=anomaly_key,
            detector_key=detector_key,
            salym_root=salym_root,
            wells=wells,
            output_dir=output_dir,
            freq=freq,
            checkpoint_every=max(1, int(args.checkpoint_every)),
            verbose=not args.quiet,
        )
        result_frames.append(result_df)
        starts_frames.append(starts_df)
        summaries.append(summary)
        _write_combined_outputs(output_dir, result_frames, starts_frames, summaries)


if __name__ == "__main__":
    main()
