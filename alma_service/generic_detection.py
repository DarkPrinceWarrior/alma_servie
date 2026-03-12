from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from alma_service.anomaly_specs import DetectionSpec, get_detection_spec
from alma_service.benchmark_metrics import (
    evaluate_predictions,
    load_intervals as load_intervals_df,
    predicted_from_mapping,
    select_interval_detection,
    summarize_splits,
    summary_score_key,
)
from alma_service.detection_artifacts import (
    DEFAULT_DETECTOR,
    DETECTOR_KEYS,
    LOCAL_DETECTOR_KEYS,
    benchmark_summary_path,
    config_path,
    legacy_summary_path,
    load_json,
    model_path,
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
    FusedDetector,
    IsolationForestDetector,
    LOFDetector,
    PCASPEDetector,
    PaAnoFeatureDetector,
    TranADGlobalDetector,
    set_seed,
)
from alma_service.onset_detection import (
    calibrate_causal_thresholds_from_reference_mask,
    detect_causal_onsets_masked,
)
from alma_service.paano_defaults import (
    LONG_PATCH,
    MIN_REFERENCE_COVERAGE,
    MIN_TOTAL_COVERAGE,
    PRESTART_TOLERANCE_HOURS,
    REFERENCE_MAX_RATIO,
    REFERENCE_MIN_DAYS,
    REFERENCE_MIN_RATIO,
)
from alma_service.paths import DB_DIR, ensure_dir, ensure_parent

DEFAULT_ONSET_CONFIG = {
    "target_far_per_day": 0.50,
    "min_run_points": 3,
    "cooldown_hours": 8.0,
    "ema_alpha": 0.08,
    "gate_mode": "score_ema",
}

ONSET_TUNE_GRID = {
    "target_far_per_day": [0.25, 0.50, 1.00],
    "min_run_points": [2, 3, 4],
    "cooldown_hours": [4.0, 8.0, 12.0],
    "ema_alpha": [0.04, 0.08, 0.12],
    "gate_mode": ["score_ema", "relaxed", "strict"],
}

PAANO_WEIGHT_GRID = [0.40, 0.60, 0.75]


@dataclass
class PreparedDetectorRun:
    prepared: PreparedWellData
    score_output: DetectorScoreOutput


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
    df = pd.read_csv(src, dtype={"well_id": str}, low_memory=False)
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
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


def _build_detector(detector_key: str, device: torch.device, verbose: bool = False):
    if detector_key == "paano_feat":
        return PaAnoFeatureDetector(device=device, verbose=verbose)
    if detector_key == "pca_spe":
        return PCASPEDetector()
    if detector_key == "lof":
        return LOFDetector()
    if detector_key == "iforest":
        return IsolationForestDetector()
    if detector_key == "fused":
        return FusedDetector(device=device, verbose=verbose)
    raise ValueError(f"Unsupported local detector: {detector_key}")


def _prepare_all_wells(
    spec: DetectionSpec,
    df: pd.DataFrame,
    intervals: pd.DataFrame,
    verbose: bool,
) -> dict[str, PreparedWellData]:
    split_map = (
        intervals[["well_id", "split"]]
        .drop_duplicates("well_id")
        .set_index("well_id")["split"]
        .to_dict()
    )
    prepared_runs: dict[str, PreparedWellData] = {}
    for well_id in sorted(df["well_id"].unique()):
        well_df = df[df["well_id"] == well_id].copy()
        prepared = prepare_engineered_well(
            anomaly_key=spec.anomaly_key,
            well_id=well_id,
            split=split_map.get(well_id, "train"),
            well_df=well_df,
            patch_size=LONG_PATCH,
            reference_min_ratio=REFERENCE_MIN_RATIO,
            reference_max_ratio=REFERENCE_MAX_RATIO,
            reference_min_days=REFERENCE_MIN_DAYS,
            min_reference_coverage=MIN_REFERENCE_COVERAGE,
            min_total_coverage=MIN_TOTAL_COVERAGE,
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


def _align_features(prepared_runs: dict[str, PreparedWellData]) -> tuple[list[str], dict[str, np.ndarray]]:
    all_columns = sorted({column for prepared in prepared_runs.values() for column in prepared.feature_columns})
    lookup = {column: idx for idx, column in enumerate(all_columns)}
    aligned: dict[str, np.ndarray] = {}
    for well_id, prepared in prepared_runs.items():
        matrix = np.zeros((len(prepared.feature_matrix), len(all_columns)), dtype=np.float32)
        local_lookup = {column: idx for idx, column in enumerate(prepared.feature_columns)}
        for column, global_idx in lookup.items():
            local_idx = local_lookup.get(column)
            if local_idx is None:
                continue
            matrix[:, global_idx] = prepared.feature_matrix[:, local_idx]
        aligned[well_id] = matrix
    return all_columns, aligned


def _build_local_runs(
    detector_key: str,
    prepared_runs: dict[str, PreparedWellData],
    device: torch.device,
    verbose: bool,
) -> dict[str, PreparedDetectorRun]:
    out: dict[str, PreparedDetectorRun] = {}
    for well_id, prepared in prepared_runs.items():
        detector = _build_detector(detector_key, device=device, verbose=verbose)
        X_ref = prepared.feature_matrix[prepared.reference_mask]
        detector.fit_reference(X_ref, mask_ref=prepared.reference_mask)
        score_output = detector.score_stream(prepared.feature_matrix, mask_all=prepared.stability_mask)
        out[well_id] = PreparedDetectorRun(prepared=prepared, score_output=score_output)
    return out


def _fit_or_load_tranad_global(
    spec: DetectionSpec,
    prepared_runs: dict[str, PreparedWellData],
    aligned_features: dict[str, np.ndarray],
    device: torch.device,
    retune: bool,
    verbose: bool,
) -> tuple[TranADGlobalDetector, list[str]]:
    train_wells = [well_id for well_id, prepared in prepared_runs.items() if prepared.split == "train"]
    detector = TranADGlobalDetector(device=device, verbose=verbose)
    model_file = ensure_parent(model_path(spec, "tranad_global"))
    if model_file.exists() and not retune:
        detector.load(model_file)
        return detector, train_wells

    train_series = [aligned_features[well_id] for well_id in train_wells]
    train_masks = [
        prepared_runs[well_id].reference_mask & prepared_runs[well_id].onset_allowed_mask for well_id in train_wells
    ]
    detector.fit_global(train_series, train_masks)
    detector.save(model_file)
    if verbose:
        print(f"Saved TranAD global model to {model_file}")
    return detector, train_wells


def _build_tranad_runs(
    spec: DetectionSpec,
    prepared_runs: dict[str, PreparedWellData],
    device: torch.device,
    retune: bool,
    verbose: bool,
) -> dict[str, PreparedDetectorRun]:
    _, aligned = _align_features(prepared_runs)
    detector, _ = _fit_or_load_tranad_global(
        spec=spec,
        prepared_runs=prepared_runs,
        aligned_features=aligned,
        device=device,
        retune=retune,
        verbose=verbose,
    )
    out: dict[str, PreparedDetectorRun] = {}
    for well_id, prepared in prepared_runs.items():
        score_output = detector.score_stream(aligned[well_id], mask_all=prepared.stability_mask)
        out[well_id] = PreparedDetectorRun(prepared=prepared, score_output=score_output)
    return out


def _score_for_config(run: PreparedDetectorRun, detector_key: str, cfg: dict[str, Any]) -> np.ndarray:
    if detector_key != "paano_feat":
        return run.score_output.primary.astype(np.float32)

    short_score = run.score_output.components.get("paano_short")
    long_score = run.score_output.components.get("paano_long")
    if short_score is None or long_score is None:
        return run.score_output.primary.astype(np.float32)
    weight = float(cfg.get("fusion_weight_short", 0.60))
    return (weight * short_score + (1.0 - weight) * long_score).astype(np.float32)


def _detect_starts_for_run(
    detector_key: str,
    run: PreparedDetectorRun,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, Any, list[pd.Timestamp]]:
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
        gate_mode=str(cfg["gate_mode"]),
    )
    return score, thresholds, starts


def _candidate_configs(detector_key: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    weight_grid = PAANO_WEIGHT_GRID if detector_key == "paano_feat" else [None]
    for target_far_per_day in ONSET_TUNE_GRID["target_far_per_day"]:
        for min_run_points in ONSET_TUNE_GRID["min_run_points"]:
            for cooldown_hours in ONSET_TUNE_GRID["cooldown_hours"]:
                for ema_alpha in ONSET_TUNE_GRID["ema_alpha"]:
                    for gate_mode in ONSET_TUNE_GRID["gate_mode"]:
                        for fusion_weight_short in weight_grid:
                            cfg = DEFAULT_ONSET_CONFIG.copy()
                            cfg.update(
                                {
                                    "target_far_per_day": float(target_far_per_day),
                                    "min_run_points": int(min_run_points),
                                    "cooldown_hours": float(cooldown_hours),
                                    "ema_alpha": float(ema_alpha),
                                    "gate_mode": str(gate_mode),
                                }
                            )
                            if fusion_weight_short is not None:
                                cfg["fusion_weight_short"] = float(fusion_weight_short)
                            candidates.append(cfg)
    return candidates


def _tune_config(
    detector_key: str,
    train_runs: dict[str, PreparedDetectorRun],
    train_intervals: pd.DataFrame,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not train_runs:
        cfg = DEFAULT_ONSET_CONFIG.copy()
        if detector_key == "paano_feat":
            cfg["fusion_weight_short"] = 0.60
        return cfg, {"message": "No train runs available"}

    best_cfg: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float, float] | None = None
    leaderboard: list[dict[str, Any]] = []

    for cfg in _candidate_configs(detector_key):
        predicted: dict[str, list[pd.Timestamp]] = {}
        for well_id, run in train_runs.items():
            _, _, starts = _detect_starts_for_run(detector_key, run, cfg)
            predicted[well_id] = starts
        pred_df = predicted_from_mapping(predicted)
        summary, _ = evaluate_predictions(
            train_intervals,
            pred_df,
            scores=None,
            prestart_hours=PRESTART_TOLERANCE_HOURS,
        )
        key = summary_score_key(summary)
        leaderboard.append({"score_key": list(key), "config": cfg, "summary": summary})
        if best_key is None or key > best_key:
            best_cfg = cfg.copy()
            best_key = key

    leaderboard = sorted(leaderboard, key=lambda row: tuple(row["score_key"]), reverse=True)
    if best_cfg is None:
        best_cfg = DEFAULT_ONSET_CONFIG.copy()
    tuning_summary = {
        "best_score_key": list(best_key) if best_key is not None else None,
        "top10": leaderboard[:10],
    }
    if verbose and tuning_summary["top10"]:
        print("  Auto-tune top configs:")
        for idx, row in enumerate(tuning_summary["top10"][:5], 1):
            summary = row["summary"]
            cfg = row["config"]
            print(
                f"    {idx}. hit={summary['hit_count']}/{summary['interval_count']}, "
                f"p90_delay_ratio={summary['p90_delay_ratio']:.3f}, "
                f"FAR/day={summary['false_alarms_per_day']:.3f}, "
                f"starts/interval={summary['avg_starts_per_interval']:.2f}, "
                f"gate={cfg['gate_mode']}, run={cfg['min_run_points']}, cd={cfg['cooldown_hours']:.0f}, "
                f"ema={cfg['ema_alpha']:.2f}"
                + (
                    f", w={cfg['fusion_weight_short']:.2f}"
                    if "fusion_weight_short" in cfg
                    else ""
                )
            )
    return best_cfg, tuning_summary


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
            return payload["config"]
        if payload:
            return payload

    cfg, tuning_summary = _tune_config(detector_key, train_runs, train_intervals, verbose=verbose)
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
        if detector_key == "paano_feat":
            components["score"] = score
            short_score = components.get("paano_short")
            long_score = components.get("paano_long")
            if short_score is not None and long_score is not None:
                components["paano_fused"] = score
        else:
            components["score"] = score

        for idx, ts in enumerate(run.prepared.timestamps):
            row = {
                "well_id": well_id,
                "timestamp": ts,
                "split": run.prepared.split,
                "score": float(score[idx]),
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


def _choose_default_detector(detector_summaries: dict[str, dict[str, Any]], legacy_payload: dict[str, Any]) -> str:
    if "fused" not in detector_summaries:
        if "paano_feat" in detector_summaries:
            return "paano_feat"
        available = sorted(detector_summaries)
        return available[0] if available else DEFAULT_DETECTOR

    fused_summary = _summary_for_payload(detector_summaries["fused"])
    baseline_summary = None
    if "paano_feat" in detector_summaries:
        baseline_summary = _summary_for_payload(detector_summaries["paano_feat"])
    if legacy_payload:
        legacy_summary = _summary_for_payload(legacy_payload)
        if baseline_summary is None or float(legacy_summary.get("hit_count", -1)) > float(
            baseline_summary.get("hit_count", -1)
        ):
            baseline_summary = legacy_summary

    if baseline_summary is None:
        return "fused"

    fused_hit = float(fused_summary.get("hit_count", 0))
    base_hit = float(baseline_summary.get("hit_count", 0))
    fused_delay_ratio = float(fused_summary.get("p90_delay_ratio", np.inf))
    base_delay_ratio = float(baseline_summary.get("p90_delay_ratio", np.inf))
    if not np.isfinite(base_delay_ratio):
        base_delay_ratio = float(baseline_summary.get("p90_abs_delay_hours", np.inf))
        fused_delay_ratio = float(fused_summary.get("p90_abs_delay_hours", np.inf))
    fused_far = float(fused_summary.get("false_alarms_per_day", np.inf))
    base_far = float(baseline_summary.get("false_alarms_per_day", np.inf))
    far_ok = True if not np.isfinite(base_far) else fused_far <= base_far * 1.25
    if fused_hit >= base_hit and fused_delay_ratio <= base_delay_ratio and far_ok:
        return "fused"
    return "paano_feat" if "paano_feat" in detector_summaries else "fused"


def _update_benchmark_summary(spec: DetectionSpec) -> dict[str, Any]:
    detector_payloads: dict[str, dict[str, Any]] = {}
    for detector_key in DETECTOR_KEYS:
        path = summary_path(spec, detector_key)
        if path.exists():
            detector_payloads[detector_key] = load_json(path)
    legacy_payload = load_json(legacy_summary_path(spec))
    selected = _choose_default_detector(detector_payloads, legacy_payload)
    payload = {
        "anomaly": spec.anomaly_key,
        "selected_default_detector": selected,
        "detectors": detector_payloads,
    }
    if legacy_payload:
        payload["legacy_paano"] = legacy_payload
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_anomaly_data(spec, source_path=source_path)
    intervals = load_intervals(spec, required=True)
    if df.empty or intervals.empty:
        raise RuntimeError("Empty data or intervals for detection.")

    prepared_runs = _prepare_all_wells(spec, df, intervals, verbose=verbose)
    if not prepared_runs:
        raise RuntimeError("No wells survived engineered preprocessing.")

    if detector_key in LOCAL_DETECTOR_KEYS:
        detector_runs = _build_local_runs(detector_key, prepared_runs, device=device, verbose=verbose)
    else:
        detector_runs = _build_tranad_runs(
            spec=spec,
            prepared_runs=prepared_runs,
            device=device,
            retune=retune,
            verbose=verbose,
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

    result_df.to_csv(output, index=False)
    print(f"\nResults saved to {output}")

    score_output_path = scores_path(spec, detector_key)
    ensure_parent(score_output_path)
    score_df.to_csv(score_output_path, index=False)
    print(f"Per-point scores saved to {score_output_path}")

    pred_output_path = predicted_starts_path(spec, detector_key)
    ensure_parent(pred_output_path)
    pred_df.to_csv(pred_output_path, index=False)
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
    split = "train"
    if not intervals.empty:
        well_intervals = intervals[intervals["well_id"] == well_id]
        if not well_intervals.empty and "split" in well_intervals.columns:
            split = str(well_intervals["split"].iloc[0]).strip().lower()

    prepared = prepare_engineered_well(
        anomaly_key=spec.anomaly_key,
        well_id=well_id,
        split=split,
        well_df=well_df,
        patch_size=LONG_PATCH,
        reference_min_ratio=REFERENCE_MIN_RATIO,
        reference_max_ratio=REFERENCE_MAX_RATIO,
        reference_min_days=REFERENCE_MIN_DAYS,
        min_reference_coverage=MIN_REFERENCE_COVERAGE,
        min_total_coverage=MIN_TOTAL_COVERAGE,
    )
    if prepared is None:
        print("No usable data after engineered preprocessing.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if detector_key in LOCAL_DETECTOR_KEYS:
        detector_obj = _build_detector(detector_key, device=device, verbose=True)
        detector_obj.fit_reference(prepared.feature_matrix[prepared.reference_mask], mask_ref=prepared.reference_mask)
        score_output = detector_obj.score_stream(prepared.feature_matrix, mask_all=prepared.stability_mask)
        run = PreparedDetectorRun(prepared=prepared, score_output=score_output)
    else:
        train_df = load_anomaly_data(spec, source_path=source_path)
        prepared_runs = _prepare_all_wells(spec, train_df, intervals, verbose=False)
        detector_runs = _build_tranad_runs(
            spec=spec,
            prepared_runs=prepared_runs,
            device=device,
            retune=retune,
            verbose=True,
        )
        run = detector_runs[well_id]

    cfg_payload = load_json(config_path(spec, detector_key))
    cfg = cfg_payload.get("config", cfg_payload) if cfg_payload else DEFAULT_ONSET_CONFIG.copy()
    if detector_key == "paano_feat" and "fusion_weight_short" not in cfg:
        cfg["fusion_weight_short"] = 0.60

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
