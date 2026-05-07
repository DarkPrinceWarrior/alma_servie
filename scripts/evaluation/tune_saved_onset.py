from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.anomaly_specs import get_detection_spec
from alma_service.benchmark_metrics import (
    load_intervals,
    load_scores,
    predicted_from_mapping,
    summarize_splits,
)
from alma_service.detection_artifacts import (
    DEFAULT_DETECTOR,
    config_path,
    load_json,
    predicted_starts_path,
    scores_path,
    tuning_path,
)
from alma_service.generic_detection import (
    BASE_ONSET_CONFIG,
    PRESSURE_TREND_WEIGHT_GRID,
    SALT_TREND_WEIGHT_GRID,
    _default_onset_config,
    _onset_tune_grid,
)
from alma_service.onset_detection import (
    calibrate_causal_thresholds_from_reference_mask,
    detect_causal_onsets_masked,
)
from alma_service.paano_defaults import PRESTART_TOLERANCE_HOURS
from alma_service.tabular_io import write_table


def _bool_values(raw: str) -> list[bool]:
    values: list[bool] = []
    for item in raw.split(","):
        normalized = item.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            values.append(True)
        elif normalized in {"0", "false", "no", "n"}:
            values.append(False)
        else:
            raise argparse.ArgumentTypeError(f"Invalid boolean value: {item}")
    return values


def _float_values(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _int_values(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _str_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _load_config(spec_key: str, detector: str) -> dict[str, Any]:
    spec = get_detection_spec(spec_key)
    payload = load_json(config_path(spec, detector))
    if isinstance(payload, dict) and "config" in payload:
        return {**_default_onset_config(spec_key, detector), **payload["config"]}
    if isinstance(payload, dict):
        return {**_default_onset_config(spec_key, detector), **payload}
    return _default_onset_config(spec_key, detector)


def _config_grid(args: argparse.Namespace, base_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = {
        "target_far_per_day": args.target_far_per_day
        or [float(base_cfg.get("target_far_per_day", BASE_ONSET_CONFIG["target_far_per_day"]))],
        "min_run_points": args.min_run_points
        or [int(base_cfg.get("min_run_points", BASE_ONSET_CONFIG["min_run_points"]))],
        "cooldown_hours": args.cooldown_hours
        or [float(base_cfg.get("cooldown_hours", BASE_ONSET_CONFIG["cooldown_hours"]))],
        "rearm_window_minutes": args.rearm_window_minutes
        or [float(base_cfg.get("rearm_window_minutes", BASE_ONSET_CONFIG["rearm_window_minutes"]))],
        "ema_alpha": args.ema_alpha or [float(base_cfg.get("ema_alpha", BASE_ONSET_CONFIG["ema_alpha"]))],
        "gate_mode": args.gate_mode or [str(base_cfg.get("gate_mode", BASE_ONSET_CONFIG["gate_mode"]))],
        "bypass_cooldown_after_clear": args.bypass_cooldown_after_clear
        or [bool(base_cfg.get("bypass_cooldown_after_clear", BASE_ONSET_CONFIG["bypass_cooldown_after_clear"]))],
    }
    physical_weight_name = None
    physical_weight_values: list[float] = []
    if args.anomaly == "pritok" and args.detector == "paano_shared":
        physical_weight_name = "pressure_trend_weight"
        physical_weight_values = args.pressure_trend_weight or [
            float(base_cfg.get("pressure_trend_weight", PRESSURE_TREND_WEIGHT_GRID[0]))
        ]
    elif args.anomaly == "salt" and args.detector == "paano_shared":
        physical_weight_name = "salt_trend_weight"
        physical_weight_values = args.salt_trend_weight or [
            float(base_cfg.get("salt_trend_weight", SALT_TREND_WEIGHT_GRID[0]))
        ]

    configs: list[dict[str, Any]] = []
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        cfg = {**base_cfg}
        cfg.update(dict(zip(keys, values, strict=True)))
        if physical_weight_name is None:
            configs.append(cfg)
            continue
        for weight in physical_weight_values:
            weighted = {**cfg, physical_weight_name: float(weight)}
            configs.append(weighted)
    return configs


def _reference_mask_from_scores(well_scores: pd.DataFrame, interval_rows: pd.DataFrame) -> np.ndarray:
    if "reference_mask" in well_scores.columns:
        return well_scores["reference_mask"].astype(bool).to_numpy()
    timestamps = pd.to_datetime(well_scores["timestamp"])
    mask = np.ones(len(well_scores), dtype=bool)
    pre_tol = pd.Timedelta(hours=float(PRESTART_TOLERANCE_HOURS))
    for _, row in interval_rows.iterrows():
        mask &= timestamps < (pd.Timestamp(row["start_date"]) - pre_tol)
    if int(mask.sum()) < max(8, int(len(mask) * 0.02)):
        first_start = pd.Timestamp(interval_rows["start_date"].min()) if not interval_rows.empty else timestamps.min()
        mask = (timestamps < first_start).to_numpy(dtype=bool)
    return np.asarray(mask, dtype=bool)


def _onset_mask_from_scores(well_scores: pd.DataFrame) -> np.ndarray:
    if "onset_allowed_mask" in well_scores.columns:
        return well_scores["onset_allowed_mask"].astype(bool).to_numpy()
    if "stability_mask" in well_scores.columns:
        return well_scores["stability_mask"].astype(bool).to_numpy()
    return np.ones(len(well_scores), dtype=bool)


def _score_for_config(well_scores: pd.DataFrame, cfg: dict[str, Any], anomaly: str, detector: str) -> np.ndarray:
    if "paano_score" in well_scores.columns:
        model = well_scores["paano_score"].to_numpy(dtype=np.float32)
    else:
        model = well_scores["score"].to_numpy(dtype=np.float32)

    if anomaly == "pritok" and detector == "paano_shared" and "pressure_trend_score" in well_scores.columns:
        weight = float(cfg.get("pressure_trend_weight", 0.0))
        return (model + weight * well_scores["pressure_trend_score"].to_numpy(dtype=np.float32)).astype(np.float32)
    if anomaly == "salt" and detector == "paano_shared" and "salt_deposition_calibrated_fusion_score" in well_scores.columns:
        weight = float(cfg.get("salt_trend_weight", 0.0))
        return (
            model
            + weight * well_scores["salt_deposition_calibrated_fusion_score"].to_numpy(dtype=np.float32)
        ).astype(np.float32)
    return model.astype(np.float32)


def _predict_from_scores(
    scores: pd.DataFrame,
    intervals: pd.DataFrame,
    cfg: dict[str, Any],
    anomaly: str,
    detector: str,
) -> pd.DataFrame:
    rows: dict[str, list[pd.Timestamp]] = {}
    for well_id, well_scores in scores.groupby("well_id", sort=True):
        well_scores = well_scores.sort_values("timestamp")
        well_intervals = intervals[intervals["well_id"] == well_id].copy()
        if well_scores.empty:
            continue
        score = _score_for_config(well_scores, cfg, anomaly, detector)
        reference_mask = _reference_mask_from_scores(well_scores, well_intervals)
        timestamps = pd.to_datetime(well_scores["timestamp"]).to_numpy()
        thresholds, diagnostics = calibrate_causal_thresholds_from_reference_mask(
            scores=score,
            timestamps=timestamps,
            reference_mask=reference_mask,
            target_far_per_day=float(cfg["target_far_per_day"]),
            min_run_points=int(cfg["min_run_points"]),
            ema_alpha=float(cfg["ema_alpha"]),
        )
        starts = detect_causal_onsets_masked(
            scores=score,
            timestamps=timestamps,
            diagnostics=diagnostics,
            thresholds=thresholds,
            reference_mask=reference_mask,
            onset_mask=_onset_mask_from_scores(well_scores),
            min_run_points=int(cfg["min_run_points"]),
            cooldown_hours=float(cfg["cooldown_hours"]),
            rearm_window_minutes=float(cfg["rearm_window_minutes"]),
            gate_mode=str(cfg["gate_mode"]),
            hysteresis_scale=float(cfg.get("hysteresis_scale", BASE_ONSET_CONFIG["hysteresis_scale"])),
            bypass_cooldown_after_clear=bool(
                cfg.get("bypass_cooldown_after_clear", BASE_ONSET_CONFIG["bypass_cooldown_after_clear"])
            ),
        )
        rows[str(well_id)] = starts
    pred_df = predicted_from_mapping(rows)
    if not pred_df.empty and "split" in scores.columns:
        split_lookup = scores.groupby("well_id")["split"].first().to_dict()
        pred_df["split"] = pred_df["well_id"].map(split_lookup).fillna("train")
    return pred_df


def _score_key(summary: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(summary.get("hit_count", 0)),
        -float(summary.get("duplicate_starts_inside_interval", 1e9)),
        -float(summary.get("false_alarms_per_day", 1e9)),
        -float(summary.get("avg_starts_per_interval", 1e9)),
        -float(summary.get("p90_delay_ratio", 1e9)),
        -float(summary.get("median_abs_delay_hours", 1e9)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune onset configs from saved detector scores without retraining.")
    parser.add_argument("--anomaly", choices=["negermet", "pritok", "salt"], required=True)
    parser.add_argument("--detector", default=DEFAULT_DETECTOR)
    parser.add_argument("--scores", default=None)
    parser.add_argument("--intervals", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    parser.add_argument("--max-p90-delay-ratio", type=float, default=None)
    parser.add_argument("--max-far-per-day", type=float, default=None)
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--write-predicted", action="store_true")
    parser.add_argument("--target-far-per-day", type=_float_values, default=None)
    parser.add_argument("--min-run-points", type=_int_values, default=None)
    parser.add_argument("--cooldown-hours", type=_float_values, default=None)
    parser.add_argument("--rearm-window-minutes", type=_float_values, default=None)
    parser.add_argument("--ema-alpha", type=_float_values, default=None)
    parser.add_argument("--gate-mode", type=_str_values, default=None)
    parser.add_argument("--bypass-cooldown-after-clear", type=_bool_values, default=None)
    parser.add_argument("--pressure-trend-weight", type=_float_values, default=None)
    parser.add_argument("--salt-trend-weight", type=_float_values, default=None)
    args = parser.parse_args()

    spec = get_detection_spec(args.anomaly)
    scores = load_scores(args.scores or scores_path(spec, args.detector))
    intervals = load_intervals(args.intervals or spec.dataset.intervals_path)
    base_cfg = _load_config(args.anomaly, args.detector)
    configs = _config_grid(args, base_cfg)
    ranked: list[dict[str, Any]] = []
    for cfg in configs:
        predictions = _predict_from_scores(scores, intervals, cfg, args.anomaly, args.detector)
        split_summaries, _ = summarize_splits(
            intervals=intervals,
            predictions=predictions,
            scores=scores,
            prestart_hours=PRESTART_TOLERANCE_HOURS,
        )
        summary = split_summaries["all"]
        hit_rate = float(summary.get("hit_rate", 0.0))
        p90 = float(summary.get("p90_delay_ratio", 1e9))
        far = float(summary.get("false_alarms_per_day", 1e9))
        if hit_rate < float(args.min_hit_rate):
            continue
        if args.max_p90_delay_ratio is not None and p90 > float(args.max_p90_delay_ratio):
            continue
        if args.max_far_per_day is not None and far > float(args.max_far_per_day):
            continue
        ranked.append({"score_key": _score_key(summary), "config": cfg, "summary": summary})
    ranked.sort(key=lambda row: row["score_key"], reverse=True)

    output = {
        "anomaly": args.anomaly,
        "detector": args.detector,
        "config_count": len(configs),
        "kept_count": len(ranked),
        "top": ranked[: args.top_k],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

    if ranked and (args.write_config or args.write_predicted):
        best = ranked[0]
        if args.write_config:
            path = config_path(spec, args.detector)
            path.write_text(
                json.dumps({"detector": args.detector, "config": best["config"]}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tune_path = tuning_path(spec, args.detector)
            tune_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved config: {path}")
            print(f"Saved tuning summary: {tune_path}")
        if args.write_predicted:
            predictions = _predict_from_scores(scores, intervals, best["config"], args.anomaly, args.detector)
            path = predicted_starts_path(spec, args.detector)
            write_table(predictions, path)
            print(f"Saved predicted starts: {path}")


if __name__ == "__main__":
    main()
