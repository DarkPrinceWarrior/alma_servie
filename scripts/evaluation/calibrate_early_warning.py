from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from alma_service.onset_detection import (
    CausalThresholds,
    calibrate_causal_thresholds_from_reference_mask,
    detect_causal_onsets_masked,
)

DB = Path("/root/projects/alma_servie/db")
ANOMALIES = ["negermet", "pritok", "salt"]
SCALES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
EARLY_COOLDOWN_H = 3.0
EARLY_REARM_MIN = 30.0


def load_cfg(anomaly: str) -> dict:
    p = DB / f"{anomaly}_paano_global_config.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    return raw.get("config", raw)


def run_one(anomaly: str) -> None:
    p = DB / f"{anomaly}_paano_global_scores.parquet"
    if not p.exists():
        print(f"## {anomaly}: scores not found — SKIP")
        return
    cfg = load_cfg(anomaly)
    df = pd.read_parquet(p)
    df["well_id"] = df["well_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    target_far = float(cfg.get("target_far_per_day", 0.5))
    min_run = int(cfg.get("min_run_points", 3))
    cooldown = float(cfg.get("cooldown_hours", 12.0))
    rearm_min = float(cfg.get("rearm_window_minutes", 60.0))
    ema_alpha = float(cfg.get("ema_alpha", 0.08))
    gate_mode = str(cfg.get("gate_mode", "score_ema"))
    hyst = float(cfg.get("hysteresis_scale", 0.60))

    per_scale = {s: {"fp_count": 0, "days": 0.0, "early_events": 0} for s in SCALES}
    crit_summary = {"crit_count": 0, "days": 0.0}

    for wid, w in df.groupby("well_id"):
        w = w.sort_values("timestamp").reset_index(drop=True)
        ts = w["timestamp"].to_numpy()
        score = w["score"].to_numpy(np.float64)
        lab = w["is_labelled_anomaly"].to_numpy(np.int64)
        pre = w["is_pre_anomaly_zone"].to_numpy(np.int64)

        ref_mask = (lab == 0) & (pre == 0)
        if ref_mask.sum() < 100:
            continue
        head = np.cumsum(ref_mask) < int(ref_mask.sum() * 0.20)
        ref_for_calib = ref_mask & head
        if ref_for_calib.sum() < 50:
            ref_for_calib = ref_mask
        try:
            thresholds, diagnostics = calibrate_causal_thresholds_from_reference_mask(
                scores=score,
                timestamps=ts,
                reference_mask=ref_for_calib,
                target_far_per_day=target_far,
                min_run_points=min_run,
                ema_alpha=ema_alpha,
            )
        except Exception as exc:
            print(f"  {anomaly}/{wid}: calibrate failed: {exc!r}")
            continue

        onset_mask_fp = (lab == 0) & (pre == 0)
        if onset_mask_fp.sum() == 0:
            continue
        normal_days = float(onset_mask_fp.sum()) * 5.0 / (60.0 * 24.0)

        crit_starts = detect_causal_onsets_masked(
            scores=score,
            timestamps=ts,
            diagnostics=diagnostics,
            thresholds=thresholds,
            reference_mask=ref_for_calib,
            onset_mask=onset_mask_fp,
            min_run_points=min_run,
            cooldown_hours=cooldown,
            rearm_window_minutes=rearm_min,
            gate_mode=gate_mode,
            hysteresis_scale=hyst,
            bypass_cooldown_after_clear=bool(cfg.get("bypass_cooldown_after_clear", True)),
        )
        crit_summary["crit_count"] += len(crit_starts)
        crit_summary["days"] += normal_days

        for scale in SCALES:
            early_thr = CausalThresholds(
                score_threshold=thresholds.score_threshold * scale,
                ema_z_threshold=thresholds.ema_z_threshold * scale,
                cusum_threshold=thresholds.cusum_threshold * scale,
                drift=thresholds.drift,
                baseline_median=thresholds.baseline_median,
                baseline_mad=thresholds.baseline_mad,
                quantile=thresholds.quantile,
            )
            try:
                early = detect_causal_onsets_masked(
                    scores=score,
                    timestamps=ts,
                    diagnostics=diagnostics,
                    thresholds=early_thr,
                    reference_mask=ref_for_calib,
                    onset_mask=onset_mask_fp,
                    min_run_points=min_run,
                    cooldown_hours=EARLY_COOLDOWN_H,
                    rearm_window_minutes=EARLY_REARM_MIN,
                    gate_mode=gate_mode,
                    hysteresis_scale=hyst,
                    bypass_cooldown_after_clear=bool(cfg.get("bypass_cooldown_after_clear", True)),
                )
            except Exception:
                early = []
            per_scale[scale]["fp_count"] += len(early)
            per_scale[scale]["days"] += normal_days

        onset_mask_lead = (lab == 0)
        for scale in SCALES:
            early_thr = CausalThresholds(
                score_threshold=thresholds.score_threshold * scale,
                ema_z_threshold=thresholds.ema_z_threshold * scale,
                cusum_threshold=thresholds.cusum_threshold * scale,
                drift=thresholds.drift,
                baseline_median=thresholds.baseline_median,
                baseline_mad=thresholds.baseline_mad,
                quantile=thresholds.quantile,
            )
            try:
                early_lead = detect_causal_onsets_masked(
                    scores=score,
                    timestamps=ts,
                    diagnostics=diagnostics,
                    thresholds=early_thr,
                    reference_mask=ref_for_calib,
                    onset_mask=onset_mask_lead,
                    min_run_points=min_run,
                    cooldown_hours=EARLY_COOLDOWN_H,
                    rearm_window_minutes=EARLY_REARM_MIN,
                    gate_mode=gate_mode,
                    hysteresis_scale=hyst,
                    bypass_cooldown_after_clear=bool(cfg.get("bypass_cooldown_after_clear", True)),
                )
            except Exception:
                early_lead = []
            extra = max(0, len(early_lead) - per_scale[scale]["fp_count"])
            per_scale[scale]["early_events"] += extra

    crit_rate = (
        crit_summary["crit_count"] / max(crit_summary["days"], 1e-9)
        if crit_summary["days"] > 0 else float("nan")
    )
    print(f"\n{'=' * 76}")
    print(f"## {anomaly.upper()} — early_warning threshold scale calibration")
    print(f"{'=' * 76}")
    print(f"Production config: target_far={target_far}, min_run={min_run}, "
          f"cooldown={cooldown}h, ema_alpha={ema_alpha}, gate={gate_mode}, hyst={hyst}")
    print(f"Critical baseline FP-rate (на normal-части): {crit_rate:.4f}/день/скв.")
    print()
    print(f"{'scale':>6s}  {'FP/день':>10s}  {'против цели 0.5':>15s}  {'доп. early events':>20s}")
    print("-" * 60)
    best = None
    for scale in SCALES:
        rec = per_scale[scale]
        rate = rec["fp_count"] / max(rec["days"], 1e-9) if rec["days"] > 0 else float("nan")
        diff = abs(rate - 0.5) if rate == rate else float("inf")
        marker = ""
        if best is None or diff < best[1]:
            best = (scale, diff, rate)
        print(f"{scale:>6.2f}  {rate:>10.4f}  {'='*max(0,min(15,int(rate*15))):>15s}  {rec['early_events']:>20d}{marker}")
    if best is not None and best[1] != float("inf"):
        print(f"\n>>> Ближайший к 0.5/день: scale={best[0]:.2f} → {best[2]:.4f} FP/день")


if __name__ == "__main__":
    for a in ANOMALIES:
        run_one(a)
    print("\nDONE")
