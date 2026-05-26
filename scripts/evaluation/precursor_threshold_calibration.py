from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from alma_service.precursor_logreg import (
    extract_features,
    load_model,
    score_proba,
)

DB = Path("/root/projects/alma_servie/db")
PRESSURE_COL = "Давление на приеме насоса кгс/см²"
FA_BUDGET_PER_DAY = 0.5
COOLDOWN_H = 3.0
MIN_RUN = 3
LEAD_LOOKBACK_HOURS = 48.0


def detect(proba: np.ndarray, thr: float, ts: np.ndarray) -> list[int]:
    fires = []
    cd_ns = int(COOLDOWN_H * 3600 * 1e9)
    last = -10**18
    run = 0
    for i in range(len(proba)):
        if proba[i] >= thr:
            run += 1
            if run >= MIN_RUN:
                t_ns = pd.Timestamp(ts[i]).value
                if t_ns - last >= cd_ns:
                    fires.append(i)
                    last = t_ns
                    run = 0
        else:
            run = 0
    return fires


def calib_one(anomaly: str) -> tuple[str, float | None]:
    print(f"\n{'=' * 70}")
    print(f"## {anomaly.upper()} — re-calibration")
    print(f"{'=' * 70}")
    model_path = DB / f"{anomaly}_paano_global_precursor.json"
    model = load_model(model_path)
    assert model is not None, f"no model at {model_path}"

    scores = pd.read_parquet(DB / f"{anomaly}_paano_global_scores.parquet")
    scores["well_id"] = scores["well_id"].astype(str)
    scores["timestamp"] = pd.to_datetime(scores["timestamp"])
    intervals = pd.read_parquet(DB / f"{anomaly}_intervals.parquet")
    intervals["well_id"] = intervals["well_id"].astype(str)
    intervals["start_date"] = pd.to_datetime(intervals["start_date"])
    raw = pd.read_parquet(
        DB / f"{anomaly}_anomaly_database_5min.parquet",
        columns=["timestamp", "well_id", PRESSURE_COL],
    )
    raw["well_id"] = raw["well_id"].astype(str)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    all_neg_proba = []
    well_data = {}
    for wid, w in scores.groupby("well_id"):
        w = w.sort_values("timestamp").reset_index(drop=True)
        ts = w["timestamp"]
        score = w["score"].to_numpy(np.float64)
        lab = w["is_labelled_anomaly"].to_numpy(np.int64)
        pre = w["is_pre_anomaly_zone"].to_numpy(np.int64)
        w_raw = raw[raw["well_id"] == wid].set_index("timestamp")
        w_raw_aligned = w_raw.reindex(ts).ffill()
        rm = w_raw_aligned[[PRESSURE_COL]].to_numpy()
        feats = extract_features(ts.to_numpy(), score, [PRESSURE_COL], rm)
        proba = score_proba(model, feats)
        well_data[str(wid)] = (ts.to_numpy(), proba, lab, pre)
        all_neg_proba.append(proba[(lab == 0) & (pre == 0)])

    neg = np.concatenate(all_neg_proba)
    thr_grid = np.quantile(neg, [0.80, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995])
    print(f"  {'thr':>10s} {'FP/день':>10s} {'TP':>5s} {'lead median':>13s} {'lead max':>10s}")
    print("-" * 56)
    best = None
    for thr in thr_grid:
        fp_total = 0
        fp_days = 0.0
        tp = 0
        leads = []
        for wid, (ts, proba, lab, pre) in well_data.items():
            normal_days = float(((lab == 0) & (pre == 0)).sum()) * 5.0 / 1440.0
            fp_days += normal_days
            fires = detect(proba, thr, ts)
            for fi in fires:
                if lab[fi] == 0 and pre[fi] == 0:
                    fp_total += 1
            w_iv = intervals[intervals["well_id"] == wid]
            for _, row in w_iv.iterrows():
                start = row["start_date"]
                lb_start = start - pd.Timedelta(hours=LEAD_LOOKBACK_HOURS)
                first_lead = None
                for fi in fires:
                    ft = pd.Timestamp(ts[fi])
                    if lb_start <= ft < start:
                        first_lead = (start - ft).total_seconds() / 3600
                        break
                if first_lead is not None:
                    tp += 1
                    leads.append(first_lead)
        fp_rate = fp_total / max(fp_days, 1e-9)
        lm = np.median(leads) if leads else float("nan")
        lx = max(leads) if leads else float("nan")
        flag = " <within" if fp_rate <= FA_BUDGET_PER_DAY else ""
        print(f"  {thr:>10.4f} {fp_rate:>10.4f} {tp:>5d} {lm:>13.2f} {lx:>10.2f}{flag}")
        if fp_rate <= FA_BUDGET_PER_DAY and (best is None or tp > best[2] or (tp == best[2] and lm > best[3])):
            best = (thr, fp_rate, tp, lm, lx)
    if best:
        thr, fp, tp, lm, lx = best
        print(f"\n  >>> Best: thr={thr:.4f}, FP/день={fp:.3f}, TP={tp}, lead median={lm:.2f}ч")
        return anomaly, thr
    print("\n  >>> No threshold within budget — keep current default")
    return anomaly, None


if __name__ == "__main__":
    results = {}
    for a in ["negermet", "pritok", "salt"]:
        anomaly, thr = calib_one(a)
        results[anomaly] = thr
    print("\nRecommended ANOMALY_PRECURSOR_DEFAULT_THRESHOLD:")
    for k, v in results.items():
        print(f"  {k}: {v if v is not None else '(no change)'}")
