from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from alma_service.precursor_logreg import (
    extract_features,
    train_lowo,
    save_model,
    score_proba,
)
from alma_service.generic_detection import PRECURSOR_TRAIN_PRESTART_HOURS

DB = Path("/root/projects/alma_servie/db")
PRESSURE_COL = "Давление на приеме насоса кгс/см²"
DEFAULT_THR = {"negermet": 0.0035, "pritok": 0.917, "salt": 0.852}


def smoke_one(anomaly: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"## {anomaly.upper()} — smoke")
    print(f"{'=' * 70}")
    scores = pd.read_parquet(DB / f"{anomaly}_paano_global_scores.parquet")
    scores["well_id"] = scores["well_id"].astype(str)
    scores["timestamp"] = pd.to_datetime(scores["timestamp"])
    intervals = pd.read_parquet(DB / f"{anomaly}_intervals.parquet")
    intervals["well_id"] = intervals["well_id"].astype(str)
    intervals["start_date"] = pd.to_datetime(intervals["start_date"])
    intervals["end_date"] = pd.to_datetime(intervals["end_date"])
    raw = pd.read_parquet(
        DB / f"{anomaly}_anomaly_database_5min.parquet",
        columns=["timestamp", "well_id", PRESSURE_COL],
    )
    raw["well_id"] = raw["well_id"].astype(str)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    pre_tol = pd.Timedelta(hours=float(PRECURSOR_TRAIN_PRESTART_HOURS))
    per_well = {}
    for wid, w in scores.groupby("well_id"):
        w = w.sort_values("timestamp").reset_index(drop=True)
        ts = w["timestamp"]
        score = w["score"].to_numpy(np.float64)
        w_raw = raw[raw["well_id"] == wid].set_index("timestamp")
        w_raw_aligned = w_raw.reindex(ts).ffill()
        raw_matrix = w_raw_aligned[[PRESSURE_COL]].to_numpy()
        raw_columns = [PRESSURE_COL]

        n = len(ts)
        labelled = np.zeros(n, dtype=bool)
        pre = np.zeros(n, dtype=bool)
        for _, row in intervals[intervals["well_id"] == wid].iterrows():
            start, end = row["start_date"], row["end_date"]
            labelled |= (ts >= start) & (ts <= end)
            pre |= (ts >= start - pre_tol) & (ts < start)
        valid = ~labelled
        target = pre.astype(int)
        feats = extract_features(ts.to_numpy(), score, raw_columns, raw_matrix)
        per_well[str(wid)] = {"features": feats, "target": target, "valid": valid}

    model = train_lowo(per_well)
    save_path = DB / f"{anomaly}_paano_global_precursor.json"
    save_model(model, save_path)
    print(f"  Model saved: {save_path}")
    print(
        f"  pos={model.train_pos}, neg={model.train_neg}, "
        f"n_wells={model.n_wells}, OOF_AUC={model.oof_auc:.4f}"
    )
    for fname, c in zip(model.feature_names, model.coef):
        print(f"    {fname:30s}: {c:+.3f}")

    sample_well = sorted(per_well.keys())[0]
    feats = per_well[sample_well]["features"]
    proba = score_proba(model, feats)
    thr = DEFAULT_THR[anomaly]
    fires = int((proba > thr).sum())
    print(f"  Sample well {sample_well}: proba range [{proba.min():.4f}..{proba.max():.4f}], points > {thr}: {fires}")


if __name__ == "__main__":
    for a in ["negermet", "pritok", "salt"]:
        smoke_one(a)
    print("\nDONE")
