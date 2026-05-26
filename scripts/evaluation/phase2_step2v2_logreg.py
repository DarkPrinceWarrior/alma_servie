from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DB = Path("/root/projects/alma_servie/db")
ANOMALIES = ["negermet", "pritok", "salt"]
PRESSURE_COL = "Давление на приеме насоса кгс/см²"
EMA_ALPHA = 0.08
ROLL_WIN_POINTS = 24
FA_BUDGET_PER_DAY = 0.5
LEAD_LOOKBACK_HOURS = 48.0


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if len(values) == 0:
        return out
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, ins, s = [], False, 0
    for i, v in enumerate(mask):
        if v and not ins:
            s, ins = i, True
        elif not v and ins:
            out.append((s, i - 1))
            ins = False
    if ins:
        out.append((s, len(mask) - 1))
    return out


def build_features(anomaly: str) -> pd.DataFrame:
    scores = pd.read_parquet(DB / f"{anomaly}_paano_global_scores.parquet")
    scores["well_id"] = scores["well_id"].astype(str)
    scores["timestamp"] = pd.to_datetime(scores["timestamp"])

    raw = pd.read_parquet(
        DB / f"{anomaly}_anomaly_database_5min.parquet",
        columns=["timestamp", "well_id", PRESSURE_COL],
    )
    raw["well_id"] = raw["well_id"].astype(str)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.rename(columns={PRESSURE_COL: "pressure"})

    df = scores.merge(raw, on=["well_id", "timestamp"], how="left")
    parts = []
    for wid, w in df.groupby("well_id", sort=True):
        w = w.sort_values("timestamp").reset_index(drop=True).copy()
        sc = w["score"].to_numpy(np.float64)
        sc = np.nan_to_num(sc, nan=0.0)
        smoothed = ema(sc, EMA_ALPHA)
        delta = np.zeros_like(smoothed)
        delta[1:] = smoothed[1:] - smoothed[:-1]
        rolling_sigma_score = (
            pd.Series(sc).rolling(ROLL_WIN_POINTS, min_periods=ROLL_WIN_POINTS // 2).std().fillna(0.0).to_numpy()
        )
        press = w["pressure"].astype(np.float64).to_numpy()
        rolling_sigma_press = (
            pd.Series(press).rolling(ROLL_WIN_POINTS, min_periods=ROLL_WIN_POINTS // 2).std().fillna(0.0).to_numpy()
        )
        w["f_smoothed_score"] = smoothed
        w["f_delta_ema_score"] = delta
        w["f_rolling_sigma_score"] = rolling_sigma_score
        w["f_rolling_sigma_pressure"] = rolling_sigma_press
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def train_eval_one(anomaly: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"## {anomaly.upper()} — 2.2-v2 logreg precursor")
    print(f"{'=' * 78}")
    df = build_features(anomaly)

    feature_cols = [
        "f_smoothed_score",
        "f_delta_ema_score",
        "f_rolling_sigma_score",
        "f_rolling_sigma_pressure",
    ]
    df["target"] = (df["is_pre_anomaly_zone"] > 0).astype(int)
    train_mask = (df["is_labelled_anomaly"] == 0) & df[feature_cols].notna().all(axis=1)
    df_tr = df[train_mask].copy()
    n_pos = int(df_tr["target"].sum())
    n_neg = int((df_tr["target"] == 0).sum())
    print(f"Точек train: {len(df_tr):,}  pos(pre_anomaly): {n_pos:,}  neg(normal): {n_neg:,}")
    print(f"Скв: {df_tr['well_id'].nunique()}")

    wells = sorted(df_tr["well_id"].unique())
    oof_proba = pd.Series(np.nan, index=df_tr.index, dtype=np.float64)
    coef_runs = []
    for hold in wells:
        train = df_tr[df_tr["well_id"] != hold]
        test = df_tr[df_tr["well_id"] == hold]
        if train["target"].sum() < 5 or train["target"].nunique() < 2:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(train[feature_cols])
        Xte = scaler.transform(test[feature_cols])
        ytr = train["target"].to_numpy()
        model = LogisticRegression(
            class_weight="balanced", max_iter=2000, solver="lbfgs"
        )
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        oof_proba.loc[test.index] = proba
        coef_runs.append(model.coef_.ravel())
    df_tr["oof_proba"] = oof_proba

    valid = df_tr["oof_proba"].notna()
    auc = _safe_auc(df_tr.loc[valid & (df_tr["target"] == 1), "oof_proba"].to_numpy(),
                    df_tr.loc[valid & (df_tr["target"] == 0), "oof_proba"].to_numpy())
    print(f"\nLOWO OOF ROC-AUC pre vs normal: {auc:.4f}")

    if coef_runs:
        coef_mean = np.mean(coef_runs, axis=0)
        print(f"Средние коэф. (нормированные):")
        for name, c in zip(feature_cols, coef_mean):
            print(f"  {name:30s}: {c:+.3f}")

    neg_proba = df_tr.loc[valid & (df_tr["target"] == 0), "oof_proba"].to_numpy()
    if len(neg_proba) == 0:
        print("Нет негативов для калибровки порога — SKIP")
        return

    full = df.copy()
    full = full.merge(
        df_tr[["well_id", "timestamp", "oof_proba"]],
        on=["well_id", "timestamp"], how="left",
    )

    cooldown_h = 3.0
    min_run = 3
    print(f"\nКалибровка порога под FA-budget ≤ {FA_BUDGET_PER_DAY}/день (cooldown={cooldown_h}ч, min_run={min_run}):")
    print(f"{'thr':>8s}  {'FP/день':>10s}  {'TP события':>11s}  {'lead median, ч':>16s}  {'lead max, ч':>13s}")
    print("-" * 70)

    intervals_path = DB / f"{anomaly}_intervals.parquet"
    intervals = pd.read_parquet(intervals_path) if intervals_path.exists() else pd.DataFrame()
    if not intervals.empty:
        intervals["well_id"] = intervals["well_id"].astype(str)
        intervals["start_date"] = pd.to_datetime(intervals["start_date"])

    thr_grid = np.quantile(neg_proba, [0.80, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999])
    best = None
    for thr in thr_grid:
        fp_total = 0
        fp_days = 0.0
        tp_events = 0
        leads = []
        for wid, w in full.groupby("well_id"):
            w = w.sort_values("timestamp").reset_index(drop=True)
            ts = w["timestamp"]
            proba = w["oof_proba"].fillna(0.0).to_numpy()
            lab = w["is_labelled_anomaly"].to_numpy(np.int64)
            pre = w["is_pre_anomaly_zone"].to_numpy(np.int64)
            normal_mask = (lab == 0) & (pre == 0)
            normal_days = float(normal_mask.sum()) * 5.0 / (60.0 * 24.0)
            fp_days += normal_days
            fires = _detect_simple(proba, thr, min_run, cooldown_h, ts)
            for fire_idx in fires:
                if normal_mask[fire_idx]:
                    fp_total += 1
            if not intervals.empty:
                w_intervals = intervals[intervals["well_id"] == wid]
                for _, row in w_intervals.iterrows():
                    start = row["start_date"]
                    lookback_start = start - pd.Timedelta(hours=LEAD_LOOKBACK_HOURS)
                    first_lead = None
                    for fire_idx in fires:
                        fire_ts = ts.iloc[fire_idx]
                        if lookback_start <= fire_ts < start:
                            first_lead = (start - fire_ts).total_seconds() / 3600
                            break
                    if first_lead is not None:
                        tp_events += 1
                        leads.append(first_lead)
        fp_rate = fp_total / max(fp_days, 1e-9)
        lead_med = float(np.median(leads)) if leads else float("nan")
        lead_max = float(np.max(leads)) if leads else float("nan")
        marker = "  ← within budget" if fp_rate <= FA_BUDGET_PER_DAY else ""
        print(f"{thr:>8.4f}  {fp_rate:>10.4f}  {tp_events:>11d}  {lead_med:>16.2f}  {lead_max:>13.2f}{marker}")
        if fp_rate <= FA_BUDGET_PER_DAY:
            if best is None or tp_events > best[2]:
                best = (thr, fp_rate, tp_events, lead_med, lead_max)
    if best:
        thr, fp, tp, lm, lx = best
        print(f"\n>>> Best within budget: thr={thr:.4f}, FP/день={fp:.3f}, TP события={tp}, lead median={lm:.2f}ч, max={lx:.2f}ч")
    else:
        print("\n>>> Ни одна точка thr_grid не вписалась в FA budget")


def _detect_simple(proba: np.ndarray, thr: float, min_run: int, cooldown_h: float, ts: pd.Series) -> list[int]:
    fires = []
    cooldown_ns = int(cooldown_h * 3600 * 1e9)
    last_fire_ns = -10**18
    run = 0
    for i in range(len(proba)):
        if proba[i] >= thr:
            run += 1
            if run >= min_run:
                t_ns = pd.Timestamp(ts.iloc[i]).value
                if t_ns - last_fire_ns >= cooldown_ns:
                    fires.append(i)
                    last_fire_ns = t_ns
                    run = 0
        else:
            run = 0
    return fires


def _safe_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n1, n0 = len(pos), len(neg)
    all_s = np.concatenate([pos, neg])
    ranks = pd.Series(all_s).rank().to_numpy()
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0))


if __name__ == "__main__":
    for a in ANOMALIES:
        try:
            train_eval_one(a)
        except Exception as exc:
            print(f"\n{a}: failed — {exc!r}")
    print("\nDONE")
