from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DB = Path("/root/projects/alma_servie/db")
ANOMALIES = ["negermet", "pritok", "salt"]


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


def safe_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n1, n0 = len(pos), len(neg)
    all_scores = np.concatenate([pos, neg])
    ranks = pd.Series(all_scores).rank().to_numpy()
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0))


def diag_one(anomaly: str) -> None:
    p = DB / f"{anomaly}_paano_global_scores.parquet"
    if not p.exists():
        print(f"## {anomaly}: scores parquet not found — SKIP")
        return
    df = pd.read_parquet(p)
    df["well_id"] = df["well_id"].astype(str)
    score = df["score"].to_numpy(np.float64)
    lab = df["is_labelled_anomaly"].to_numpy(np.int64)
    pre = df["is_pre_anomaly_zone"].to_numpy(np.int64)

    m_norm = (lab == 0) & (pre == 0)
    m_pre = (pre == 1) & (lab == 0)
    m_lab = lab == 1

    s_norm, s_pre, s_lab = score[m_norm], score[m_pre], score[m_lab]

    print(f"\n{'=' * 78}")
    print(f"## {anomaly.upper()} — paano_global scores")
    print(f"{'=' * 78}")
    print(f"Точек всего: {len(df):,}  скв.: {df['well_id'].nunique()}")
    print(f"  normal           : n={len(s_norm):>9,}  ({100*len(s_norm)/len(df):5.2f}%)")
    print(f"  pre_anomaly_zone : n={len(s_pre):>9,}  ({100*len(s_pre)/len(df):5.2f}%)")
    print(f"  labelled_anomaly : n={len(s_lab):>9,}  ({100*len(s_lab)/len(df):5.2f}%)")

    print(f"\nРаспределение score:")
    for name, s in [("normal", s_norm), ("pre_anomaly", s_pre), ("labelled", s_lab)]:
        if len(s) == 0:
            print(f"  {name:14s}: ПУСТО")
            continue
        print(
            f"  {name:14s}: mean={s.mean():.4f}  median={np.median(s):.4f}  "
            f"q95={np.quantile(s, 0.95):.4f}  q99={np.quantile(s, 0.99):.4f}  max={s.max():.4f}"
        )

    auc_pre_vs_norm = safe_auc(s_pre, s_norm)
    auc_lab_vs_norm = safe_auc(s_lab, s_norm)
    auc_lab_vs_pre = safe_auc(s_lab, s_pre)
    print(f"\nРазделимость (ROC-AUC):")
    print(f"  pre_anomaly vs normal   : {auc_pre_vs_norm:.4f}  ← главный сигнал для Фазы 2")
    print(f"  labelled    vs normal   : {auc_lab_vs_norm:.4f}  (baseline — текущая задача)")
    print(f"  labelled    vs pre_anom : {auc_lab_vs_pre:.4f}  (внутри подозрительной зоны)")

    thr99 = float(np.quantile(s_norm, 0.99))
    above_pre = (s_pre > thr99).sum()
    above_lab = (s_lab > thr99).sum()
    print(f"\nДоля выше q99(normal) = {thr99:.4f}:")
    print(f"  pre_anomaly : {above_pre}/{len(s_pre)} = {100*above_pre/max(len(s_pre),1):.1f}%")
    print(f"  labelled    : {above_lab}/{len(s_lab)} = {100*above_lab/max(len(s_lab),1):.1f}%")

    print(f"\nLead-time по labelled-событиям (первый момент score > q99(normal)):")
    leads = []
    for wid, w in df.groupby("well_id"):
        w = w.sort_values("timestamp").reset_index(drop=True)
        ts = pd.to_datetime(w["timestamp"])
        sc = w["score"].to_numpy(np.float64)
        wlab = w["is_labelled_anomaly"].to_numpy(np.int64)
        for s_idx, e_idx in runs(wlab > 0):
            start_ts = ts.iloc[s_idx]
            look_back = max(0, s_idx - 1440)
            pre_sc = sc[look_back:s_idx]
            pre_ts = ts.iloc[look_back:s_idx]
            above = np.where(pre_sc > thr99)[0]
            if len(above):
                first_t = pre_ts.iloc[int(above[0])]
                lead_h = (start_ts - first_t).total_seconds() / 3600
                leads.append((str(wid), lead_h))
            else:
                leads.append((str(wid), float("nan")))
    if leads:
        ld = pd.Series([l for _, l in leads])
        with_signal = ld.dropna()
        print(f"  Событий с сигналом до старта: {len(with_signal)}/{len(leads)}")
        if len(with_signal):
            print(
                f"  Lead-time (часы): "
                f"median={with_signal.median():.2f}  p25={with_signal.quantile(.25):.2f}  "
                f"p75={with_signal.quantile(.75):.2f}  max={with_signal.max():.2f}"
            )
        for wid, lh in sorted(leads, key=lambda x: -(x[1] if x[1] == x[1] else -1))[:10]:
            tag = f"{lh:+6.2f}ч" if lh == lh else "— нет"
            print(f"    скв. {wid:>6s}: lead = {tag}")


if __name__ == "__main__":
    for a in ANOMALIES:
        diag_one(a)
    print("\nDONE")
