from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

DB = Path("/root/projects/alma_servie/db")
# Скоры paano_global — общая 5-мин сетка для всех типов аномалий.
GRID_MIN = {"negermet": 5, "pritok": 5, "salt": 5}
ANOMALIES = ("negermet", "pritok", "salt")
VUS_BUFFERS = (0, 5, 10, 20, 40)
TOL_PTS = 24
Q_GRID = np.round(np.arange(0.90, 0.999, 0.01), 3)


@dataclass
class Well:
    anomaly: str
    well_id: str
    split: str
    score: np.ndarray
    label: np.ndarray


@dataclass
class Panel:
    name: str
    rows: list[dict] = field(default_factory=list)


def load_paano_wells(anomaly: str) -> dict[str, Well]:
    df = pd.read_parquet(DB / f"{anomaly}_paano_global_scores.parquet")
    df = df.sort_values(["well_id", "timestamp"])
    wells: dict[str, Well] = {}
    for wid, g in df.groupby("well_id"):
        wells[str(wid)] = Well(
            anomaly=anomaly,
            well_id=str(wid),
            split=str(g["split"].iloc[0]),
            score=g["score"].to_numpy(np.float64),
            label=g["is_labelled_anomaly"].to_numpy(np.int64),
        )
    return wells


def labelled_wells(wells: dict[str, Well]) -> dict[str, Well]:
    return {w: x for w, x in wells.items() if x.label.sum() > 0}


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    inside = False
    start = 0
    for i, v in enumerate(mask):
        if v and not inside:
            start, inside = i, True
        elif not v and inside:
            out.append((start, i - 1))
            inside = False
    if inside:
        out.append((start, len(mask) - 1))
    return out


def dilate(label: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return label.astype(bool)
    out = label.astype(bool).copy()
    for s, e in runs(label > 0):
        out[max(0, s - k):min(len(label), e + k + 1)] = True
    return out


def vus_pr(score: np.ndarray, label: np.ndarray) -> float:
    if label.sum() == 0 or label.sum() == len(label):
        return float("nan")
    vals = []
    for k in VUS_BUFFERS:
        dl = dilate(label, k).astype(int)
        if 0 < dl.sum() < len(dl):
            vals.append(average_precision_score(dl, score))
    return float(np.mean(vals)) if vals else float("nan")


def event_metrics(score: np.ndarray, label: np.ndarray, thr: float, grid_min: int) -> dict:
    intervals = runs(label > 0)
    alarm = score >= thr
    alarm_events = runs(alarm)
    detected = 0
    onsets: list[float] = []
    for s, e in intervals:
        lo = max(0, s - TOL_PTS)
        hit = np.where(alarm[lo:e + 1])[0]
        if len(hit):
            detected += 1
            first = lo + int(hit[0])
            onsets.append((first - s) * grid_min / 60.0)
    tp_ev = 0
    for s, e in alarm_events:
        if any(not (e < (i_s - TOL_PTS) or s > i_e) for i_s, i_e in intervals):
            tp_ev += 1
    recall = detected / len(intervals) if intervals else float("nan")
    precision = tp_ev / len(alarm_events) if alarm_events else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and not np.isnan(precision) and not np.isnan(recall) else 0.0)
    return {
        "event_recall": recall,
        "event_precision": precision,
        "event_f1": f1,
        "onset_offset_h": float(np.median(onsets)) if onsets else float("nan"),
        "n_intervals": len(intervals),
        "n_alarm_events": len(alarm_events),
    }


def fp_alarm_rate(score: np.ndarray, label: np.ndarray, thr: float) -> float:
    normal = label == 0
    if normal.sum() == 0:
        return float("nan")
    return 100.0 * float((score[normal] >= thr).mean())


def well_threshold(w: Well, q: float) -> float:
    normal = w.score[w.label == 0]
    base = normal if len(normal) else w.score
    return float(np.quantile(base, q))


def fp_holdout(w: Well, q: float) -> float:
    half = len(w.score) // 2
    if half < 50:
        return float("nan")
    thr = float(np.quantile(w.score[:half], q))
    return 100.0 * float((w.score[half:] >= thr).mean())


def deployed_false_alarms(anomaly: str, normal_ids: list[str]) -> dict[str, int]:
    ps = pd.read_parquet(DB / f"{anomaly}_paano_global_predicted_starts.parquet")
    ps["well_id"] = ps["well_id"].astype(str)
    out: dict[str, int] = {}
    for w in normal_ids:
        g = ps[ps["well_id"] == w]
        if len(g) == 0:
            out[w] = 0
        elif "actionable_alert" in g.columns and "start_class" in g.columns:
            out[w] = int((g["actionable_alert"] & (g["start_class"] == "anomaly_candidate")).sum())
        else:
            out[w] = len(g)
    return out


def lowo_select_quantile(train: dict[str, Well], grid_min: int) -> tuple[float, float]:
    best_q, best_f1 = float(Q_GRID[0]), -1.0
    for q in Q_GRID:
        fold_f1 = []
        for wv in train.values():
            thr = well_threshold(wv, q)
            fold_f1.append(event_metrics(wv.score, wv.label, thr, grid_min)["event_f1"])
        m = float(np.nanmean(fold_f1))
        if m > best_f1:
            best_q, best_f1 = float(q), m
    return best_q, best_f1


def evaluate(anomaly: str) -> Panel:
    grid_min = GRID_MIN[anomaly]
    all_wells = load_paano_wells(anomaly)
    wells = labelled_wells(all_wells)
    normals = {w: x for w, x in all_wells.items() if x.label.sum() == 0}
    train = {w: x for w, x in wells.items() if x.split == "train"}
    test = {w: x for w, x in wells.items() if x.split == "test"}
    panel = Panel(name=f"paano_global / {anomaly}")

    q, cv_f1 = lowo_select_quantile(train, grid_min)
    panel.fp_normal = {wid: round(fp_holdout(x, q), 1) for wid, x in normals.items()}
    panel.deployed_fp = deployed_false_alarms(anomaly, list(normals))

    for tag, group in (("train(LOWO)", train), ("test", test)):
        for wid, w in group.items():
            thr = well_threshold(w, q)
            em = event_metrics(w.score, w.label, thr, grid_min)
            panel.rows.append({
                "split": tag, "well": wid,
                "VUS_PR": round(vus_pr(w.score, w.label), 3),
                "point_PR_AUC": round(average_precision_score(w.label, w.score), 3)
                if 0 < w.label.sum() < len(w.label) else float("nan"),
                "event_recall": round(em["event_recall"], 2),
                "event_precision": round(em["event_precision"], 2),
                "onset_offset_h": round(em["onset_offset_h"], 1)
                if not np.isnan(em["onset_offset_h"]) else None,
                "FP_norm_pts_%": round(fp_alarm_rate(w.score, w.label, thr), 1),
            })
    panel.q = q
    panel.cv_f1 = cv_f1
    return panel


def print_panel(panel: Panel) -> None:
    print("=" * 78)
    print(f"### {panel.name}  | квантиль порога(LOWO)={panel.q:.3f}  CV event-F1={panel.cv_f1:.3f}")
    cols = ["split", "well", "VUS_PR", "point_PR_AUC", "event_recall",
            "event_precision", "onset_offset_h", "FP_norm_pts_%"]
    print("  " + "  ".join(f"{c:>14s}" for c in cols))
    for r in panel.rows:
        print("  " + "  ".join(f"{str(r.get(c)):>14s}" for c in cols))
    tr = [r for r in panel.rows if r["split"].startswith("train")]
    te = [r for r in panel.rows if r["split"] == "test"]

    def avg(rows, key):
        vals = [r[key] for r in rows if isinstance(r[key], (int, float)) and not np.isnan(r[key])]
        return round(float(np.mean(vals)), 3) if vals else float("nan")

    for tag, rows in (("СРЕДНЕЕ train(LOWO)", tr), ("СРЕДНЕЕ test", te)):
        if rows:
            print(f"  -> {tag}: VUS-PR={avg(rows,'VUS_PR')} "
                  f"event-recall={avg(rows,'event_recall')} "
                  f"event-precision={avg(rows,'event_precision')}")
    fp = getattr(panel, "fp_normal", {})
    if fp:
        base = round(100 * (1 - panel.q), 1)
        vals = [v for v in fp.values() if not np.isnan(v)]
        mean_fp = round(float(np.mean(vals)), 1) if vals else float("nan")
        print(f"  -> ЛОЖНЫЕ ТРЕВОГИ на нормальных скв. (калибровка на 1-й половине, "
              f"замер на 2-й; идеал ≈ {base}% = 1-q):")
        for wid, v in fp.items():
            print(f"       {wid}: {v}%")
        print(f"     среднее: {mean_fp}%  (это дрейф сырого скора, не развёрнутые тревоги)")
    dfp = getattr(panel, "deployed_fp", {})
    if dfp:
        print(f"  -> РАЗВЁРНУТЫЕ ЛОЖНЫЕ ТРЕВОГИ (actionable anomaly_candidate в predicted_starts):")
        for wid, n in dfp.items():
            print(f"       {wid}: {n} ложных тревог")
        print(f"     всего: {sum(dfp.values())} на {len(dfp)} нормальных скв.")


if __name__ == "__main__":
    sel = sys.argv[1:] or list(ANOMALIES)
    for a in sel:
        print_panel(evaluate(a))
