from __future__ import annotations

import json
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

# Режимы построения alarm-сигнала из score+threshold.
# "raw"   — alarm = score >= thr (старая логика, базовый стенд).
# "onset" — EMA+min_run+cooldown+hysteresis-rearm, приближение production-onset
#           (alma_service/onset_detection.py). Цель — сравнить harness с продом.
ALARM_MODES = ("raw", "onset")

# Pre-transform скоринга перед построением alarm.
# "raw"      — без трансформации.
# "robust_z" — per-well z = (score - median) / (MAD * 1.4826). FATE техника №1:
#              выравнивает узкие распределения (как у скв. 5021), делает скоры
#              сопоставимыми между скв. и порог — единым в сигма-единицах.
TRANSFORMS = ("raw", "robust_z")


def transform_score(score: np.ndarray, kind: str) -> np.ndarray:
    s = np.asarray(score, dtype=np.float64)
    if kind == "raw":
        return s
    if kind == "robust_z":
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med))) * 1.4826
        if mad < 1e-12:
            return np.zeros_like(s)
        return (s - med) / mad
    raise ValueError(f"unknown transform: {kind!r}")

# Параметры onset берутся из db/{a}_paano_global_config.json (текущий
# production-tuned). Cache lazy.
_ONSET_CFG_CACHE: dict[str, dict] = {}


def onset_cfg(anomaly: str) -> dict:
    if anomaly in _ONSET_CFG_CACHE:
        return _ONSET_CFG_CACHE[anomaly]
    payload = json.load(open(DB / f"{anomaly}_paano_global_config.json"))
    body = payload.get("config", payload)
    cfg = {
        "ema_alpha": float(body.get("ema_alpha", 0.08)),
        "min_run_points": int(body.get("min_run_points", 3)),
        "cooldown_hours": float(body.get("cooldown_hours", 24.0)),
        "rearm_window_minutes": float(body.get("rearm_window_minutes", 120.0)),
        "hysteresis_scale": float(body.get("hysteresis_scale", 0.6)),
    }
    _ONSET_CFG_CACHE[anomaly] = cfg
    return cfg


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


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def onset_alarm(
    score: np.ndarray,
    thr: float,
    *,
    grid_min: int,
    ema_alpha: float,
    min_run_points: int,
    cooldown_hours: float,
    rearm_window_minutes: float,
    hysteresis_scale: float,
) -> np.ndarray:
    """Latched onset-эмуляция: EMA(score) >= thr ≥ min_run_points подряд → выстрел;
    держится пока EMA >= thr*hysteresis. После rearm_window_minutes без sustain
    re-arm; новый выстрел запрещён в течение cooldown_hours от прошлого.
    """
    n = len(score)
    if n == 0:
        return np.zeros(0, dtype=bool)
    ema = _ema(score.astype(np.float64), float(ema_alpha))
    cooldown_pts = max(1, int(round(cooldown_hours * 60.0 / grid_min)))
    clear_pts = max(1, int(round(rearm_window_minutes / grid_min)))
    sustain_thr = float(thr) * float(hysteresis_scale)

    alarm = np.zeros(n, dtype=bool)
    armed = True
    run_len = 0
    clear_len = 0
    last_start_i = -10**12
    rearmed = False

    for i in range(n):
        if armed:
            if ema[i] >= thr:
                run_len += 1
                if run_len >= int(min_run_points):
                    start_i = i - run_len + 1
                    if rearmed or (i - last_start_i) >= cooldown_pts:
                        alarm[i] = True
                        last_start_i = start_i
                        armed = False
                        rearmed = False
                        clear_len = 0
                        run_len = 0
                    else:
                        run_len = 0
            else:
                run_len = 0
        else:
            alarm[i] = True
            if ema[i] >= sustain_thr:
                clear_len = 0
            else:
                clear_len += 1
            if clear_len >= clear_pts:
                armed = True
                rearmed = True
                clear_len = 0
                alarm[i] = False
    return alarm


def build_alarm(
    score: np.ndarray,
    thr: float,
    *,
    grid_min: int,
    mode: str,
    anomaly: str | None = None,
) -> np.ndarray:
    if mode == "raw":
        return score >= thr
    if mode == "onset":
        if anomaly is None:
            raise ValueError("onset mode requires anomaly key")
        return onset_alarm(score, thr, grid_min=grid_min, **onset_cfg(anomaly))
    raise ValueError(f"unknown mode: {mode}")


def event_metrics(
    score: np.ndarray,
    label: np.ndarray,
    thr: float,
    grid_min: int,
    *,
    mode: str = "raw",
    anomaly: str | None = None,
) -> dict:
    intervals = runs(label > 0)
    alarm = build_alarm(score, thr, grid_min=grid_min, mode=mode, anomaly=anomaly)
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


def fp_alarm_rate(
    score: np.ndarray,
    label: np.ndarray,
    thr: float,
    *,
    grid_min: int = 5,
    mode: str = "raw",
    anomaly: str | None = None,
) -> float:
    normal = label == 0
    if normal.sum() == 0:
        return float("nan")
    alarm = build_alarm(score, thr, grid_min=grid_min, mode=mode, anomaly=anomaly)
    return 100.0 * float(alarm[normal].mean())


CALIBRATIONS = ("per_well", "global")


def well_threshold(w: Well, q: float) -> float:
    normal = w.score[w.label == 0]
    base = normal if len(normal) else w.score
    return float(np.quantile(base, q))


def global_threshold(wells: dict[str, Well], q: float) -> float:
    """Единый порог на пуле нормальных точек всех train-скв.
    Имеет смысл только в связке с per-well robust_z трансформацией —
    скоры сопоставимы между скв. и σ-порог работает один на всех.
    """
    chunks = []
    for w in wells.values():
        chunks.append(w.score[w.label == 0])
    pool = np.concatenate(chunks) if chunks else np.zeros(0)
    if pool.size == 0:
        pool = np.concatenate([w.score for w in wells.values()]) if wells else np.zeros(0)
    return float(np.quantile(pool, q)) if pool.size else 0.0


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


def lowo_select_quantile(
    train: dict[str, Well],
    grid_min: int,
    *,
    mode: str = "raw",
    anomaly: str | None = None,
    calibration: str = "per_well",
) -> tuple[float, float]:
    best_q, best_f1 = float(Q_GRID[0]), -1.0
    for q in Q_GRID:
        if calibration == "global":
            # Leave-one-well-out: для каждой fold-скважины порог считается на остальных
            fold_f1 = []
            wells_list = list(train.values())
            for i, wv in enumerate(wells_list):
                others = {w.well_id: w for j, w in enumerate(wells_list) if j != i}
                thr = global_threshold(others, q) if others else well_threshold(wv, q)
                em = event_metrics(wv.score, wv.label, thr, grid_min, mode=mode, anomaly=anomaly)
                fold_f1.append(em["event_f1"])
        else:
            fold_f1 = []
            for wv in train.values():
                thr = well_threshold(wv, q)
                em = event_metrics(wv.score, wv.label, thr, grid_min, mode=mode, anomaly=anomaly)
                fold_f1.append(em["event_f1"])
        m = float(np.nanmean(fold_f1))
        if m > best_f1:
            best_q, best_f1 = float(q), m
    return best_q, best_f1


def _apply_transform(wells: dict[str, Well], transform: str) -> dict[str, Well]:
    if transform == "raw":
        return wells
    out: dict[str, Well] = {}
    for wid, w in wells.items():
        out[wid] = Well(
            anomaly=w.anomaly,
            well_id=w.well_id,
            split=w.split,
            score=transform_score(w.score, transform),
            label=w.label,
        )
    return out


def evaluate(
    anomaly: str,
    *,
    mode: str = "raw",
    transform: str = "raw",
    calibration: str = "per_well",
) -> Panel:
    grid_min = GRID_MIN[anomaly]
    all_wells_raw = load_paano_wells(anomaly)
    all_wells = _apply_transform(all_wells_raw, transform)
    wells = labelled_wells(all_wells)
    normals = {w: x for w, x in all_wells.items() if x.label.sum() == 0}
    train = {w: x for w, x in wells.items() if x.split == "train"}
    test = {w: x for w, x in wells.items() if x.split == "test"}
    panel = Panel(
        name=f"paano_global / {anomaly} [transform={transform} mode={mode} calib={calibration}]"
    )

    q, cv_f1 = lowo_select_quantile(
        train, grid_min, mode=mode, anomaly=anomaly, calibration=calibration
    )
    panel.fp_normal = {wid: round(fp_holdout(x, q), 1) for wid, x in normals.items()}
    panel.deployed_fp = deployed_false_alarms(anomaly, list(normals))
    global_thr_val = global_threshold(train, q) if calibration == "global" else None

    for tag, group in (("train(LOWO)", train), ("test", test)):
        for wid, w in group.items():
            if calibration == "global":
                thr = float(global_thr_val) if global_thr_val is not None else well_threshold(w, q)
            else:
                thr = well_threshold(w, q)
            em = event_metrics(w.score, w.label, thr, grid_min, mode=mode, anomaly=anomaly)
            panel.rows.append({
                "split": tag, "well": wid,
                "VUS_PR": round(vus_pr(w.score, w.label), 3),
                "point_PR_AUC": round(average_precision_score(w.label, w.score), 3)
                if 0 < w.label.sum() < len(w.label) else float("nan"),
                "event_recall": round(em["event_recall"], 2),
                "event_precision": round(em["event_precision"], 2),
                "onset_offset_h": round(em["onset_offset_h"], 1)
                if not np.isnan(em["onset_offset_h"]) else None,
                "FP_norm_pts_%": round(
                    fp_alarm_rate(w.score, w.label, thr, grid_min=grid_min, mode=mode, anomaly=anomaly), 1
                ),
            })
    panel.q = q
    panel.cv_f1 = cv_f1
    panel.mode = mode
    panel.transform = transform
    panel.calibration = calibration
    panel.global_thr = global_thr_val
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
    # CLI: eval_harness.py [anomaly ...] [--mode raw|onset|both] [--transform raw|robust_z|both]
    args = list(sys.argv[1:])

    def pop_opt(name: str, default: str) -> str:
        nonlocal_args = args  # noqa: F841 — for readability
        if name in args:
            i = args.index(name)
            val = args[i + 1]
            del args[i:i + 2]
            return val
        return default

    mode = pop_opt("--mode", "raw")
    transform = pop_opt("--transform", "raw")
    calibration = pop_opt("--calibration", "per_well")
    sel = args or list(ANOMALIES)
    modes = ALARM_MODES if mode == "both" else (mode,)
    transforms = TRANSFORMS if transform == "both" else (transform,)
    calibrations = CALIBRATIONS if calibration == "both" else (calibration,)
    for m in modes:
        if m not in ALARM_MODES:
            raise SystemExit(f"unknown mode {m!r}; expected one of {ALARM_MODES} or 'both'")
    for t in transforms:
        if t not in TRANSFORMS:
            raise SystemExit(f"unknown transform {t!r}; expected one of {TRANSFORMS} or 'both'")
    for c in calibrations:
        if c not in CALIBRATIONS:
            raise SystemExit(f"unknown calibration {c!r}; expected one of {CALIBRATIONS} or 'both'")
    for t in transforms:
        for c in calibrations:
            for m in modes:
                for a in sel:
                    print_panel(evaluate(a, mode=m, transform=t, calibration=c))
