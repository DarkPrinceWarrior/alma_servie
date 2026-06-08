from __future__ import annotations

# Пересчёт таблицы «Префиксы N аномальных скважин» из физ-отчётного .md.
# Источник правды — db/{class}_anomaly_database_5min.parquet + db/{class}_intervals.parquet
# (intervals содержат summary_row и даты с применёнными overrides сводной).
#
# Метрики префикса (валидированы против текущей таблицы, см. docs/физика_…):
#   - длительность префикса: data_start → старт аномалии;
#   - размах и сигма ДАВЛЕНИЯ по 12-часовым медианам;
#   - суммарный линейный дрейф давления по 12-часовым медианам, % от медианы;
#   - доля остановок: f < 1 Гц;
#   - максимальный скачок частоты по 6-часовым медианам.
#
# Правило пиков (только приток): зоны влияния остановок (гидростатика в остановке +
# переходный процесс после запуска) вырезаются из статистики — та же логика, что в
# scripts/reports/generate_anomaly_physics_report.py (detect_stop_influence_zones).
#
# Новая разметка применяется автоматически: anomaly_start берётся из intervals (overrides
# уже применены при сборке). Для приток-подинтервалов в префиксе соляных скважин (строки 4,
# 7 — extra_intervals в configs/alma_summary_overrides.json) конец нормального префикса
# сдвигается на старт притока (нормальная часть кончается там, где начинается приток).

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.paths import DB_DIR  # noqa: E402

PRESSURE = "Давление на приеме насоса кгс/см²"
FREQUENCY = "Выходная частота"
STOP_HZ = 1.0
INFLUENCE_BASE_WINDOW_HOURS = 12.0
INFLUENCE_MAX_TAIL_HOURS = 36.0
INFLUENCE_FREQ_RECOVERY_RATIO = 0.95
INFLUENCE_PRESSURE_RECOVERY_RATIO = 1.005
STOP_MIN_SAMPLES = 2

CLASS_RU = {"negermet": "Негерметичность НКТ", "pritok": "Приток", "salt": "Соли"}
SPLIT_RU = {"train": "обучающая", "test": "отложенная проверочная"}
OVERRIDES_PATH = PROJECT_ROOT / "configs" / "alma_summary_overrides.json"


def stop_influence_zones(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    f = frame.set_index("timestamp")[[FREQUENCY, PRESSURE]].dropna().sort_index()
    if len(f) < 50:
        return []
    freq, pres = f[FREQUENCY], f[PRESSURE]
    stopped = freq < STOP_HZ
    if not stopped.any():
        return []
    bw = pd.Timedelta(hours=INFLUENCE_BASE_WINDOW_HOURS)
    mt = pd.Timedelta(hours=INFLUENCE_MAX_TAIL_HOURS)
    groups = (stopped != stopped.shift()).cumsum()
    zones: list[dict] = []
    for _, core in freq[stopped].groupby(groups[stopped]):
        if len(core) < STOP_MIN_SAMPLES:
            continue
        cs, ce = core.index[0], core.index[-1]
        before_f = freq.loc[cs - bw:cs]
        before_f = before_f[before_f >= STOP_HZ]
        before_p = pres.loc[cs - bw:cs]
        if before_f.empty or before_p.empty:
            continue
        wf = float(before_f.median())
        base = float(before_p.median())
        normal_before = freq.loc[:cs]
        normal_before = normal_before[normal_before >= wf * INFLUENCE_FREQ_RECOVERY_RATIO]
        zs = normal_before.index[-1] if len(normal_before) else cs
        after = f.loc[ce:]
        rec = (after[FREQUENCY] >= wf * INFLUENCE_FREQ_RECOVERY_RATIO) & (
            after[PRESSURE] <= base * INFLUENCE_PRESSURE_RECOVERY_RATIO
        )
        rt = after.index[rec]
        limit = ce + mt
        ze = rt[0] if (len(rt) and rt[0] <= limit) else min(f.index[-1], limit)
        zones.append({"start": zs, "end": ze})
    merged: list[dict] = []
    for z in sorted(zones, key=lambda z: z["start"]):
        if merged and z["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], z["end"])
        else:
            merged.append(dict(z))
    return [(z["start"], z["end"]) for z in merged]


def median_series(values: np.ndarray, ts: pd.Series, hours: float) -> pd.Series:
    s = pd.DataFrame({"t": pd.to_datetime(ts), "v": values}).dropna()
    return s.set_index("t")["v"].resample(f"{hours}h").median().dropna()


def prefix_metrics(pre: pd.DataFrame, exclude_zones: list[tuple[pd.Timestamp, pd.Timestamp]]):
    mask = pd.Series(True, index=pre.index)
    for zs, ze in exclude_zones:
        mask &= ~((pre["timestamp"] >= zs) & (pre["timestamp"] <= ze))
    clean = pre[mask]
    pressure = clean[PRESSURE].astype(float).values
    m12 = median_series(pressure, clean["timestamp"], 12.0)
    if len(m12) < 3:
        return None
    med = float(np.median(m12.values))
    rng = float(m12.max() - m12.min())
    sigma = float(np.std(m12.values))
    x = (m12.index.astype("int64") / 1e9 / 86400.0).to_numpy()
    x = x - x[0]
    slope = float(np.polyfit(x, m12.values, 1)[0])
    drift_pct = slope * (x[-1] - x[0]) / med * 100.0
    freq = pre[FREQUENCY].astype(float).values
    stop_pct = float(np.mean(freq < STOP_HZ) * 100.0)
    f6 = median_series(freq, pre["timestamp"], 6.0)
    fjump = float(np.nanmax(np.abs(np.diff(f6.values)))) if len(f6) > 1 else 0.0
    return {
        "range_kgs": rng,
        "range_pct": rng / med * 100.0,
        "sigma": sigma,
        "drift_pct": drift_pct,
        "stop_pct": stop_pct,
        "freq_jump": fjump,
        "excluded_pct": float((~mask).mean() * 100.0),
    }


def load_extra_prefix_ends() -> dict[int, pd.Timestamp]:
    if not OVERRIDES_PATH.exists():
        return {}
    payload = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    ends: dict[int, pd.Timestamp] = {}
    for extra in payload.get("extra_intervals", []):
        if str(extra.get("anomaly_type", "")).strip().lower().startswith("приток"):
            ends[int(extra["summary_row"])] = pd.Timestamp(extra["start"])
    return ends


def recompute() -> pd.DataFrame:
    extra_ends = load_extra_prefix_ends()
    rows: list[dict] = []
    for cls in ("negermet", "pritok", "salt"):
        db = pd.read_parquet(DB_DIR / f"{cls}_anomaly_database_5min.parquet")
        iv = pd.read_parquet(DB_DIR / f"{cls}_intervals.parquet")
        db["timestamp"] = pd.to_datetime(db["timestamp"])
        for _, r in iv.iterrows():
            well = str(r["well_id"])
            srow = int(r["summary_row"]) if "summary_row" in iv.columns and pd.notna(r["summary_row"]) else -1
            ds = pd.Timestamp(r["data_start"])
            anomaly_start = pd.Timestamp(r["start_date"])
            prefix_end = anomaly_start
            note = ""
            if srow in extra_ends and extra_ends[srow] < prefix_end:
                prefix_end = extra_ends[srow]
                note = f"конец нормы сдвинут на старт притока {prefix_end:%d.%m}"
            g = db[db["well_id"] == well].sort_values("timestamp")
            if g.empty:
                continue
            series_span_days = (g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 86400.0
            pre = g[(g["timestamp"] >= ds) & (g["timestamp"] < prefix_end)]
            if pre.empty:
                continue
            dur_days = (prefix_end - ds).total_seconds() / 86400.0
            zones = stop_influence_zones(pre[["timestamp", FREQUENCY, PRESSURE]].copy()) if cls == "pritok" else []
            m = prefix_metrics(pre, zones)
            if m is None:
                continue
            rows.append({
                "summary_row": srow,
                "well": well,
                "class": CLASS_RU[cls],
                "split": SPLIT_RU.get(str(r.get("split", "train")), "обучающая"),
                "dur_days": dur_days,
                "dur_pct": dur_days / series_span_days * 100.0 if series_span_days else float("nan"),
                "n_zones": len(zones),
                "note": note,
                **m,
            })
    out = pd.DataFrame(rows).sort_values("summary_row").reset_index(drop=True)
    return out


def verdict(r: pd.Series) -> str:
    if r["dur_days"] < 7.0:
        return f"короткий ({r['dur_days']:.1f} сут)"
    if abs(r["drift_pct"]) >= 8.0:
        return f"дрейф давления {r['drift_pct']:+.0f}%"
    if r["range_pct"] > 40.0:
        return f"размах давления {r['range_pct']:.0f}%"
    return "чистый"


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "| Строка | Скважина | Тип | Разбиение | Длит. префикса | Размах давления | Дрейф | Остановки | Скачок частоты | Вырезано пиков | Вердикт |",
        "|---:|---|---|---|---|---|---:|---|---|---:|---|",
    ]
    for _, r in df.iterrows():
        stop = "нет" if r["stop_pct"] < 0.05 else f"{r['stop_pct']:.1f}%"
        fj = "нет" if r["freq_jump"] < 0.5 else f"{r['freq_jump']:.1f} Гц"
        excl = "—" if r["n_zones"] == 0 else f"{r['n_zones']} зон / {r['excluded_pct']:.1f}%"
        lines.append(
            f"| {r['summary_row']} | {r['well']} | {r['class']} | {r['split']} | "
            f"{r['dur_days']:.0f} сут ({r['dur_pct']:.0f}% ряда) | "
            f"{r['range_kgs']:.1f} кгс/см² ({r['range_pct']:.0f}%) | "
            f"{r['drift_pct']:+.1f}% | {stop} | {fj} | {excl} | {verdict(r)} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None, help="optional CSV dump")
    args = ap.parse_args()
    df = recompute()
    print(to_markdown(df))
    print(f"\n# rows: {len(df)}")
    notes = df[df["note"] != ""]
    if not notes.empty:
        print("\n# приток-подинтервалы (конец нормы сдвинут):")
        for _, r in notes.iterrows():
            print(f"#   строка {r['summary_row']} {r['well']}: {r['note']}")
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n# CSV: {args.csv}")


if __name__ == "__main__":
    main()
