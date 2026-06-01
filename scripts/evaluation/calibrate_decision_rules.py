"""Калибровка порогов самостоятельных решающих правил по сводной.

Постановка (онлайн-детекция): детектор стоит в точке времени T и смотрит назад.

- Негерметичность: скачок давления за последние часы
  (окно [T-2ч, T] против [T-26ч, T-2ч]).
- Трендовое событие (приток/соли): наклон давления по 12-часовым медианам
  в окне [T-N суток, T], нормированный в процентах от медианы в сутки.
  Различение приток/соли — по поведению частоты и токов в том же окне.

Полнота: старт аномалии считается обнаруженным, если правило сработало хотя бы
в одной точке проверки внутри размеченной аномалии (негерметичность — в течение
24 часов после начала; тренды — в любой момент аномалии, потому что по эксперту
аномалия продолжается до конца выгрузки). Задержка обнаружения фиксируется
как метрика качества.

Ложные срабатывания: проверки на полностью нормальных скважинах и в нормальных
префиксах аномальных скважин (до размеченного начала минус лаг). Цель — ноль.

Результат: configs/alma_decision_rules_standalone.json + печать сводки.
"""

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

from alma_service.engineered_features import FREQ_COL, PRESSURE_COL
from alma_service.paths import DB_DIR, PROJECT_ROOT as ALMA_ROOT
from alma_service.tabular_io import read_table

ANOMALY_KEYS = ("negermet", "pritok", "salt")
PHASE_CURRENT_COLS = ("Ток на фазе А", "Ток на фазе В", "Ток на фазе С")
LOAD_COL = "Коэффициент загрузки ПЭД"

# Параметры постановки (не калибруются)
NEGERMET_STEP_HOURS = 2.0          # окно скачка
NEGERMET_BASE_HOURS = 24.0         # базовое окно перед скачком
TREND_WINDOW_DAYS = 14.0           # окно тренда, смотрим назад
NEGERMET_DETECT_LAG_HOURS = 24.0   # допустимый лаг обнаружения негермета
TREND_DETECT_LAG_DAYS = 7.0        # допустимый лаг обнаружения тренда
NEGERMET_CHECK_STEP_HOURS = 2.0    # шаг проверки негермета = окну скачка
TREND_CHECK_STEP_HOURS = 12.0      # шаг проверки трендов
STOPPED_FREQ_THRESHOLD = 1.0       # частота ниже — насос стоит

# Источники нормы, исключённые из калибровки по результатам проверки префиксов
# (docs/физика_аномалий_по_сводной_и_экспертным_комментариям.md):
# их «нормальные» префиксы содержат реальные неразмеченные тренды, и срабатывание
# правил на них — не ложное срабатывание, а корректная детекция явления.
CALIBRATION_NORM_EXCLUSIONS = {
    "negermet:5271г": "префикс содержит тренд -30.8% при постоянной частоте (неразмеченный приток?)",
    "salt:408": "префикс содержит нисходящий тренд -17.7% за 33 суток (снижение пластового давления)",
    "salt:3245__summary004": "префикс содержит тренд -16.8% (снижение пластового давления, комментарий эксперта)",
    "salt:3245(2)": "префикс содержит тренд -15.1%",
    "salt:3245__summary046": "префикс содержит тренд -20.4%",
    "salt:3244г": "префикс содержит тренд -12.6%",
    "pritok:610": "хвост префикса содержит начало тренда притока (+8.9%, лаг разметки)",
}


def _series(well_df: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in well_df.columns:
        return None
    s = well_df.set_index("timestamp")[column].dropna()
    return s if len(s) else None


def _median_in(series: pd.Series | None, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    if series is None:
        return float("nan")
    window = series.loc[t0:t1]
    return float(window.median()) if len(window) else float("nan")


def _trend_pct_per_day(
    series: pd.Series | None, t0: pd.Timestamp, t1: pd.Timestamp
) -> float:
    """Наклон по 12-часовым медианам, % от медианы окна в сутки."""
    if series is None:
        return float("nan")
    window = series.loc[t0:t1]
    if len(window) < 10:
        return float("nan")
    med12 = window.resample("12h").median().dropna()
    if len(med12) < 4:
        return float("nan")
    base = float(window.median())
    if abs(base) < 1e-9:
        return float("nan")
    x = np.arange(len(med12), dtype=float) / 2.0  # шаг 12ч = 0.5 суток
    slope = float(np.polyfit(x, med12.values, 1)[0])
    return slope / abs(base) * 100.0


def _stopped_fraction_in(
    freq: pd.Series | None, t0: pd.Timestamp, t1: pd.Timestamp
) -> float:
    if freq is None:
        return float("nan")
    window = freq.loc[t0:t1]
    if not len(window):
        return float("nan")
    return float((window < STOPPED_FREQ_THRESHOLD).mean())


def step_features(well_df: pd.DataFrame, when: pd.Timestamp) -> dict[str, float] | None:
    """Признаки скачка (негермет): короткое окно против базового, смотрим назад."""
    pressure = _series(well_df, PRESSURE_COL)
    freq = _series(well_df, FREQ_COL)
    if pressure is None:
        return None
    step_t0 = when - pd.Timedelta(hours=NEGERMET_STEP_HOURS)
    base_t0 = when - pd.Timedelta(hours=NEGERMET_BASE_HOURS + NEGERMET_STEP_HOURS)
    base_t1 = step_t0
    pressure_step_now = _median_in(pressure, step_t0, when)
    pressure_step_base = _median_in(pressure, base_t0, base_t1)
    if np.isfinite(pressure_step_now) and np.isfinite(pressure_step_base) and abs(pressure_step_base) > 1e-9:
        step_pct = (pressure_step_now - pressure_step_base) / abs(pressure_step_base) * 100.0
    else:
        step_pct = float("nan")
    freq_step_now = _median_in(freq, step_t0, when)
    freq_step_base = _median_in(freq, base_t0, base_t1)
    if np.isfinite(freq_step_now) and np.isfinite(freq_step_base) and abs(freq_step_base) > 1e-9:
        freq_step_pct = (freq_step_now - freq_step_base) / abs(freq_step_base) * 100.0
    else:
        freq_step_pct = float("nan")
    stopped = _stopped_fraction_in(freq, when - pd.Timedelta(hours=NEGERMET_STEP_HOURS), when)
    return {"step_pct": step_pct, "freq_step_pct": freq_step_pct, "stopped_fraction": stopped}


def trend_features(
    well_df: pd.DataFrame, when: pd.Timestamp, window_days: float
) -> dict[str, float] | None:
    """Признаки тренда (приток/соли): наклон за окно, смотрим назад."""
    pressure = _series(well_df, PRESSURE_COL)
    freq = _series(well_df, FREQ_COL)
    if pressure is None:
        return None
    trend_t0 = when - pd.Timedelta(days=window_days)
    pressure_trend = _trend_pct_per_day(pressure, trend_t0, when)
    freq_now = _median_in(freq, when - pd.Timedelta(days=1), when)
    freq_ago = _median_in(freq, trend_t0, trend_t0 + pd.Timedelta(days=1))
    if np.isfinite(freq_now) and np.isfinite(freq_ago) and abs(freq_ago) > 1e-9:
        freq_change_pct = (freq_now - freq_ago) / abs(freq_ago) * 100.0
    else:
        freq_change_pct = float("nan")
    current_changes = []
    for col in (*PHASE_CURRENT_COLS, LOAD_COL):
        s = _series(well_df, col)
        now = _median_in(s, when - pd.Timedelta(days=1), when)
        ago = _median_in(s, trend_t0, trend_t0 + pd.Timedelta(days=1))
        if np.isfinite(now) and np.isfinite(ago) and abs(ago) > 1e-9:
            current_changes.append((now - ago) / abs(ago) * 100.0)
    load_change_pct = float(np.median(current_changes)) if current_changes else float("nan")
    stopped = _stopped_fraction_in(freq, when - pd.Timedelta(days=1), when)
    return {
        "pressure_trend_pct_per_day": pressure_trend,
        "freq_change_pct": freq_change_pct,
        "load_change_pct": load_change_pct,
        "stopped_fraction": stopped,
    }


# ---------------- решающие правила ----------------

def fires_negermet(f: dict[str, float], th: dict[str, float]) -> bool:
    if np.isfinite(f["stopped_fraction"]) and f["stopped_fraction"] >= 0.5:
        return False
    if np.isfinite(f["freq_step_pct"]) and abs(f["freq_step_pct"]) >= th["frequency_step_max_pct"]:
        return False
    return np.isfinite(f["step_pct"]) and f["step_pct"] >= th["negermet_step_min_pct"]


def fires_trend(f: dict[str, float], th: dict[str, float]) -> bool:
    """Трендовое событие — общий триггер для притока и солей."""
    if np.isfinite(f["stopped_fraction"]) and f["stopped_fraction"] >= 0.5:
        return False
    trend = f["pressure_trend_pct_per_day"]
    if not np.isfinite(trend) or abs(trend) < th["trend_min_pct_per_day"]:
        return False
    # фильтр реакции на частоту: тренд давления противоположен изменению частоты
    freq_change = f["freq_change_pct"]
    if (
        np.isfinite(freq_change)
        and abs(freq_change) >= th["frequency_change_max_pct"]
        and trend * freq_change < 0
    ):
        return False
    return True


def classify_trend(f: dict[str, float], th: dict[str, float]) -> str:
    """Уточнение класса трендового события: приток или соли."""
    trend = f["pressure_trend_pct_per_day"]
    freq_change = f["freq_change_pct"]
    load_change = f["load_change_pct"]
    # соли: рост давления при росте/неизменности частоты, либо падение токов
    if trend > 0:
        freq_up_or_flat = not np.isfinite(freq_change) or freq_change >= -th["frequency_change_max_pct"]
        load_drop = np.isfinite(load_change) and load_change <= -th["salt_load_drop_min_pct"]
        if (np.isfinite(freq_change) and freq_change >= th["frequency_change_max_pct"]) or load_drop:
            return "salt"
        if freq_up_or_flat:
            return "trend_up_review"  # рост при стабильной частоте: ранние соли или приток вверх
    return "pritok"


def case_fires(
    case_points: list[dict[str, Any]],
    anomaly_key: str,
    th: dict[str, float],
) -> bool:
    """Сработало ли правило хотя бы в одной точке проверки случая."""
    for point in case_points:
        kind = point.get("_kind")
        if kind == "step" and fires_negermet(point, th):
            return True
        if kind == "trend" and fires_trend(point, th):
            return True
    return False


# ---------------- сбор точек ----------------

def _case_check_points(
    well_df: pd.DataFrame,
    anomaly_key: str,
    anomaly_start: pd.Timestamp,
    anomaly_end: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Точки проверки внутри аномалии: негермет — частые проверки скачка,
    тренды — проверки наклона; для негермета добавляем и трендовые проверки
    (резкий скачок виден и как тренд)."""
    points: list[dict[str, Any]] = []
    if anomaly_key == "negermet":
        lag = pd.Timedelta(hours=NEGERMET_DETECT_LAG_HOURS)
        detect_until = min(anomaly_start + lag, anomaly_end)
        for when in pd.date_range(anomaly_start, detect_until, freq=f"{int(NEGERMET_CHECK_STEP_HOURS)}h"):
            when = pd.Timestamp(when)
            features = step_features(well_df, when)
            if features:
                features["_kind"] = "step"
                features["_when"] = when
                points.append(features)
    else:
        # эксперт: аномалия продолжается до конца выгрузки -> детекция допустима
        # в любой момент аномалии; задержка фиксируется как метрика качества
        for when in pd.date_range(anomaly_start, anomaly_end, freq=f"{int(TREND_CHECK_STEP_HOURS)}h"):
            when = pd.Timestamp(when)
            features = trend_features(well_df, when, TREND_WINDOW_DAYS)
            if features:
                features["_kind"] = "trend"
                features["_when"] = when
                points.append(features)
    return points


def collect_check_points() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(случаи аномалий с точками проверки, нормальные точки)"""
    anomaly_cases: list[dict[str, Any]] = []
    normal_points: list[dict[str, Any]] = []

    for anomaly_key in ANOMALY_KEYS:
        df = read_table(DB_DIR / f"{anomaly_key}_anomaly_database_5min.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        intervals = read_table(DB_DIR / f"{anomaly_key}_intervals.parquet")
        for _, interval in intervals.iterrows():
            well_id = str(interval["well_id"])
            well_df = df[df["well_id"] == well_id].sort_values("timestamp")
            if well_df.empty:
                continue
            data_start = pd.Timestamp(interval["data_start"])
            anomaly_start = pd.Timestamp(interval["start_date"])
            anomaly_end = pd.Timestamp(interval["end_date"])

            anomaly_cases.append(
                {
                    "anomaly_key": anomaly_key,
                    "well_id": well_id,
                    "summary_row": int(interval["summary_row"]),
                    "split": str(interval["split"]),
                    "anomaly_start": anomaly_start,
                    "points": _case_check_points(well_df, anomaly_key, anomaly_start, anomaly_end),
                }
            )

            # нормальные точки префикса: до начала аномалии, исключая известные
            # проблемные префиксы (см. CALIBRATION_NORM_EXCLUSIONS)
            prefix_key = f"{anomaly_key}:{well_id}"
            if prefix_key in CALIBRATION_NORM_EXCLUSIONS:
                continue
            norm_until = anomaly_start - pd.Timedelta(hours=1)
            norm_from = data_start + pd.Timedelta(days=TREND_WINDOW_DAYS)
            if norm_until > norm_from:
                for when in pd.date_range(norm_from, norm_until, freq="1D"):
                    when = pd.Timestamp(when)
                    step = step_features(well_df, when)
                    if step:
                        step["_kind"] = "step"
                        normal_points.append({"well_id": prefix_key, "group": "prefix", **step})
                    trend = trend_features(well_df, when, TREND_WINDOW_DAYS)
                    if trend:
                        trend["_kind"] = "trend"
                        normal_points.append({"well_id": prefix_key, "group": "prefix", **trend})

    # полностью нормальные скважины
    norm_df = read_table(DB_DIR / "norm_work_database_5min.parquet")
    norm_df["timestamp"] = pd.to_datetime(norm_df["timestamp"])
    for well_id in sorted(norm_df["well_id"].astype(str).unique()):
        well_df = norm_df[norm_df["well_id"] == well_id].sort_values("timestamp")
        t0, t1 = well_df["timestamp"].min(), well_df["timestamp"].max()
        # для коротких нормальных рядов трендовое окно адаптируется к доступной длине,
        # но не короче 5 суток
        series_days = (t1 - t0).total_seconds() / 86400.0
        effective_window = min(TREND_WINDOW_DAYS, max(series_days - 1.0, 5.0))
        start = t0 + pd.Timedelta(days=effective_window)
        if t1 <= start:
            continue
        for when in pd.date_range(start, t1, freq="1D"):
            when = pd.Timestamp(when)
            step = step_features(well_df, when)
            if step:
                step["_kind"] = "step"
                normal_points.append({"well_id": well_id, "group": "norm_work", **step})
            trend = trend_features(well_df, when, effective_window)
            if trend:
                trend["_kind"] = "trend"
                normal_points.append({"well_id": well_id, "group": "norm_work", **trend})
    return anomaly_cases, normal_points


# ---------------- оценка ----------------

def evaluate(
    anomaly_cases: list[dict[str, Any]],
    normal_points: list[dict[str, Any]],
    th: dict[str, float],
) -> dict[str, Any]:
    false_accepts: list[dict[str, Any]] = []
    for point in normal_points:
        kind = point.get("_kind")
        fired = (kind == "step" and fires_negermet(point, th)) or (
            kind == "trend" and fires_trend(point, th)
        )
        if fired:
            false_accepts.append(point)

    by_class = {key: {"total": 0, "hit": 0, "missed": [], "delays_days": {}} for key in ANOMALY_KEYS}
    for case in anomaly_cases:
        cls = case["anomaly_key"]
        by_class[cls]["total"] += 1
        first_fire: pd.Timestamp | None = None
        for point in case["points"]:
            kind = point.get("_kind")
            fired = (kind == "step" and fires_negermet(point, th)) or (
                kind == "trend" and fires_trend(point, th)
            )
            if fired:
                first_fire = point["_when"]
                break
        if first_fire is not None:
            by_class[cls]["hit"] += 1
            anomaly_start = case["anomaly_start"]
            delay_days = (first_fire - anomaly_start).total_seconds() / 86400.0
            by_class[cls]["delays_days"][case["well_id"]] = round(delay_days, 2)
        else:
            by_class[cls]["missed"].append(case["well_id"])
    total_hits = sum(v["hit"] for v in by_class.values())
    return {
        "false_accepts": len(false_accepts),
        "false_details": false_accepts,
        "by_class": by_class,
        "total_hits": total_hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Калибровка порогов решающих правил (скользящие окна).")
    parser.add_argument(
        "--output",
        default=str(ALMA_ROOT / "configs" / "alma_decision_rules_standalone.json"),
    )
    args = parser.parse_args()

    print("Сбор точек проверки по сводной (скользящие окна, смотрим назад)...")
    anomaly_cases, normal_points = collect_check_points()
    n_anomaly_points = sum(len(c["points"]) for c in anomaly_cases)
    print(f"  аномальных случаев: {len(anomaly_cases)} (точек проверки: {n_anomaly_points})")
    print(f"  нормальных точек: {len(normal_points)}")

    grid = {
        "negermet_step_min_pct": [10.0, 15.0, 18.0, 25.0],
        "frequency_step_max_pct": [1.0, 2.0, 3.0],
        "trend_min_pct_per_day": [0.3, 0.5, 0.8, 1.0, 1.5],
        "frequency_change_max_pct": [0.5, 1.0, 2.0],
        "salt_load_drop_min_pct": [5.0, 10.0],
    }
    keys = list(grid.keys())
    results: list[tuple[int, int, dict[str, float]]] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        th = dict(zip(keys, combo))
        res = evaluate(anomaly_cases, normal_points, th)
        results.append((res["false_accepts"], res["total_hits"], th))

    # сортировка: сперва минимум ложных, затем максимум полноты
    results.sort(key=lambda item: (item[0], -item[1]))
    best_false, best_hits, best_th = results[0]
    zero_false = [r for r in results if r[0] == 0]
    print(f"\nконфигураций с нулём ложных: {len(zero_false)}")
    if zero_false:
        best_false, best_hits, best_th = zero_false[0]

    final = evaluate(anomaly_cases, normal_points, best_th)
    print("\n=== ИТОГОВАЯ КОНФИГУРАЦИЯ ===")
    print(json.dumps(best_th, ensure_ascii=False, indent=2))
    print("\n=== КАЧЕСТВО ===")
    print(f"ложных срабатываний на норме: {final['false_accepts']} из {len(normal_points)}")
    for cls, stats in final["by_class"].items():
        delays = list(stats["delays_days"].values())
        delay_info = ""
        if delays:
            delay_info = f"; задержка медиана {np.median(delays):.1f} сут, макс {max(delays):.1f} сут"
        print(f"  {cls}: {stats['hit']}/{stats['total']}; пропущены: {stats['missed']}{delay_info}")
    if final["false_accepts"]:
        print("\nдетали ложных:")
        for d in final["false_details"][:15]:
            kind = d.get("_kind")
            if kind == "trend":
                print(f"  {d['well_id']} ({d['group']}, тренд): {d['pressure_trend_pct_per_day']:.2f}%/сут, частота {d['freq_change_pct']:.2f}%")
            else:
                print(f"  {d['well_id']} ({d['group']}, скачок): {d['step_pct']:.1f}%, частота {d['freq_step_pct']:.2f}%")

    payload = {
        "name": "alma_decision_rules_standalone",
        "description": (
            "Пороги самостоятельных решающих правил по трём классам аномалий "
            "(онлайн-постановка: детектор смотрит назад от текущей точки). "
            "Откалиброваны по объединённой сводной: ноль ложных срабатываний на "
            "источниках нормы; полнота считается по срабатыванию в любой момент "
            "размеченной аномалии (негерметичность — в первые 24 часа), задержка "
            "обнаружения фиксируется отдельно."
        ),
        "calibrated_at": "2026-06-01",
        "structure": {
            "level_1_immediate": "негерметичность: скачок давления за 2 часа против базовых 24 часов",
            "level_1_trend": "трендовое событие: наклон давления по 12-часовым медианам за 14 суток",
            "level_2_classify": "уточнение класса тренда: соли (частота растёт или токи падают) против приток (токи стабильны)",
            "filters": [
                "остановка скважины (частота < 1 Гц более половины суток) — отказ",
                "реакция на частоту: тренд давления противоположен изменению частоты — отказ",
            ],
        },
        "windows": {
            "negermet_step_hours": NEGERMET_STEP_HOURS,
            "negermet_base_hours": NEGERMET_BASE_HOURS,
            "trend_window_days": TREND_WINDOW_DAYS,
            "negermet_detect_lag_hours": NEGERMET_DETECT_LAG_HOURS,
            "trend_detect_lag_days": TREND_DETECT_LAG_DAYS,
        },
        "calibration_norm_exclusions": CALIBRATION_NORM_EXCLUSIONS,
        "thresholds": best_th,
        "calibration_quality": {
            "normal_points": len(normal_points),
            "false_accepts": final["false_accepts"],
            "by_class": {
                cls: {
                    "hit": stats["hit"],
                    "total": stats["total"],
                    "missed": stats["missed"],
                    "delays_days": stats["delays_days"],
                }
                for cls, stats in final["by_class"].items()
            },
        },
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nКонфигурация сохранена: {output_path}")


if __name__ == "__main__":
    main()
