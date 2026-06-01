"""Бэк-тест слоя доменных решающих правил по всем строкам объединённой сводной.

Проверяет правила в изоляции от PaAno-детектора:

1. Аномальные скважины (negermet/pritok/salt): правило применяется в точке
   размеченного экспертом начала аномалии. Ожидание — accept (кандидат
   соответствующего класса). Допустимо uncertain (на ревью); reject — промах.
2. Полностью нормальные скважины (norm_work): правило применяется в нескольких
   точках по всему ряду (раз в сутки). Любой accept — ложное срабатывание.
   Цель: ноль ложных accept на всех источниках нормы.
3. Нормальные префиксы аномальных скважин: правило применяется в середине
   префикса. Любой accept — ложное срабатывание.

Результат: artifacts/results/decision_rules_backtest.json + печать сводки.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.domain_decision_layer import assess_domain_start
from alma_service.engineered_features import (
    FREQ_COL,
    PRESSURE_COL,
    PreparedWellData,
)
from alma_service.paths import DB_DIR, RESULTS_DIR
from alma_service.tabular_io import read_table

ANOMALY_KEYS = ("negermet", "pritok", "salt")
ACCEPT_VERDICTS = {"pritok_candidate", "salt_candidate", "negermet_candidate"}


def _prepared_from_frame(well_id: str, well_df: pd.DataFrame) -> PreparedWellData:
    frame = well_df.sort_values("timestamp").reset_index(drop=True)
    timestamps = frame["timestamp"].to_numpy()
    columns = [c for c in frame.columns if c not in ("timestamp", "well_id")]
    matrix = frame[columns].to_numpy(dtype=float)
    points = len(frame)
    return PreparedWellData(
        well_id=well_id,
        split="backtest",
        timestamps=timestamps,
        raw_columns=columns,
        feature_columns=columns,
        raw_matrix=matrix,
        feature_matrix=matrix,
        reference_end_idx=points,
        reference_mask=np.ones(points, dtype=bool),
        stability_mask=np.ones(points, dtype=bool),
        onset_allowed_mask=np.ones(points, dtype=bool),
        detail={},
    )


def _start_row(well_id: str, detected_time: pd.Timestamp) -> pd.Series:
    return pd.Series(
        {
            "well_id": well_id,
            "detected_time": detected_time,
            "start_class": "anomaly_candidate",
            "is_bad_data": False,
            "is_regime_event": False,
        }
    )


def _assess(anomaly_key: str, prepared: PreparedWellData, when: pd.Timestamp) -> dict[str, Any]:
    return assess_domain_start(
        anomaly_key=anomaly_key,
        prepared=prepared,
        start_row=_start_row(prepared.well_id, when),
    )


def backtest_anomaly_starts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for anomaly_key in ANOMALY_KEYS:
        df = read_table(DB_DIR / f"{anomaly_key}_anomaly_database_5min.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        intervals = read_table(DB_DIR / f"{anomaly_key}_intervals.parquet")
        for _, interval in intervals.iterrows():
            well_id = str(interval["well_id"])
            well_df = df[df["well_id"] == well_id]
            if well_df.empty:
                continue
            prepared = _prepared_from_frame(well_id, well_df)
            start_time = pd.Timestamp(interval["start_date"])
            result = _assess(anomaly_key, prepared, start_time)
            verdict = result["domain_verdict"]
            outcome = (
                "accept"
                if verdict in ACCEPT_VERDICTS
                else ("uncertain" if result["domain_action"] == "uncertain" else "reject")
            )
            rows.append(
                {
                    "group": "anomaly_start",
                    "anomaly_key": anomaly_key,
                    "well_id": well_id,
                    "summary_row": int(interval["summary_row"]),
                    "split": str(interval["split"]),
                    "checked_at": str(start_time),
                    "verdict": verdict,
                    "action": result["domain_action"],
                    "reason": result["domain_reason"],
                    "outcome": outcome,
                    "pressure_delta_pct": result["domain_pressure_post_vs_pre_pct"],
                    "frequency_delta_pct": result["domain_frequency_post_vs_pre_pct"],
                    "stopped_fraction": result.get("domain_stopped_fraction"),
                }
            )
    return rows


def backtest_normal_wells(check_every_days: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    df = read_table(DB_DIR / "norm_work_database_5min.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for well_id in sorted(df["well_id"].astype(str).unique()):
        well_df = df[df["well_id"] == well_id]
        prepared = _prepared_from_frame(well_id, well_df)
        t0 = well_df["timestamp"].min()
        t1 = well_df["timestamp"].max()
        # точки проверки: каждые check_every_days, с отступом от краёв,
        # чтобы окна pre/post имели данные
        checks = pd.date_range(
            t0 + pd.Timedelta(days=3),
            t1 - pd.Timedelta(days=3),
            freq=f"{check_every_days}D",
        )
        # каждая нормальная скважина проверяется по всем трём классам правил
        for when in checks:
            for anomaly_key in ANOMALY_KEYS:
                result = _assess(anomaly_key, prepared, pd.Timestamp(when))
                verdict = result["domain_verdict"]
                is_false_accept = verdict in ACCEPT_VERDICTS
                rows.append(
                    {
                        "group": "normal_well",
                        "anomaly_key": anomaly_key,
                        "well_id": well_id,
                        "checked_at": str(when),
                        "verdict": verdict,
                        "action": result["domain_action"],
                        "reason": result["domain_reason"],
                        "outcome": "false_accept" if is_false_accept else "ok",
                        "pressure_delta_pct": result["domain_pressure_post_vs_pre_pct"],
                        "frequency_delta_pct": result["domain_frequency_post_vs_pre_pct"],
                    }
                )
    return rows


def backtest_anomaly_prefixes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for anomaly_key in ANOMALY_KEYS:
        df = read_table(DB_DIR / f"{anomaly_key}_anomaly_database_5min.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        intervals = read_table(DB_DIR / f"{anomaly_key}_intervals.parquet")
        for _, interval in intervals.iterrows():
            well_id = str(interval["well_id"])
            well_df = df[df["well_id"] == well_id]
            if well_df.empty:
                continue
            data_start = pd.Timestamp(interval["data_start"])
            anomaly_start = pd.Timestamp(interval["start_date"])
            prefix_days = (anomaly_start - data_start).total_seconds() / 86400.0
            if prefix_days < 8:
                continue
            prepared = _prepared_from_frame(well_id, well_df)
            # середина префикса — заведомо нормальная точка по разметке
            midpoint = data_start + (anomaly_start - data_start) / 2
            for check_key in ANOMALY_KEYS:
                result = _assess(check_key, prepared, midpoint)
                verdict = result["domain_verdict"]
                is_false_accept = verdict in ACCEPT_VERDICTS
                rows.append(
                    {
                        "group": "anomaly_prefix_midpoint",
                        "anomaly_key": check_key,
                        "well_id": f"{anomaly_key}:{well_id}",
                        "summary_row": int(interval["summary_row"]),
                        "checked_at": str(midpoint),
                        "verdict": verdict,
                        "action": result["domain_action"],
                        "reason": result["domain_reason"],
                        "outcome": "false_accept" if is_false_accept else "ok",
                        "pressure_delta_pct": result["domain_pressure_post_vs_pre_pct"],
                        "frequency_delta_pct": result["domain_frequency_post_vs_pre_pct"],
                    }
                )
    return rows


def summarize(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(all_rows)
    summary: dict[str, Any] = {}

    starts = frame[frame["group"] == "anomaly_start"]
    by_class: dict[str, Any] = {}
    for anomaly_key in ANOMALY_KEYS:
        sub = starts[starts["anomaly_key"] == anomaly_key]
        by_class[anomaly_key] = {
            "total": int(len(sub)),
            "accept": int((sub["outcome"] == "accept").sum()),
            "uncertain": int((sub["outcome"] == "uncertain").sum()),
            "reject": int((sub["outcome"] == "reject").sum()),
            "rejected_wells": sub[sub["outcome"] == "reject"]["well_id"].tolist(),
            "uncertain_wells": sub[sub["outcome"] == "uncertain"]["well_id"].tolist(),
        }
    summary["anomaly_starts"] = by_class

    norm = frame[frame["group"] == "normal_well"]
    false_accepts = norm[norm["outcome"] == "false_accept"]
    summary["normal_wells"] = {
        "checks_total": int(len(norm)),
        "false_accepts": int(len(false_accepts)),
        "false_accept_details": false_accepts[
            ["well_id", "anomaly_key", "checked_at", "reason"]
        ].to_dict("records"),
    }

    prefixes = frame[frame["group"] == "anomaly_prefix_midpoint"]
    prefix_false = prefixes[prefixes["outcome"] == "false_accept"]
    summary["anomaly_prefixes"] = {
        "checks_total": int(len(prefixes)),
        "false_accepts": int(len(prefix_false)),
        "false_accept_details": prefix_false[
            ["well_id", "anomaly_key", "checked_at", "reason"]
        ].to_dict("records"),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Бэк-тест доменных решающих правил по сводной.")
    parser.add_argument("--check-every-days", type=int, default=1)
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "decision_rules_backtest.json"),
    )
    args = parser.parse_args()

    print("=== Бэк-тест 1: размеченные старты аномалий ===")
    start_rows = backtest_anomaly_starts()
    print(f"  проверено стартов: {len(start_rows)}")

    print("=== Бэк-тест 2: полностью нормальные скважины ===")
    normal_rows = backtest_normal_wells(check_every_days=int(args.check_every_days))
    print(f"  проверок: {len(normal_rows)}")

    print("=== Бэк-тест 3: середины нормальных префиксов аномальных скважин ===")
    prefix_rows = backtest_anomaly_prefixes()
    print(f"  проверок: {len(prefix_rows)}")

    all_rows = start_rows + normal_rows + prefix_rows
    summary = summarize(all_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"summary": summary, "rows": all_rows}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n=== СВОДКА ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nПолный результат: {output_path}")


if __name__ == "__main__":
    main()
