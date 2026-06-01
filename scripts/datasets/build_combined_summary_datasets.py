from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.dataset_builder import _build_well_frame, parse_parameter_series
from alma_service.dataset_config import split_for_well
from alma_service.paths import DB_DIR, DATA_DIR, ensure_dir
from alma_service.tabular_io import write_dataset_tables, write_table


DEFAULT_SUMMARY_PATH = DATA_DIR / "reference" / "Сводная информация_объединённая.xlsx"
DEFAULT_FREQ = "5min"
MIN_REFERENCE_POINTS = 768
MIN_REFERENCE_DAYS = MIN_REFERENCE_POINTS * 5 / 60 / 24

ANOMALY_TYPE_KEYS = {
    "негермет": "negermet",
    "приток": "pritok",
    "соли": "salt",
}
NORMAL_TYPE_NEEDLES = (
    "нормальная работа",
    "нормальное поведение",
)
ANOMALY_RAW_DIRS = {
    "negermet": "negermet",
    "pritok": "pritok",
    "salt": "salt",
}
OUTPUT_KEYS = ("negermet", "pritok", "salt", "norm_work")


@dataclass(frozen=True)
class SummaryCase:
    output_key: str
    case_id: str
    source_well_id: str
    summary_row: int
    anomaly_type: str
    source_kind: str
    source_file: str
    source_path: Path
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    clip_start: pd.Timestamp
    clip_end: pd.Timestamp
    anomaly_start: pd.Timestamp | None
    anomaly_end: pd.Timestamp | None
    normal_reference_days: float
    split: str


def _normalize_id(value: object) -> str:
    return str(value).strip().lower()


def _parse_date(value: object) -> pd.Timestamp:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return pd.NaT
    return pd.to_datetime(value, dayfirst=True, errors="coerce")


def _anomaly_key(type_text: object) -> str | None:
    normalized = _normalize_id(type_text)
    for needle, key in ANOMALY_TYPE_KEYS.items():
        if needle in normalized:
            return key
    return None


def _is_full_normal(type_text: object) -> bool:
    normalized = _normalize_id(type_text)
    return any(needle in normalized for needle in NORMAL_TYPE_NEEDLES)


def _min_timestamp(*values: pd.Timestamp) -> pd.Timestamp:
    valid = [pd.Timestamp(value) for value in values if pd.notna(value)]
    if not valid:
        return pd.NaT
    return min(valid)


def _resolve_source_path(row: pd.Series, output_key: str) -> Path:
    source_file = str(row["Файл скважины"]).strip()
    source_kind = _normalize_id(row["Источник данных"])

    candidates: list[Path] = []
    if source_kind == "новая":
        candidates.append(DATA_DIR / "67_days" / source_file)
    elif source_kind == "norm_work":
        candidates.append(DATA_DIR / "raw" / "norm_work" / source_file)
    elif output_key in ANOMALY_RAW_DIRS:
        candidates.append(DATA_DIR / "raw" / ANOMALY_RAW_DIRS[output_key] / source_file)
        candidates.append(DATA_DIR / "67_days" / source_file)
    else:
        candidates.extend(
            [
                DATA_DIR / "raw" / "norm_work" / source_file,
                DATA_DIR / "raw" / "pritok" / source_file,
                DATA_DIR / "67_days" / source_file,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(DATA_DIR.rglob(source_file))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Source file not found for summary row {int(row.name) + 4}: {source_file}")


def _load_summary_cases(summary_path: Path, min_reference_days: float) -> list[SummaryCase]:
    df = pd.read_excel(summary_path, header=2)
    candidates: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        anomaly_type = str(row["Тип аномалии"]).strip()
        output_key = _anomaly_key(anomaly_type)
        full_normal = _is_full_normal(anomaly_type)
        if output_key is None and not full_normal:
            continue
        if full_normal:
            output_key = "norm_work"

        source_well_id = _normalize_id(row["Скважина"])
        data_start = _parse_date(row["Дата начала выгрузки"])
        data_end = _parse_date(row["Дата конца выгрузки"])
        anomaly_start = _parse_date(row.get("Дата начала аномалии"))
        anomaly_end = _parse_date(row.get("Дата конца аномалии"))
        if pd.isna(data_start) or pd.isna(data_end):
            continue

        if output_key == "norm_work":
            normal_reference_days = float((data_end - data_start).total_seconds() / 86400)
            clip_end = data_end
            split = "train"
            interval_start: pd.Timestamp | None = None
            interval_end: pd.Timestamp | None = None
        else:
            if pd.isna(anomaly_start) or pd.isna(anomaly_end):
                continue
            normal_reference_days = float((anomaly_start - data_start).total_seconds() / 86400)
            clip_end = _min_timestamp(data_end, anomaly_end)
            split = split_for_well(str(output_key), source_well_id)
            interval_start = anomaly_start
            interval_end = anomaly_end

        if normal_reference_days <= min_reference_days:
            continue

        candidates.append(
            {
                "output_key": str(output_key),
                "source_well_id": source_well_id,
                "summary_row": int(idx) + 4,
                "anomaly_type": anomaly_type,
                "source_kind": str(row["Источник данных"]).strip(),
                "source_file": str(row["Файл скважины"]).strip(),
                "source_path": _resolve_source_path(row, str(output_key)),
                "data_start": data_start,
                "data_end": data_end,
                "clip_start": data_start,
                "clip_end": clip_end,
                "anomaly_start": interval_start,
                "anomaly_end": interval_end,
                "normal_reference_days": normal_reference_days,
                "split": split,
            }
        )

    duplicate_counts: dict[tuple[str, str], int] = {}
    for item in candidates:
        key = (str(item["output_key"]), str(item["source_well_id"]))
        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1

    cases: list[SummaryCase] = []
    for item in candidates:
        dup_key = (str(item["output_key"]), str(item["source_well_id"]))
        if duplicate_counts[dup_key] > 1:
            case_id = f"{item['source_well_id']}__summary{int(item['summary_row']):03d}"
        else:
            case_id = str(item["source_well_id"])
        cases.append(SummaryCase(case_id=case_id, **item))
    return cases


def _build_output_dataset(output_key: str, cases: list[SummaryCase], freq: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    interval_rows: list[dict[str, Any]] = []
    all_columns: set[str] = set()
    build_rows: list[dict[str, Any]] = []

    for case in cases:
        series_list = parse_parameter_series(case.case_id, case.source_path)
        if not series_list:
            raise RuntimeError(f"{case.case_id}: no telemetry parsed from {case.source_path}")
        built = _build_well_frame(
            series_list,
            freq,
            clip_start=case.clip_start,
            clip_end=case.clip_end,
        )
        if built is None:
            raise RuntimeError(
                f"{case.case_id}: too short after clipping "
                f"{case.clip_start} — {case.clip_end}"
            )
        frame, detail = built
        frame["well_id"] = case.case_id
        for column in frame.columns:
            if column not in {"timestamp", "well_id"}:
                all_columns.add(column)
        frames.append(frame)
        build_rows.append(
            {
                "case_id": case.case_id,
                "source_well_id": case.source_well_id,
                "summary_row": case.summary_row,
                "source_file": case.source_file,
                "source_path": str(case.source_path),
                "start": detail["start"],
                "end": detail["end"],
                "grid_points": int(detail["grid_points"]),
                "normal_reference_days": case.normal_reference_days,
            }
        )

        if output_key != "norm_work":
            interval_rows.append(
                {
                    "well_id": case.case_id,
                    "source_well_id": case.source_well_id,
                    "summary_row": case.summary_row,
                    "source_kind": case.source_kind,
                    "source_file": case.source_file,
                    "start_date": case.anomaly_start,
                    "end_date": case.anomaly_end,
                    "data_start": case.data_start,
                    "data_end": case.data_end,
                    "split": case.split,
                    "interval_idx": 1,
                }
            )

    if not frames:
        raise RuntimeError(f"No frames built for {output_key}")

    dataset = pd.concat(frames, ignore_index=True)
    numeric_cols = sorted(all_columns)
    dataset = dataset[["timestamp", "well_id"] + [c for c in numeric_cols if c in dataset.columns]]

    if output_key == "norm_work":
        intervals = pd.DataFrame()
    else:
        intervals = (
            pd.DataFrame(interval_rows)
            .sort_values(["well_id", "start_date"])
            .reset_index(drop=True)
        )

    return dataset, intervals


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def build_combined_summary_datasets(
    summary_path: Path,
    *,
    freq: str = DEFAULT_FREQ,
    min_reference_days: float = MIN_REFERENCE_DAYS,
) -> None:
    ensure_dir(DB_DIR)
    cases = _load_summary_cases(summary_path, min_reference_days)
    by_output = {key: [case for case in cases if case.output_key == key] for key in OUTPUT_KEYS}

    summary: dict[str, Any] = {
        "summary_path": str(summary_path),
        "freq": freq,
        "min_reference_points": MIN_REFERENCE_POINTS,
        "min_reference_days": float(min_reference_days),
        "total_cases": len(cases),
        "outputs": {},
    }

    for output_key in OUTPUT_KEYS:
        output_cases = by_output[output_key]
        print(f"\n=== {output_key}: {len(output_cases)} cases ===")
        dataset, intervals = _build_output_dataset(output_key, output_cases, freq)
        freq_label = freq.replace(" ", "")
        if output_key == "norm_work":
            dataset_path = DB_DIR / f"norm_work_database_{freq_label}.parquet"
        else:
            dataset_path = DB_DIR / f"{output_key}_anomaly_database_{freq_label}.parquet"
        write_dataset_tables(dataset, parquet_path=dataset_path)
        print(f"Dataset: {dataset_path} ({len(dataset)} rows, {dataset['well_id'].nunique()} cases)")

        output_summary: dict[str, Any] = {
            "cases": len(output_cases),
            "rows": int(len(dataset)),
            "wells": int(dataset["well_id"].nunique()),
            "dataset_path": str(dataset_path),
            "case_ids": sorted(dataset["well_id"].astype(str).unique()),
        }
        if output_key != "norm_work":
            intervals_path = DB_DIR / f"{output_key}_intervals.parquet"
            write_table(intervals, intervals_path)
            print(f"Intervals: {intervals_path} ({len(intervals)} rows)")
            output_summary["intervals_path"] = str(intervals_path)
            output_summary["intervals"] = int(len(intervals))
            output_summary["split_counts"] = intervals["split"].value_counts().to_dict()
        summary["outputs"][output_key] = output_summary

    summary_path_out = DB_DIR / "combined_summary_dataset_summary.json"
    summary_path_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"\nSummary: {summary_path_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 5min datasets from the combined well summary.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--freq", default=DEFAULT_FREQ)
    parser.add_argument("--min-reference-days", type=float, default=MIN_REFERENCE_DAYS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_combined_summary_datasets(
        args.summary,
        freq=str(args.freq),
        min_reference_days=float(args.min_reference_days),
    )


if __name__ == "__main__":
    main()
