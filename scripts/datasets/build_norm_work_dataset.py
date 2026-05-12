from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from alma_service.dataset_builder import _build_well_frame, parse_parameter_series
from alma_service.paths import DB_DIR, ensure_dir
from alma_service.tabular_io import write_dataset_tables


DEFAULT_INPUT_DIR = Path("data/raw/norm_work")
DEFAULT_FREQS = ("2min", "10min", "15min")


def _well_id_from_path(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"_?Нормальная работа$", "", stem, flags=re.IGNORECASE).strip("_ ")
    stem = re.sub(r"_?ЮЯ$", "", stem, flags=re.IGNORECASE).strip("_ ")
    return stem


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def build_norm_work_dataset(input_dir: Path, freqs: list[str]) -> None:
    files = sorted(input_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {input_dir}")

    ensure_dir(DB_DIR)
    well_series: dict[str, list[pd.Series]] = {}
    parse_rows: list[dict[str, Any]] = []

    print(f"=== Сборка датасета нормальной работы: {input_dir} ===")
    for path in files:
        well_id = _well_id_from_path(path)
        series_list = parse_parameter_series(well_id, path)
        well_series[well_id] = series_list
        starts = [series.index.min() for series in series_list if not series.empty]
        ends = [series.index.max() for series in series_list if not series.empty]
        counts = [len(series) for series in series_list if not series.empty]
        parse_rows.append(
            {
                "well_id": well_id,
                "source_file": str(path),
                "parameters": len(series_list),
                "start": min(starts) if starts else pd.NaT,
                "end": max(ends) if ends else pd.NaT,
                "min_points_per_parameter": min(counts) if counts else 0,
                "max_points_per_parameter": max(counts) if counts else 0,
            }
        )

    parse_summary = pd.DataFrame(parse_rows).sort_values("well_id")
    summary_payload: dict[str, Any] = {
        "input_dir": str(input_dir),
        "source_files": len(files),
        "wells": parse_summary.to_dict("records"),
        "outputs": {},
    }

    for freq in freqs:
        frames: list[pd.DataFrame] = []
        build_rows: list[dict[str, Any]] = []
        print(f"\nПостроение norm_work с шагом {freq}...")
        for well_id in sorted(well_series):
            built = _build_well_frame(well_series[well_id], freq)
            if built is None:
                build_rows.append({"well_id": well_id, "status": "skipped", "reason": "too_short"})
                print(f"  {well_id}: ПРОПУСК — слишком короткий диапазон")
                continue
            frame, detail = built
            frame["well_id"] = well_id
            columns = ["timestamp", "well_id"] + sorted(
                column for column in frame.columns if column not in {"timestamp", "well_id"}
            )
            frame = frame[columns]
            frames.append(frame)
            build_rows.append(
                {
                    "well_id": well_id,
                    "status": "ok",
                    "start": detail["start"],
                    "end": detail["end"],
                    "grid_points": int(detail["grid_points"]),
                    "parameters": int(len([c for c in frame.columns if c not in {"timestamp", "well_id"}])),
                }
            )
            print(
                f"  {well_id}: {detail['start']} — {detail['end']}, "
                f"grid={detail['grid_points']}"
            )

        if not frames:
            raise RuntimeError(f"No norm_work frames built for freq={freq}")

        dataset = pd.concat(frames, ignore_index=True)
        freq_label = freq.replace(" ", "")
        out_path = DB_DIR / f"norm_work_database_{freq_label}.parquet"
        write_dataset_tables(dataset, parquet_path=out_path)
        summary_payload["outputs"][freq_label] = {
            "path": str(out_path),
            "rows": int(len(dataset)),
            "wells": int(dataset["well_id"].nunique()),
            "build": build_rows,
        }
        print(f"Database: {out_path} ({len(dataset)} rows)")

    summary_path = DB_DIR / "norm_work_dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"\nSummary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build expert-confirmed normal-work dataset from Excel files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--freqs", default=",".join(DEFAULT_FREQS), help="Comma-separated pandas frequencies.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freqs = [item.strip() for item in str(args.freqs).split(",") if item.strip()]
    build_norm_work_dataset(args.input_dir, freqs)


if __name__ == "__main__":
    main()
