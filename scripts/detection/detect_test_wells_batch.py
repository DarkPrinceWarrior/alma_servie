"""Batch-прогон детекции по папке data/raw/test_wells (blind, без разметки).

Обходит все xlsx из --xlsx-dir, для каждой скв. выполняет тот же путь, что
`detect_uploaded_well.py`: парсит ряды → parquet → прогоняет указанный детектор
по всем трём типам аномалий. Per-well результат пишется в
`<output-root>/<well_id>/<anomaly>/{scores,predicted_starts,summary}.json|parquet`.

Дополнительно собирается `<output-root>/_batch_summary.json` — компактная
сводка по всем скв. для генератора сводного HTML-отчёта.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.detection_artifacts import DETECTOR_KEYS
from alma_service.paths import ensure_dir
from scripts.detection.detect_uploaded_well import (
    ANOMALIES,
    BLIND_REFERENCE_FRACTION_DEFAULT,
    DETECTOR_AUTO,
    run_for_anomaly,
)

_WELL_ID_CLEAN = re.compile(r"\s*\(\d+\)\s*$")


def derive_well_id(xlsx_path: Path) -> str:
    stem = xlsx_path.stem.strip()
    stem = _WELL_ID_CLEAN.sub("", stem)
    return stem.lower()


def list_xlsx(xlsx_dir: Path) -> list[Path]:
    return sorted(p for p in xlsx_dir.iterdir() if p.suffix.lower() == ".xlsx" and not p.name.startswith("~"))


def collect_well_summary(well_dir: Path, well_id: str, anomalies: tuple[str, ...]) -> dict[str, object]:
    per_anomaly: dict[str, object] = {}
    for anomaly in anomalies:
        anomaly_dir = well_dir / anomaly
        summary_path = anomaly_dir / "summary.json"
        error_path = anomaly_dir / "error.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            per_anomaly[anomaly] = {
                "status": "ok",
                "detector": payload.get("detector"),
                "n_points": payload.get("n_points"),
                "n_detected": payload.get("n_detected"),
                "n_actionable_detected": payload.get("n_actionable_detected"),
                "n_incidents": payload.get("n_incidents"),
                "score_min": payload.get("score_min"),
                "score_median": payload.get("score_median"),
                "score_max": payload.get("score_max"),
                "time_start": payload.get("time_start"),
                "time_end": payload.get("time_end"),
            }
        elif error_path.exists():
            payload = json.loads(error_path.read_text(encoding="utf-8"))
            per_anomaly[anomaly] = {"status": "error", "error": payload.get("error")}
        else:
            per_anomaly[anomaly] = {"status": "missing"}
    return {"well_id": well_id, "anomalies": per_anomaly}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-детекция blind-скв. из папки.")
    parser.add_argument("--xlsx-dir", default="data/raw/test_wells")
    parser.add_argument("--output-root", default="artifacts/test_wells")
    parser.add_argument(
        "--detector",
        default="paano_global",
        choices=sorted({DETECTOR_AUTO, *DETECTOR_KEYS}),
    )
    parser.add_argument("--anomalies", default=",".join(ANOMALIES))
    parser.add_argument(
        "--freq",
        default=None,
        help="Override resampling freq (e.g. 5min). Default: 5min for paano_global.",
    )
    parser.add_argument(
        "--normal-reference-fraction",
        type=float,
        default=BLIND_REFERENCE_FRACTION_DEFAULT,
        help=f"Blind wells: first N points fraction used as reference (default: {BLIND_REFERENCE_FRACTION_DEFAULT}).",
    )
    parser.add_argument(
        "--use-population-memory-bank",
        action="store_true",
        help="Use pre-built population memory bank instead of local reference.",
    )
    parser.add_argument(
        "--trusted-local-reference",
        action="store_true",
        help="Treat each local normal-reference window as expert-confirmed and append it to the population bank with a cap.",
    )
    args = parser.parse_args()

    xlsx_dir = Path(args.xlsx_dir)
    if not xlsx_dir.is_dir():
        raise FileNotFoundError(xlsx_dir)

    requested_anomalies = tuple(a.strip().lower() for a in args.anomalies.split(",") if a.strip())
    unknown = [a for a in requested_anomalies if a not in ANOMALIES]
    if unknown:
        raise ValueError(f"Unknown anomalies: {unknown}; supported: {ANOMALIES}")

    output_root = Path(args.output_root)
    ensure_dir(output_root)

    files = list_xlsx(xlsx_dir)
    if not files:
        print(f"Нет xlsx в {xlsx_dir}")
        sys.exit(1)

    print(f"Найдено {len(files)} файлов в {xlsx_dir}; detector={args.detector}")
    well_summaries: list[dict[str, object]] = []
    overall_start = time.time()

    for idx, xlsx_path in enumerate(files, start=1):
        well_id = derive_well_id(xlsx_path)
        well_dir = output_root / well_id
        ensure_dir(well_dir)
        print(f"\n=== [{idx}/{len(files)}] {xlsx_path.name} → well_id={well_id} ===")
        for anomaly in requested_anomalies:
            try:
                run_for_anomaly(
                    anomaly,
                    xlsx_path,
                    well_id,
                    well_dir,
                    args.detector,
                    args.freq,
                    args.normal_reference_fraction,
                    args.use_population_memory_bank,
                    args.trusted_local_reference,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[{anomaly}] FATAL: {exc}")
                traceback.print_exc()
                (well_dir / anomaly / "error.json").write_text(
                    json.dumps(
                        {"anomaly": anomaly, "well_id": well_id, "error": f"{type(exc).__name__}: {exc}"},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        well_summaries.append(collect_well_summary(well_dir, well_id, requested_anomalies))

    elapsed = time.time() - overall_start
    batch_summary = {
        "xlsx_dir": str(xlsx_dir),
        "output_root": str(output_root),
        "detector_choice": args.detector,
        "freq_override": args.freq,
        "normal_reference_fraction": args.normal_reference_fraction,
        "use_population_memory_bank": args.use_population_memory_bank,
        "trusted_local_reference": args.trusted_local_reference,
        "anomalies": list(requested_anomalies),
        "elapsed_seconds": round(elapsed, 2),
        "wells": well_summaries,
    }
    (output_root / "_batch_summary.json").write_text(
        json.dumps(batch_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone. Total elapsed: {elapsed:.1f}s; summary: {output_root / '_batch_summary.json'}")


if __name__ == "__main__":
    main()
