"""Detection on an uploaded single-well Excel (unlabeled well).

Mirrors the raw -> parquet step of dataset_builder.build_dataset for a single
file, then runs the production single-well detector for all three anomaly
classes and saves per-class scores / predicted starts / summary into
``<output-dir>/<anomaly>/``.

The user does not pick an anomaly class — the upload is scored against
negermet, pritok and salt. Per-class failures (e.g. the resampled series is
too short for that class' frequency) are caught and recorded as ``error.json``
so the rest of the run still completes.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.anomaly_specs import get_dataset_spec
from alma_service.dataset_builder import _build_well_frame, parse_parameter_series
from alma_service.detection_artifacts import default_detector_for
from alma_service.generic_detection import run_single_well
from alma_service.paths import ensure_dir
from alma_service.tabular_io import write_table

ANOMALIES = ("negermet", "pritok", "salt")


def build_single_well_parquet(anomaly: str, excel_path: Path, well_id: str, out_dir: Path) -> Path:
    spec = get_dataset_spec(anomaly)
    freq = spec.default_freq
    series_list = parse_parameter_series(well_id, excel_path)
    if not series_list:
        raise RuntimeError(f"Не удалось распарсить параметры из {excel_path}")
    built = _build_well_frame(series_list, freq)
    if built is None:
        raise RuntimeError("Слишком короткий диапазон данных после ресемплинга.")
    df_well, _meta = built
    df_well["well_id"] = well_id
    numeric_cols = sorted(c for c in df_well.columns if c not in {"timestamp", "well_id"})
    df_well = df_well[["timestamp", "well_id", *numeric_cols]]

    ensure_dir(out_dir)
    out_path = out_dir / "source.parquet"
    write_table(df_well, out_path)
    print(
        f"[{anomaly}] single-well dataset built: {out_path} "
        f"({len(df_well)} rows, {len(numeric_cols)} channels, freq={freq})"
    )
    return out_path


def run_for_anomaly(anomaly: str, excel_path: Path, well_id: str, output_dir: Path) -> bool:
    anomaly_dir = output_dir / anomaly
    try:
        detector = default_detector_for(anomaly)
        source_parquet = build_single_well_parquet(anomaly, excel_path, well_id, anomaly_dir)
        run_single_well(
            anomaly_key=anomaly,
            well_id=well_id,
            detector=detector,
            source_path=str(source_parquet),
            save_dir=str(anomaly_dir),
        )
        if not (anomaly_dir / "summary.json").exists():
            raise RuntimeError("Детектор не сформировал результат (недостаточно данных).")
        print(f"[{anomaly}] OK")
        return True
    except Exception as exc:  # noqa: BLE001
        ensure_dir(anomaly_dir)
        (anomaly_dir / "error.json").write_text(
            json.dumps(
                {"anomaly": anomaly, "well_id": well_id, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[{anomaly}] FAILED: {exc}")
        traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Detection on an uploaded single-well Excel.")
    parser.add_argument("--excel", required=True, help="Path to the raw single-well xlsx")
    parser.add_argument("--well-id", required=True, help="Well identifier")
    parser.add_argument("--output-dir", required=True, help="Directory for per-class results")
    args = parser.parse_args()

    excel_path = Path(args.excel).resolve()
    if not excel_path.exists():
        raise FileNotFoundError(excel_path)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    ok_count = 0
    for anomaly in ANOMALIES:
        if run_for_anomaly(anomaly, excel_path, args.well_id, output_dir):
            ok_count += 1

    print(f"Done: {ok_count}/{len(ANOMALIES)} anomaly classes scored.")
    if ok_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
