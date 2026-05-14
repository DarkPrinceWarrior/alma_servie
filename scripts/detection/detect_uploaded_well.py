"""Detection on an uploaded single-well Excel (unlabeled well).

Mirrors the raw -> parquet step of dataset_builder.build_dataset for a single
file, then runs the production single-well detector and saves scores / predicted
starts / summary into an output directory.
"""

from __future__ import annotations

import argparse
import sys
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
        f"Single-well dataset built: {out_path} "
        f"({len(df_well)} rows, {len(numeric_cols)} channels, freq={freq})"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Detection on an uploaded single-well Excel.")
    parser.add_argument("--anomaly", choices=["negermet", "pritok", "salt"], required=True)
    parser.add_argument("--excel", required=True, help="Path to the raw single-well xlsx")
    parser.add_argument("--well-id", required=True, help="Well identifier")
    parser.add_argument("--output-dir", required=True, help="Directory for scores/starts/summary")
    parser.add_argument("--detector", default=None)
    args = parser.parse_args()

    detector = args.detector or default_detector_for(args.anomaly)
    excel_path = Path(args.excel).resolve()
    if not excel_path.exists():
        raise FileNotFoundError(excel_path)

    out_dir = Path(args.output_dir)
    source_parquet = build_single_well_parquet(args.anomaly, excel_path, args.well_id, out_dir)
    run_single_well(
        anomaly_key=args.anomaly,
        well_id=args.well_id,
        detector=detector,
        source_path=str(source_parquet),
        save_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
