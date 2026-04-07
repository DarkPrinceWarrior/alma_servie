from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.salym_raw_pipeline import SELECTED_PARAMS, YEARS, build_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Build separate raw Salym dataset.")
    parser.add_argument("--output-root", default="salym_prepared", help="Separate output directory for Salym artifacts.")
    parser.add_argument("--metadata-only", action="store_true", help="Build only metadata/anomaly tables without raw telemetry extraction.")
    parser.add_argument("--skip-metadata", action="store_true", help="Reuse existing metadata files instead of rebuilding them.")
    parser.add_argument("--skip-anomalies", action="store_true", help="Reuse existing anomaly metadata files instead of rebuilding them.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction and reuse existing raw fragment files.")
    parser.add_argument("--param", action="append", choices=SELECTED_PARAMS, help="Limit extraction to selected params.")
    parser.add_argument("--year", action="append", choices=YEARS, help="Limit extraction to selected years.")
    parser.add_argument("--batch-size", type=int, default=250_000, help="PyArrow batch size for parquet extraction.")
    parser.add_argument("--limit-wells", type=int, default=None, help="Optional limit for smoke runs.")
    args = parser.parse_args()

    outputs = build_all(
        args.output_root,
        metadata_only=args.metadata_only,
        params=args.param,
        years=args.year,
        batch_size=args.batch_size,
        limit_wells=args.limit_wells,
        skip_metadata=args.skip_metadata,
        skip_anomalies=args.skip_anomalies,
        skip_extract=args.skip_extract,
    )
    print(outputs)


if __name__ == "__main__":
    main()
