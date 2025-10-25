"""Command-line entry point for the pipeline package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from . import anomalies, clean, events, extract, ingest, qc, align


def cmd_ingest(args: argparse.Namespace) -> None:
    manifest = ingest.ingest(Path(args.source), Path(args.config))
    print(f"Ingested {len(manifest)} files into raw storage.")


def cmd_extract(args: argparse.Namespace) -> None:
    wells: Optional[List[str]] = None
    if args.wells:
        wells = [w.strip() for w in args.wells.split(",") if w.strip()]
    extract.run_import(Path(args.config), wells=wells, combine=not args.skip_combine, combine_only=args.combine_only)


def cmd_qc(args: argparse.Namespace) -> None:
    qc.generate_quality_reports(Path(args.config))


def cmd_clean(args: argparse.Namespace) -> None:
    summary = clean.run_cleaning(Path(args.config))
    print(f"Cleaned datasets saved. AGZU rows: {summary['agzu_rows']}, SU rows: {summary['su_rows']}")


def cmd_align(args: argparse.Namespace) -> None:
    stats = align.run_alignment(Path(args.config))
    print(
        "Aligned datasets: "
        f"AGZU={stats['agzu_resampled_rows']}, SU={stats['su_resampled_rows']}, merged={stats['merged_rows']}"
    )


def cmd_anomalies(args: argparse.Namespace) -> None:
    result = anomalies.run_anomaly_analysis(Path(args.config), workbook_override=args.source)
    print(f"Detected {len(result)} anomaly segments.")


def cmd_events(args: argparse.Namespace) -> None:
    summary = events.run_reference_analysis(Path(args.config), workbook_override=args.source)
    print(f"Reference analysis completed. Records: {summary['records']}")


def cmd_full(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if args.source:
        cmd_ingest(argparse.Namespace(source=args.source, config=config_path))
    extract.run_import(config_path, wells=None, combine=False, combine_only=False)
    clean.run_cleaning(config_path)
    align.run_alignment(config_path)
    anomaly_path = args.anomaly_source if args.anomaly_source else None
    anomalies.run_anomaly_analysis(config_path, workbook_override=anomaly_path)
    if args.calibrate:
        events.run_reference_analysis(config_path, workbook_override=anomaly_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for the anomaly detection pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Copy raw files into the project and log metadata.")
    ingest_parser.add_argument("--source", required=True, help="Directory containing raw Excel files.")
    ingest_parser.add_argument("--config", default="config/pipeline.yaml")
    ingest_parser.set_defaults(func=cmd_ingest)

    extract_parser = subparsers.add_parser("extract", help="Parse raw Excel files and build interim tables.")
    extract_parser.add_argument("--config", default="config/pipeline.yaml")
    extract_parser.add_argument("--wells", help="Comma separated list of wells to process.")
    extract_parser.add_argument("--combine-only", action="store_true", help="Only rebuild combined parquet tables.")
    extract_parser.add_argument("--skip-combine", action="store_true", help="Skip rebuilding combined parquet tables.")
    extract_parser.set_defaults(func=cmd_extract)

    qc_parser = subparsers.add_parser("qc", help="Generate quality control reports.")
    qc_parser.add_argument("--config", default="config/pipeline.yaml")
    qc_parser.set_defaults(func=cmd_qc)

    clean_parser = subparsers.add_parser("clean", help="Clean interim datasets.")
    clean_parser.add_argument("--config", default="config/pipeline.yaml")
    clean_parser.set_defaults(func=cmd_clean)

    align_parser = subparsers.add_parser("align", help="Resample and align cleaned datasets.")
    align_parser.add_argument("--config", default="config/pipeline.yaml")
    align_parser.set_defaults(func=cmd_align)

    anomaly_parser = subparsers.add_parser("anomalies", help="Detect anomalies in merged hourly data using rule-based logic.")
    anomaly_parser.add_argument("--config", default="config/pipeline.yaml")
    anomaly_parser.add_argument(
        "--source",
        type=Path,
        help="(Необязательно) путь к эталонному workbook; для правил не используется, оставлено для совместимости.",
    )
    anomaly_parser.set_defaults(func=cmd_anomalies)

    events_parser = subparsers.add_parser("events", help="Analyse reference workbook (normal vs abnormal).")
    events_parser.add_argument("--config", default="config/pipeline.yaml")
    events_parser.add_argument("--source", type=Path, help="Override path to reference workbook.")
    events_parser.set_defaults(func=cmd_events)

    full_parser = subparsers.add_parser("full", help="Run extract, clean, align, and anomaly steps sequentially.")
    full_parser.add_argument("--config", default="config/pipeline.yaml")
    full_parser.add_argument("--source", help="Optional directory with raw excel files to ingest before running.")
    full_parser.add_argument("--anomaly-source", type=Path, help="Override path to anomaly workbook.")
    full_parser.add_argument("--calibrate", action="store_true", help="Run reference events analysis after anomalies step.")
    full_parser.set_defaults(func=cmd_full)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
