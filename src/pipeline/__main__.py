"""Command-line entry point for the pipeline package."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import anomalies, events


def cmd_anomalies(args: argparse.Namespace) -> None:
    result = anomalies.run_anomaly_analysis(Path(args.config), workbook_override=args.source)
    anomaly_count = int((result["segment_type"] == "anomaly").sum()) if not result.empty else 0
    normal_count = int((result["segment_type"] == "normal").sum()) if not result.empty else 0
    print(
        f"Detected {anomaly_count} anomaly segments and {normal_count} normal segments "
        f"(total {len(result)})."
    )


def cmd_events(args: argparse.Namespace) -> None:
    summary = events.run_reference_analysis(Path(args.config), workbook_override=args.source)
    print(f"Reference analysis completed. Records: {summary['records']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for anomaly detection based on alma workbook.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    anomaly_parser = subparsers.add_parser("anomalies", help="Detect anomalies from alma/Общая_таблица.xlsx.")
    anomaly_parser.add_argument("--config", default="config/pipeline.yaml")
    anomaly_parser.add_argument("--source", type=Path, help="Override path to workbook.")
    anomaly_parser.set_defaults(func=cmd_anomalies)

    events_parser = subparsers.add_parser("events", help="Summarise reference intervals from the workbook.")
    events_parser.add_argument("--config", default="config/pipeline.yaml")
    events_parser.add_argument("--source", type=Path, help="Override path to workbook.")
    events_parser.set_defaults(func=cmd_events)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
