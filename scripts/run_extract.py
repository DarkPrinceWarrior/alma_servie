import argparse
from pathlib import Path

from pipeline.extract import run_import


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extract step for selected wells.")
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"))
    parser.add_argument("--wells", type=str, required=False, help="Comma separated list of wells.")
    parser.add_argument("--combine", action="store_true", help="Combine per-well tables after processing.")
    args = parser.parse_args()

    wells = None
    if args.wells:
        wells = [w.strip() for w in args.wells.split(",") if w.strip()]

    run_import(args.config, wells=wells, combine=args.combine)


if __name__ == "__main__":
    main()
