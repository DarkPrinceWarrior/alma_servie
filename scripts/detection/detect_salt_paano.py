from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.paano_pipeline import run_detection, run_single_well


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified PaAno detection for salt.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--well", type=str, default=None)
    parser.add_argument("--retune", action="store_true")
    args = parser.parse_args()

    if args.well:
        run_single_well("salt", args.well, args.source, args.retune)
        return
    run_detection("salt", output_path=args.output, source_path=args.source, retune=args.retune, verbose=True)


if __name__ == "__main__":
    main()
