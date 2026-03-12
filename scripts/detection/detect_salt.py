from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.detection_artifacts import DEFAULT_DETECTOR, DETECTOR_KEYS
from alma_service.generic_detection import run_detection, run_single_well


def main() -> None:
    parser = argparse.ArgumentParser(description="Engineered-feature detection for salt.")
    parser.add_argument("--detector", choices=sorted(DETECTOR_KEYS), default=DEFAULT_DETECTOR)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--well", type=str, default=None)
    parser.add_argument("--retune", action="store_true")
    args = parser.parse_args()

    if args.well:
        run_single_well("salt", args.well, detector=args.detector, source_path=args.source, retune=args.retune)
        return
    run_detection(
        "salt",
        detector=args.detector,
        output_path=args.output,
        source_path=args.source,
        retune=args.retune,
        verbose=True,
    )


if __name__ == "__main__":
    main()
