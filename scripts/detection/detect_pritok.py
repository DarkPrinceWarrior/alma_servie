from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.detection_artifacts import DETECTOR_KEYS, default_detector_for
from alma_service.engineered_features import REFERENCE_POLICIES, REFERENCE_POLICY_NORMAL_WINDOWS
from alma_service.generic_detection import run_detection, run_single_well


def main() -> None:
    parser = argparse.ArgumentParser(description="Engineered-feature detection for pritok.")
    parser.add_argument("--detector", choices=sorted(DETECTOR_KEYS), default=default_detector_for("pritok"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--well", type=str, default=None)
    parser.add_argument("--retune", action="store_true")
    parser.add_argument("--reference-policy", choices=sorted(REFERENCE_POLICIES), default=REFERENCE_POLICY_NORMAL_WINDOWS)
    args = parser.parse_args()

    if args.well:
        run_single_well(
            "pritok",
            args.well,
            detector=args.detector,
            source_path=args.source,
            retune=args.retune,
            reference_policy=args.reference_policy,
        )
        return
    run_detection(
        "pritok",
        detector=args.detector,
        output_path=args.output,
        source_path=args.source,
        retune=args.retune,
        verbose=True,
        reference_policy=args.reference_policy,
    )


if __name__ == "__main__":
    main()
