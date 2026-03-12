"""
Build negermet dataset on a uniform grid.

This wrapper keeps the old command stable but delegates the logic to the
shared dataset builder used by all anomaly types.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.anomaly_specs import get_dataset_spec
from alma_service.dataset_builder import build_dataset


def main(freq: str = "15s") -> None:
    build_dataset(get_dataset_spec("negermet"), freq=freq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", default="15s", help="Resample frequency (e.g. 15s, 2min)")
    args = parser.parse_args()
    main(freq=args.freq)
