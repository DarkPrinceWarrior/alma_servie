from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.anomaly_specs import get_detection_spec
from alma_service.generic_detection import (
    _prepare_all_wells,
    _resolve_torch_device,
    _runtime_config,
    load_anomaly_data,
    load_intervals,
)
from alma_service.generic_detectors import set_seed
from alma_service.shared_encoder import save_shared_encoder_state, shared_encoder_path, train_shared_encoder


def _parse_anomalies(value: str) -> list[str]:
    anomalies = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"negermet", "pritok", "salt"}
    unknown = sorted(set(anomalies) - allowed)
    if unknown:
        raise ValueError(f"Unknown anomaly keys: {unknown}")
    return anomalies


def build_encoder(anomaly_key: str, *, force: bool, verbose: bool) -> Path:
    output_path = shared_encoder_path(anomaly_key)
    if output_path.exists() and not force:
        print(f"Skip {anomaly_key}: artifact exists: {output_path}")
        return output_path

    spec = get_detection_spec(anomaly_key)
    device = _resolve_torch_device("paano_shared", verbose=True)
    df = load_anomaly_data(spec)
    intervals = load_intervals(spec, required=True)
    intervals = (
        intervals.sort_values(["well_id", "start_date", "interval_idx"])
        .groupby("well_id", as_index=False)
        .first()
    )
    prepared = _prepare_all_wells(
        spec,
        df,
        intervals,
        verbose=verbose,
        zone_aware=True,
    )
    runtime_cfg = _runtime_config(anomaly_key)
    state = train_shared_encoder(
        prepared_wells=prepared,
        patch_short=int(runtime_cfg["paano_patch_short"]),
        patch_long=int(runtime_cfg["paano_patch_long"]),
        anomaly_key=anomaly_key,
        device=device,
        verbose=verbose,
    )
    saved_path = save_shared_encoder_state(state, output_path)
    print(f"Saved {anomaly_key} shared encoder: {saved_path}")
    print(f"  detail: {state.detail}")
    return saved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen PaAno shared encoder artifacts.")
    parser.add_argument("--anomalies", default="negermet,pritok,salt")
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    for anomaly_key in _parse_anomalies(args.anomalies):
        build_encoder(anomaly_key, force=args.force, verbose=not args.quiet)


if __name__ == "__main__":
    main()
