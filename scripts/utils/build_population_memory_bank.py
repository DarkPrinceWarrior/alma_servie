"""Собрать population memory bank для blind-инференса.

Для `paano_global` строится один общий банк в пространстве
`global_normality_paano_shared_encoder.pt`, общий для negermet/pritok/salt.

Для остальных детекторов источники нормы (per-anomaly):
- labelled train скв.: первая часть до первого labelled_start (через
  prepare_engineered_well + anomaly_intervals → ветка
  REFERENCE_POLICY_NORMAL_WINDOWS).
- norm_work скв.: ВСЁ как норма (normal_reference_fraction=1.0).

Артефакты:
- models/population_memory_bank_paano_global_global_normality.npz
- models/population_memory_bank_<detector>_<anomaly>.npz
{ features: (N, C) float32, shared_channels: object[], sources: object[][well_id,count] }
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.anomaly_specs import get_detection_spec
from alma_service.engineered_features import (
    REFERENCE_POLICY_NORMAL_WINDOWS,
    prepare_engineered_well,
)
from alma_service.generic_detection import (
    MIN_REFERENCE_COVERAGE,
    MIN_TOTAL_COVERAGE,
    REFERENCE_MAX_RATIO,
    REFERENCE_MIN_DAYS,
    REFERENCE_MIN_RATIO,
    _runtime_config,
)
from alma_service.global_normality import (
    GLOBAL_DETECTOR_KEY,
    GLOBAL_ENCODER_KEY,
    prepare_global_normality_runtime,
)
from alma_service.paths import DB_DIR, MODELS_DIR
from alma_service.shared_encoder import load_shared_encoder_state, select_shared_columns
from alma_service.tabular_io import read_table


def _collect_one(
    anomaly_key: str,
    well_id: str,
    well_df: pd.DataFrame,
    intervals: pd.DataFrame | None,
    normal_reference_fraction: float | None,
):
    return prepare_engineered_well(
        anomaly_key=anomaly_key,
        well_id=well_id,
        split="population",
        well_df=well_df,
        patch_size=int(_runtime_config(anomaly_key)["prepare_patch_size"]),
        reference_min_ratio=REFERENCE_MIN_RATIO,
        reference_max_ratio=REFERENCE_MAX_RATIO,
        reference_min_days=REFERENCE_MIN_DAYS,
        min_reference_coverage=MIN_REFERENCE_COVERAGE,
        min_total_coverage=MIN_TOTAL_COVERAGE,
        anomaly_intervals=intervals,
        reference_policy=REFERENCE_POLICY_NORMAL_WINDOWS,
        normal_reference_fraction=normal_reference_fraction,
    )


def _stratified_subsample(
    chunks: list[np.ndarray], max_total: int, seed: int = 0
) -> tuple[np.ndarray, list[int]]:
    """Sample max_total points proportionally across source chunks."""
    rng = np.random.default_rng(seed)
    sizes = np.array([len(c) for c in chunks], dtype=np.int64)
    total = int(sizes.sum())
    if total <= max_total:
        return np.concatenate(chunks, axis=0), [len(c) for c in chunks]
    # quotas proportional to source size, minimum 50 per source
    quotas = np.maximum((sizes / total * max_total).astype(np.int64), np.minimum(sizes, 50))
    # rescale if sum > max_total
    while quotas.sum() > max_total:
        idx = int(np.argmax(quotas))
        quotas[idx] -= 1
    sampled: list[np.ndarray] = []
    taken: list[int] = []
    for chunk, q in zip(chunks, quotas):
        q = int(min(q, len(chunk)))
        if q <= 0:
            taken.append(0)
            continue
        idxs = rng.choice(len(chunk), size=q, replace=False)
        sampled.append(chunk[idxs])
        taken.append(q)
    return np.concatenate(sampled, axis=0), taken


def build_for_anomaly(anomaly_key: str, detector: str, out_path: Path, max_points: int | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_state = load_shared_encoder_state(anomaly_key, device=device, verbose=False)
    shared_channels = list(shared_state.shared_channels)
    print(f"  shared_channels: {len(shared_channels)} channels")

    nw_df = read_table(DB_DIR / "norm_work_database_5min.parquet")
    nw_df["well_id"] = nw_df["well_id"].astype(str).str.lower()
    nw_wells = sorted(nw_df["well_id"].unique())

    pool: list[np.ndarray] = []
    sources: list[list[str]] = []

    # Cross-anomaly labelled donors: каждая labelled-скв ВСЕХ типов даёт
    # эмбеддинги нормы до first_start своей собственной аномалии.
    for donor_anomaly in ("negermet", "pritok", "salt"):
        lab_df = read_table(DB_DIR / f"{donor_anomaly}_anomaly_database_5min.parquet")
        lab_df["well_id"] = lab_df["well_id"].astype(str).str.lower()
        intv = read_table(DB_DIR / f"{donor_anomaly}_intervals.parquet")
        intv["well_id"] = intv["well_id"].astype(str).str.lower()
        train_wells = sorted(intv[intv["split"] == "train"]["well_id"].unique())
        kind = "self" if donor_anomaly == anomaly_key else "cross"
        print(f"  [labelled-{donor_anomaly} ({kind})] {len(train_wells)} wells (до first_start)")
        for well_id in train_wells:
            wdf = lab_df[lab_df["well_id"] == well_id]
            wintv = intv[intv["well_id"] == well_id]
            prepared = _collect_one(anomaly_key, well_id, wdf, wintv, normal_reference_fraction=None)
            if prepared is None:
                print(f"    {well_id}: SKIP (prepare returned None)")
                continue
            X = select_shared_columns(prepared.feature_columns, prepared.feature_matrix, shared_channels)
            ref = X[prepared.reference_mask].astype(np.float32)
            print(f"    {donor_anomaly}:{well_id}: {len(ref):>7d} points")
            pool.append(ref)
            sources.append([f"labelled-{donor_anomaly}:{well_id}", str(len(ref))])

    print(f"  [norm_work] {len(nw_wells)} wells (full series)")
    for well_id in nw_wells:
        wdf = nw_df[nw_df["well_id"] == well_id]
        prepared = _collect_one(anomaly_key, well_id, wdf, intervals=None, normal_reference_fraction=1.0)
        if prepared is None:
            print(f"    {well_id}: SKIP (prepare returned None)")
            continue
        X = select_shared_columns(prepared.feature_columns, prepared.feature_matrix, shared_channels)
        ref = X[prepared.reference_mask].astype(np.float32)
        print(f"    {well_id}: {len(ref):>7d} points")
        pool.append(ref)
        sources.append([f"norm_work:{well_id}", str(len(ref))])

    if not pool:
        raise RuntimeError(f"No sources collected for {anomaly_key}")

    raw_total = sum(len(c) for c in pool)
    if max_points is None or max_points >= raw_total:
        population = np.concatenate(pool, axis=0).astype(np.float32)
        sources_with_taken = [[s[0], s[1], s[1]] for s in sources]
        suffix = f"(no subsample, {raw_total} points)"
    else:
        population, taken_per_source = _stratified_subsample(pool, max_total=max_points)
        sources_with_taken = [[s[0], s[1], str(t)] for s, t in zip(sources, taken_per_source)]
        population = population.astype(np.float32)
        suffix = f"(subsampled from {raw_total} to {max_points})"
    np.savez_compressed(
        out_path,
        features=population,
        shared_channels=np.array(shared_channels, dtype=object),
        sources=np.array(sources_with_taken, dtype=object),
    )
    total_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n  Saved {anomaly_key}: shape={population.shape} {suffix}, file={total_mb:.1f} MB → {out_path}")


def build_global_normality_bank(
    detector: str,
    out_path: Path,
    max_points: int | None,
    anomaly_key: str = "negermet",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = prepare_global_normality_runtime(anomaly_key, device=device, verbose=True)
    shared_channels = list(runtime.shared_state.shared_channels)
    print(f"  global shared_channels: {len(shared_channels)} channels")

    pool: list[np.ndarray] = []
    sources: list[list[str]] = []
    for pool_key, prepared in sorted(runtime.population_pool.items()):
        X = select_shared_columns(prepared.feature_columns, prepared.feature_matrix, shared_channels)
        ref = X[np.asarray(prepared.reference_mask, dtype=bool)].astype(np.float32)
        if len(ref) == 0:
            continue
        print(f"    {pool_key}: {len(ref):>7d} points")
        pool.append(ref)
        sources.append([str(pool_key), str(len(ref))])

    if not pool:
        raise RuntimeError("No sources collected for global normality population bank")

    raw_total = sum(len(c) for c in pool)
    if max_points is None or max_points >= raw_total:
        population = np.concatenate(pool, axis=0).astype(np.float32)
        sources_with_taken = [[s[0], s[1], s[1]] for s in sources]
        suffix = f"(no subsample, {raw_total} points)"
    else:
        population, taken_per_source = _stratified_subsample(pool, max_total=max_points)
        sources_with_taken = [[s[0], s[1], str(t)] for s, t in zip(sources, taken_per_source)]
        population = population.astype(np.float32)
        suffix = f"(subsampled from {raw_total} to {max_points})"

    np.savez_compressed(
        out_path,
        features=population,
        shared_channels=np.array(shared_channels, dtype=object),
        sources=np.array(sources_with_taken, dtype=object),
    )
    total_mb = out_path.stat().st_size / 1024 / 1024
    print(
        f"\n  Saved {detector}:{GLOBAL_ENCODER_KEY}: shape={population.shape} "
        f"{suffix}, file={total_mb:.1f} MB → {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build population memory bank for paano_global blind inference.")
    parser.add_argument("--anomaly", default="all", choices=["all", "negermet", "pritok", "salt"])
    parser.add_argument("--detector", default="paano_global")
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional cap: stratified subsample to this many points per bank. Default: no subsample.",
    )
    args = parser.parse_args()

    anomalies = ("negermet", "pritok", "salt") if args.anomaly == "all" else (args.anomaly,)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cap = "no_cap" if args.max_points is None else str(args.max_points)
    if args.detector == GLOBAL_DETECTOR_KEY:
        out = MODELS_DIR / f"population_memory_bank_{args.detector}_{GLOBAL_ENCODER_KEY}.npz"
        print(f"\n=== {GLOBAL_ENCODER_KEY} (max_points={cap}) ===")
        # при --anomaly <класс> runtime готовится под этот класс: вместе с
        # ALMA_GLOBAL_POOL_CLASS_ONLY=1 это даёт банк только из выбранного класса + нормы
        bank_anomaly = args.anomaly if args.anomaly != "all" else "negermet"
        build_global_normality_bank(args.detector, out, args.max_points, anomaly_key=bank_anomaly)
        return
    for a in anomalies:
        out = MODELS_DIR / f"population_memory_bank_{args.detector}_{a}.npz"
        print(f"\n=== {a} (max_points={cap}) ===")
        build_for_anomaly(a, args.detector, out, args.max_points)


if __name__ == "__main__":
    main()
