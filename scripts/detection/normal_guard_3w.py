"""3W cross-instance NORMAL guard.

Builds an aggregated per-channel quantile envelope from all class 0 (NORMAL)
instances of the Petrobras 3W dataset, and exposes a callable that, given a
test-time feature matrix, returns a `looks_like_normal` indicator per timestep.

Idea: if at a given timestep enough channels fall inside the [p10, p90] band of
the NORMAL corpus, the onset detector should suppress an alarm there. This
provides an additional, cross-instance prior beyond each well's local reference
window — useful for instances whose local reference is short or contaminated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRIOR_CHANNELS = [
    "P-PDG", "P-TPT", "P-MON-CKP", "P-JUS-CKP",
    "P-JUS-CKGL", "P-ANULAR", "T-PDG", "T-TPT", "T-JUS-CKP", "T-MON-CKP",
    "QGL",
]


def fit_prior(processed_dir: Path, manifest: pd.DataFrame, quantiles: tuple[float, float] = (0.10, 0.90)) -> dict:
    """Fit per-channel quantile envelope from class 0 (NORMAL) raw rows.

    Reads the per-class features parquet for class 0 if available; otherwise
    fall back to scanning instance-level parquet files under data/raw/3w.
    Returns a dict mapping channel -> {p_low, p_high, median}.
    """
    p = processed_dir / "class_0" / "features.parquet"
    if p.exists():
        df = pd.read_parquet(p, engine="pyarrow")
    else:
        # Stitch from manifest raw paths
        normals = manifest[manifest["folder_label"].astype(int) == 0]
        frames = []
        for _, row in normals.iterrows():
            raw_path = PROJECT_ROOT / row["path"]
            if not raw_path.exists():
                continue
            sub = pd.read_parquet(raw_path, engine="pyarrow")
            frames.append(sub)
            if sum(len(f) for f in frames) > 1_000_000:
                break
        if not frames:
            raise FileNotFoundError("Could not assemble NORMAL prior from manifest paths")
        df = pd.concat(frames, axis=0, ignore_index=True)

    prior: dict[str, dict[str, float]] = {}
    for ch in PRIOR_CHANNELS:
        if ch not in df.columns:
            continue
        v = pd.to_numeric(df[ch], errors="coerce").to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        if len(v) < 100:
            continue
        prior[ch] = {
            "p_low": float(np.quantile(v, quantiles[0])),
            "p_high": float(np.quantile(v, quantiles[1])),
            "median": float(np.median(v)),
        }
    return prior


def normal_likeness(features_df: pd.DataFrame, prior: dict) -> np.ndarray:
    """Fraction of prior channels within [p_low, p_high] envelope at each row."""
    chans = [c for c in prior if c in features_df.columns]
    if not chans:
        return np.zeros(len(features_df), dtype=np.float32)
    counts = np.zeros(len(features_df), dtype=np.int32)
    for ch in chans:
        v = pd.to_numeric(features_df[ch], errors="coerce").to_numpy(dtype=np.float64)
        lo, hi = prior[ch]["p_low"], prior[ch]["p_high"]
        ok = (v >= lo) & (v <= hi)
        counts = counts + ok.astype(np.int32)
    return (counts / len(chans)).astype(np.float32)


def save_prior(prior: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(prior, indent=2, ensure_ascii=False), encoding="utf-8")


def load_prior(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Fit NORMAL prior from 3W class 0.")
    parser.add_argument("--processed-dir", default="data/processed/3w")
    parser.add_argument("--out", default="artifacts/3w/normal_prior.json")
    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / args.processed_dir
    manifest = pd.read_parquet(processed_dir / "manifest.parquet", engine="pyarrow")
    prior = fit_prior(processed_dir, manifest)
    out_path = PROJECT_ROOT / args.out
    save_prior(prior, out_path)
    print(f"[normal_guard] prior over {len(prior)} channels -> {out_path}")
    for ch, p in prior.items():
        print(f"  {ch}: [{p['p_low']:.3f}, {p['p_high']:.3f}]  median={p['median']:.3f}")


if __name__ == "__main__":
    main()
