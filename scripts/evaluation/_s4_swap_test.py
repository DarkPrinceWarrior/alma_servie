"""Sprint 4 helper: swap an A/B transfer encoder into production path,
run the ALMA detector, snapshot results, then restore the previous encoder."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path("/root/projects/alma_servie")


def rewrite_anomaly_key(src: Path, dst: Path, target_key: str) -> None:
    payload = torch.load(src, map_location="cpu", weights_only=False)
    payload["anomaly_key"] = target_key
    if "detail" in payload and isinstance(payload["detail"], dict):
        payload["detail"]["deployed_as"] = target_key
    torch.save(payload, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-path", required=True)
    ap.add_argument("--target-anomaly", required=True, choices=["negermet", "pritok", "salt"])
    ap.add_argument("--snapshot-tag", required=True)
    args = ap.parse_args()

    target_path = PROJECT_ROOT / "models" / f"{args.target_anomaly}_paano_shared_encoder.pt"
    backup_path = PROJECT_ROOT / "models" / f"{args.target_anomaly}_paano_shared_encoder.pt.s4backup"
    snapshot_dir = PROJECT_ROOT / "artifacts" / "results" / "transfers" / "s4_swap" / args.snapshot_tag
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not backup_path.exists():
        shutil.copy2(target_path, backup_path)
        print(f"[swap] backup -> {backup_path}", flush=True)

    rewrite_anomaly_key(Path(args.variant_path), target_path, args.target_anomaly)
    print(f"[swap] deployed {args.variant_path} -> {target_path} key={args.target_anomaly}", flush=True)

    t0 = time.time()
    detect_script = PROJECT_ROOT / "scripts" / "detection" / f"detect_{args.target_anomaly}.py"
    cmd = ["uv", "run", "python", str(detect_script), "--detector", "paano_shared"]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"[swap] detect_{args.target_anomaly} returncode={proc.returncode} t={elapsed:.1f}s", flush=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], flush=True)
        print("STDERR:", proc.stderr[-2000:], flush=True)
        sys.exit(1)

    results = PROJECT_ROOT / "artifacts" / "results" / f"{args.target_anomaly}_paano_shared_results.parquet"
    summary = PROJECT_ROOT / "artifacts" / "results" / f"{args.target_anomaly}_paano_shared_results.summary.json"
    if results.exists():
        shutil.copy2(results, snapshot_dir / results.name)
    if summary.exists():
        shutil.copy2(summary, snapshot_dir / summary.name)
        snap = json.loads(summary.read_text(encoding="utf-8"))
        all_blk = snap.get("all", snap.get("test", {}))
        print(
            f"[swap] {args.snapshot_tag} | hit={all_blk.get('hit_rate', 0):.3f} "
            f"far={all_blk.get('false_alarms_per_day', 0):.4f} "
            f"starts={all_blk.get('avg_starts_per_interval', 0):.2f} "
            f"mae_h={all_blk.get('delay_mae_hours', 0):.2f}",
            flush=True,
        )

    if backup_path.exists():
        shutil.copy2(backup_path, target_path)
        print(f"[swap] restored baseline from {backup_path}", flush=True)


if __name__ == "__main__":
    main()
