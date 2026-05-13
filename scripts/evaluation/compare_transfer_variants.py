"""Compile A/B comparison of ALMA transfer variants."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VARIANTS = [
    ("baseline (no transfer)", "baseline_pre_hook"),
    ("3w class 9 (per-class)", "3w_class_9_v2"),
    ("3w global NORMAL", "global_normal"),
    ("global + injection 0.3", "global_inject"),
]
ANOMS = ["negermet", "pritok", "salt"]


def fmt(value: float | None, width: int, precision: int) -> str:
    if value is None or not isinstance(value, (int, float)):
        return f"{'nan':>{width}}"
    return f"{value:{width}.{precision}f}"


def main() -> None:
    transfers_dir = PROJECT_ROOT / "artifacts" / "results" / "transfers"
    for a in ANOMS:
        print(f"\n## {a.upper()}")
        header = f"{'variant':<28} | {'split':<5} | {'hit':>6} | {'FAR/d':>7} | {'starts':>6} | {'mae_h':>7}"
        print(header)
        print("-" * len(header))
        for name, dirn in VARIANTS:
            p = transfers_dir / dirn / f"{a}_paano_shared_results.summary.json"
            if not p.exists():
                print(f"{name:<28} | (missing)")
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            for sp in ("all", "train", "test"):
                s = data["splits"].get(sp, {})
                if not s:
                    continue
                row = (
                    f"{name:<28} | {sp:<5} | "
                    f"{fmt(s.get('hit_rate'), 6, 4)} | "
                    f"{fmt(s.get('false_alarms_per_day'), 7, 4)} | "
                    f"{fmt(s.get('avg_starts_per_interval'), 6, 3)} | "
                    f"{fmt(s.get('delay_mae_hours'), 7, 3)}"
                )
                print(row)


if __name__ == "__main__":
    main()
