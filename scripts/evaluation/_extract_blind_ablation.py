from __future__ import annotations

import glob
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "artifacts/test_wells_phys_ablation"

variants = sorted(
    d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))
)
wells: set[str] = set()
data: dict[str, dict[str, dict]] = {}
for v in variants:
    data[v] = {}
    for sp in glob.glob(f"{ROOT}/{v}/*/pritok/summary.json"):
        well = sp.split("/")[-3]
        wells.add(well)
        s = json.load(open(sp, encoding="utf-8"))
        data[v][well] = {
            "starts": s.get("detected_starts", []),
            "n_act": s.get("n_actionable_detected"),
            "n_det": s.get("n_detected"),
        }

for well in sorted(wells):
    print(f"\n==== {well} ====")
    for v in variants:
        rec = data[v].get(well, {})
        starts = rec.get("starts", [])
        first = starts[0] if starts else "—"
        print(f"  {v:26s} first={str(first):20s} n_det={rec.get('n_det')} "
              f"n_actionable={rec.get('n_act')} all={starts}")
