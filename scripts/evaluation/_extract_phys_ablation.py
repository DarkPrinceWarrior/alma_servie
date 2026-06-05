from __future__ import annotations

import glob
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "artifacts/results/global_phys_ablation"
KIND = sys.argv[2] if len(sys.argv) > 2 else "pritok"


def fmt(m: dict) -> str:
    if not m:
        return "—"
    return (
        f"hit={m.get('hit_count')}/{m.get('interval_count')} "
        f"(rate={m.get('hit_rate')}) FAR/d={m.get('false_alarms_per_day')} "
        f"medDelay={m.get('median_abs_delay_hours')}h "
        f"p90Delay={m.get('p90_abs_delay_hours')}h starts={m.get('start_count')}"
    )


for path in sorted(glob.glob(f"{ROOT}/*/global_normality_benchmark.json")):
    variant = os.path.basename(os.path.dirname(path))
    payload = json.load(open(path, encoding="utf-8"))
    cls = payload.get("classes", {}).get(KIND, {})
    pv = cls.get("physics_variant", {})
    splits = cls.get("splits", {})
    print(f"### {variant} | branch={pv.get('pressure_branch')} "
          f"w={pv.get('pressure_trend_weight')} gate={pv.get('domain_gate')}")
    print(f"    TEST: {fmt(splits.get('test', {}))}")
    print(f"    ALL : {fmt(splits.get('all', {}))}")
