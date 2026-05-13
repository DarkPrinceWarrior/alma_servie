from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def per_class_split_composition(manifest: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    merged = manifest.merge(splits[["instance_id", "split"]], on="instance_id", how="inner")
    rows: list[dict] = []
    for label, group in merged.groupby("folder_label"):
        for sp in ("train", "val", "test"):
            sub = group[group["split"].eq(sp)]
            if len(sub) == 0:
                continue
            rows.append(
                {
                    "class": int(label),
                    "split": sp,
                    "total": int(len(sub)),
                    "real": int((sub["source_type"] == "real").sum()),
                    "hand_drawn": int((sub["source_type"] == "hand_drawn").sum()),
                    "simulated": int((sub["source_type"] == "simulated").sum()),
                    "unknown": int((sub["source_type"] == "unknown").sum()),
                    "has_transient": int(sub["has_transient"].sum()),
                    "has_event": int(sub["has_event"].sum()),
                }
            )
    return pd.DataFrame(rows)


def per_class_score_stats(scores_dir: Path, class_id: int) -> dict:
    p = scores_dir / f"class_{class_id}_scores.parquet"
    if not p.exists():
        return {"class": class_id, "available": False}
    df = pd.read_parquet(p, engine="pyarrow")
    out = {"class": class_id, "available": True, "rows": int(len(df))}
    for sp in ("train", "val", "test"):
        sub = df[df["split"].eq(sp)]
        if len(sub) == 0:
            continue
        out[f"{sp}_n_wells"] = int(sub["well_id"].nunique())
        out[f"{sp}_score_median"] = float(np.median(sub["score"]))
        out[f"{sp}_score_p90"] = float(np.quantile(sub["score"], 0.90))
        out[f"{sp}_score_p99"] = float(np.quantile(sub["score"], 0.99))
        out[f"{sp}_score_max"] = float(np.max(sub["score"]))
        if "reference_mask" in sub.columns:
            out[f"{sp}_ref_frac"] = float(sub["reference_mask"].astype(bool).mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose 3W train->test gap across all event classes.")
    parser.add_argument("--config", default="configs/3w_paano.json")
    parser.add_argument("--out", default="docs/3w_phase05_diagnostic.md")
    args = parser.parse_args()

    cfg = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    event_labels = {int(k): v for k, v in cfg["dataset"]["event_labels"].items()}
    processed_dir = PROJECT_ROOT / cfg["dataset"]["processed_dir"]
    scores_dir = PROJECT_ROOT / "artifacts" / "3w" / "scores"
    metrics_dir = PROJECT_ROOT / "artifacts" / "3w" / "metrics"

    manifest = pd.read_parquet(processed_dir / "manifest.parquet", engine="pyarrow")
    splits = pd.read_parquet(processed_dir / "splits.parquet", engine="pyarrow")

    comp_df = per_class_split_composition(manifest, splits)
    score_rows = [per_class_score_stats(scores_dir, c) for c in sorted(event_labels) if c != 0]
    score_df = pd.DataFrame([r for r in score_rows if r.get("available")])

    metric_rows: list[dict] = []
    for c in sorted(event_labels):
        if c == 0:
            continue
        mp = metrics_dir / f"class_{c}_metrics.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        for sp, d in m.get("splits", {}).items():
            metric_rows.append(
                {
                    "class": c,
                    "split": sp,
                    "hit_rate": float(d.get("hit_rate") or 0.0),
                    "far_per_day": float(d.get("false_alarms_per_day") or 0.0),
                    "starts_per_event": float(d.get("avg_starts_per_interval") or 0.0),
                }
            )
    metric_df = pd.DataFrame(metric_rows)

    print("=== Split composition by source_type ===")
    print(comp_df.to_string(index=False))
    print()
    print("=== Score distribution by split (after fusion) ===")
    if not score_df.empty:
        cols_to_show = [c for c in score_df.columns if not c.endswith("_n_wells") and not c.endswith("_ref_frac")]
        print(score_df[cols_to_show].to_string(index=False))
    print()
    print("=== Hit-rate train vs test ===")
    if not metric_df.empty:
        pivot = metric_df.pivot_table(index="class", columns="split", values="hit_rate")
        print(pivot.fillna(np.nan).to_string())

    out_md = PROJECT_ROOT / args.out
    lines: list[str] = []
    lines.append("# 3W Phase 0.5 — диагностика train->test gap")
    lines.append("")
    lines.append("Дата: " + pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    lines.append("")
    lines.append("## Композиция splits по source_type")
    lines.append("")
    lines.append("| Класс | Сплит | Всего | real | hand_drawn | simulated | unknown |")
    lines.append("|------:|:-----:|------:|-----:|-----------:|----------:|--------:|")
    for _, row in comp_df.iterrows():
        lines.append(
            f"| {row['class']} | {row['split']} | {row['total']} | "
            f"{row['real']} | {row['hand_drawn']} | {row['simulated']} | {row['unknown']} |"
        )
    lines.append("")
    lines.append("## Распределение score после fusion (медиана / P90 / P99 / max)")
    lines.append("")
    if not score_df.empty:
        lines.append("| Класс | train med | train P99 | train max | test med | test P99 | test max | ratio test_max/train_max |")
        lines.append("|------:|---------:|----------:|----------:|---------:|---------:|---------:|------------------------:|")
        for _, row in score_df.iterrows():
            tr_max = row.get("train_score_max", 0.0)
            te_max = row.get("test_score_max", 0.0)
            ratio = (te_max / tr_max) if (tr_max and tr_max > 0) else float("nan")
            lines.append(
                f"| {int(row['class'])} | "
                f"{row.get('train_score_median', float('nan')):.3f} | "
                f"{row.get('train_score_p99', float('nan')):.3f} | "
                f"{row.get('train_score_max', float('nan')):.3f} | "
                f"{row.get('test_score_median', float('nan')):.3f} | "
                f"{row.get('test_score_p99', float('nan')):.3f} | "
                f"{row.get('test_score_max', float('nan')):.3f} | "
                f"{ratio:.3f} |"
            )
    lines.append("")
    lines.append("## Hit-rate train vs test")
    lines.append("")
    if not metric_df.empty:
        pivot = metric_df.pivot_table(index="class", columns="split", values="hit_rate")
        lines.append("| Класс | train | val | test |")
        lines.append("|------:|------:|----:|-----:|")
        for cls, row in pivot.iterrows():
            tr = row.get("train", float("nan"))
            vl = row.get("val", float("nan"))
            te = row.get("test", float("nan"))
            lines.append(f"| {int(cls)} | {tr:.3f} | {vl if isinstance(vl, float) and not pd.isna(vl) else '—'} | {te:.3f} |")
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[diagnostic] {out_md}")


if __name__ == "__main__":
    main()
