from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ESTADO_COLS = (
    "ESTADO-DHSV", "ESTADO-M1", "ESTADO-M2", "ESTADO-PXO",
    "ESTADO-SDV-GL", "ESTADO-SDV-P", "ESTADO-W1", "ESTADO-W2", "ESTADO-XO",
)
LABEL_COLS = ("class", "state")
SOURCE_RE = re.compile(r"^(WELL-\d+|DRAWN|SIMULATED)")


@dataclass
class InstanceMeta:
    instance_id: str
    path: str
    folder_label: int
    event_name: str
    source_type: str
    n_rows: int
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    duration_hours: float
    columns_available: list[str]
    is_valid: bool
    validation_error: str
    has_normal: bool
    has_transient: bool
    has_event: bool
    undesirable_start_ts: pd.Timestamp | None
    transient_start_ts: pd.Timestamp | None
    event_start_ts: pd.Timestamp | None


def parse_source_type(filename: str) -> str:
    m = SOURCE_RE.match(filename)
    if not m:
        return "unknown"
    token = m.group(1)
    if token.startswith("WELL-"):
        return "real"
    if token == "DRAWN":
        return "hand_drawn"
    if token == "SIMULATED":
        return "simulated"
    return "unknown"


def find_usable_columns(df: pd.DataFrame, drop_estado: bool, min_ratio: float) -> list[str]:
    cols = []
    for c in df.columns:
        if c in LABEL_COLS:
            continue
        if drop_estado and c in ESTADO_COLS:
            continue
        if not np.issubdtype(df[c].dtype, np.floating):
            continue
        nn = df[c].notna().sum()
        if len(df) > 0 and nn / len(df) >= min_ratio:
            cols.append(c)
    return cols


def resample_instance(df: pd.DataFrame, freq: str, usable_cols: list[str]) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    numeric = df[usable_cols].astype("float64")
    resampled = numeric.resample(freq).median()
    class_series = df["class"].astype("Int64").resample(freq).agg(
        lambda s: int(s.dropna().max()) if s.dropna().size else pd.NA
    )
    state_series = df["state"].astype("Int64").resample(freq).agg(
        lambda s: int(s.dropna().max()) if s.dropna().size else pd.NA
    )
    out = resampled.copy()
    out["class"] = class_series
    out["state"] = state_series
    return out


def extract_label_starts(
    resampled: pd.DataFrame, folder_label: int, transient_offset: int
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    cls = resampled["class"]
    transient_label = folder_label + transient_offset
    transient_idx = cls.eq(transient_label)
    event_idx = cls.eq(folder_label)
    non_normal_idx = cls.notna() & cls.ne(0)
    transient_start = resampled.index[transient_idx][0] if transient_idx.any() else None
    event_start = resampled.index[event_idx][0] if event_idx.any() else None
    undesirable_candidates = [t for t in (transient_start, event_start) if t is not None]
    if not undesirable_candidates and non_normal_idx.any():
        undesirable_candidates = [resampled.index[non_normal_idx][0]]
    undesirable = min(undesirable_candidates) if undesirable_candidates else None
    return transient_start, event_start, undesirable


def build_features_from_resampled(
    resampled: pd.DataFrame,
    usable_cols: list[str],
    rolling_windows_minutes: list[int],
    rolling_ops: list[str],
    freq_minutes: int,
) -> tuple[np.ndarray, list[str]]:
    feature_columns: list[str] = []
    feature_arrays: list[np.ndarray] = []
    for c in usable_cols:
        feature_columns.append(c)
        feature_arrays.append(resampled[c].to_numpy(dtype=np.float32))
    for win_min in rolling_windows_minutes:
        n = max(int(win_min / max(freq_minutes, 1)), 2)
        for c in usable_cols:
            ser = resampled[c]
            if "mean" in rolling_ops:
                feature_columns.append(f"{c}_roll{win_min}m_mean")
                feature_arrays.append(ser.rolling(n, min_periods=1).mean().to_numpy(dtype=np.float32))
            if "std" in rolling_ops:
                feature_columns.append(f"{c}_roll{win_min}m_std")
                feature_arrays.append(
                    ser.rolling(n, min_periods=2).std().fillna(0.0).to_numpy(dtype=np.float32)
                )
    matrix = np.stack(feature_arrays, axis=1)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, feature_columns


def process_one_instance(
    path: Path,
    folder_label: int,
    event_name: str,
    cfg: dict,
) -> tuple[InstanceMeta, pd.DataFrame | None]:
    rel_path = str(path)
    source_type = parse_source_type(path.name)
    instance_id = f"c{folder_label}_{path.stem}"
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        return (
            InstanceMeta(
                instance_id=instance_id, path=rel_path, folder_label=folder_label,
                event_name=event_name, source_type=source_type, n_rows=0,
                start_ts=pd.Timestamp("NaT"), end_ts=pd.Timestamp("NaT"),
                duration_hours=0.0, columns_available=[], is_valid=False,
                validation_error=f"read_failed: {exc}",
                has_normal=False, has_transient=False, has_event=False,
                undesirable_start_ts=None, transient_start_ts=None, event_start_ts=None,
            ),
            None,
        )
    if "class" not in df.columns or "state" not in df.columns:
        return (
            InstanceMeta(
                instance_id=instance_id, path=rel_path, folder_label=folder_label,
                event_name=event_name, source_type=source_type, n_rows=len(df),
                start_ts=df.index[0] if len(df) else pd.Timestamp("NaT"),
                end_ts=df.index[-1] if len(df) else pd.Timestamp("NaT"),
                duration_hours=0.0, columns_available=list(df.columns), is_valid=False,
                validation_error="missing_class_or_state",
                has_normal=False, has_transient=False, has_event=False,
                undesirable_start_ts=None, transient_start_ts=None, event_start_ts=None,
            ),
            None,
        )
    usable = find_usable_columns(
        df,
        drop_estado=cfg["preprocess"]["drop_estado_columns"],
        min_ratio=cfg["preprocess"]["min_non_null_ratio_per_channel"],
    )
    if len(usable) == 0:
        return (
            InstanceMeta(
                instance_id=instance_id, path=rel_path, folder_label=folder_label,
                event_name=event_name, source_type=source_type, n_rows=len(df),
                start_ts=df.index[0], end_ts=df.index[-1],
                duration_hours=(df.index[-1] - df.index[0]).total_seconds() / 3600.0,
                columns_available=list(df.columns), is_valid=False,
                validation_error="no_usable_channels",
                has_normal=False, has_transient=False, has_event=False,
                undesirable_start_ts=None, transient_start_ts=None, event_start_ts=None,
            ),
            None,
        )
    freq = cfg["preprocess"]["resample_freq"]
    resampled = resample_instance(df, freq, usable)
    if len(resampled) < cfg["preprocess"]["min_points_per_instance"]:
        return (
            InstanceMeta(
                instance_id=instance_id, path=rel_path, folder_label=folder_label,
                event_name=event_name, source_type=source_type, n_rows=len(resampled),
                start_ts=resampled.index[0], end_ts=resampled.index[-1],
                duration_hours=(resampled.index[-1] - resampled.index[0]).total_seconds() / 3600.0,
                columns_available=usable, is_valid=False,
                validation_error=f"too_short_after_resample({len(resampled)})",
                has_normal=False, has_transient=False, has_event=False,
                undesirable_start_ts=None, transient_start_ts=None, event_start_ts=None,
            ),
            None,
        )
    transient_start, event_start, undesirable = extract_label_starts(
        resampled, folder_label, cfg["dataset"]["transient_offset"]
    )
    cls = resampled["class"]
    has_normal = bool(cls.eq(0).any())
    has_transient = transient_start is not None
    has_event = event_start is not None
    freq_minutes = int(pd.Timedelta(freq).total_seconds() // 60) or 1
    matrix, feature_columns = build_features_from_resampled(
        resampled, usable,
        cfg["features"]["rolling_windows_minutes"],
        cfg["features"]["rolling_ops"],
        freq_minutes=freq_minutes,
    )
    timestamps = resampled.index.values
    is_normal_point = cls.fillna(-1).to_numpy() == 0
    rows = pd.DataFrame(
        {
            "instance_id": instance_id,
            "timestamp": timestamps,
            "class_value": cls.astype("Int64").to_numpy(),
            "is_normal": is_normal_point,
        }
    )
    for i, col in enumerate(feature_columns):
        rows[col] = matrix[:, i]
    meta = InstanceMeta(
        instance_id=instance_id, path=rel_path, folder_label=folder_label,
        event_name=event_name, source_type=source_type, n_rows=len(resampled),
        start_ts=pd.Timestamp(resampled.index[0]),
        end_ts=pd.Timestamp(resampled.index[-1]),
        duration_hours=float((resampled.index[-1] - resampled.index[0]).total_seconds() / 3600.0),
        columns_available=usable, is_valid=True, validation_error="",
        has_normal=has_normal, has_transient=has_transient, has_event=has_event,
        undesirable_start_ts=undesirable, transient_start_ts=transient_start, event_start_ts=event_start,
    )
    return meta, rows


def per_class_common_channels(
    per_class_frames: dict[int, list[pd.DataFrame]],
    metas: list[InstanceMeta],
    min_presence_fraction: float = 0.50,
) -> dict[int, list[str]]:
    """Determine the canonical feature columns for each class.

    A column is kept for the class only if it is actually populated (has any
    finite values) in at least min_presence_fraction of the class's valid
    instances. This avoids the v1 bug where the class-level features parquet
    silently kept channels that were fully NaN for a subset of instances
    (typically real wells lacking columns that simulated wells had).
    """
    out: dict[int, list[str]] = {}
    meta_by_id = {m.instance_id: m for m in metas if m.is_valid}
    for label, frames in per_class_frames.items():
        valid_frames = [f for f in frames if f["instance_id"].iloc[0] in meta_by_id]
        if not valid_frames:
            out[label] = []
            continue
        all_cols: set[str] = set()
        for f in valid_frames:
            all_cols.update(
                c for c in f.columns
                if c not in ("instance_id", "timestamp", "class_value", "is_normal")
            )
        presence: dict[str, int] = {c: 0 for c in all_cols}
        for f in valid_frames:
            cols_in_frame = [c for c in all_cols if c in f.columns]
            mat = f[cols_in_frame].to_numpy(dtype=np.float32)
            has_any = np.any(np.isfinite(mat) & (mat != 0.0), axis=0)
            for ci, c in enumerate(cols_in_frame):
                if has_any[ci]:
                    presence[c] += 1
        n_valid = len(valid_frames)
        threshold = max(int(np.ceil(n_valid * min_presence_fraction)), 1)
        kept = sorted([c for c, n in presence.items() if n >= threshold])
        out[label] = kept
    return out


def make_splits_balanced(manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stratify per (folder_label, source_type) so each split gets a
    proportional mix of real / hand_drawn / simulated instances.

    This replaces the v1 'prefer_real_in_test' logic that pushed all real
    instances into test for several classes, leaving train+val 100% simulated
    and causing severe distribution shift between train and test.
    """
    rng = np.random.default_rng(cfg["splits"]["seed"])
    train_ratio = cfg["splits"]["train_ratio"]
    val_ratio = cfg["splits"]["val_ratio"]
    valid = manifest[manifest["is_valid"]].copy()
    rows: list[dict] = []
    for label, group in valid.groupby("folder_label"):
        for source_type, sub in group.groupby("source_type"):
            ids = sub["instance_id"].tolist()
            rng.shuffle(ids)
            n_total = len(ids)
            if n_total == 0:
                continue
            n_test = max(int(round(n_total * (1 - train_ratio - val_ratio))), 0)
            n_val = max(int(round(n_total * val_ratio)), 0)
            if n_total <= 3:
                n_train = max(n_total - 1, 1)
                n_val = max(min(1, n_total - n_train), 0)
                n_test = max(n_total - n_train - n_val, 0)
            else:
                if n_test == 0 and n_total >= 4:
                    n_test = 1
                if n_val == 0 and n_total >= 4:
                    n_val = 1
            n_train = n_total - n_test - n_val
            train_ids = ids[:n_train]
            val_ids = ids[n_train: n_train + n_val]
            test_ids = ids[n_train + n_val:]
            for iid in train_ids:
                rows.append({"instance_id": iid, "folder_label": label, "split": "train", "source_type": source_type})
            for iid in val_ids:
                rows.append({"instance_id": iid, "folder_label": label, "split": "val", "source_type": source_type})
            for iid in test_ids:
                rows.append({"instance_id": iid, "folder_label": label, "split": "test", "source_type": source_type})
    return pd.DataFrame(rows)


def make_intervals(manifest: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    merged = manifest.merge(splits[["instance_id", "split"]], on="instance_id", how="inner")
    rows: list[dict] = []
    for _, row in merged.iterrows():
        if pd.isna(row["undesirable_start_ts"]) or row["folder_label"] == 0:
            continue
        rows.append(
            {
                "well_id": row["instance_id"],
                "start_date": pd.Timestamp(row["undesirable_start_ts"]),
                "end_date": pd.Timestamp(row["end_ts"]),
                "data_start": pd.Timestamp(row["start_ts"]),
                "data_end": pd.Timestamp(row["end_ts"]),
                "split": row["split"],
                "interval_idx": 1,
                "folder_label": int(row["folder_label"]),
                "event_name": row["event_name"],
                "source_type": row["source_type"],
                "transient_start_ts": row["transient_start_ts"],
                "event_start_ts": row["event_start_ts"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Petrobras 3W manifest, intervals, splits, per-class features.")
    parser.add_argument("--config", default="configs/3w_paano.json")
    parser.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated folder labels to include.")
    parser.add_argument("--max-instances-per-class", type=int, default=0, help="0 = all")
    parser.add_argument("--min-channel-presence", type=float, default=0.50,
                        help="Drop class-level columns populated in fewer than this fraction of instances.")
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"]
    processed_dir = PROJECT_ROOT / cfg["dataset"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    target_labels = [int(x) for x in args.classes.split(",") if x.strip() != ""]
    event_labels = {int(k): v for k, v in cfg["dataset"]["event_labels"].items()}

    metas: list[InstanceMeta] = []
    per_class_rows: dict[int, list[pd.DataFrame]] = {}
    for folder_label in target_labels:
        event_name = event_labels.get(folder_label, f"UNKNOWN_{folder_label}")
        folder = raw_dir / str(folder_label)
        if not folder.exists():
            print(f"[warn] folder missing: {folder}", flush=True)
            continue
        files = sorted(folder.glob("*.parquet"))
        if args.max_instances_per_class > 0:
            files = files[: args.max_instances_per_class]
        print(f"[class {folder_label}] {len(files)} files...", flush=True)
        per_class_rows.setdefault(folder_label, [])
        for i, fp in enumerate(files, 1):
            meta, rows = process_one_instance(fp, folder_label, event_name, cfg)
            metas.append(meta)
            if rows is not None:
                per_class_rows[folder_label].append(rows)
            if i % 25 == 0 or i == len(files):
                valid_count = sum(1 for m in metas if m.folder_label == folder_label and m.is_valid)
                print(f"  [class {folder_label}] {i}/{len(files)} valid_so_far={valid_count}", flush=True)

    manifest_rows = []
    for m in metas:
        manifest_rows.append(
            {
                "instance_id": m.instance_id,
                "path": m.path,
                "folder_label": m.folder_label,
                "event_name": m.event_name,
                "source_type": m.source_type,
                "n_rows": m.n_rows,
                "start_ts": m.start_ts,
                "end_ts": m.end_ts,
                "duration_hours": m.duration_hours,
                "n_columns": len(m.columns_available),
                "columns_available": ",".join(m.columns_available),
                "is_valid": m.is_valid,
                "validation_error": m.validation_error,
                "has_normal": m.has_normal,
                "has_transient": m.has_transient,
                "has_event": m.has_event,
                "undesirable_start_ts": m.undesirable_start_ts,
                "transient_start_ts": m.transient_start_ts,
                "event_start_ts": m.event_start_ts,
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = processed_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, engine="pyarrow", compression="brotli")
    print(f"[manifest] {manifest_path}  total={len(manifest)} valid={int(manifest['is_valid'].sum())}", flush=True)

    splits = make_splits_balanced(manifest, cfg)
    splits_path = processed_dir / "splits.parquet"
    splits.to_parquet(splits_path, engine="pyarrow", compression="brotli")
    print(f"[splits] {splits_path}  rows={len(splits)}", flush=True)

    intervals = make_intervals(manifest, splits)
    intervals_path = processed_dir / "intervals.parquet"
    intervals.to_parquet(intervals_path, engine="pyarrow", compression="brotli")
    print(f"[intervals] {intervals_path}  rows={len(intervals)}", flush=True)

    common_per_class = per_class_common_channels(
        per_class_rows, metas, min_presence_fraction=args.min_channel_presence,
    )

    for folder_label, frames in per_class_rows.items():
        if not frames:
            continue
        out_dir = processed_dir / f"class_{folder_label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        valid_ids = set(manifest[(manifest.folder_label == folder_label) & manifest.is_valid]["instance_id"])
        kept = [f for f in frames if f["instance_id"].iloc[0] in valid_ids]
        if not kept:
            print(f"[class {folder_label}] no valid frames", flush=True)
            continue
        kept_columns = common_per_class.get(folder_label, [])
        usable_frames = []
        for f in kept:
            sub = f[["instance_id", "timestamp", "class_value", "is_normal"]].copy()
            for c in kept_columns:
                if c in f.columns:
                    sub[c] = f[c].to_numpy()
                else:
                    sub[c] = 0.0
            usable_frames.append(sub)
        if not usable_frames:
            print(f"[class {folder_label}] no usable frames", flush=True)
            continue
        df = pd.concat(usable_frames, axis=0, ignore_index=True)
        mat = df[kept_columns].to_numpy(dtype=np.float32)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        for i, c in enumerate(kept_columns):
            df[c] = mat[:, i]
        out_path = out_dir / "features.parquet"
        df.to_parquet(out_path, engine="pyarrow", compression="brotli")
        print(
            f"[class {folder_label}] features rows={len(df)} channels={len(kept_columns)} "
            f"instances={df['instance_id'].nunique()} -> {out_path}",
            flush=True,
        )

    summary = {
        "config": str(cfg_path),
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "manifest_rows": int(len(manifest)),
        "valid_rows": int(manifest["is_valid"].sum()),
        "min_channel_presence": float(args.min_channel_presence),
        "classes": sorted(set(int(x) for x in manifest["folder_label"])),
        "by_class": {
            int(label): {
                "total": int((manifest.folder_label == label).sum()),
                "valid": int(((manifest.folder_label == label) & manifest.is_valid).sum()),
                "real": int(((manifest.folder_label == label) & manifest.is_valid & (manifest.source_type == "real")).sum()),
                "drawn": int(((manifest.folder_label == label) & manifest.is_valid & (manifest.source_type == "hand_drawn")).sum()),
                "simulated": int(((manifest.folder_label == label) & manifest.is_valid & (manifest.source_type == "simulated")).sum()),
                "canonical_channels": len(common_per_class.get(int(label), [])),
            }
            for label in sorted(set(manifest["folder_label"]))
        },
        "split_counts_by_split_source": (
            [
                {"split": str(k[0]), "source_type": str(k[1]), "count": int(v)}
                for k, v in splits.groupby(["split", "source_type"]).size().to_dict().items()
            ]
            if len(splits)
            else []
        ),
        "intervals_count": int(len(intervals)),
    }
    summary_path = processed_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
