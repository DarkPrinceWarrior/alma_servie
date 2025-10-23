"""
Utilities for importing raw Excel files into the project workspace and logging
their metadata. The script can be used both as a module and as a CLI entry
point.

Example:
    python -m pipeline.ingest --source /path/to/excel_dump
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from .config import load_config


def compute_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Return the hex digest for *file_path* using the requested hash algorithm."""
    hasher = hashlib.new(algorithm)
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def iter_excel_files(source_dir: Path) -> Iterable[Path]:
    """Yield Excel files (*.xlsx, *.xls) from *source_dir* recursively."""
    patterns = ("*.xlsx", "*.xls", "*.xlsm")
    for pattern in patterns:
        yield from source_dir.rglob(pattern)


def build_manifest_rows(excel_files: Iterable[Path], dest_dir: Path, hash_algorithm: str) -> List[Dict]:
    """Copy Excel files into *dest_dir* and collect manifest rows."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for file_path in excel_files:
        dest_path = dest_dir / file_path.name
        if file_path.resolve() != dest_path.resolve():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_path)
        stat = dest_path.stat()
        rows.append(
            {
                "file_name": dest_path.name,
                "relative_path": str(dest_path.relative_to(dest_dir.parent)),
                "size_bytes": stat.st_size,
                "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "hash_algorithm": hash_algorithm,
                "hash_value": compute_hash(dest_path, hash_algorithm),
                "ingested_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
    return rows


def write_manifest(manifest_path: Path, rows: Iterable[Dict]) -> None:
    """Save manifest rows as CSV. Append to existing manifest if present."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return

    existing_rows: List[Dict] = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            existing_rows = list(reader)

    fieldnames = [
        "file_name",
        "relative_path",
        "size_bytes",
        "modified_at_utc",
        "hash_algorithm",
        "hash_value",
        "ingested_at_utc",
    ]

    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        for row in rows:
            writer.writerow(row)


def ingest(source_dir: Path, config_path: Path) -> List[Dict]:
    """Ingest Excel files from *source_dir* according to configuration."""
    config = load_config(config_path)
    destination_dir = Path(config["paths"]["raw_dir"])
    hash_algorithm = config.get("metadata", {}).get("hash_algorithm", "sha256")
    manifest_path = Path(config.get("metadata", {}).get("manifest_file", destination_dir / "manifest.csv"))

    excel_files = list(iter_excel_files(source_dir))
    manifest_rows = build_manifest_rows(excel_files, destination_dir, hash_algorithm)
    write_manifest(manifest_path, manifest_rows)
    return manifest_rows


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy raw Excel files into project storage and log metadata.")
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Directory containing raw Excel files (scanned recursively).",
    )
    parser.add_argument(
        "--config",
        default=Path("config/pipeline.yaml"),
        type=Path,
        help="Path to the pipeline configuration YAML file.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    rows = ingest(args.source, args.config)
    print(f"Ingested {len(rows)} Excel files into raw storage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
