"""
Import raw Excel files into consolidated interim datasets with basic structure validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .config import DEFAULT_CONFIG_PATH, load_config


@dataclass
class ValidationIssue:
    file: str
    level: str
    issue_type: str
    message: str
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        data = {
            "file": self.file,
            "level": self.level,
            "issue_type": self.issue_type,
            "message": self.message,
        }
        if self.details:
            data["details"] = self.details
        return data


def scan_raw_directory(raw_dir: Path, wells: Optional[Iterable[str]] = None) -> List[Path]:
    files: List[Path] = []
    patterns = ("*.xlsx", "*.xls", "*.xlsm")
    targets = {str(well) for well in wells} if wells else None
    for pattern in patterns:
        for path in raw_dir.glob(pattern):
            if targets and path.stem not in targets:
                continue
            files.append(path)
    return sorted(files)


def validate_agzu(df: pd.DataFrame, expected_columns: List[str], file_path: Path) -> Tuple[pd.DataFrame, List[ValidationIssue]]:
    issues: List[ValidationIssue] = []
    available = [col for col in expected_columns if col in df.columns]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        issues.append(
            ValidationIssue(
                file=str(file_path),
                level="warning",
                issue_type="missing_columns",
                message="АГЗУ: отсутствуют ожидаемые колонки",
                details={"columns": missing},
            )
        )
    filtered = df[available].copy() if available else pd.DataFrame(columns=expected_columns)
    return filtered, issues


def validate_su(df: pd.DataFrame, expected_columns: List[str], file_path: Path) -> Tuple[pd.DataFrame, List[ValidationIssue]]:
    issues: List[ValidationIssue] = []
    available = [col for col in expected_columns if col in df.columns]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        issues.append(
            ValidationIssue(
                file=str(file_path),
                level="warning",
                issue_type="missing_columns",
                message="СУ: отсутствуют ожидаемые колонки",
                details={"columns": missing},
            )
        )
    filtered = df[available].copy() if available else pd.DataFrame(columns=expected_columns)
    return filtered, issues


def process_excel_file(file_path: Path, config: Dict) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[ValidationIssue]]:
    issues: List[ValidationIssue] = []
    agzu_conf = config["agzu"]
    su_conf = config["su"]
    well_column = agzu_conf["well_column"]

    try:
        excel = pd.ExcelFile(file_path)
    except Exception as exc:  # pragma: no cover - logged and skipped
        issues.append(
            ValidationIssue(
                file=str(file_path),
                level="error",
                issue_type="read_error",
                message=f"Не удалось открыть Excel: {exc}",
            )
        )
        return None, None, issues

    well_number = file_path.stem
    agzu_df: Optional[pd.DataFrame] = None
    su_df: Optional[pd.DataFrame] = None

    # АГЗУ sheet
    agzu_sheet = agzu_conf["sheet_name"]
    if agzu_sheet in excel.sheet_names:
        agzu_df_full = excel.parse(agzu_sheet)
        filtered, agzu_issues = validate_agzu(agzu_df_full, agzu_conf["key_columns"], file_path)
        filtered[well_column] = well_number
        agzu_df = filtered
        issues.extend(agzu_issues)
    else:
        issues.append(
            ValidationIssue(
                file=str(file_path),
                level="warning",
                issue_type="missing_sheet",
                message=f"АГЗУ: лист '{agzu_sheet}' не найден",
            )
        )

    # СУ sheet
    su_sheet = su_conf["sheet_name"]
    if su_sheet in excel.sheet_names:
        su_df_full = excel.parse(su_sheet)
        filtered, su_issues = validate_su(su_df_full, su_conf["key_columns"], file_path)
        filtered[su_conf["well_column"]] = well_number
        su_df = filtered
        issues.extend(su_issues)
    else:
        issues.append(
            ValidationIssue(
                file=str(file_path),
                level="warning",
                issue_type="missing_sheet",
                message=f"СУ: лист '{su_sheet}' не найден",
            )
        )

    return agzu_df, su_df, issues


def collect_raw_data(raw_dir: Path, config: Dict, wells: Optional[Iterable[str]] = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], List[ValidationIssue]]:
    agzu_frames: Dict[str, pd.DataFrame] = {}
    su_frames: Dict[str, pd.DataFrame] = {}
    all_issues: List[ValidationIssue] = []

    for file_path in scan_raw_directory(raw_dir, wells=wells):
        agzu_df, su_df, issues = process_excel_file(file_path, config)
        all_issues.extend(issues)
        if agzu_df is not None:
            agzu_frames[file_path.stem] = agzu_df
        if su_df is not None:
            su_frames[file_path.stem] = su_df

    return agzu_frames, su_frames, all_issues


def save_dataframe(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(dest, index=False)
    except Exception:  # fallback without breaking the flow
        fallback = dest.with_suffix(".pkl")
        df.to_pickle(fallback)
        print(f"[warn] Не удалось сохранить {dest.name} в parquet, использован pickle: {fallback.name}")


def write_issues(issues: List[ValidationIssue], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict] = []
    if dest.exists():
        existing = json.loads(dest.read_text(encoding="utf-8"))
    incoming = [issue.to_dict() for issue in issues]

    processed_files = {issue["file"] for issue in incoming}
    remaining = [issue for issue in existing if issue.get("file") not in processed_files]
    remaining.extend(incoming)
    dest.write_text(json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8")


def save_per_well_tables(agzu_frames: Dict[str, pd.DataFrame], su_frames: Dict[str, pd.DataFrame], config: Dict) -> None:
    interim_dir = Path(config["paths"]["interim_dir"])
    agzu_dir = interim_dir / "agzu_wells"
    su_dir = interim_dir / "su_wells"
    for well, df in agzu_frames.items():
        save_dataframe(df, agzu_dir / f"{well}.parquet")
    for well, df in su_frames.items():
        save_dataframe(df, su_dir / f"{well}.parquet")


def combine_well_tables(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    interim_dir = Path(config["paths"]["interim_dir"])
    agzu_dir = interim_dir / "agzu_wells"
    su_dir = interim_dir / "su_wells"

    agzu_frames: List[pd.DataFrame] = []
    for file_path in sorted(agzu_dir.glob("*.parquet")):
        agzu_frames.append(pd.read_parquet(file_path))
    su_frames: List[pd.DataFrame] = []
    for file_path in sorted(su_dir.glob("*.parquet")):
        su_frames.append(pd.read_parquet(file_path))

    agzu_combined = pd.concat(agzu_frames, ignore_index=True) if agzu_frames else pd.DataFrame()
    su_combined = pd.concat(su_frames, ignore_index=True) if su_frames else pd.DataFrame()

    if not agzu_combined.empty:
        save_dataframe(agzu_combined, interim_dir / "agzu_raw.parquet")
    if not su_combined.empty:
        save_dataframe(su_combined, interim_dir / "su_raw.parquet")

    return agzu_combined, su_combined


def run_import(config_path: Path, wells: Optional[Iterable[str]] = None, combine: bool = True, combine_only: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[ValidationIssue]]:
    config = load_config(config_path)
    raw_dir = Path(config["paths"]["raw_dir"])

    aggregated_agzu: Optional[pd.DataFrame] = None
    aggregated_su: Optional[pd.DataFrame] = None
    issues: List[ValidationIssue] = []

    if not combine_only:
        agzu_frames, su_frames, issues = collect_raw_data(raw_dir, config, wells=wells)
        save_per_well_tables(agzu_frames, su_frames, config)
        write_issues(issues, Path(config["paths"]["interim_dir"]) / "raw_validation.json")

    if combine:
        aggregated_agzu, aggregated_su = combine_well_tables(config)

    return aggregated_agzu, aggregated_su, issues


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import raw Excel files and perform structural validation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to pipeline configuration file.",
    )
    parser.add_argument(
        "--wells",
        type=str,
        help="Comma-separated list of well identifiers to process (process all when omitted).",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Skip raw Excel parsing and rebuild combined parquet tables from per-well caches.",
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Do not rebuild combined parquet tables (useful when processing many wells sequentially).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    wells = None
    if args.wells:
        wells = [w.strip() for w in args.wells.split(",") if w.strip()]

    combine = not args.skip_combine
    agzu_df, su_df, issues = run_import(
        args.config,
        wells=wells,
        combine=combine,
        combine_only=args.combine_only,
    )

    warnings = sum(1 for issue in issues if issue.level.lower() == "warning") if issues else 0
    errors = sum(1 for issue in issues if issue.level.lower() == "error") if issues else 0
    print(f"Import completed with {errors} errors and {warnings} warnings.")

    if agzu_df is not None and not agzu_df.empty:
        print(f"АГЗУ записей: {len(agzu_df)}")
    if su_df is not None and not su_df.empty:
        print(f"СУ записей: {len(su_df)}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
