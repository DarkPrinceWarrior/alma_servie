from __future__ import annotations

import importlib
import inspect
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.anomaly_specs import DATASET_SPECS  # noqa: E402
from alma_service.paths import DB_DIR  # noqa: E402
from alma_service.tabular_io import read_table  # noqa: E402


METADATA_COLUMNS = {"timestamp", "well_id"}
PRIMARY_DATASETS = ("negermet", "pritok", "salt")
COMMON_SCHEMA_DATASETS = ("negermet", "pritok", "salt", "norm_work")
MIN_GOOD_WELL_COVERAGE = 0.95


@dataclass(frozen=True)
class SourceAudit:
    dataset: str
    role: str
    path: str
    exists: bool
    rows: int | None = None
    wells: int | None = None
    start: str | None = None
    end: str | None = None
    channels: int | None = None
    median_step_seconds: float | None = None


@dataclass(frozen=True)
class ChannelAudit:
    dataset: str
    source_role: str
    channel: str
    status: str
    wells_total: int
    wells_with_any: int
    wells_good_coverage: int
    overall_coverage: float
    min_well_coverage: float
    median_well_coverage: float
    max_well_coverage: float
    first_valid_timestamp: str | None
    reason: str


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _freq_label_from_path(path: Path) -> str:
    stem = path.stem
    marker = "_database_"
    if marker in stem:
        return stem.split(marker, 1)[1]
    return "unknown"


def _wrapper_default_freq(anomaly_key: str) -> str | None:
    module_name = f"scripts.datasets.build_{anomaly_key}_dataset"
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    main = getattr(module, "main", None)
    if main is None:
        return None
    signature = inspect.signature(main)
    freq_param = signature.parameters.get("freq")
    if freq_param is None or freq_param.default is inspect.Parameter.empty:
        return None
    return str(freq_param.default)


def _norm_work_defaults() -> tuple[str, ...]:
    try:
        module = importlib.import_module("scripts.datasets.build_norm_work_dataset")
    except Exception:
        return ()
    return tuple(getattr(module, "DEFAULT_FREQS", ()))


def _full_build_default_freqs() -> dict[str, str | None]:
    script_path = Path("scripts/run_full_dataset_build.sh")
    if not script_path.exists():
        return {"negermet": None, "pritok": None, "salt": None}
    text = script_path.read_text(encoding="utf-8")
    mapping = {
        "negermet": "NEGERMET_FREQ",
        "pritok": "PRITOK_FREQ",
        "salt": "SALT_FREQ",
    }
    defaults: dict[str, str | None] = {}
    for dataset, env_name in mapping.items():
        match = re.search(rf'{env_name}="\$\{{{env_name}:-([^}}]+)\}}"', text)
        defaults[dataset] = match.group(1) if match else None
    return defaults


def _read_dataset(path: Path) -> pd.DataFrame:
    df = read_table(path, dtypes={"well_id": str}, parse_dates=["timestamp"])
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values(["well_id", "timestamp"]).reset_index(drop=True)


def _median_step_seconds(df: pd.DataFrame) -> float | None:
    steps: list[float] = []
    for _, part in df.groupby("well_id", sort=False):
        deltas = part["timestamp"].diff().dropna().dt.total_seconds()
        if not deltas.empty:
            steps.append(float(deltas.median()))
    if not steps:
        return None
    return float(np.median(steps))


def _source_audit(dataset: str, role: str, path: Path) -> tuple[SourceAudit, pd.DataFrame | None]:
    if not path.exists():
        return SourceAudit(dataset=dataset, role=role, path=str(path), exists=False), None
    df = _read_dataset(path)
    channels = [column for column in df.columns if column not in METADATA_COLUMNS]
    audit = SourceAudit(
        dataset=dataset,
        role=role,
        path=str(path),
        exists=True,
        rows=int(len(df)),
        wells=int(df["well_id"].nunique()),
        start=str(df["timestamp"].min()),
        end=str(df["timestamp"].max()),
        channels=int(len(channels)),
        median_step_seconds=_median_step_seconds(df),
    )
    return audit, df


def _channel_audits(dataset: str, source_role: str, df: pd.DataFrame) -> list[ChannelAudit]:
    wells = sorted(df["well_id"].dropna().unique())
    wells_total = len(wells)
    audits: list[ChannelAudit] = []
    for channel in sorted(column for column in df.columns if column not in METADATA_COLUMNS):
        coverage_by_well: list[float] = []
        first_valids: list[pd.Timestamp] = []
        for well_id, part in df.groupby("well_id", sort=True):
            values = pd.to_numeric(part[channel], errors="coerce")
            coverage = float(values.notna().mean()) if len(values) else 0.0
            coverage_by_well.append(coverage)
            non_null = part.loc[values.notna(), "timestamp"]
            if not non_null.empty:
                first_valids.append(non_null.min())

        coverage_array = np.asarray(coverage_by_well, dtype=float)
        wells_with_any = int((coverage_array > 0).sum())
        wells_good = int((coverage_array >= MIN_GOOD_WELL_COVERAGE).sum())
        overall_coverage = float(pd.to_numeric(df[channel], errors="coerce").notna().mean())
        if wells_with_any == 0:
            status = "drop"
            reason = "канал полностью пустой"
        elif wells_with_any < wells_total:
            status = "partial"
            reason = "канал отсутствует хотя бы в одной скважине"
        elif wells_good < wells_total:
            status = "partial"
            reason = f"coverage ниже {MIN_GOOD_WELL_COVERAGE:.0%} хотя бы в одной скважине"
        else:
            status = "keep_candidate"
            reason = "канал есть во всех скважинах с высоким покрытием"

        audits.append(
            ChannelAudit(
                dataset=dataset,
                source_role=source_role,
                channel=channel,
                status=status,
                wells_total=wells_total,
                wells_with_any=wells_with_any,
                wells_good_coverage=wells_good,
                overall_coverage=overall_coverage,
                min_well_coverage=float(coverage_array.min()) if len(coverage_array) else 0.0,
                median_well_coverage=float(np.median(coverage_array)) if len(coverage_array) else 0.0,
                max_well_coverage=float(coverage_array.max()) if len(coverage_array) else 0.0,
                first_valid_timestamp=str(min(first_valids)) if first_valids else None,
                reason=reason,
            )
        )
    return audits


def _primary_source_for(anomaly_key: str) -> Path | None:
    spec = DATASET_SPECS[anomaly_key]
    for name in spec.source_candidates:
        path = DB_DIR / name
        if path.exists():
            return path
    return None


def _existing_norm_work_sources() -> list[Path]:
    paths: list[Path] = []
    for freq in _norm_work_defaults():
        path = DB_DIR / f"norm_work_database_{freq.replace(' ', '')}.parquet"
        if path.exists():
            paths.append(path)
    return paths


def _channel_summary(channel_audits: list[ChannelAudit]) -> dict[str, Any]:
    primary = [
        audit
        for audit in channel_audits
        if audit.dataset in COMMON_SCHEMA_DATASETS and audit.source_role == "primary"
    ]
    by_dataset: dict[str, dict[str, ChannelAudit]] = {}
    for audit in primary:
        by_dataset.setdefault(audit.dataset, {})[audit.channel] = audit

    channel_union = sorted({audit.channel for audit in primary})
    canonical_candidates: list[str] = []
    rejected: dict[str, list[str]] = {}
    for channel in channel_union:
        reasons: list[str] = []
        for dataset in COMMON_SCHEMA_DATASETS:
            audit = by_dataset.get(dataset, {}).get(channel)
            if audit is None:
                reasons.append(f"{dataset}: нет колонки")
                continue
            if audit.status != "keep_candidate":
                reasons.append(f"{dataset}: {audit.reason}")
        if reasons:
            rejected[channel] = reasons
        else:
            canonical_candidates.append(channel)

    return {
        "canonical_candidates": canonical_candidates,
        "canonical_candidate_count": len(canonical_candidates),
        "rejected_count": len(rejected),
        "rejected": rejected,
    }


def _defaults_summary() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    full_build_defaults = _full_build_default_freqs()
    for anomaly_key in PRIMARY_DATASETS:
        spec = DATASET_SPECS[anomaly_key]
        wrapper_default = _wrapper_default_freq(anomaly_key)
        full_build_default = full_build_defaults.get(anomaly_key)
        first_existing = _primary_source_for(anomaly_key)
        status = "ok"
        if wrapper_default != spec.default_freq or full_build_default != spec.default_freq:
            status = "mismatch"
        rows.append(
            {
                "dataset": anomaly_key,
                "spec_default_freq": spec.default_freq,
                "wrapper_default_freq": wrapper_default,
                "full_build_default_freq": full_build_default,
                "source_candidates": spec.source_candidates,
                "first_existing_source": str(first_existing) if first_existing else None,
                "status": status,
            }
        )
    rows.append(
        {
            "dataset": "norm_work",
            "spec_default_freq": None,
            "wrapper_default_freq": _norm_work_defaults(),
            "full_build_default_freq": None,
            "source_candidates": tuple(str(path.name) for path in _existing_norm_work_sources()),
            "first_existing_source": str(_existing_norm_work_sources()[0]) if _existing_norm_work_sources() else None,
            "status": "informational",
        }
    )
    return rows


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    defaults = report["defaults"]
    sources = report["sources"]
    summary = report["channel_summary"]
    channels = report["channels"]

    lines: list[str] = []
    lines.append("# Аудит defaults и доступности каналов")
    lines.append("")
    lines.append("Файл сгенерирован скриптом `scripts/datasets/audit_preprocessing_defaults_and_channels.py`.")
    lines.append("")
    lines.append("## Defaults")
    lines.append("")
    lines.append("| Датасет | default в spec | default wrapper | default full build | первый существующий source | статус |")
    lines.append("|---|---:|---:|---:|---|---|")
    for row in defaults:
        lines.append(
            "| {dataset} | {spec} | {wrapper} | {full_build} | {source} | {status} |".format(
                dataset=row["dataset"],
                spec=row["spec_default_freq"],
                wrapper=row["wrapper_default_freq"],
                full_build=row["full_build_default_freq"],
                source=row["first_existing_source"],
                status=row["status"],
            )
        )

    lines.append("")
    lines.append("## Источники")
    lines.append("")
    lines.append("| Датасет | Роль | Строк | Скважин | Каналов | Медианный шаг, сек | Период |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for item in sources:
        if not item["exists"]:
            continue
        lines.append(
            "| {dataset} | {role} | {rows} | {wells} | {channels} | {step} | {start} - {end} |".format(
                dataset=item["dataset"],
                role=item["role"],
                rows=item["rows"],
                wells=item["wells"],
                channels=item["channels"],
                step=round(item["median_step_seconds"], 3) if item["median_step_seconds"] is not None else "",
                start=item["start"],
                end=item["end"],
            )
        )

    lines.append("")
    lines.append("## Кандидаты в общий набор каналов")
    lines.append("")
    lines.append(f"Кандидатов, присутствующих во всех primary-источниках: {summary['canonical_candidate_count']}.")
    if summary["canonical_candidates"]:
        lines.append("")
        for channel in summary["canonical_candidates"]:
            lines.append(f"- {channel}")

    lines.append("")
    lines.append("## Отброшенные/спорные каналы")
    lines.append("")
    lines.append(f"Каналов с проблемами: {summary['rejected_count']}.")
    for channel, reasons in summary["rejected"].items():
        joined = "; ".join(reasons)
        lines.append(f"- {channel}: {joined}")

    lines.append("")
    lines.append("## Детализация по каналам")
    lines.append("")
    lines.append("| Датасет | Канал | Статус | Скважин с каналом | Good coverage | Median coverage | Причина |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for item in channels:
        if item["source_role"] != "primary":
            continue
        lines.append(
            "| {dataset} | {channel} | {status} | {any}/{total} | {good}/{total} | {median:.3f} | {reason} |".format(
                dataset=item["dataset"],
                channel=item["channel"],
                status=item["status"],
                any=item["wells_with_any"],
                total=item["wells_total"],
                good=item["wells_good_coverage"],
                median=item["median_well_coverage"],
                reason=item["reason"],
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report() -> dict[str, Any]:
    sources: list[SourceAudit] = []
    channel_audits: list[ChannelAudit] = []

    for anomaly_key in PRIMARY_DATASETS:
        primary = _primary_source_for(anomaly_key)
        if primary is not None:
            audit, df = _source_audit(anomaly_key, "primary", primary)
            sources.append(audit)
            if df is not None:
                channel_audits.extend(_channel_audits(anomaly_key, "primary", df))
        spec = DATASET_SPECS[anomaly_key]
        for name in spec.source_candidates:
            candidate = DB_DIR / name
            if primary is not None and candidate == primary:
                continue
            audit, _ = _source_audit(anomaly_key, "candidate", candidate)
            sources.append(audit)

    norm_sources = _existing_norm_work_sources()
    if norm_sources:
        primary_norm = next((path for path in norm_sources if path.name.endswith("_10min.parquet")), norm_sources[0])
        for path in norm_sources:
            role = "primary" if path == primary_norm else "candidate"
            audit, df = _source_audit("norm_work", role, path)
            sources.append(audit)
            if role == "primary" and df is not None:
                channel_audits.extend(_channel_audits("norm_work", role, df))

    report = {
        "defaults": _defaults_summary(),
        "sources": [asdict(item) for item in sources],
        "channels": [asdict(item) for item in channel_audits],
        "channel_summary": _channel_summary(channel_audits),
    }
    return report


def main() -> None:
    output_json = Path("artifacts/analysis/preprocessing_defaults_and_channels.json")
    output_md = Path("artifacts/analysis/preprocessing_defaults_and_channels.md")
    report = build_report()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_markdown(report, output_md)
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")
    print(
        "Canonical candidates:",
        report["channel_summary"]["canonical_candidate_count"],
        "Rejected:",
        report["channel_summary"]["rejected_count"],
    )
    mismatches = [row for row in report["defaults"] if row["status"] == "mismatch"]
    if mismatches:
        print("Default mismatches:")
        for row in mismatches:
            print(
                f"  {row['dataset']}: spec={row['spec_default_freq']} "
                f"wrapper={row['wrapper_default_freq']} "
                f"full_build={row['full_build_default_freq']} "
                f"first_source={row['first_existing_source']}"
            )


if __name__ == "__main__":
    main()
