from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from alma_service.paths import PROJECT_ROOT, SALYM_PREPARED_DIR, SALYM_SOURCE_DIR, ensure_dir
from alma_service.tabular_io import write_table

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None


SELECTED_PARAM_MAP: dict[str, str] = {
    "ESP.Motor.VibrationX": "Вибрация Х",
    "ESP.Motor.VibrationY": "Вибрация Y",
    "ESP.Motor.VibrationZ": "Вибрация Z",
    "ESP.Frequency": "Выходная частота",
    "ESP.IntakePressure": "Давление на приеме насоса кгс/см²",
    "ESP.Motor.CurrentUnbalance": "Дисбаланс токов",
    "ESP.Motor.Load": "Коэффициент загрузки ПЭД",
    "ESP.Motor.VoltageAB": "Линейное напряжение по фазе АB",
    "ESP.Motor.VoltageBC": "Линейное напряжение по фазе ВC",
    "ESP.Motor.VoltageCA": "Линейное напряжение по фазе СA",
    "ESP.IntakeTemperature": "Температура на приёме насоса",
    "ESP.Motor.CurrentU": "Ток на фазе А",
    "ESP.Motor.CurrentV": "Ток на фазе В",
    "ESP.Motor.CurrentW": "Ток на фазе С",
    "ESP.Motor.Temperature": "Температура масла двигателя",
}

SELECTED_PARAMS = tuple(SELECTED_PARAM_MAP.keys())
YEARS = ("2015", "2016", "2017", "2018")

TAG_RE = re.compile(r"^[^.]+\.([^.]+)\.([^.]+)\.(.+)$")
PAD_RE = re.compile(r"(?i)^wellpad(\d+)([A-Za-z]*)$")
WELL_RE = re.compile(r"(?i)^well(\d+)$")


@dataclass(frozen=True)
class SalymLayout:
    root: Path
    metadata_dir: Path
    anomalies_dir: Path
    qc_dir: Path
    raw_wells_dir: Path


def _log(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)


def _iter_progress(iterable, *, desc: str, total: int | None = None, leave: bool = False):
    if tqdm is None:
        _log(desc)
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave)


def salym_layout(root: str | Path | None = None) -> SalymLayout:
    base = Path(root) if root is not None else SALYM_PREPARED_DIR
    return SalymLayout(
        root=base,
        metadata_dir=base / "metadata",
        anomalies_dir=base / "anomalies",
        qc_dir=base / "qc",
        raw_wells_dir=base / "raw_wells",
    )


def ensure_salym_layout(root: str | Path | None = None) -> SalymLayout:
    layout = salym_layout(root)
    for path in (
        layout.root,
        layout.metadata_dir,
        layout.anomalies_dir,
        layout.qc_dir,
        layout.raw_wells_dir,
    ):
        ensure_dir(path)
    return layout


def _clean_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _normalize_suffix(value: Any) -> str | None:
    text = _clean_text(value)
    return text.upper() if text else None


def _suffix_key(value: Any) -> str:
    normalized = _normalize_suffix(value)
    return normalized if normalized is not None else ""


def _extract_tag_parts(tag: str) -> tuple[str | None, str | None, str | None]:
    match = TAG_RE.match(tag)
    if not match:
        return None, None, None
    return match.group(1), match.group(2), match.group(3)


def _parse_pad(pad_tag: str | None) -> tuple[int | None, str | None]:
    if not pad_tag:
        return None, None
    match = PAD_RE.match(pad_tag.strip())
    if not match:
        return None, None
    suffix = match.group(2) or None
    return int(match.group(1)), suffix.upper() if suffix else None


def _parse_well(well_tag: str | None) -> int | None:
    if not well_tag:
        return None
    match = WELL_RE.match(well_tag.strip())
    if not match:
        return None
    return int(match.group(1))


def _excel_registry_path() -> Path:
    return SALYM_SOURCE_DIR / "сводная_информация_салим.xlsx"


def _param_file(param_key: str, year: str) -> Path:
    return SALYM_SOURCE_DIR / "parquet" / f"{param_key}_{year}.parquet"


def _fragment_dir(layout: SalymLayout, well_id: str, param_key: str) -> Path:
    return layout.raw_wells_dir / well_id / "_fragments" / param_key


def _final_param_path(layout: SalymLayout, well_id: str, param_key: str) -> Path:
    return layout.raw_wells_dir / well_id / f"{param_key}.parquet"


def _read_fragment_frame(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    expected = {
        "timestamp": pl.Datetime("ns"),
        "value": pl.Float64,
        "value_text": pl.Utf8,
        "quality_flag": pl.Utf8,
        "source_tag": pl.Utf8,
        "source_year": pl.Int64,
        "well_id": pl.Utf8,
        "param_key": pl.Utf8,
    }
    for column, dtype in expected.items():
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return frame.select(
        [
            pl.col("timestamp").cast(pl.Datetime("ns"), strict=False),
            pl.col("value").cast(pl.Float64, strict=False),
            pl.col("value_text").cast(pl.Utf8, strict=False),
            pl.col("quality_flag").cast(pl.Utf8, strict=False),
            pl.col("source_tag").cast(pl.Utf8, strict=False),
            pl.col("source_year").cast(pl.Int64, strict=False),
            pl.col("well_id").cast(pl.Utf8, strict=False),
            pl.col("param_key").cast(pl.Utf8, strict=False),
        ]
    )


def _safe_float_from_text(value_text: Any) -> float | None:
    text = _clean_text(value_text)
    if text is None:
        return None
    normalized = text.replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def _quality_and_value(value: Any, value_text: Any) -> tuple[float | None, str]:
    vt = _clean_text(value_text)
    if vt == "Bad":
        return None, "bad_text"
    if vt in {"-9999", "-32768"}:
        return None, f"sentinel_{vt}"
    if value is not None and not (isinstance(value, float) and pd.isna(value)):
        numeric = float(value)
        if numeric == -9999.0:
            return None, "sentinel_-9999"
        if numeric == -32768.0:
            return None, "sentinel_-32768"
        return numeric, "ok"
    fallback = _safe_float_from_text(vt)
    if fallback is not None:
        if fallback == -9999.0:
            return None, "sentinel_-9999"
        if fallback == -32768.0:
            return None, "sentinel_-32768"
        return fallback, "text_only"
    return None, "missing_nonphysical"


def _load_excel_registry() -> pd.DataFrame:
    df = pd.read_excel(_excel_registry_path(), header=5)
    df = df[df["Well"].notna()].copy()
    df["Well"] = df["Well"].astype(str).str.strip()
    df["Pad"] = df["Pad"].astype(str).str.strip()
    df = df[~df["Well"].isin(["Well", "Скважина"])]
    df = df[~df["Unique Job identifier"].astype(str).str.contains("Идентификационный номер ремонта", na=False)]
    df["pad_num"] = df["Pad"].str.extract(r"(\d+)").astype("Int64")
    df["pad_suffix"] = df["Pad"].str.extract(r"([A-Za-z]+)$")[0].map(_normalize_suffix)
    df["pad_suffix_key"] = df["pad_suffix"].map(_suffix_key)
    df["well_num"] = df["Well"].str.extract(r"(\d+)").astype("Int64")
    df["field_name"] = df["Field"].astype(str).str.strip()
    df["canonical_well_id"] = df["Well"]
    return df


def _build_frequency_presence() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in _iter_progress(YEARS, desc="Metadata: ESP.Frequency years", total=len(YEARS), leave=True):
        pf = pq.ParquetFile(_param_file("ESP.Frequency", year))
        wells = set()
        batches = pf.iter_batches(columns=["Tag"], batch_size=250_000)
        for batch in _iter_progress(
            batches,
            desc=f"  Scan tags {year}",
            total=pf.metadata.num_row_groups,
            leave=False,
        ):
            for tag in batch.column(0).to_pylist():
                if not tag:
                    continue
                pad_tag, well_tag, _ = _extract_tag_parts(str(tag).strip())
                pad_num, pad_suffix = _parse_pad(pad_tag)
                well_num = _parse_well(well_tag)
                if pad_num is None or well_num is None:
                    continue
                wells.add((pad_tag, well_tag, pad_num, pad_suffix, well_num))
        for pad_tag, well_tag, pad_num, pad_suffix, well_num in sorted(wells):
            rows.append(
                {
                    "source_pad_tag": pad_tag,
                    "source_well_tag": well_tag,
                    "pad_num": pad_num,
                    "pad_suffix": pad_suffix,
                    "pad_suffix_key": _suffix_key(pad_suffix),
                    "well_num": well_num,
                    "year": int(year),
                    "is_present": True,
                }
            )
    return pd.DataFrame(rows)


def build_metadata(output_root: str | Path | None = None) -> dict[str, Path]:
    layout = ensure_salym_layout(output_root)
    _log(f"Metadata: reading Excel registry from {_excel_registry_path()}")
    excel = _load_excel_registry()
    _log("Metadata: scanning stable wells from ESP.Frequency parquet files")
    presence = _build_frequency_presence()

    year_presence = (
        presence.pivot_table(
            index=["source_pad_tag", "source_well_tag", "pad_num", "pad_suffix_key", "well_num"],
            columns="year",
            values="is_present",
            aggfunc="max",
            fill_value=False,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    year_presence["pad_suffix"] = year_presence["pad_suffix_key"].map(lambda s: s if s else None)
    for year in YEARS:
        key = int(year)
        if key not in year_presence.columns:
            year_presence[key] = False
        year_presence[f"is_present_{year}"] = year_presence[key].astype(bool)
    year_presence["is_stable_4y_frequency"] = year_presence[[f"is_present_{year}" for year in YEARS]].all(axis=1)

    excel_unique = excel[
        [
            "field_name",
            "Pad",
            "Well",
            "canonical_well_id",
            "pad_num",
            "pad_suffix",
            "pad_suffix_key",
            "well_num",
        ]
    ].drop_duplicates()
    ambiguous = (
        excel_unique.groupby(["pad_num", "pad_suffix_key", "well_num"], dropna=False)
        .agg(n_names=("canonical_well_id", "nunique"))
        .reset_index()
    )
    ambiguous = ambiguous[ambiguous["n_names"] > 1].copy()

    well_map = year_presence.merge(
        excel_unique,
        on=["pad_num", "pad_suffix_key", "well_num"],
        how="left",
        suffixes=("", "_excel"),
    )
    well_map["match_status"] = well_map["canonical_well_id"].notna().map({True: "matched", False: "unmatched"})
    well_map = well_map.merge(
        ambiguous.assign(is_excluded_ambiguous=True)[["pad_num", "pad_suffix_key", "well_num", "is_excluded_ambiguous"]],
        on=["pad_num", "pad_suffix_key", "well_num"],
        how="left",
    )
    well_map["is_excluded_ambiguous"] = well_map["is_excluded_ambiguous"].eq(True)

    selected = well_map[
        well_map["is_stable_4y_frequency"]
        & (~well_map["is_excluded_ambiguous"])
        & (well_map["match_status"] == "matched")
    ].copy()
    selected["selected_reason"] = "stable_4y_frequency_non_ambiguous"

    excluded = well_map[well_map["is_excluded_ambiguous"]].copy()

    outputs = {
        "well_map": layout.metadata_dir / "well_map.parquet",
        "excluded": layout.metadata_dir / "excluded_ambiguous_wells.parquet",
        "selected": layout.metadata_dir / "selected_wells.parquet",
    }
    write_table(well_map, outputs["well_map"])
    write_table(excluded, outputs["excluded"])
    write_table(selected, outputs["selected"])
    _log(
        "Metadata ready: "
        f"well_map={len(well_map)}, selected={len(selected)}, excluded_ambiguous={len(excluded)}"
    )
    return outputs


def _selected_lookup(selected: pd.DataFrame) -> dict[tuple[int, str, int], dict[str, Any]]:
    lookup: dict[tuple[int, str, int], dict[str, Any]] = {}
    for row in selected.to_dict("records"):
        key = (int(row["pad_num"]), _suffix_key(row.get("pad_suffix")), int(row["well_num"]))
        lookup[key] = row
    return lookup


def _load_selected_wells(output_root: str | Path | None = None) -> pd.DataFrame:
    layout = salym_layout(output_root)
    path = layout.metadata_dir / "selected_wells.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Selected wells metadata not found: {path}")
    return pl.read_parquet(path).to_pandas()


def _metadata_paths(layout: SalymLayout) -> dict[str, Path]:
    return {
        "well_map": layout.metadata_dir / "well_map.parquet",
        "excluded": layout.metadata_dir / "excluded_ambiguous_wells.parquet",
        "selected": layout.metadata_dir / "selected_wells.parquet",
    }


def _anomaly_paths(layout: SalymLayout) -> dict[str, Path]:
    return {
        "equipment_cycles": layout.anomalies_dir / "equipment_cycles.parquet",
        "anomaly_periods": layout.anomalies_dir / "anomaly_periods.parquet",
    }


def extract_raw_telemetry(
    output_root: str | Path | None = None,
    *,
    params: list[str] | None = None,
    years: list[str] | None = None,
    batch_size: int = 250_000,
    limit_wells: int | None = None,
) -> dict[str, int]:
    layout = ensure_salym_layout(output_root)
    selected = _load_selected_wells(output_root)
    if limit_wells is not None:
        selected = selected.head(limit_wells).copy()
    lookup = _selected_lookup(selected)

    total_rows = 0
    total_fragments = 0
    params = params or list(SELECTED_PARAMS)
    years = years or list(YEARS)

    total_param_years = len(params) * len(years)
    current_param_year = 0
    for param_key in params:
        for year in years:
            current_param_year += 1
            src = _param_file(param_key, year)
            if not src.exists():
                continue
            _log(f"Extract: [{current_param_year}/{total_param_years}] {param_key} {year}")
            pf = pq.ParquetFile(src)
            fragment_idx = 0
            batches = pf.iter_batches(columns=["Tag", "TimeStamp", "ValueText", "Value"], batch_size=batch_size)
            for batch in _iter_progress(
                batches,
                desc=f"  Batches {param_key} {year}",
                total=pf.metadata.num_row_groups,
                leave=False,
            ):
                records_by_well: dict[str, list[dict[str, Any]]] = {}
                tags = batch.column(0).to_pylist()
                timestamps = batch.column(1).to_pylist()
                value_texts = batch.column(2).to_pylist()
                values = batch.column(3).to_pylist()
                for tag, ts, value_text, value in zip(tags, timestamps, value_texts, values):
                    if not tag or ts is None:
                        continue
                    pad_tag, well_tag, _ = _extract_tag_parts(str(tag).strip())
                    pad_num, pad_suffix = _parse_pad(pad_tag)
                    well_num = _parse_well(well_tag)
                    if pad_num is None or well_num is None:
                        continue
                    meta = lookup.get((pad_num, _suffix_key(pad_suffix), well_num))
                    if meta is None:
                        continue
                    numeric_value, quality_flag = _quality_and_value(value, value_text)
                    well_id = str(meta["canonical_well_id"])
                    records_by_well.setdefault(well_id, []).append(
                        {
                            "timestamp": pd.Timestamp(ts),
                            "value": numeric_value,
                            "value_text": _clean_text(value_text),
                            "quality_flag": quality_flag,
                            "source_tag": str(tag).strip(),
                            "source_year": int(year),
                            "well_id": well_id,
                            "param_key": param_key,
                        }
                    )
                if not records_by_well:
                    continue
                fragment_idx += 1
                for well_id, rows in records_by_well.items():
                    fragment_dir = _fragment_dir(layout, well_id, param_key)
                    ensure_dir(fragment_dir)
                    fragment_path = fragment_dir / f"{year}_{fragment_idx:06d}.parquet"
                    write_table(pd.DataFrame(rows), fragment_path)
                    total_fragments += 1
                    total_rows += len(rows)
            _log(
                f"Extract done: {param_key} {year}, "
                f"rows_written_so_far={total_rows}, fragments_written_so_far={total_fragments}"
            )

    return {"rows_written": total_rows, "fragments_written": total_fragments}


def compact_raw_telemetry(output_root: str | Path | None = None, *, cleanup_fragments: bool = False) -> dict[str, int]:
    layout = ensure_salym_layout(output_root)
    files_written = 0
    qc_rows: list[dict[str, Any]] = []

    well_dirs = sorted(path for path in layout.raw_wells_dir.iterdir() if path.is_dir())
    for well_dir in _iter_progress(well_dirs, desc="Compact: wells", total=len(well_dirs), leave=True):
        if not well_dir.is_dir():
            continue
        fragments_root = well_dir / "_fragments"
        if not fragments_root.exists():
            continue
        param_dirs = sorted(path for path in fragments_root.iterdir() if path.is_dir())
        for param_dir in _iter_progress(
            param_dirs,
            desc=f"  Compact params {well_dir.name}",
            total=len(param_dirs),
            leave=False,
        ):
            if not param_dir.is_dir():
                continue
            fragment_paths = sorted(param_dir.glob("*.parquet"))
            if not fragment_paths:
                continue
            frames = [_read_fragment_frame(path) for path in fragment_paths]
            frame = pl.concat(frames, how="vertical_relaxed", rechunk=True).sort("timestamp")
            if frame.is_empty():
                continue
            final_path = _final_param_path(layout, well_dir.name, param_dir.name)
            frame.write_parquet(final_path, compression="zstd")
            files_written += 1
            quality_counts = (
                frame.group_by("quality_flag")
                .len()
                .to_dicts()
            )
            quality_map = {row["quality_flag"]: row["len"] for row in quality_counts}
            qc_rows.append(
                {
                    "well_id": well_dir.name,
                    "param_key": param_dir.name,
                    "min_timestamp": frame["timestamp"].min(),
                    "max_timestamp": frame["timestamp"].max(),
                    "n_rows": frame.height,
                    "n_valid_rows": int(frame.filter(pl.col("quality_flag") == "ok").height),
                    "n_missing_rows": int(frame.filter(pl.col("value").is_null()).height),
                    "n_bad_text": int(quality_map.get("bad_text", 0)),
                    "n_sentinel_9999": int(quality_map.get("sentinel_-9999", 0)),
                    "n_sentinel_32768": int(quality_map.get("sentinel_-32768", 0)),
                    "years_present": sorted(set(frame["source_year"].to_list())),
                }
            )
        if cleanup_fragments and fragments_root.exists():
            for path in sorted(fragments_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    if qc_rows:
        write_table(pd.DataFrame(qc_rows), layout.qc_dir / "well_parameter_coverage.parquet")
    _log(f"Compact ready: param_files_written={files_written}, qc_rows_written={len(qc_rows)}")
    return {"param_files_written": files_written, "qc_rows_written": len(qc_rows)}


def build_equipment_cycles(output_root: str | Path | None = None) -> Path:
    layout = ensure_salym_layout(output_root)
    _log("Anomalies: building equipment_cycles.parquet")
    selected = _load_selected_wells(output_root)
    selected = selected.copy()
    selected["pad_suffix_key"] = selected["pad_suffix"].map(_suffix_key)
    selected_keys = selected[["pad_num", "pad_suffix_key", "well_num", "canonical_well_id", "field_name"]].drop_duplicates()
    excel = _load_excel_registry()
    cycles = excel.merge(selected_keys, on=["pad_num", "pad_suffix_key", "well_num"], how="inner", suffixes=("", "_selected"))
    result = pd.DataFrame(
        {
            "well_id": cycles["canonical_well_id_selected"].fillna(cycles["canonical_well_id"]),
            "field_name": cycles["field_name_selected"].fillna(cycles["field_name"]),
            "pad": cycles["Pad"],
            "well": cycles["Well"],
            "job_id": cycles.get("Unique Job identifier"),
            "installed_at": pd.to_datetime(cycles.get("Installed"), errors="coerce"),
            "started_at": pd.to_datetime(cycles.get("Started"), errors="coerce"),
            "failed_at": pd.to_datetime(cycles.get("Failed"), errors="coerce"),
            "pulled_at": pd.to_datetime(cycles.get("Pulled"), errors="coerce"),
            "runtime": cycles.get("Runtime"),
            "runlife": cycles.get("Runlife"),
            "reason": cycles.get("Reason"),
            "kind_workover": cycles.get("Kind workover"),
            "jobs_done": cycles.get("Jobs Done"),
            "stop_reason": cycles.get("The reason of a stop "),
            "failed_equipment": cycles.get("Failed Equipment"),
            "motor_id": cycles.get("Motor / ПЭД"),
        }
    )
    path = layout.anomalies_dir / "equipment_cycles.parquet"
    write_table(result, path)
    _log(f"Anomalies ready: equipment_cycles rows={len(result)}")
    return path


def build_anomaly_periods(output_root: str | Path | None = None) -> Path:
    layout = ensure_salym_layout(output_root)
    _log("Anomalies: building anomaly_periods.parquet")
    cycles = pl.read_parquet(layout.anomalies_dir / "equipment_cycles.parquet").to_pandas()
    event_ts = pd.to_datetime(cycles["failed_at"], errors="coerce")
    event_ts = event_ts.fillna(pd.to_datetime(cycles["pulled_at"], errors="coerce"))
    result = pd.DataFrame(
        {
            "well_id": cycles["well_id"],
            "interval_start": event_ts,
            "interval_end": event_ts,
            "data_start": pd.to_datetime(cycles["started_at"], errors="coerce").fillna(pd.to_datetime(cycles["installed_at"], errors="coerce")),
            "data_end": event_ts,
            "interval_kind": "failure_point",
            "interval_source": "excel_runlife_proxy",
            "is_proxy_interval": True,
            "job_id": cycles["job_id"],
            "reason": cycles["reason"],
            "kind_workover": cycles["kind_workover"],
            "stop_reason": cycles["stop_reason"],
            "failed_equipment": cycles["failed_equipment"],
        }
    )
    result = result[result["interval_start"].notna()].reset_index(drop=True)
    path = layout.anomalies_dir / "anomaly_periods.parquet"
    write_table(result, path)
    _log(f"Anomalies ready: anomaly_periods rows={len(result)}")
    return path


def build_all(
    output_root: str | Path | None = None,
    *,
    metadata_only: bool = False,
    params: list[str] | None = None,
    years: list[str] | None = None,
    batch_size: int = 250_000,
    limit_wells: int | None = None,
    skip_metadata: bool = False,
    skip_anomalies: bool = False,
    skip_extract: bool = False,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    layout = ensure_salym_layout(output_root)
    _log(f"Salym build root: {Path(output_root) if output_root is not None else SALYM_PREPARED_DIR}")

    metadata_paths = _metadata_paths(layout)
    if skip_metadata:
        missing = [str(path) for path in metadata_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"skip_metadata requested, but metadata files are missing: {missing}")
        _log("Stage 1/4: metadata (skip, reuse existing files)")
        outputs["metadata"] = metadata_paths
    else:
        _log("Stage 1/4: metadata")
        outputs["metadata"] = build_metadata(output_root)

    anomaly_paths = _anomaly_paths(layout)
    if skip_anomalies:
        missing = [str(path) for path in anomaly_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"skip_anomalies requested, but anomaly files are missing: {missing}")
        _log("Stage 2/4 + 3/4: anomalies (skip, reuse existing files)")
        outputs["equipment_cycles"] = str(anomaly_paths["equipment_cycles"])
        outputs["anomaly_periods"] = str(anomaly_paths["anomaly_periods"])
    else:
        _log("Stage 2/4: equipment cycles")
        outputs["equipment_cycles"] = str(build_equipment_cycles(output_root))
        _log("Stage 3/4: anomaly periods")
        outputs["anomaly_periods"] = str(build_anomaly_periods(output_root))

    if metadata_only:
        _log("Build complete: metadata-only mode")
        return outputs
    if skip_extract:
        _log("Stage 4/4: raw telemetry extraction (skip, reuse existing fragments)")
    else:
        _log("Stage 4/4: raw telemetry extraction")
        outputs["extract"] = extract_raw_telemetry(
            output_root,
            params=params,
            years=years,
            batch_size=batch_size,
            limit_wells=limit_wells,
        )
    _log("Stage 4b/4: compact raw telemetry")
    outputs["compact"] = compact_raw_telemetry(output_root)
    _log("Build complete")
    return outputs


__all__ = [
    "SELECTED_PARAM_MAP",
    "SELECTED_PARAMS",
    "YEARS",
    "build_all",
    "build_anomaly_periods",
    "build_equipment_cycles",
    "build_metadata",
    "compact_raw_telemetry",
    "extract_raw_telemetry",
    "salym_layout",
]
