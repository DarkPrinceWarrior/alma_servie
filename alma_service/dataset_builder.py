from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from alma_service.anomaly_specs import DatasetSpec
from alma_service.dataset_config import CLIP_TO_SUMMARY_BOUNDS_WELLS, normalize_param_name, split_for_well
from alma_service.paths import DB_DIR, RAW_CACHE_DIR, SUMMARY_INFO_PATH, ensure_dir
from alma_service.tabular_io import read_excel_sheet, read_excel_workbook, read_table, write_dataset_tables, write_table


def _looks_like_datetime(value: object) -> bool:
    if value is None:
        return False
    try:
        parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    except Exception:
        return False
    return pd.notna(parsed)


def _raw_cache_path(filepath: Path) -> Path:
    return RAW_CACHE_DIR / filepath.parent.name / f"{filepath.stem}.parquet"


def _parse_workbook_to_cache_frame(filepath: Path) -> pd.DataFrame:
    """Parse multi-sheet xlsx into long-form cache DataFrame.

    Vectorized: each sheet's timestamp + value columns are converted in one
    pandas call instead of row-by-row pd.to_datetime() in a Python loop.
    Speedup ~50-100× on large sheets (10K+ rows).
    """
    workbook = read_excel_workbook(filepath, has_header=False, infer_schema_length=20)
    frames: list[pd.DataFrame] = []
    for sname, sheet_df in workbook.items():
        if sheet_df is None or len(sheet_df) < 2:
            continue
        # Row 0 (Python idx 0): usually empty / title.
        # Row 1 (Python idx 1): header row with full parameter name in col 0.
        # Row 2 (Python idx 2): may be either column label or first data row.
        try:
            header_value = sheet_df.iloc[1, 0]
        except (IndexError, KeyError):
            continue
        full_name = str(header_value) if pd.notna(header_value) else ""
        parts = full_name.rsplit(".", 1)
        param_name = parts[-1].strip() if len(parts) > 1 else full_name.strip()
        param_name = normalize_param_name(param_name)

        # Decide where data starts: idx=2 if it looks like a datetime, else idx=3.
        third_value = sheet_df.iloc[2, 0] if len(sheet_df) >= 3 else None
        data_start = 2 if _looks_like_datetime(third_value) else 3
        if data_start >= len(sheet_df) or sheet_df.shape[1] < 2:
            continue

        body = sheet_df.iloc[data_start:, :2].copy()
        body.columns = ["_ts", "_val"]
        body["_ts"] = pd.to_datetime(body["_ts"], dayfirst=True, errors="coerce")
        body["_val"] = pd.to_numeric(body["_val"], errors="coerce")
        body = body.dropna(subset=["_ts"])
        if body.empty:
            continue

        frame = pd.DataFrame(
            {
                "sheet_name": sname,
                "param_name": param_name,
                "timestamp": body["_ts"].to_numpy(),
                "value": body["_val"].to_numpy(dtype=np.float32),
            }
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["sheet_name", "param_name", "timestamp", "value"])
    cache_df = pd.concat(frames, ignore_index=True)
    cache_df = cache_df.sort_values(["param_name", "timestamp"]).reset_index(drop=True)
    return cache_df


def parse_parameter_series(well_id: str, filepath: Path) -> list[pd.Series]:
    print(f"  Парсинг {well_id} из {filepath}...")
    series_list: list[pd.Series] = []
    cache_path = _raw_cache_path(filepath)
    cache_is_fresh = cache_path.exists() and cache_path.stat().st_mtime >= filepath.stat().st_mtime
    if cache_is_fresh:
        cache_df = read_table(
            cache_path,
            dtypes={"sheet_name": str, "param_name": str},
            parse_dates=["timestamp"],
        )
    else:
        cache_df = _parse_workbook_to_cache_frame(filepath)
        ensure_dir(cache_path.parent)
        write_table(cache_df, cache_path)

    if cache_df.empty:
        print(f"    {well_id}: 0 параметров из xlsx")
        return series_list

    for param_name, param_df in cache_df.groupby("param_name", sort=True):
        series = pd.Series(
            param_df["value"].to_numpy(dtype=np.float32),
            index=pd.DatetimeIndex(param_df["timestamp"]),
            name=str(param_name),
        )
        series = series[~series.index.duplicated(keep="first")].sort_index()
        if not series.empty:
            series_list.append(series)
    print(f"    {well_id}: {len(series_list)} параметров из xlsx")
    return series_list


def _parse_summary_date(value: object) -> pd.Timestamp:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value)
    text = str(value).strip()
    if not text:
        return pd.NaT
    # ISO-like (YYYY-...): use default parser to avoid dayfirst quirks
    if len(text) >= 4 and text[:4].isdigit() and (len(text) == 4 or text[4] in "-/. "):
        return pd.to_datetime(text, errors="coerce")
    # DD-MM-YYYY / DD.MM.YYYY / DD/MM/YYYY
    return pd.to_datetime(text, dayfirst=True, errors="coerce")


def build_intervals(spec: DatasetSpec) -> pd.DataFrame:
    print("Парсинг сводной информации...")
    svod = read_excel_sheet(
        SUMMARY_INFO_PATH,
        sheet_id=1,
        has_header=False,
        infer_schema_length=50,
        raise_if_empty=False,
    )

    header_row = None
    for i in range(min(10, len(svod))):
        vals = [str(v).strip() if pd.notna(v) else "" for v in svod.iloc[i]]
        if "Скважина" in vals and ("Аномалия" in vals or "Тип аномалии" in vals):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Не найдена строка заголовков в сводной информации")

    headers = [str(v).strip() if pd.notna(v) else "" for v in svod.iloc[header_row]]
    data = svod.iloc[header_row + 1:].reset_index(drop=True)
    data.columns = headers + [f"_col{j}" for j in range(len(headers), data.shape[1])]
    type_col = "Аномалия" if "Аномалия" in headers else "Тип аномалии"
    known_wells = {str(k).strip().lower() for k in spec.well_files.keys()}

    intervals: list[dict[str, object]] = []
    for _, row in data.iterrows():
        anom_type = str(row[type_col]).strip() if pd.notna(row[type_col]) else ""
        if spec.summary_match not in anom_type.lower():
            continue
        well_id = str(row["Скважина"]).strip().lower()
        if well_id not in known_wells:
            continue
        data_start = _parse_summary_date(row["Дата начала выгрузки"])
        data_end = _parse_summary_date(row["Дата конца выгрузки"])
        s1 = _parse_summary_date(row["Дата начала аномалии"])
        e1 = _parse_summary_date(row["Дата конца аномалии"])
        if pd.notna(s1) and pd.notna(e1):
            intervals.append(
                {
                    "well_id": well_id,
                    "start_date": s1,
                    "end_date": e1,
                    "data_start": data_start,
                    "data_end": data_end,
                    "split": split_for_well(spec.anomaly_key, well_id),
                }
            )

    idf = pd.DataFrame(intervals)
    idf = idf.sort_values(["well_id", "start_date"]).reset_index(drop=True)
    idf["interval_idx"] = idf.groupby("well_id").cumcount() + 1
    return idf


def _load_well_bounds(spec: DatasetSpec) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    svod = read_excel_sheet(
        SUMMARY_INFO_PATH,
        sheet_id=1,
        has_header=False,
        infer_schema_length=50,
        raise_if_empty=False,
    )
    header_row = None
    for i in range(min(10, len(svod))):
        vals = [str(v).strip() if pd.notna(v) else "" for v in svod.iloc[i]]
        if "Скважина" in vals and ("Аномалия" in vals or "Тип аномалии" in vals):
            header_row = i
            break
    if header_row is None:
        return {}
    headers = [str(v).strip() if pd.notna(v) else "" for v in svod.iloc[header_row]]
    data = svod.iloc[header_row + 1:].reset_index(drop=True)
    data.columns = headers + [f"_col{j}" for j in range(len(headers), data.shape[1])]
    type_col = "Аномалия" if "Аномалия" in headers else "Тип аномалии"
    known_wells = {str(k).strip().lower() for k in spec.well_files.keys()}

    bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for _, row in data.iterrows():
        anom_type = str(row[type_col]).strip() if pd.notna(row[type_col]) else ""
        if spec.summary_match not in anom_type.lower():
            continue
        well_id = str(row["Скважина"]).strip().lower()
        if well_id not in known_wells:
            continue
        ds = _parse_summary_date(row["Дата начала выгрузки"])
        de = _parse_summary_date(row["Дата конца выгрузки"])
        if pd.notna(ds) and pd.notna(de):
            bounds[well_id] = (ds, de)
    return bounds


def _build_well_frame(
    slist: list[pd.Series],
    freq: str,
    clip_start: pd.Timestamp | None = None,
    clip_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]] | None:
    if clip_start is not None or clip_end is not None:
        clipped = []
        for series in slist:
            s = series
            if clip_start is not None:
                s = s[s.index >= clip_start]
            if clip_end is not None:
                s = s[s.index <= clip_end]
            if len(s) > 0:
                clipped.append(s)
        if not clipped:
            return None
        slist = clipped

    starts = [series.index.min() for series in slist]
    ends = [series.index.max() for series in slist]
    union_start = min(starts)
    union_end = max(ends)

    grid_start = union_start.ceil(freq)
    grid_end = union_end.floor(freq)
    grid = pd.date_range(grid_start, grid_end, freq=freq)
    if len(grid) < 2:
        return None

    df_well = pd.DataFrame({"timestamp": grid})
    for series in slist:
        trimmed = series[(series.index >= union_start) & (series.index <= union_end)]
        if len(trimmed) < 2:
            df_well[series.name] = np.nan
            continue
        combined_idx = trimmed.index.union(grid)
        reindexed = trimmed.reindex(combined_idx).sort_index()
        causal_filled = reindexed.ffill()
        df_well[series.name] = causal_filled.reindex(grid).values

    return df_well, {
        "start": union_start,
        "end": union_end,
        "grid_points": len(grid),
    }


def build_dataset(spec: DatasetSpec, freq: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq = freq or spec.default_freq
    ensure_dir(DB_DIR)

    print(f"=== Построение датасета {spec.display_name} (шаг={freq}) ===\n")

    well_series: dict[str, list[pd.Series]] = {}
    for well_id, path in spec.well_files.items():
        slist = parse_parameter_series(well_id, path)
        if slist:
            well_series[well_id] = slist

    if not well_series:
        raise RuntimeError("Нет данных для построения датасета.")

    frames: list[pd.DataFrame] = []
    all_param_names: set[str] = set()
    well_bounds = _load_well_bounds(spec)
    print(f"\nПостроение датасета (полный диапазон по скважине, шаг {freq}; обрезка по сводке для {sorted(CLIP_TO_SUMMARY_BOUNDS_WELLS)})...")
    for well_id in sorted(well_series):
        if well_id in CLIP_TO_SUMMARY_BOUNDS_WELLS:
            cs, ce = well_bounds.get(well_id, (None, None))
        else:
            cs, ce = None, None
        built = _build_well_frame(well_series[well_id], freq, clip_start=cs, clip_end=ce)
        if built is None:
            print(f"  {well_id}: ПРОПУСК — слишком короткий диапазон")
            continue
        df_well, meta = built
        for column in df_well.columns:
            if column not in {"timestamp", "well_id"}:
                all_param_names.add(column)
        df_well["well_id"] = well_id
        frames.append(df_well)
        print(
            f"  {well_id}: диапазон {meta['start']} — {meta['end']} "
            f"({meta['end'] - meta['start']}), сетка {meta['grid_points']} точек"
        )

    if not frames:
        raise RuntimeError("Нет данных после интерполяции.")

    result = pd.concat(frames, ignore_index=True)
    numeric_cols = sorted(all_param_names)
    result = result[["timestamp", "well_id"] + [c for c in numeric_cols if c in result.columns]]

    freq_label = freq.replace(" ", "")
    parquet_out_path = DB_DIR / f"{spec.output_prefix}_anomaly_database_{freq_label}.parquet"
    write_dataset_tables(
        result,
        parquet_path=parquet_out_path,
    )
    print(f"\nDatabase: {parquet_out_path} ({len(result)} rows)")

    intervals = build_intervals(spec)
    intervals_path = DB_DIR / f"{spec.output_prefix}_intervals.parquet"
    write_table(intervals, intervals_path)
    print(f"\nIntervals: {intervals_path}")
    print(intervals.to_string())

    print(f"\nВерификация датасета ({freq}):")
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    for well_id in sorted(result["well_id"].unique()):
        sub = result[result["well_id"] == well_id].sort_values("timestamp")
        deltas = sub["timestamp"].diff().dropna()
        usable_cols = [c for c in numeric_cols if not sub[c].isna().all()]
        all_nan_cols = [c for c in numeric_cols if sub[c].isna().all()]
        print(
            f"  {well_id}: median={deltas.median()}, count={len(sub)}, "
            f"usable={len(usable_cols)}, all_nan={len(all_nan_cols)}"
        )
        if all_nan_cols:
            print(f"    {well_id}: all-NaN каналы: {all_nan_cols}")

    return result, intervals
