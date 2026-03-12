"""
Build negermet (НКТ leak) dataset from xlsx files.
Produces:
  - db/negermet_anomaly_database_<freq>.csv   (interpolated on uniform grid)
  - db/negermet_intervals.csv                 (ground truth intervals)

Usage:
  python scripts/datasets/build_negermet_dataset.py                  # default 15s
  python scripts/datasets/build_negermet_dataset.py --freq 2min      # 2-minute grid
  python scripts/datasets/build_negermet_dataset.py --freq 15s       # 15-second grid
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.dataset_config import (
    NEGREMET_WELL_FILES,
    load_model_parameters,
    normalize_param_name,
    split_for_well,
)
from alma_service.paths import DB_DIR, SUMMARY_INFO_PATH, ensure_dir

warnings.filterwarnings('ignore')

WELL_FILES = NEGREMET_WELL_FILES
ALLOWED_PARAMS = load_model_parameters()


def parse_xlsx(well_id, filepath):
    """
    Parse xlsx into per-parameter Series list.
    Returns list of (param_name, pd.Series with DatetimeIndex).
    """
    print(f'  Парсинг {well_id} из {filepath}...')
    wb = openpyxl.load_workbook(filepath, read_only=True)
    series_list = []
    skipped_params = set()

    for sname in wb.sheetnames:
        ws = wb[sname]
        row_iter = ws.iter_rows(values_only=True)
        _ = next(row_iter, None)
        header_row = next(row_iter, None)
        if header_row is None:
            continue

        full_name = str(header_row[0]) if header_row[0] else ''
        parts = full_name.rsplit('.', 1)
        param_name = parts[-1].strip() if len(parts) > 1 else full_name.strip()
        param_name = normalize_param_name(param_name)
        if param_name not in ALLOWED_PARAMS:
            skipped_params.add(param_name)
            continue

        times = []
        values = []
        for r in row_iter:
            if r[0] is None:
                continue
            try:
                ts = pd.to_datetime(r[0], dayfirst=True)
                val = float(r[1]) if r[1] is not None else np.nan
                times.append(ts)
                values.append(val)
            except (ValueError, TypeError):
                continue

        if times:
            s = pd.Series(values, index=pd.DatetimeIndex(times), name=param_name)
            s = s[~s.index.duplicated(keep='first')].sort_index()
            series_list.append(s)

    wb.close()
    print(f'    {well_id}: {len(series_list)} параметров из xlsx')
    if skipped_params:
        print(f'    {well_id}: пропущены параметры вне списка модели: {sorted(skipped_params)}')
    return series_list


def build_intervals():
    print('Парсинг сводной информации...')
    svod = pd.read_excel(SUMMARY_INFO_PATH, header=None)

    # Find header row (contains '№' and 'Скважина')
    header_row = None
    for i in range(min(10, len(svod))):
        vals = [str(v).strip() if pd.notna(v) else '' for v in svod.iloc[i]]
        if '№' in vals and 'Скважина' in vals:
            header_row = i
            break

    if header_row is None:
        raise ValueError('Не найдена строка заголовков в сводной информации')

    # Columns: 0=№, 1=Месторождение, 2=Скважина, 3=Тип аномалии,
    # 4=Дата начала выгрузки, 5=Дата конца выгрузки, 6=Комментарий,
    # 7=Дата начала аномалии 1, 8=Дата конца аномалии 1,
    # 9=Дата начала аномалии 2, 10=Дата конца аномалии 2
    data = svod.iloc[header_row + 1:].reset_index(drop=True)

    intervals = []
    for _, row in data.iterrows():
        anom_type = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ''
        if 'Негерметичность' not in anom_type:
            continue

        well_id = str(row.iloc[2]).strip().lower()
        data_start = pd.to_datetime(row.iloc[4], errors='coerce')
        data_end = pd.to_datetime(row.iloc[5], errors='coerce')

        # First anomaly interval (cols 7, 8)
        s1 = pd.to_datetime(row.iloc[7], errors='coerce')
        e1 = pd.to_datetime(row.iloc[8], errors='coerce')
        if pd.notna(s1) and pd.notna(e1):
            intervals.append({
                'well_id': well_id,
                'start_date': s1,
                'end_date': e1,
                'data_start': data_start,
                'data_end': data_end,
                'split': split_for_well('negermet', well_id),
            })

        # Second anomaly interval (cols 9, 10) if present
        if len(row) > 10:
            s2 = pd.to_datetime(row.iloc[9], errors='coerce')
            e2 = pd.to_datetime(row.iloc[10], errors='coerce')
            if pd.notna(s2) and pd.notna(e2):
                intervals.append({
                    'well_id': well_id,
                    'start_date': s2,
                    'end_date': e2,
                    'data_start': data_start,
                    'data_end': data_end,
                    'split': split_for_well('negermet', well_id),
                })

    idf = pd.DataFrame(intervals)
    idf = idf.sort_values(['well_id', 'start_date']).reset_index(drop=True)
    idf['interval_idx'] = idf.groupby('well_id').cumcount() + 1
    return idf


def main(freq='15s'):
    print(f'=== Построение датасета Негерметичность НКТ (шаг={freq}) ===\n')
    ensure_dir(DB_DIR)

    # 1. Parse all xlsx -> per-well list of Series
    well_series = {}
    for wid, fpath in WELL_FILES.items():
        slist = parse_xlsx(wid, fpath)
        if slist:
            well_series[wid] = slist

    if not well_series:
        print('Нет данных!')
        return

    # 2. For each well: find common time range and interpolate
    #    Common start = latest start among all params
    #    Common end   = earliest end among all params
    print(f'\nПостроение датасета (общий диапазон по параметрам, шаг {freq})...')
    frames = []
    all_param_names = set()

    for wid in sorted(well_series.keys()):
        slist = well_series[wid]

        param_starts = []
        param_ends = []
        for s in slist:
            param_starts.append(s.index.min())
            param_ends.append(s.index.max())

        common_start = max(param_starts)
        common_end = min(param_ends)

        if common_start >= common_end:
            print(f'  {wid}: ПРОПУСК — нет общего диапазона')
            continue

        grid_start = common_start.ceil(freq)
        grid_end = common_end.floor(freq)
        grid = pd.date_range(grid_start, grid_end, freq=freq)

        if len(grid) < 2:
            print(f'  {wid}: ПРОПУСК — слишком короткий общий диапазон')
            continue

        print(f'  {wid}: общий диапазон {common_start} — {common_end} '
              f'({common_end - common_start}), сетка {len(grid)} точек')

        df_well = pd.DataFrame({'timestamp': grid})
        for s in slist:
            param_name = s.name
            all_param_names.add(param_name)
            trimmed = s[(s.index >= common_start) & (s.index <= common_end)]
            if len(trimmed) < 2:
                df_well[param_name] = np.nan
                continue
            combined_idx = trimmed.index.union(grid)
            reindexed = trimmed.reindex(combined_idx).sort_index()
            interpolated = reindexed.interpolate(method='time')
            interpolated = interpolated.ffill().bfill()
            df_well[param_name] = interpolated.reindex(grid).values

        df_well['well_id'] = wid
        frames.append(df_well)

    if not frames:
        print('Нет данных после интерполяции!')
        return

    result = pd.concat(frames, ignore_index=True)
    numeric_cols = sorted(all_param_names)
    result = result[['timestamp', 'well_id'] + [c for c in numeric_cols if c in result.columns]]

    freq_label = freq.replace(' ', '')
    out_path = DB_DIR / f'negermet_anomaly_database_{freq_label}.csv'
    result.to_csv(out_path, index=False)
    print(f'\nDatabase: {out_path} ({len(result)} rows)')

    # 3. Build intervals
    idf = build_intervals()
    intervals_path = DB_DIR / 'negermet_intervals.csv'
    idf.to_csv(intervals_path, index=False)
    print(f'\nIntervals: {intervals_path}')
    print(idf.to_string())

    # 4. Verify
    print(f'\nВерификация датасета ({freq}):')
    result['timestamp'] = pd.to_datetime(result['timestamp'])
    for wid in sorted(result['well_id'].unique()):
        sub = result[result['well_id'] == wid].sort_values('timestamp')
        deltas = sub['timestamp'].diff().dropna()
        nan_total = sub[numeric_cols].isna().sum().sum() if all(c in sub.columns for c in numeric_cols) else 0
        print(f'  {wid}: median={deltas.median()}, count={len(sub)}, NaN={nan_total}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', default='15s', help='Resample frequency (e.g. 15s, 2min)')
    args = parser.parse_args()
    main(freq=args.freq)
