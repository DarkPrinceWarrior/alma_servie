"""
PaAno-based anomaly detection for Негерметичность НКТ (NKT leak).
Mirrors detect_salt_paano.py but tuned for short/sharp anomalies.
"""
import sys
import os
import time
import random
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'paano'))

import torch
import torch.nn.functional as F
from model import PatchEncoder
from train import train_model
from utils.data_preprocess import PatchCreator, preprocess_to_patches
from utils.utils import create_memory_bank
from utils.evaluation import calculate_anomaly_scores, distribute_patch_scores_to_points

warnings.filterwarnings('ignore')

SEED = 2027
PATCH_SIZE = 128      # 128 x 15s = 32 min per patch
NUM_ITERS = 200
BATCH_SIZE = 256
LR = 1e-4
TOP_K = 3
MEMORY_BANK_RATIO = 0.1


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_negermet_data(source_path=None):
    if source_path is not None:
        src = Path(source_path)
    else:
        candidates = [
            Path('db/negermet_anomaly_database_15s.csv'),
            Path('db/negermet_anomaly_database_2min.csv'),
        ]
        src = next((p for p in candidates if p.exists()), None)
    if src is None or not src.exists():
        print(f"Файл не найден: {src}")
        return pd.DataFrame()
    print(f"Loading negermet data from: {src}")
    df = pd.read_csv(src, dtype={'well_id': str}, low_memory=False)
    df['well_id'] = df['well_id'].astype(str).str.strip().str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    return df


def load_negermet_intervals():
    src = Path('db/negermet_intervals.csv')
    if not src.exists():
        return pd.DataFrame()
    intervals = pd.read_csv(src, dtype={'well_id': str})
    intervals['well_id'] = intervals['well_id'].astype(str).str.strip().str.lower()
    intervals['start_date'] = pd.to_datetime(intervals['start_date'], errors='coerce')
    intervals['end_date'] = pd.to_datetime(intervals['end_date'], errors='coerce')
    intervals['data_start'] = pd.to_datetime(intervals.get('data_start'), errors='coerce')
    intervals['data_end'] = pd.to_datetime(intervals.get('data_end'), errors='coerce')
    if 'interval_idx' not in intervals.columns:
        intervals['interval_idx'] = intervals.groupby('well_id').cumcount() + 1
    return intervals.sort_values(['well_id', 'start_date', 'interval_idx']).reset_index(drop=True)


def prepare_well_data(well_df, intervals_df, well_id):
    wd = well_df.sort_values('timestamp').reset_index(drop=True)
    timestamps = wd['timestamp'].values
    numeric_cols = [c for c in wd.columns if c not in ('timestamp', 'well_id')]
    data = wd[numeric_cols].values.astype(np.float32)

    for col_idx in range(data.shape[1]):
        col = data[:, col_idx]
        mask = np.isnan(col)
        if mask.all():
            data[:, col_idx] = 0.0
            continue
        if mask.any():
            indices = np.where(~mask, np.arange(len(col)), 0)
            np.maximum.accumulate(indices, out=indices)
            data[:, col_idx] = col[indices]
            still_nan = np.isnan(data[:, col_idx])
            if still_nan.any():
                first_valid = col[~mask][0]
                data[still_nan, col_idx] = first_valid

    well_intervals = intervals_df[intervals_df['well_id'] == well_id].sort_values('start_date')
    labels = np.zeros(len(data), dtype=np.float32)
    for _, row in well_intervals.iterrows():
        idx_mask = (timestamps >= np.datetime64(row['start_date'])) & \
                   (timestamps <= np.datetime64(row['end_date']))
        labels[idx_mask] = 1.0

    normal_mask = labels == 0.0
    train_segments = []
    seg_start = None
    for i in range(len(normal_mask)):
        if normal_mask[i]:
            if seg_start is None:
                seg_start = i
        else:
            if seg_start is not None:
                if i - seg_start >= PATCH_SIZE:
                    train_segments.append(data[seg_start:i])
                seg_start = None
    if seg_start is not None and len(data) - seg_start >= PATCH_SIZE:
        train_segments.append(data[seg_start:])

    if train_segments:
        train_data = np.concatenate(train_segments, axis=0)
    else:
        train_data = data[normal_mask]
        if len(train_data) < PATCH_SIZE * 2:
            train_data = data[:max(PATCH_SIZE * 2, len(data) // 3)]

    return data, labels, timestamps, train_data


def detect_negermet_paano_well(well_df, intervals_df, well_id, verbose=True):
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data, labels, timestamps, train_data = prepare_well_data(well_df, intervals_df, well_id)

    if len(data) < PATCH_SIZE * 3:
        return None, timestamps, "Not enough data for PaAno"

    full_data = data
    train_mean = np.mean(train_data, axis=0, keepdims=True).astype(np.float32)
    train_std = np.std(train_data, axis=0, keepdims=True).astype(np.float32)
    train_std = np.where(train_std == 0.0, 1e-8, train_std)

    full_data_norm = (full_data - train_mean) / train_std
    train_data_norm = (train_data - train_mean) / train_std

    in_channels = data.shape[1]
    if verbose:
        print(f"  PaAno: {well_id} | points={len(data)}, "
              f"train={len(train_data)} (all normal), channels={in_channels}")

    patch_creator = PatchCreator(L=PATCH_SIZE, s=1, random_seed=SEED)
    train_loader, full_loader, _ = patch_creator.create_dataloaders(
        train_data_norm, full_data_norm, labels, batch_size=BATCH_SIZE)

    model = PatchEncoder(in_channels=in_channels, use_revin=True).to(device)

    t0 = time.time()
    train_patches = preprocess_to_patches(train_data_norm, patch_size=PATCH_SIZE, stride=1)
    train_model(model, train_loader, train_patches, device,
                num_iter=NUM_ITERS, pretext_step=PATCH_SIZE, lr=LR, see_loss=False)
    t_train = time.time() - t0

    memory_bank, _ = create_memory_bank(model, train_loader, device, num_cores=MEMORY_BANK_RATIO)
    all_scores = calculate_anomaly_scores(model, full_loader, memory_bank, top_k=TOP_K, device=device)
    dist_scores = distribute_patch_scores_to_points(all_scores, patch_size=PATCH_SIZE, num_points=len(data))

    detail = (f"PaAno (patch={PATCH_SIZE}, iters={NUM_ITERS}, channels={in_channels}, "
              f"train_pts={len(train_data)}, train_time={t_train:.1f}s)")
    if verbose:
        print(f"  PaAno done: {detail}")
    return dist_scores, timestamps, detail


def find_anomaly_starts_from_scores(scores, timestamps, threshold_quantile=0.90,
                                     min_run_points=3, cooldown_hours=4):
    """Tuned for short/sharp anomalies: q=0.90 to handle high anomaly-ratio wells."""
    if scores is None or len(scores) == 0:
        return []

    threshold = np.quantile(scores, threshold_quantile)
    high = scores >= threshold

    starts = []
    cooldown = pd.Timedelta(hours=cooldown_hours)
    i = 0
    while i < len(high):
        if high[i]:
            run_start = i
            run_len = 0
            while i < len(high) and high[i]:
                run_len += 1
                i += 1
            if run_len >= min_run_points:
                ts = pd.Timestamp(timestamps[run_start])
                if not starts or ts - starts[-1] >= cooldown:
                    starts.append(ts)
        else:
            i += 1
    return starts


def run_negermet_paano_detection(output_path='negermet_paano_results.csv',
                                  source_path=None, verbose=True):
    print("=== PaAno Negermet (НКТ leak) Detection ===")
    df = load_negermet_data(source_path=source_path)
    if df.empty:
        print("No negermet data found.")
        return

    intervals = load_negermet_intervals()
    if intervals.empty:
        print("No negermet intervals found.")
        return

    all_wells = sorted(df['well_id'].unique())
    prestart_tolerance = pd.Timedelta(hours=2)
    early_status_tolerance = pd.Timedelta(hours=2)

    results = []
    scores_dict = {}

    for wid in all_wells:
        well_data = df[df['well_id'] == wid]
        well_intervals = intervals[intervals['well_id'] == wid]

        scores, timestamps, detail = detect_negermet_paano_well(
            well_data, intervals, wid, verbose=verbose)
        scores_dict[wid] = (scores, timestamps)

        if scores is None:
            for _, row in well_intervals.iterrows():
                results.append({
                    'well_id': wid,
                    'interval_idx': int(row.get('interval_idx', 1)),
                    'type': 'Negermet_NKT',
                    'detected_time': None,
                    'actual_type': 'Negermet_NKT',
                    'actual_start': row['start_date'],
                    'actual_end': row['end_date'],
                    'status': 'Not found',
                    'detail': detail,
                })
            continue

        pred_starts = find_anomaly_starts_from_scores(scores, timestamps)
        used = [False] * len(pred_starts)

        if verbose:
            print(f"  Predictions for {wid}: {len(pred_starts)} anomaly starts")

        for _, row in well_intervals.sort_values(['start_date', 'interval_idx']).iterrows():
            interval_idx = int(row.get('interval_idx', 1))
            start_dt = row['start_date']
            end_dt = row['end_date']

            detected_time = None
            candidates = []
            for i, ts in enumerate(pred_starts):
                if used[i]:
                    continue
                if (start_dt - prestart_tolerance) <= ts <= end_dt:
                    dist = abs((ts - start_dt).total_seconds())
                    candidates.append((dist, i, ts))

            if candidates:
                _, idx, detected_time = min(candidates, key=lambda x: x[0])
                used[idx] = True

            if detected_time is None:
                status = 'Not found'
            elif detected_time < start_dt - early_status_tolerance:
                status = 'Early detected'
            else:
                status = 'Detected'

            results.append({
                'well_id': wid,
                'interval_idx': interval_idx,
                'type': 'Negermet_NKT',
                'detected_time': detected_time,
                'actual_type': 'Negermet_NKT',
                'actual_start': start_dt,
                'actual_end': end_dt,
                'status': status,
                'detail': detail,
            })
            if verbose:
                print(f"  Interval {interval_idx}: {status} | Detected: {detected_time} | "
                      f"Actual: {start_dt} - {end_dt}")

    res_df = pd.DataFrame(results)
    print("\n=== PaAno Negermet Detection Results ===")
    print(res_df[['well_id', 'interval_idx', 'detected_time', 'actual_start',
                   'actual_end', 'status']].to_string())
    res_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    scores_output = Path('db/negermet_paano_scores.csv')
    score_rows = []
    for wid, (scores, timestamps) in scores_dict.items():
        if scores is not None:
            for ts, sc in zip(timestamps, scores):
                score_rows.append({'well_id': wid, 'timestamp': ts, 'paano_score': sc})
    if score_rows:
        pd.DataFrame(score_rows).to_csv(scores_output, index=False)
        print(f"Per-point scores saved to {scores_output}")

    return res_df, scores_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PaAno Negermet NKT Detection')
    parser.add_argument('--output', type=str, default='negermet_paano_results.csv')
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--well', type=str, default=None)
    args = parser.parse_args()

    if args.well:
        df = load_negermet_data(source_path=args.source)
        intervals = load_negermet_intervals()
        wid = args.well.strip().lower()
        well_data = df[df['well_id'] == wid]
        if well_data.empty:
            print(f"No data for well {wid}")
            sys.exit(1)
        scores, timestamps, detail = detect_negermet_paano_well(well_data, intervals, wid)
        starts = find_anomaly_starts_from_scores(scores, timestamps)
        print(f"\nAnomaly starts: {starts}")
        print(f"Detail: {detail}")
    else:
        run_negermet_paano_detection(output_path=args.output, source_path=args.source)
