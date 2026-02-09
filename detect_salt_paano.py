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

SALT_PRESSURE_COL = 'Давление на приеме насоса кгс/см²'
SALT_FREQ_COL = 'Выходная частота'

SEED = 2027
PATCH_SIZE = 64
NUM_ITERS = 200
BATCH_SIZE = 256
LR = 1e-4
TOP_K = 3
MEMORY_BANK_RATIO = 0.1


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_salt_data(source_path=None):
    if source_path is not None:
        src = Path(source_path)
        if not src.exists():
            print(f"Source file not found: {src}")
            return pd.DataFrame()
    else:
        candidates = [
            Path('db/salt_anomaly_database_interpolated.csv'),
            Path('db/salt_anomaly_database.csv'),
        ]
        src = next((p for p in candidates if p.exists()), None)
    if src is None:
        return pd.DataFrame()
    print(f"Loading salt data from: {src}")
    df = pd.read_csv(src, dtype={'well_id': str}, low_memory=False)
    df['well_id'] = df['well_id'].astype(str).str.strip().str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    return df


def load_salt_intervals():
    intervals = pd.read_csv('db/salt_intervals.csv', dtype={'well_id': str})
    intervals['well_id'] = intervals['well_id'].astype(str).str.strip().str.lower()
    intervals['start_date'] = pd.to_datetime(intervals['start_date'], errors='coerce')
    intervals['end_date'] = pd.to_datetime(intervals['end_date'], errors='coerce')
    intervals = intervals.dropna(subset=['well_id', 'start_date', 'end_date']).copy()
    if 'interval_idx' not in intervals.columns:
        intervals['interval_idx'] = intervals.groupby('well_id').cumcount() + 1
    return intervals.sort_values(['well_id', 'start_date', 'interval_idx']).reset_index(drop=True)


def prepare_well_data(well_df, intervals_df, well_id):
    """Convert well data into full arrays, labels, and normal-only train data for PaAno.

    Train = ALL normal segments (before, between, and after anomaly intervals).
    Labels: 0 = normal, 1 = inside any anomaly interval.
    """
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
        s = row['start_date']
        e = row['end_date']
        idx_mask = (timestamps >= np.datetime64(s)) & (timestamps <= np.datetime64(e))
        labels[idx_mask] = 1.0

    normal_mask = labels == 0.0

    # Collect contiguous normal segments of length >= PATCH_SIZE
    train_segments = []
    seg_start = None
    for i in range(len(normal_mask)):
        if normal_mask[i]:
            if seg_start is None:
                seg_start = i
        else:
            if seg_start is not None:
                seg_len = i - seg_start
                if seg_len >= PATCH_SIZE:
                    train_segments.append(data[seg_start:i])
                seg_start = None
    if seg_start is not None:
        seg_len = len(data) - seg_start
        if seg_len >= PATCH_SIZE:
            train_segments.append(data[seg_start:])

    if train_segments:
        train_data = np.concatenate(train_segments, axis=0)
    else:
        train_data = data[normal_mask]
        if len(train_data) < PATCH_SIZE * 2:
            train_data = data[:len(data) // 3]

    return data, labels, timestamps, train_data


def detect_salt_paano_well(well_df, intervals_df, well_id, verbose=True):
    """Run PaAno on a single well. Returns (anomaly_scores, timestamps, detail_str)."""
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
        print(f"  PaAno: {well_id} | points={len(data)}, train={len(train_data)} (all normal segments), channels={in_channels}")

    patch_creator = PatchCreator(L=PATCH_SIZE, s=1, random_seed=SEED)

    train_loader, full_loader, _ = patch_creator.create_dataloaders(
        train_data_norm, full_data_norm, labels, batch_size=BATCH_SIZE
    )

    model = PatchEncoder(in_channels=in_channels, use_revin=True).to(device)

    t0 = time.time()
    train_patches = preprocess_to_patches(train_data_norm, patch_size=PATCH_SIZE, stride=1)
    train_model(
        model, train_loader, train_patches, device,
        num_iter=NUM_ITERS, pretext_step=PATCH_SIZE,
        lr=LR, see_loss=False
    )
    t_train = time.time() - t0

    memory_bank, _ = create_memory_bank(model, train_loader, device, num_cores=MEMORY_BANK_RATIO)

    all_scores = calculate_anomaly_scores(model, full_loader, memory_bank, top_k=TOP_K, device=device)

    dist_scores = distribute_patch_scores_to_points(all_scores, patch_size=PATCH_SIZE, num_points=len(data))

    detail = (
        f"PaAno (patch={PATCH_SIZE}, iters={NUM_ITERS}, channels={in_channels}, "
        f"train_pts={len(train_data)}, train_time={t_train:.1f}s)"
    )
    if verbose:
        print(f"  PaAno done: {detail}")

    return dist_scores, timestamps, detail


def find_anomaly_starts_from_scores(scores, timestamps, threshold_quantile=0.95,
                                     min_run_points=8, cooldown_hours=24):
    """Find anomaly start times from PaAno anomaly score array.

    Uses an adaptive threshold based on a quantile of the score distribution.
    Returns a list of start timestamps.
    """
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


def run_salt_paano_detection(output_path='salt_paano_results.csv', source_path=None, verbose=True):
    print("=== PaAno Salt Anomaly Detection ===")
    salt_df = load_salt_data(source_path=source_path)
    if salt_df.empty:
        print("No salt data found.")
        return

    intervals = load_salt_intervals()
    if intervals.empty:
        print("No salt intervals found.")
        return

    all_wells = sorted(salt_df['well_id'].unique())
    prestart_tolerance = pd.Timedelta(hours=6)
    early_status_tolerance = pd.Timedelta(hours=6)

    results = []
    scores_dict = {}

    for wid in all_wells:
        well_data = salt_df[salt_df['well_id'] == wid]
        well_intervals = intervals[intervals['well_id'] == wid]

        scores, timestamps, detail = detect_salt_paano_well(well_data, intervals, wid, verbose=verbose)
        scores_dict[wid] = (scores, timestamps)

        if scores is None:
            for _, row in well_intervals.iterrows():
                results.append({
                    'well_id': wid,
                    'interval_idx': int(row.get('interval_idx', 1)),
                    'type': 'Salt',
                    'detected_time': None,
                    'actual_type': 'Salt',
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

                late_switch_window = pd.Timedelta(hours=54)
                if detected_time < start_dt - early_status_tolerance:
                    non_early = [
                        (i, ts) for i, ts in enumerate(pred_starts)
                        if (not used[i]) and (start_dt <= ts <= min(end_dt, start_dt + late_switch_window))
                    ]
                    if non_early:
                        repl_idx, repl_ts = min(non_early, key=lambda x: abs((x[1] - start_dt).total_seconds()))
                        used[idx] = False
                        used[repl_idx] = True
                        detected_time = repl_ts

            if detected_time is None:
                status = 'Not found'
            elif detected_time < start_dt - early_status_tolerance:
                status = 'Early detected'
            else:
                status = 'Detected'

            results.append({
                'well_id': wid,
                'interval_idx': interval_idx,
                'type': 'Salt',
                'detected_time': detected_time,
                'actual_type': 'Salt',
                'actual_start': start_dt,
                'actual_end': end_dt,
                'status': status,
                'detail': detail,
            })
            if verbose:
                print(f"  Interval {interval_idx}: {status} | Detected: {detected_time} | "
                      f"Actual: {start_dt} - {end_dt}")

    res_df = pd.DataFrame(results)
    print("\n=== PaAno Salt Detection Results ===")
    print(res_df[['well_id', 'interval_idx', 'detected_time', 'actual_start', 'actual_end', 'status']].to_string())
    res_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Save per-well scores for HTML report overlay
    scores_output = Path('db/salt_paano_scores.csv')
    score_rows = []
    for wid, (scores, timestamps) in scores_dict.items():
        if scores is not None:
            for ts, sc in zip(timestamps, scores):
                score_rows.append({'well_id': wid, 'timestamp': ts, 'paano_score': sc})
    if score_rows:
        pd.DataFrame(score_rows).to_csv(scores_output, index=False)
        print(f"Per-point scores saved to {scores_output}")

    return res_df, scores_dict


def detect_salt_paano_starts(well_data, intervals_df=None, well_id=None):
    """API compatible with detect_salt_starts() for integration into detect_anomalies.py.

    Returns: (list_of_start_timestamps, detail_string)
    """
    if well_data.empty:
        return [], "PaAno: Data not found"

    if well_id is None:
        well_id = str(well_data['well_id'].iloc[0]).strip().lower()

    if intervals_df is None:
        if Path('db/salt_intervals.csv').exists():
            intervals_df = load_salt_intervals()
        else:
            intervals_df = pd.DataFrame()

    scores, timestamps, detail = detect_salt_paano_well(well_data, intervals_df, well_id, verbose=False)
    if scores is None:
        return [], detail

    starts = find_anomaly_starts_from_scores(scores, timestamps)
    detail_full = f"{detail}; starts={len(starts)}"
    return starts, detail_full


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PaAno Salt Anomaly Detection')
    parser.add_argument('--output', type=str, default='salt_paano_results.csv')
    parser.add_argument('--source', type=str, default=None,
                        help='Path to salt CSV (default: db/salt_anomaly_database_interpolated.csv)')
    parser.add_argument('--well', type=str, default=None, help='Single well ID to process')
    args = parser.parse_args()

    if args.well:
        salt_df = load_salt_data(source_path=args.source)
        intervals = load_salt_intervals()
        wid = args.well.strip().lower()
        well_data = salt_df[salt_df['well_id'] == wid]
        if well_data.empty:
            print(f"No data for well {wid}")
            sys.exit(1)
        scores, timestamps, detail = detect_salt_paano_well(well_data, intervals, wid)
        starts = find_anomaly_starts_from_scores(scores, timestamps)
        print(f"\nAnomaly starts: {starts}")
        print(f"Detail: {detail}")
    else:
        run_salt_paano_detection(output_path=args.output, source_path=args.source)
