"""
Feature importance report for PaAno pritok (inflow change) detection.
"""
import sys
import os
import io
import base64
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAANO_ROOT = PROJECT_ROOT / 'paano'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PAANO_ROOT) not in sys.path:
    sys.path.insert(0, str(PAANO_ROOT))

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model import PatchEncoder
from train import train_model
from utils.data_preprocess import PatchCreator, preprocess_to_patches
from utils.utils import create_memory_bank
from utils.evaluation import calculate_anomaly_scores, distribute_patch_scores_to_points
from alma_service.paths import DB_DIR, REPORTS_DIR, ensure_parent
from alma_service.well_features import get_well_feature_columns

warnings.filterwarnings('ignore')

SEED = 2027
PATCH_SIZE = 64
NUM_ITERS = 200
BATCH_SIZE = 256
LR = 1e-4
TOP_K = 3
MEMORY_BANK_RATIO = 0.1


def set_seed():
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_data(src_path):
    df = pd.read_csv(src_path, dtype={'well_id': str}, low_memory=False)
    df['well_id'] = df['well_id'].astype(str).str.strip().str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    return df


def load_intervals():
    intervals = pd.read_csv(DB_DIR / 'pritok_intervals.csv', dtype={'well_id': str})
    intervals['well_id'] = intervals['well_id'].astype(str).str.strip().str.lower()
    intervals['start_date'] = pd.to_datetime(intervals['start_date'], errors='coerce')
    intervals['end_date'] = pd.to_datetime(intervals['end_date'], errors='coerce')
    intervals['data_start'] = pd.to_datetime(intervals.get('data_start'), errors='coerce')
    intervals['data_end'] = pd.to_datetime(intervals.get('data_end'), errors='coerce')
    if 'interval_idx' not in intervals.columns:
        intervals['interval_idx'] = intervals.groupby('well_id').cumcount() + 1
    return intervals.sort_values(['well_id', 'start_date']).reset_index(drop=True)


def prepare_well_arrays(well_df, intervals_df, well_id):
    wd = well_df.sort_values('timestamp').reset_index(drop=True)
    timestamps = wd['timestamp'].values
    numeric_cols = get_well_feature_columns(wd)
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
                data[still_nan, col_idx] = col[~mask][0]

    wi = intervals_df[intervals_df['well_id'] == well_id].sort_values('start_date')
    labels = np.zeros(len(data), dtype=np.float32)
    for _, row in wi.iterrows():
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

    return data, labels, timestamps, train_data, numeric_cols


def score_full_data(model, full_data_norm, train_data_norm, memory_bank, device):
    patch_creator = PatchCreator(L=PATCH_SIZE, s=1, random_seed=SEED)
    dummy_labels = np.zeros(len(full_data_norm), dtype=np.float32)
    _, full_loader, _ = patch_creator.create_dataloaders(
        train_data_norm, full_data_norm, dummy_labels, batch_size=BATCH_SIZE)
    all_scores = calculate_anomaly_scores(model, full_loader, memory_bank, top_k=TOP_K, device=device)
    return distribute_patch_scores_to_points(all_scores, patch_size=PATCH_SIZE, num_points=len(full_data_norm))


def compute_feature_importance(well_df, intervals_df, well_id, device, verbose=True):
    set_seed()
    data, labels, timestamps, train_data, col_names = prepare_well_arrays(
        well_df, intervals_df, well_id)

    if not col_names:
        if verbose:
            print(f'  {well_id}: нет пригодных каналов для анализа')
        return None

    if len(data) < PATCH_SIZE * 3:
        if verbose:
            print(f'  {well_id}: недостаточно данных')
        return None

    anomaly_mask = labels == 1.0
    if not anomaly_mask.any():
        if verbose:
            print(f'  {well_id}: нет аномальных точек')
        return None

    train_mean = np.mean(train_data, axis=0, keepdims=True).astype(np.float32)
    train_std = np.std(train_data, axis=0, keepdims=True).astype(np.float32)
    train_std = np.where(train_std == 0.0, 1e-8, train_std)

    full_norm = (data - train_mean) / train_std
    train_norm = (train_data - train_mean) / train_std

    in_channels = data.shape[1]
    if verbose:
        print(f'  {well_id}: обучение модели ({len(train_data)} train, {in_channels} каналов)...')

    patch_creator = PatchCreator(L=PATCH_SIZE, s=1, random_seed=SEED)
    train_loader, _, _ = patch_creator.create_dataloaders(
        train_norm, full_norm, labels, batch_size=BATCH_SIZE)

    model = PatchEncoder(in_channels=in_channels, use_revin=True).to(device)
    train_patches = preprocess_to_patches(train_norm, patch_size=PATCH_SIZE, stride=1)
    train_model(model, train_loader, train_patches, device,
                num_iter=NUM_ITERS, pretext_step=PATCH_SIZE, lr=LR, see_loss=False)

    memory_bank, _ = create_memory_bank(model, train_loader, device, num_cores=MEMORY_BANK_RATIO)

    baseline_scores = score_full_data(model, full_norm, train_norm, memory_bank, device)
    baseline_anom_score = float(np.mean(baseline_scores[anomaly_mask]))

    if verbose:
        print(f'    Baseline anomaly score: {baseline_anom_score:.6f}')
        print(f'    Вычисление важности {in_channels} каналов...')

    importance = {}
    for ch_idx in range(in_channels):
        perturbed = full_norm.copy()
        perturbed[:, ch_idx] = 0.0

        perturbed_scores = score_full_data(model, perturbed, train_norm, memory_bank, device)
        perturbed_anom_score = float(np.mean(perturbed_scores[anomaly_mask]))

        drop = baseline_anom_score - perturbed_anom_score
        importance[col_names[ch_idx]] = {
            'drop': drop,
            'baseline': baseline_anom_score,
            'perturbed': perturbed_anom_score,
            'pct_drop': (drop / baseline_anom_score * 100) if baseline_anom_score > 0 else 0.0,
        }
        if verbose:
            print(f'      [{ch_idx+1:2d}/{in_channels}] {col_names[ch_idx]}: '
                  f'drop={drop:+.6f} ({importance[col_names[ch_idx]]["pct_drop"]:+.1f}%)')

    return {
        'well_id': well_id,
        'col_names': col_names,
        'importance': importance,
        'baseline_scores': baseline_scores,
        'timestamps': timestamps,
        'labels': labels,
        'data': data,
    }


def make_importance_bar_b64(result, top_n=15):
    imp = result['importance']
    df = pd.DataFrame([
        {'channel': k, 'drop_pct': v['pct_drop']}
        for k, v in imp.items()
    ]).sort_values('drop_pct', ascending=False)

    df_top = df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df_top))))
    colors = ['#e67e22' if v > 0 else '#bdc3c7' for v in df_top['drop_pct']]
    ax.barh(range(len(df_top)), df_top['drop_pct'].values, color=colors)
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top['channel'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Снижение PaAno Score при отключении канала, %')
    ax.set_title(f"Скважина {result['well_id']} \u2014 Важность каналов для детекции изменения притока")
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def make_top_channels_timeseries_b64(result, intervals_df, top_n=6):
    well_id = result['well_id']
    timestamps = result['timestamps']
    imp = result['importance']

    sorted_channels = sorted(imp.keys(), key=lambda k: imp[k]['pct_drop'], reverse=True)
    top_channels = sorted_channels[:top_n]

    wi = intervals_df[intervals_df['well_id'] == well_id].sort_values('start_date')

    fig, axes = plt.subplots(top_n + 1, 1, figsize=(14, 3.0 * (top_n + 1)), sharex=True)

    ts_pd = pd.to_datetime(timestamps)

    ax0 = axes[0]
    ax0.fill_between(ts_pd, 0, result['baseline_scores'], color='#e67e22', alpha=0.25)
    ax0.plot(ts_pd, result['baseline_scores'], color='#e67e22', linewidth=0.6)
    for _, row in wi.iterrows():
        ax0.axvspan(row['start_date'], row['end_date'], color='red', alpha=0.12)
        ax0.axvline(row['start_date'], color='green', linewidth=0.8)
    ax0.set_ylabel('PaAno Score\n(все каналы)', fontsize=8)
    ax0.set_title(f"Скважина {well_id} \u2014 Вклад топ-{top_n} каналов (изменение притока)",
                  fontsize=11, fontweight='bold')
    ax0.grid(True, alpha=0.3)

    data = result['data']
    col_names = result['col_names']

    for idx, ch_name in enumerate(top_channels):
        ax = axes[idx + 1]
        ch_idx = col_names.index(ch_name)
        raw_values = data[:, ch_idx]
        drop_pct = imp[ch_name]['pct_drop']

        ax.plot(ts_pd, raw_values, color='tab:blue', linewidth=0.5, alpha=0.8)
        ax.set_ylabel(f'{ch_name}\n(drop {drop_pct:+.1f}%)', fontsize=7)
        for _, row in wi.iterrows():
            ax.axvspan(row['start_date'], row['end_date'], color='red', alpha=0.12)
            ax.axvline(row['start_date'], color='green', linewidth=0.8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Время')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_report(source_path, output_path, report_title):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Устройство: {device}')

    df = load_data(source_path)
    intervals = load_intervals()

    all_results = []
    for wid in sorted(df['well_id'].unique()):
        print(f'\nСкважина {wid}:')
        well_data = df[df['well_id'] == wid]
        result = compute_feature_importance(well_data, intervals, wid, device)
        if result is not None:
            all_results.append(result)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>{report_title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; color: #222; background: #fafafa; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #e67e22; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 35px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; background: #fff; }}
        th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 0.9em; }}
        th {{ background-color: #e67e22; color: #fff; }}
        tr:nth-child(even) {{ background-color: #fef5ed; }}
        .positive {{ color: #e67e22; font-weight: bold; }}
        .negative {{ color: #95a5a6; }}
        .plot-container {{ margin-bottom: 40px; background: #fff; border: 1px solid #ddd;
                          border-radius: 6px; padding: 15px; }}
        img {{ max-width: 100%; height: auto; }}
        .method-box {{ background: #fef5ed; border-left: 4px solid #e67e22; padding: 15px 20px;
                       margin: 20px 0; border-radius: 4px; line-height: 1.7; }}
        .method-box b {{ color: #a04000; }}
    </style>
</head>
<body>
    <h1>{report_title}</h1>

    <div class="method-box">
        <h3>Методика оценки влияния переменных</h3>
        <p>Для каждого канала (переменной) измеряем, <b>насколько снижается PaAno Score в зоне аномалии</b>
        при &laquo;отключении&raquo; этого канала (замена его значений на среднее по нормальным данным).</p>
        <ul>
            <li><b>Большое положительное снижение (%)</b> &mdash; канал <b>важен</b> для детекции:
                без него модель хуже видит аномалию.</li>
            <li><b>Около нуля или отрицательное</b> &mdash; канал малозначим или вносит шум.</li>
        </ul>
    </div>
"""

    for result in all_results:
        wid = result['well_id']
        imp = result['importance']

        html += f'<h2>Скважина {wid}</h2>\n'

        bar_b64 = make_importance_bar_b64(result, top_n=15)
        html += f'<div class="plot-container"><img src="data:image/png;base64,{bar_b64}"></div>\n'

        ts_b64 = make_top_channels_timeseries_b64(result, intervals, top_n=6)
        html += f'<div class="plot-container"><img src="data:image/png;base64,{ts_b64}"></div>\n'

        sorted_imp = sorted(imp.items(), key=lambda x: x[1]['pct_drop'], reverse=True)
        html += """
    <table>
        <tr><th>#</th><th>Канал</th><th>Baseline Score</th><th>Score без канала</th>
            <th>Снижение</th><th>Снижение, %</th></tr>
"""
        for rank, (ch, v) in enumerate(sorted_imp, 1):
            cls = 'positive' if v['pct_drop'] > 0.5 else 'negative'
            html += f"""
        <tr><td>{rank}</td><td>{ch}</td><td>{v['baseline']:.6f}</td><td>{v['perturbed']:.6f}</td>
            <td class="{cls}">{v['drop']:+.6f}</td><td class="{cls}">{v['pct_drop']:+.1f}%</td></tr>
"""
        html += '    </table>\n'

    html += '</body></html>'
    ensure_parent(Path(output_path)).write_text(html, encoding='utf-8')
    print(f'\nОтчёт сохранён: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=str(DB_DIR / 'pritok_anomaly_database_2min.csv'))
    parser.add_argument('--output', default=str(REPORTS_DIR / 'pritok_paano_feature_importance.html'))
    parser.add_argument('--title', default='Влияние переменных на детекцию изменения притока (PaAno)')
    args = parser.parse_args()
    generate_report(args.source, args.output, args.title)
