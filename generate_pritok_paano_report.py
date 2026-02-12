"""
PaAno report for Изменение притока (inflow change).
"""
import base64
import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['font.size'] = 10

PRESSURE_COL = 'Давление на приеме насоса кгс/см²'
FREQ_COL = 'Выходная частота'


def load_data(src_path):
    df = pd.read_csv(src_path, dtype={'well_id': str}, low_memory=False)
    df['well_id'] = df['well_id'].astype(str).str.strip().str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values(['well_id', 'timestamp'])
    return df


def load_intervals():
    src = Path('db/pritok_intervals.csv')
    intervals = pd.read_csv(src, dtype={'well_id': str})
    intervals['well_id'] = intervals['well_id'].astype(str).str.strip().str.lower()
    intervals['start_date'] = pd.to_datetime(intervals['start_date'], errors='coerce')
    intervals['end_date'] = pd.to_datetime(intervals['end_date'], errors='coerce')
    intervals['data_start'] = pd.to_datetime(intervals.get('data_start'), errors='coerce')
    intervals['data_end'] = pd.to_datetime(intervals.get('data_end'), errors='coerce')
    if 'interval_idx' not in intervals.columns:
        intervals['interval_idx'] = intervals.groupby('well_id').cumcount() + 1
    intervals['interval_idx'] = pd.to_numeric(intervals['interval_idx'], errors='coerce').fillna(1).astype(int)
    return intervals.sort_values(['well_id', 'interval_idx']).reset_index(drop=True)


def load_paano_detected(path):
    src = Path(path)
    if not src.exists():
        return {}
    res = pd.read_csv(src, dtype={'well_id': str})
    res['well_id'] = res['well_id'].astype(str).str.strip().str.lower()
    if 'interval_idx' in res.columns:
        res['interval_idx'] = pd.to_numeric(res['interval_idx'], errors='coerce').fillna(1).astype(int)
    else:
        res['interval_idx'] = 1
    res['detected_time'] = pd.to_datetime(res['detected_time'], errors='coerce')
    return {(row['well_id'], row['interval_idx']): row['detected_time'] for _, row in res.iterrows()}


def load_paano_scores(path):
    src = Path(path)
    if not src.exists():
        return {}
    df = pd.read_csv(src, dtype={'well_id': str})
    df['well_id'] = df['well_id'].astype(str).str.strip().str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    return {wid: grp.sort_values('timestamp') for wid, grp in df.groupby('well_id')}


def create_plot_base64(well_data, start_dt, end_dt, paano_detected_dt, title,
                       x_min, x_max, paano_scores_df=None):
    cols_present = [c for c in [PRESSURE_COL, FREQ_COL] if c in well_data.columns]
    if not cols_present:
        return None

    df = well_data[['timestamp'] + cols_present].copy()
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    if df.empty:
        return None

    has_paano = paano_scores_df is not None and not paano_scores_df.empty
    n_rows = len(cols_present) + (1 if has_paano else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    label_map = {
        PRESSURE_COL: ('Давление на приёме насоса, кгс/см\u00b2', 'tab:blue'),
        FREQ_COL: ('Выходная частота, Гц', 'tab:orange'),
    }

    for idx, col in enumerate(cols_present):
        ax = axes[idx]
        label, color = label_map.get(col, (col, 'tab:blue'))
        line_df = df[['timestamp', col]].dropna(subset=[col])
        if line_df.empty:
            ax.text(0.5, 0.5, f'Нет данных: {label}', transform=ax.transAxes, ha='center', va='center')
        else:
            ax.plot(line_df['timestamp'], line_df[col], color=color, linewidth=0.6, label=label)
        ax.axvspan(start_dt, end_dt, color='tab:red', alpha=0.15, label='Интервал аномалии')
        ax.axvline(start_dt, color='tab:green', linestyle='-', linewidth=1.0, label='Факт. начало')
        if pd.notna(x_min) and pd.notna(x_max) and x_min < x_max:
            ax.set_xlim(x_min, x_max)
        if pd.notna(paano_detected_dt):
            ax.axvline(paano_detected_dt, color='purple', linestyle='--', linewidth=1.3, label='PaAno: обнаружено')
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        handles, labels_leg = ax.get_legend_handles_labels()
        uniq = dict()
        for h, l in zip(handles, labels_leg):
            if l not in uniq:
                uniq[l] = h
        ax.legend(uniq.values(), uniq.keys(), loc='upper right', fontsize=8)

    if has_paano:
        ax_p = axes[-1]
        score_df = paano_scores_df
        if pd.notna(x_min) and pd.notna(x_max):
            score_df = score_df[(score_df['timestamp'] >= x_min) & (score_df['timestamp'] <= x_max)]
        ax_p.fill_between(score_df['timestamp'], 0, score_df['paano_score'], color='purple', alpha=0.25)
        ax_p.plot(score_df['timestamp'], score_df['paano_score'], color='purple', linewidth=0.6, label='PaAno Score')
        ax_p.axvspan(start_dt, end_dt, color='tab:red', alpha=0.15, label='Интервал аномалии')
        ax_p.axvline(start_dt, color='tab:green', linestyle='-', linewidth=1.0, label='Факт. начало')
        if pd.notna(paano_detected_dt):
            ax_p.axvline(paano_detected_dt, color='purple', linestyle='--', linewidth=1.3, label='PaAno: обнаружено')
        ax_p.set_ylabel('PaAno Score (отклонение от нормы)')
        if pd.notna(x_min) and pd.notna(x_max) and x_min < x_max:
            ax_p.set_xlim(x_min, x_max)
        ax_p.grid(True, alpha=0.3)
        handles_p, labels_p = ax_p.get_legend_handles_labels()
        uniq_p = dict()
        for h, l in zip(handles_p, labels_p):
            if l not in uniq_p:
                uniq_p[l] = h
        ax_p.legend(uniq_p.values(), uniq_p.keys(), loc='upper right', fontsize=8)

    axes[0].set_title(title, fontsize=12, fontweight='bold')
    axes[-1].set_xlabel('Время')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_html(source_path, output_path, report_title, paano_results_path, paano_scores_path):
    print(f'Генерация отчёта из: {source_path}')
    data_df = load_data(source_path)
    intervals = load_intervals()
    paano_map = load_paano_detected(paano_results_path)
    scores_map = load_paano_scores(paano_scores_path)

    detected_count = sum(1 for v in paano_map.values() if pd.notna(v))
    total_intervals = len(intervals)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>{report_title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; color: #222; background: #fafafa; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #e67e22; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 35px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 25px; background: #fff; }}
        th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
        th {{ background-color: #e67e22; color: #fff; }}
        tr:nth-child(even) {{ background-color: #fef5ed; }}
        .status-detected {{ color: #27ae60; font-weight: bold; }}
        .status-notfound {{ color: #c0392b; font-weight: bold; }}
        .plot-container {{ margin-bottom: 40px; background: #fff; border: 1px solid #ddd;
                          border-radius: 6px; padding: 15px; }}
        img {{ max-width: 100%; height: auto; }}
        .method-box {{ background: #fef5ed; border-left: 4px solid #e67e22; padding: 15px 20px;
                       margin: 20px 0; border-radius: 4px; line-height: 1.7; }}
        .method-box b {{ color: #a04000; }}
        .summary {{ font-size: 1.1em; margin: 15px 0; }}
    </style>
</head>
<body>
    <h1>{report_title}</h1>

    <div class="method-box">
        <h3>Что такое PaAno Score?</h3>
        <p><b>PaAno</b> (Patch-based Anomaly detection) &mdash; нейросетевой метод обнаружения аномалий
        во временных рядах (ICLR 2026).</p>
        <ol>
            <li><b>Обучение на норме.</b> Все нормальные участки данных скважины нарезаются на короткие
                фрагменты (патчи по 64 точки = 128 мин). 1D-CNN кодирует каждый патч в вектор-эмбеддинг.
                Из тренировочных эмбеддингов формируется банк нормальных паттернов.</li>
            <li><b>Скоринг.</b> Для каждого фрагмента вычисляется расстояние до ближайших эталонов
                в банке. Чем дальше &mdash; тем выше <b>PaAno Score</b>.</li>
            <li><b>Детекция.</b> Устойчивое превышение порога score фиксируется как начало аномалии.</li>
        </ol>
        <p><b>Особенности для изменения притока:</b> используется патч 64 точки (128 мин)
        на 2-минутной сетке для ловли медленных трендовых аномалий.</p>
    </div>

    <p class="summary">Результат: обнаружено <b>{detected_count} из {total_intervals}</b> интервалов аномалий.</p>

    <h2>Сводная таблица</h2>
    <table>
        <tr>
            <th>Скважина</th>
            <th>Интервал</th>
            <th>Начало данных</th>
            <th>Конец данных</th>
            <th>Факт. начало аномалии</th>
            <th>Факт. конец аномалии</th>
            <th>PaAno: обнаружено</th>
            <th>Статус</th>
        </tr>
"""

    for _, row in intervals.iterrows():
        key = (row['well_id'], int(row['interval_idx']))
        det = paano_map.get(key)
        if pd.notna(det):
            status, cls = 'Обнаружено', 'status-detected'
        else:
            status, cls = 'Не найдено', 'status-notfound'

        det_str = det.strftime('%Y-%m-%d %H:%M') if pd.notna(det) else '&mdash;'
        ds = row['data_start'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['data_start']) else '&mdash;'
        de = row['data_end'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['data_end']) else '&mdash;'

        html += f"""
        <tr>
            <td>{row['well_id']}</td>
            <td>{int(row['interval_idx'])}</td>
            <td>{ds}</td>
            <td>{de}</td>
            <td>{row['start_date'].strftime('%Y-%m-%d %H:%M')}</td>
            <td>{row['end_date'].strftime('%Y-%m-%d %H:%M')}</td>
            <td>{det_str}</td>
            <td class="{cls}">{status}</td>
        </tr>
"""

    html += """
    </table>
    <h2>Графики по скважинам</h2>
"""

    total = len(intervals)
    for i, (_, row) in enumerate(intervals.iterrows()):
        wid = row['well_id']
        idx = int(row['interval_idx'])
        start_dt = row['start_date']
        end_dt = row['end_date']

        print(f'  График {i + 1}/{total}: скв. {wid}, интервал {idx}')
        well_data = data_df[data_df['well_id'] == wid]
        if well_data.empty:
            html += f'<div class="plot-container"><h3>Скважина {wid}</h3><p>Нет данных.</p></div>\n'
            continue

        data_start = row['data_start'] if pd.notna(row['data_start']) else well_data['timestamp'].min()
        data_end = row['data_end'] if pd.notna(row['data_end']) else well_data['timestamp'].max()
        plot_df = well_data[(well_data['timestamp'] >= data_start) & (well_data['timestamp'] <= data_end)]

        title = f"Скважина {wid} \u2014 изменение притока, интервал {idx}"
        x_min = data_start
        x_max = data_end
        det = paano_map.get((wid, idx))
        scores = scores_map.get(wid)

        b64 = create_plot_base64(plot_df, start_dt, end_dt, det, title, x_min, x_max, paano_scores_df=scores)
        if b64:
            html += f'<div class="plot-container"><h3>{title}</h3><img src="data:image/png;base64,{b64}"></div>\n'
        else:
            html += f'<div class="plot-container"><h3>{title}</h3><p>Нет данных для графика.</p></div>\n'

    html += '</body></html>'
    Path(output_path).write_text(html, encoding='utf-8')
    print(f'Отчёт сгенерирован: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='db/pritok_anomaly_database_2min.csv')
    parser.add_argument('--output', default='pritok_paano_report.html')
    parser.add_argument('--title', default='Детекция изменения притока методом PaAno (AI)')
    parser.add_argument('--paano-results', default='pritok_paano_results.csv')
    parser.add_argument('--paano-scores', default='db/pritok_paano_scores.csv')
    args = parser.parse_args()
    generate_html(args.source, args.output, args.title, args.paano_results, args.paano_scores)
