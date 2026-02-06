import base64
import argparse
import io
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

SALT_PRESSURE_COL = 'Давление на приеме насоса кгс/см²'
SALT_FREQ_COL = 'Выходная частота'


def load_salt_data(src_path):
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f'Missing file: {src}')

    df = pd.read_csv(src, dtype={'well_id': str}, low_memory=False)
    df['well_id'] = df['well_id'].astype(str).str.strip().str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values(['well_id', 'timestamp'])
    return df


def load_salt_intervals():
    src = Path('db/salt_intervals.csv')
    if not src.exists():
        raise FileNotFoundError(f'Missing file: {src}')

    intervals = pd.read_csv(src, dtype={'well_id': str})
    intervals['well_id'] = intervals['well_id'].astype(str).str.strip().str.lower()
    intervals['start_date'] = pd.to_datetime(intervals['start_date'], errors='coerce')
    intervals['end_date'] = pd.to_datetime(intervals['end_date'], errors='coerce')
    intervals['data_start'] = pd.to_datetime(intervals.get('data_start'), errors='coerce')
    intervals['data_end'] = pd.to_datetime(intervals.get('data_end'), errors='coerce')
    if 'interval_idx' not in intervals.columns:
        intervals['interval_idx'] = intervals.groupby('well_id').cumcount() + 1
    intervals['interval_idx'] = pd.to_numeric(intervals['interval_idx'], errors='coerce').fillna(1).astype(int)
    intervals = intervals.dropna(subset=['well_id', 'start_date', 'end_date'])
    return intervals.sort_values(['well_id', 'interval_idx']).reset_index(drop=True)


def load_detected_salt_points():
    src = Path('anomaly_detection_results.csv')
    if not src.exists():
        return {}

    res = pd.read_csv(src, dtype={'well_id': str})
    if 'type' not in res.columns:
        return {}

    res = res[res['type'] == 'Salt'].copy()
    if res.empty:
        return {}

    res['well_id'] = res['well_id'].astype(str).str.strip().str.lower()
    if 'interval_idx' in res.columns:
        res['interval_idx'] = pd.to_numeric(res['interval_idx'], errors='coerce').fillna(1).astype(int)
    else:
        res['interval_idx'] = 1
    res['detected_time'] = pd.to_datetime(res['detected_time'], errors='coerce')

    return {
        (row['well_id'], row['interval_idx']): row['detected_time']
        for _, row in res.iterrows()
    }


def create_plot_base64(well_data, start_dt, end_dt, detected_dt, title, x_min, x_max):
    if SALT_PRESSURE_COL not in well_data.columns or SALT_FREQ_COL not in well_data.columns:
        return None

    df = well_data[['timestamp', SALT_PRESSURE_COL, SALT_FREQ_COL]].copy()
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    if df.empty:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    series_map = [
        (SALT_PRESSURE_COL, 'Intake Pressure', 'tab:blue'),
        (SALT_FREQ_COL, 'Output Frequency', 'tab:orange'),
    ]

    for ax, (col, label, color) in zip(axes, series_map):
        line_df = df[['timestamp', col]].dropna(subset=[col])
        if line_df.empty:
            ax.text(0.5, 0.5, f'No data for {label}', transform=ax.transAxes, ha='center', va='center')
        else:
            ax.plot(line_df['timestamp'], line_df[col], color=color, linewidth=0.7, label=label)
        ax.axvspan(start_dt, end_dt, color='tab:red', alpha=0.12, label='Anomaly interval')
        ax.axvline(start_dt, color='tab:green', linestyle='-', linewidth=1.0, label='Actual start')
        if pd.notna(x_min) and pd.notna(x_max) and x_min < x_max:
            ax.set_xlim(x_min, x_max)

        if pd.notna(detected_dt):
            ax.axvline(detected_dt, color='black', linestyle='--', linewidth=1.0, label='Detected start')

        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        ax.legend(uniq.values(), uniq.keys(), loc='upper right')

    axes[0].set_title(title)
    axes[-1].set_xlabel('Time')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_html(
    source_path='db/salt_anomaly_database_interpolated.csv',
    output_path='salt_anomaly_report_interpolated.html',
    report_title='Salt Anomaly Report (Interpolated Dataset)',
):
    print(f'Generating Salt report from: {source_path}')
    salt_df = load_salt_data(source_path)
    intervals = load_salt_intervals()
    detected_map = load_detected_salt_points()

    html = """
    <html>
    <head>
        <title>Salt Anomaly Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .plot-container { margin-bottom: 40px; border: 1px solid #eee; padding: 10px; text-align: center; }
            img { max-width: 100%; height: auto; }
            h2 { color: #333; }
        </style>
    </head>
    <body>
        <h1>{report_title}</h1>
        <h2>Summary Table</h2>
        <table>
            <tr>
                <th>Well ID</th>
                <th>Interval</th>
                <th>Data Start</th>
                <th>Data End</th>
                <th>Anomaly Start</th>
                <th>Anomaly End</th>
                <th>Detected Start</th>
                <th>Status</th>
            </tr>
    """

    for _, row in intervals.iterrows():
        key = (row['well_id'], int(row['interval_idx']))
        detected = detected_map.get(key)

        status = 'Not found'
        if pd.notna(detected):
            status = 'Detected' if row['start_date'] <= detected <= row['end_date'] else 'Out of interval'

        html += f"""
            <tr>
                <td>{row['well_id']}</td>
                <td>{int(row['interval_idx'])}</td>
                <td>{row['data_start']}</td>
                <td>{row['data_end']}</td>
                <td>{row['start_date']}</td>
                <td>{row['end_date']}</td>
                <td>{detected}</td>
                <td>{status}</td>
            </tr>
        """

    html += """
        </table>
        <h2>Plots</h2>
    """

    total = len(intervals)
    for i, row in intervals.iterrows():
        well_id = row['well_id']
        interval_idx = int(row['interval_idx'])
        start_dt = row['start_date']
        end_dt = row['end_date']
        detected = detected_map.get((well_id, interval_idx))

        print(f'Plot {i + 1}/{total}: well={well_id}, interval={interval_idx}')
        well_data = salt_df[salt_df['well_id'] == well_id]
        if well_data.empty:
            html += f"""
            <div class="plot-container">
                <h3>Well {well_id} (interval {interval_idx})</h3>
                <p>No data available for plotting.</p>
            </div>
            """
            continue

        # Use original extraction window if present, otherwise full well range.
        data_start = row['data_start'] if pd.notna(row['data_start']) else well_data['timestamp'].min()
        data_end = row['data_end'] if pd.notna(row['data_end']) else well_data['timestamp'].max()
        plot_df = well_data[(well_data['timestamp'] >= data_start) & (well_data['timestamp'] <= data_end)]

        title = f"Well {well_id} - Salt interval {interval_idx}"
        x_min = data_start if pd.notna(data_start) else plot_df['timestamp'].min()
        x_max = data_end if pd.notna(data_end) else plot_df['timestamp'].max()
        b64_img = create_plot_base64(plot_df, start_dt, end_dt, detected, title, x_min, x_max)
        if b64_img is None:
            html += f"""
            <div class="plot-container">
                <h3>{title}</h3>
                <p>No data available for plotting.</p>
            </div>
            """
        else:
            html += f"""
            <div class="plot-container">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{b64_img}" alt="{title}">
            </div>
            """

    html += """
    </body>
    </html>
    """

    Path(output_path).write_text(html, encoding='utf-8')
    print(f'Report generated: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source',
        default='db/salt_anomaly_database_interpolated.csv',
        help='Path to salt CSV file for plotting',
    )
    parser.add_argument(
        '--output',
        default='salt_anomaly_report_interpolated.html',
        help='Path to output HTML report',
    )
    parser.add_argument(
        '--title',
        default='Salt Anomaly Report (Interpolated Dataset)',
        help='Title displayed in HTML report',
    )
    args = parser.parse_args()
    generate_html(source_path=args.source, output_path=args.output, report_title=args.title)
