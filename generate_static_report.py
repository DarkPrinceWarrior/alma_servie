import pandas as pd
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

MEHA_COLUMN_MAP = {
    'Затрубное давление': 'annulus_pressure',
    'Линейное давление': 'line_pressure',
    'Давление на приеме насоса': 'intake_pressure',
    'Объемный дебит жидкости, м3/сут': 'flow_rate',
    'Ток фазы A': 'current',
    'Коэффициент загрузки': 'load_coef',
    'Температура двигателя': 'motor_temperature',
    'Рабочая частота': 'frequency',
}

def load_legacy_data():
    print("Loading legacy database for plotting...")
    df = pd.read_csv('db/wells_database.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def load_meha_data():
    print("Loading Meha data for plotting...")
    frames = []
    for path in sorted(Path('db').glob('*_МЕХА.feather')):
        well_id = path.stem.split('_')[0]
        df = pd.read_feather(path)
        df = df.rename(columns={'index': 'timestamp', **MEHA_COLUMN_MAP})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['well_id'] = str(well_id)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def load_data(results):
    if 'type' in results.columns and (results['type'] == 'Meha').any():
        return load_meha_data()
    return load_legacy_data()

def create_plot_base64(well_id, anomaly_type, detected_time, actual_start, actual_end, df):
    """
    Generates a plot for the given well and returns it as a base64 string.
    """
    well_data = df[df['well_id'] == str(well_id)].sort_values('timestamp')
    if well_data.empty:
         well_data = df[df['well_id'] == well_id].sort_values('timestamp')
         
    if well_data.empty:
        return None

    if anomaly_type == 'Meha':
        well_data = well_data.set_index('timestamp').resample('1h').mean(numeric_only=True).dropna().reset_index()

        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        series_map = [
            ('intake_pressure', 'Intake Pressure', 'tab:blue'),
            ('load_coef', 'Load Coefficient', 'tab:orange'),
            ('current', 'Current', 'tab:green'),
            ('motor_temperature', 'Motor Temperature', 'tab:red'),
        ]
        for ax, (col, label, color) in zip(axes, series_map):
            ax.plot(well_data['timestamp'], well_data[col], label=label, color=color, linewidth=0.8)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            if pd.notna(detected_time):
                try:
                    dt = pd.to_datetime(detected_time)
                    ax.axvline(dt, color='black', linestyle='--', linewidth=1.0)
                except Exception:
                    pass

            if pd.notna(actual_start) and pd.notna(actual_end):
                try:
                    start = pd.to_datetime(actual_start)
                    end = pd.to_datetime(actual_end)
                    ax.axvspan(start, end, color='tab:red', alpha=0.1)
                except Exception:
                    pass

        axes[0].set_title(f"Well {well_id} - {anomaly_type}")
        axes[-1].set_xlabel("Time")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
    else:
        # Handle missing values for plotting to avoid gaps
        well_data['intake_pressure'] = well_data['intake_pressure'].ffill().bfill()

        plt.figure(figsize=(10, 5))
        plt.plot(well_data['timestamp'], well_data['intake_pressure'], label='Intake Pressure', color='blue')
        
        if pd.notna(detected_time):
            try:
                dt = pd.to_datetime(detected_time)
                plt.axvline(dt, color='red', linestyle='--', label=f'Detected: {dt.strftime("%Y-%m-%d %H:%M")}')
            except Exception:
                pass
                
        if pd.notna(actual_start):
            try:
                at = pd.to_datetime(actual_start)
                plt.axvline(at, color='green', linestyle='-', label=f'Actual: {at.strftime("%Y-%m-%d %H:%M")}')
            except Exception:
                pass

        plt.title(f"Well {well_id} - {anomaly_type}")
        plt.xlabel("Time")
        plt.ylabel("Pressure")
        plt.legend()
        plt.grid(True)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
    
    buf.seek(0)
    b64_string = base64.b64encode(buf.read()).decode('utf-8')
    return b64_string

def generate_static_html():
    print("Generating static HTML report...")
    results = pd.read_csv('anomaly_detection_results.csv')
    full_data = load_data(results)
    has_interval = 'actual_start' in results.columns and 'actual_end' in results.columns
    
    html_content = """
    <html>
    <head>
        <title>Anomaly Detection Report (Static)</title>
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
        <h1>Anomaly Detection Report</h1>
        
        <h2>Summary Table</h2>
        <table>
            <tr>
                <th>Well ID</th>
                <th>Type</th>
                <th>Detected Time</th>
                <th>Actual Start</th>
                <th>Actual End</th>
                <th>Status</th>
            </tr>
    """

    if not has_interval:
        html_content = html_content.replace("<th>Actual Start</th>\n                <th>Actual End</th>", "<th>Actual Time</th>")
    
    for _, row in results.iterrows():
        if has_interval:
            actual_start = row['actual_start']
            actual_end = row['actual_end']
            html_content += f"""
            <tr>
                <td>{row['well_id']}</td>
                <td>{row['type']}</td>
                <td>{row['detected_time']}</td>
                <td>{actual_start}</td>
                <td>{actual_end}</td>
                <td>{row['status']}</td>
            </tr>
            """
        else:
            actual_time = row.get('actual_time')
            html_content += f"""
            <tr>
                <td>{row['well_id']}</td>
                <td>{row['type']}</td>
                <td>{row['detected_time']}</td>
                <td>{actual_time}</td>
                <td>{row['status']}</td>
            </tr>
            """
        
    html_content += """
        </table>
        <h2>Plots</h2>
    """
    
    total_plots = len(results)
    print(f"Generating plots for {total_plots} wells...")
    
    for idx, row in results.iterrows():
        well_id = str(row['well_id'])
        anomaly_type = row['type']
        detected_time = row['detected_time']
        actual_start = row['actual_start'] if has_interval else row.get('actual_time')
        actual_end = row['actual_end'] if has_interval else None
        
        print(f"Processing plot {idx + 1}/{total_plots}: Well {well_id}")
        
        b64_img = create_plot_base64(well_id, anomaly_type, detected_time, actual_start, actual_end, full_data)
        
        if b64_img:
            html_content += f"""
            <div class="plot-container">
                <h3>Well {well_id} ({anomaly_type})</h3>
                <img src="data:image/png;base64,{b64_img}" alt="Plot for {well_id}">
            </div>
            """
        else:
            html_content += f"""
            <div class="plot-container">
                <h3>Well {well_id} ({anomaly_type})</h3>
                <p>No data available for plotting.</p>
            </div>
            """

    html_content += """
    </body>
    </html>
    """
    
    with open('anomaly_report_static.html', 'w') as f:
        f.write(html_content)
        
    print("Report generated: anomaly_report_static.html")

if __name__ == "__main__":
    generate_static_html()
