import pandas as pd
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def load_data():
    print("Loading database for plotting...")
    df = pd.read_csv('db/wells_database.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def create_plot_base64(well_id, anomaly_type, detected_time, actual_time, df):
    """
    Generates a plot for the given well and returns it as a base64 string.
    """
    well_data = df[df['well_id'] == str(well_id)].sort_values('timestamp')
    if well_data.empty:
         well_data = df[df['well_id'] == well_id].sort_values('timestamp')
         
    if well_data.empty:
        return None

    # Handle missing values for plotting to avoid gaps
    # We use forward fill then backward fill to ensure continuity
    well_data['intake_pressure'] = well_data['intake_pressure'].ffill().bfill()

    plt.figure(figsize=(10, 5))
    plt.plot(well_data['timestamp'], well_data['intake_pressure'], label='Intake Pressure', color='blue')
    
    # Plot Detected Time
    if pd.notna(detected_time):
        try:
            dt = pd.to_datetime(detected_time)
            plt.axvline(dt, color='red', linestyle='--', label=f'Detected: {dt.strftime("%Y-%m-%d %H:%M")}')
        except:
            pass
            
    # Plot Actual Time
    if pd.notna(actual_time) and str(actual_time) != 'Not found':
        try:
            at = pd.to_datetime(actual_time)
            plt.axvline(at, color='green', linestyle='-', label=f'Actual: {at.strftime("%Y-%m-%d %H:%M")}')
        except:
            pass

    plt.title(f"Well {well_id} - {anomaly_type}")
    plt.xlabel("Time")
    plt.ylabel("Pressure")
    plt.legend()
    plt.grid(True)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    # Save to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    buf.seek(0)
    b64_string = base64.b64encode(buf.read()).decode('utf-8')
    return b64_string

def generate_static_html():
    print("Generating static HTML report...")
    results = pd.read_csv('anomaly_detection_results.csv')
    full_data = load_data()
    
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
                <th>Actual Time</th>
                <th>Status</th>
            </tr>
    """
    
    for _, row in results.iterrows():
        html_content += f"""
            <tr>
                <td>{row['well_id']}</td>
                <td>{row['type']}</td>
                <td>{row['detected_time']}</td>
                <td>{row['actual_time']}</td>
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
        actual_time = row['actual_time']
        
        print(f"Processing plot {idx + 1}/{total_plots}: Well {well_id}")
        
        b64_img = create_plot_base64(well_id, anomaly_type, detected_time, actual_time, full_data)
        
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
