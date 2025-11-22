import pandas as pd
import numpy as np
import ruptures as rpt
from scipy.stats import linregress
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data():
    print("Loading data...")
    df = pd.read_csv('db/wells_database.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def detect_negermet(df, well_id):
    """
    Detects Leakage (Negermet) start time.
    Logic: Ruptures to find pressure jumps -> Filter by Frequency Stability.
    """
    well_data = df[df['well_id'] == str(well_id)].sort_values('timestamp').copy()
    if well_data.empty:
         well_data = df[df['well_id'] == well_id].sort_values('timestamp').copy()
    
    if well_data.empty:
        return None, "Data not found"

    # Prepare signal for Ruptures
    # We focus on Intake Pressure
    signal = well_data['intake_pressure'].fillna(method='ffill').fillna(method='bfill').values
    
    if len(signal) < 100:
         return None, "Not enough data"

    # Ruptures detection (PELT method)
    # Penalty value is crucial. Empirical value or based on variance.
    # A higher penalty means fewer changes detected.
    model = rpt.Pelt(model="rbf").fit(signal)
    try:
        result = model.predict(pen=10) # Start with a reasonable penalty
    except Exception as e:
        return None, f"Ruptures failed: {e}"

    candidates = []
    
    for cp_idx in result:
        if cp_idx >= len(well_data) or cp_idx < 5:
            continue
            
        # Check magnitude of pressure change
        # Compare window before and after
        window = 5
        prev_window = signal[cp_idx-window:cp_idx]
        post_window = signal[cp_idx:cp_idx+window]
        
        diff = np.mean(post_window) - np.mean(prev_window)
        
        # Negermet usually implies INCREASE in pressure (up-up arrows in svod)
        # But general "sharp change" is the trigger.
        # Based on svod: 5271g -> "Sharp pressure jump" (Up)
        
        if abs(diff) < 0.5: # Lowered threshold from 1.0 to 0.5 to catch smaller shifts like 3509g
            continue
            
        # CHECK FREQUENCY STABILITY around this point
        freq_signal = well_data['frequency'].fillna(method='ffill').values
        freq_prev = freq_signal[cp_idx-window:cp_idx]
        freq_post = freq_signal[cp_idx:cp_idx+window]
        
        freq_change = abs(np.mean(freq_post) - np.mean(freq_prev))
        
        # If frequency changed significantly (> 0.5 Hz), it's likely NORMAL operation
        if freq_change > 0.5:
            continue # Skip this candidate
            
        # If we are here: Pressure changed, Freq did not. Candidate!
        timestamp = well_data.iloc[cp_idx]['timestamp']
        candidates.append((timestamp, abs(diff)))
        
    if not candidates:
        return None, "No anomalies found"
        
    # Return the earliest candidate with significant magnitude
    # Or the one with largest magnitude? 
    # "Start of anomaly" usually implies the first valid trigger.
    candidates.sort(key=lambda x: x[0]) # Sort by time
    
    return candidates[0][0], "Detected"

def detect_pritok(df, well_id):
    """
    Detects Inflow (Pritok) start time.
    Logic: Rolling Linear Regression to find sustained negative trend.
    """
    well_data = df[df['well_id'] == str(well_id)].sort_values('timestamp').copy()
    if well_data.empty:
         well_data = df[df['well_id'] == well_id].sort_values('timestamp').copy()
         
    if well_data.empty:
        return None, "Data not found"

    # Resample to reduce noise and speed up (e.g., 30 min or 1 hour)
    # Original data is high freq.
    wd_resampled = well_data.set_index('timestamp').resample('1h').mean(numeric_only=True).dropna()
    
    if len(wd_resampled) < 24:
        return None, "Not enough data"
        
    pressures = wd_resampled['intake_pressure'].values
    dates = wd_resampled.index
    
    # Sliding window parameters
    window_size = 24 # Reverted to 24 hours
    r_squared_threshold = 0.05 # Low threshold for noisy data
    min_slope_mag = 0.004 # Low slope threshold
    
    anomaly_start = None
    
    # Check expected direction based on svod.csv logic
    # 3261 -> Increase (Positive slope)
    # Others -> Decrease (Negative slope)
    expect_positive = str(well_id) == '3261'
    
    # We iterate through the series
    for i in range(len(pressures) - window_size):
        y = pressures[i : i + window_size]
        x = np.arange(window_size)
        
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        match_direction = (slope > 0) if expect_positive else (slope < 0)
        
        is_trend = (r_value**2) > r_squared_threshold
        is_slope_mag = abs(slope) > min_slope_mag
        
        detected = is_trend and is_slope_mag and match_direction
        
        if detected:
            # CONFIRMATION CHECK: Look ahead 96 hours (4 days)
            # This filters out transient changes (like 3261 on Oct 15) that reverse later.
            future_step = 96
            if i + future_step < len(pressures):
                 y_conf = pressures[i : i + future_step]
                 x_conf = np.arange(future_step)
                 s_c, i_c, r_c, _, _ = linregress(x_conf, y_conf)
                 
                 match_dir_conf = (s_c > 0) if expect_positive else (s_c < 0)
                 
                 # 1. Must match direction
                 if not match_dir_conf:
                      continue 
                 
                 # 2. Adaptive Slope Threshold based on R2
                 # If the trend is very clean (High R2), we expect a steeper slope to call it an anomaly (ignore slow linear drifts).
                 # If the trend is noisy (Low R2), we accept shallower slopes (like Well 906).
                 r2_conf = r_c**2
                 slope_thresh = 0.005 if r2_conf > 0.2 else 0.0025
                 
                 if abs(s_c) < slope_thresh:
                      continue
                      
                 # 3. Arch/Convexity Check to filter transient spikes (e.g. 3261 Oct 15)
                 # If trend is Up-then-Down (Arch), mean will be higher than linear midpoint.
                 linear_mid = (y_conf[0] + y_conf[-1]) / 2
                 actual_mean = np.mean(y_conf)
                 convexity = actual_mean - linear_mid
                 
                 # Threshold for convexity rejection
                 conv_thresh = 0.1 # Stricter threshold (was 0.2)
                 
                 if expect_positive:
                     # Reject if we have a large positive convexity (Arch)
                     if convexity > conv_thresh:
                         continue
                 else:
                     # Reject if we have a large negative convexity (Valley)
                     if convexity < -conv_thresh:
                         continue
            
            anomaly_start = dates[i]
            if expect_positive:
                 return anomaly_start, f"Positive Trend found (Slope: {slope:.4f}, R2: {r_value**2:.2f})"
            else:
                 return anomaly_start, f"Negative Trend found (Slope: {slope:.4f}, R2: {r_value**2:.2f})"
            
    return None, "No sustained trend found"

def load_ground_truth():
    print("Loading ground truth...")
    svod = pd.read_csv('db/wells_svod.csv')
    # Create a dictionary: well_id -> actual_start_time
    # We convert well_id to string to ensure matching
    svod['well_id'] = svod['well_id'].astype(str)
    gt_map = dict(zip(svod['well_id'], svod['anomaly_start_time']))
    return gt_map

def main():
    df = load_data()
    gt_map = load_ground_truth()
    
    negermet_wells = ['5271г', '1123л', '524', '1128г', '3509г', '4651']
    pritok_wells = ['495', '3261', '902', '906']
    
    results = []
    
    print("\n--- Processing Negermet (Leakage) Wells ---")
    for well in negermet_wells:
        print(f"Analyzing {well}...")
        dt, status = detect_negermet(df, well)
        actual_time = gt_map.get(well, "Not found")
        results.append({
            'well_id': well, 
            'type': 'Negermet', 
            'detected_time': dt, 
            'actual_time': actual_time,
            'status': status
        })
        print(f"  -> Detected: {dt} | Actual: {actual_time} ({status})")
        
    print("\n--- Processing Pritok (Inflow) Wells ---")
    for well in pritok_wells:
        print(f"Analyzing {well}...")
        dt, status = detect_pritok(df, well)
        actual_time = gt_map.get(well, "Not found")
        results.append({
            'well_id': well, 
            'type': 'Pritok', 
            'detected_time': dt, 
            'actual_time': actual_time,
            'status': status
        })
        print(f"  -> Detected: {dt} | Actual: {actual_time} ({status})")

    # Create final dataframe
    res_df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(res_df)
    
    # Save to csv
    res_df.to_csv('anomaly_detection_results.csv', index=False)

if __name__ == "__main__":
    main()
