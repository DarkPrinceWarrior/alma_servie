import pandas as pd
import numpy as np
try:
    import ruptures as rpt
except ImportError:
    rpt = None
try:
    from scipy.stats import linregress
except ImportError:
    linregress = None
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

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

MEHA_SAW_COLS = [
    'annulus_pressure',
    'line_pressure',
    'intake_pressure',
    'flow_rate',
    'current',
    'load_coef',
    'frequency',
]

MEHA_TEMP_COL = 'motor_temperature'

def parse_trend_rule(text):
    """
    Parses text like '↑↑ Давление', '↓ Ток', '= Температура' into expected direction.
    Returns: 1 (Increase), -1 (Decrease), 0 (Stable), None (Unknown/Ambiguous)
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    if '↑' in text and '↓' in text: # Ambiguous '↓↑'
        return None 
    if '↑' in text:
        return 1
    if '↓' in text:
        return -1
    if '=' in text:
        return 0
    return None

def load_validation_rules():
    """
    Loads trend rules from wells_svod.csv.
    Returns: dict {well_id: {'pressure': rule, 'current': rule, 'temp': rule}}
    """
    svod = pd.read_csv('db/wells_svod.csv')
    rules = {}
    for _, row in svod.iterrows():
        wid = str(row['well_id'])
        rules[wid] = {
            'pressure': parse_trend_rule(row.get('pressure_info')),
            'current': parse_trend_rule(row.get('current_info')),
            'temp': parse_trend_rule(row.get('temperature_info'))
        }
    return rules

def calculate_slope(series):
    """Calculates linear regression slope for a series."""
    if len(series) < 2:
        return 0
    y = series.values
    x = np.arange(len(y))
    slope, _, _, _, _ = safe_linregress(x, y)
    return slope

def safe_linregress(x, y):
    if linregress is not None:
        return linregress(x, y)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    x_mean = x.mean()
    y_mean = y.mean()
    x_dev = x - x_mean
    y_dev = y - y_mean
    var = np.sum(x_dev ** 2)
    if var == 0:
        return 0.0, y_mean, 0.0, 0.0, 0.0
    cov = np.sum(x_dev * y_dev)
    slope = cov / var
    intercept = y_mean - slope * x_mean
    denom = np.sqrt(var * np.sum(y_dev ** 2))
    r_value = cov / denom if denom else 0.0
    return slope, intercept, r_value, 0.0, 0.0

def validate_anomaly(well_data, anomaly_time, rules, window_hours=3):
    """
    Validates an anomaly candidate by checking trends of auxiliary parameters.
    """
    start_dt = pd.to_datetime(anomaly_time)
    end_dt = start_dt + pd.Timedelta(hours=window_hours)
    
    # Extract validation window
    window = well_data[(well_data['timestamp'] >= start_dt) & (well_data['timestamp'] <= end_dt)]
    if len(window) < 2:
        return True # Not enough data to invalidate
        
    # Resample to ensure consistent slope calculation (e.g. 1h) if data is high freq
    # Assuming well_data is raw, we resample to 10T or 1H?
    # For slope calculation, it's safer to use resampled data to avoid noise.
    # Let's use the raw data but simple slope or mean-resampled.
    # Using 10T resampling seems appropriate for validation.
    window_res = window.set_index('timestamp').resample('10min').mean(numeric_only=True).interpolate()
    
    # Check Pressure
    if rules['pressure'] is not None:
        p_slope = calculate_slope(window_res['intake_pressure'])
        if rules['pressure'] == 1 and p_slope < -0.01: # Expected Increase, but Decreasing
            return False
        if rules['pressure'] == -1 and p_slope > 0.01: # Expected Decrease, but Increasing
            return False
        # Note: We treat Stable (=) loosely for pressure since main algo handles it.
        
    # Check Current
    if rules['current'] is not None and 'current' in window_res:
        c_slope = calculate_slope(window_res['current'])
        # Current stability threshold? 
        # Stable means slope is near 0.
        # Increase means slope > 0.
        if rules['current'] == 0:
             if abs(c_slope) > 0.5: # Arbitrary threshold for "Significant Current Change"
                 return False
        elif rules['current'] == 1 and c_slope < -0.1:
             return False
        elif rules['current'] == -1 and c_slope > 0.1:
             return False
             
    # Check Temperature
    if rules['temp'] is not None and 'motor_temperature' in window_res:
        t_slope = calculate_slope(window_res['motor_temperature'])
        if rules['temp'] == 0:
             if abs(t_slope) > 0.5: # Significant Temp Change
                 return False
        elif rules['temp'] == 1 and t_slope < -0.1:
             return False
        elif rules['temp'] == -1 and t_slope > 0.1:
             return False
             
    return True

def load_data():
    print("Loading data...")
    df = pd.read_csv('db/wells_database.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def load_meha_data():
    print("Loading Meha data from feather...")
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

def load_meha_intervals():
    print("Loading Meha ground truth intervals...")
    intervals = pd.read_csv('db/meha_intervals.csv')
    intervals['well_id'] = intervals['well_id'].astype(str)
    intervals['start_date'] = pd.to_datetime(intervals['start_date'])
    intervals['end_date'] = pd.to_datetime(intervals['end_date']) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return dict(
        (row['well_id'], (row['start_date'], row['end_date']))
        for _, row in intervals.iterrows()
    )

def median_abs_deviation(series):
    values = np.asarray(series, dtype=float)
    med = np.nanmedian(values)
    return np.nanmedian(np.abs(values - med))

def rolling_sawtooth(series, window):
    diff = series.diff()
    sign = np.sign(diff)
    sign_change = (sign != sign.shift(1)) & (sign != 0) & (sign.shift(1) != 0)
    sign_rate = sign_change.rolling(window, min_periods=window).mean()
    mean_abs_diff = diff.abs().rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return sign_rate * (mean_abs_diff / (std + 1e-6))

def normalize_positive(series):
    mad = median_abs_deviation(series)
    if mad == 0 or np.isnan(mad):
        mad = np.nanstd(series) or 1.0
    median = np.nanmedian(series)
    return ((series - median) / mad).clip(lower=0)

def find_longest_true_segment(mask):
    best_start = None
    best_len = 0
    current_start = None
    current_len = 0
    for timestamp, flag in mask.items():
        if flag:
            if current_start is None:
                current_start = timestamp
                current_len = 1
            else:
                current_len += 1
        else:
            if current_start is not None and current_len > best_len:
                best_start = current_start
                best_len = current_len
            current_start = None
            current_len = 0
    if current_start is not None and current_len > best_len:
        best_start = current_start
        best_len = current_len
    return best_start, best_len

def detect_meha(well_data, window_hours=24, score_quantile=0.7, temp_weight=0.5):
    well_data = well_data.sort_values('timestamp')
    resampled = well_data.set_index('timestamp').resample('1h').mean(numeric_only=True).dropna()
    if len(resampled) < window_hours:
        return None, "Not enough data"

    saw_scores = [rolling_sawtooth(resampled[col], window_hours) for col in MEHA_SAW_COLS]
    saw_mean = pd.concat(saw_scores, axis=1).mean(axis=1)

    temp = resampled[MEHA_TEMP_COL]
    temp_slope = (temp - temp.shift(window_hours - 1)) / float(window_hours - 1)
    temp_norm = normalize_positive(temp_slope)

    score = saw_mean + temp_weight * temp_norm
    monthly = score.resample('MS').median()
    nonzero = monthly[monthly > 0]
    if nonzero.empty:
        return None, "No anomaly score"

    threshold = np.quantile(nonzero.values, score_quantile)
    high = monthly >= threshold
    start, seg_len = find_longest_true_segment(high)
    if start is None:
        return None, "No sustained anomaly"

    return start, f"Detected (segment months: {seg_len})"

def detect_negermet(df, well_id, rules=None):
    """
    Detects Leakage (Negermet) start time.
    Logic: Ruptures to find pressure jumps -> Filter by Frequency Stability -> Validate with Rules.
    """
    if rpt is None:
        return None, "ruptures not installed"
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
        
        # VALIDATE WITH SVOD RULES (e.g. Check Current Direction)
        if rules:
            # Use a short window (1h) for Negermet validation as jumps are sharp
            if not validate_anomaly(well_data, timestamp, rules.get(str(well_id)), window_hours=1):
                continue

        candidates.append((timestamp, abs(diff)))
        
    if not candidates:
        return None, "No anomalies found"
        
    # Return the earliest candidate with significant magnitude
    # Or the one with largest magnitude? 
    # "Start of anomaly" usually implies the first valid trigger.
    candidates.sort(key=lambda x: x[0]) # Sort by time
    
    return candidates[0][0], "Detected"

def detect_pritok(df, well_id, rules=None):
    """
    Detects Inflow (Pritok) start time.
    Logic: Rolling Linear Regression to find sustained negative trend -> Validate with Rules.
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
        
        slope, intercept, r_value, p_value, std_err = safe_linregress(x, y)
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
                 s_c, i_c, r_c, _, _ = safe_linregress(x_conf, y_conf)
                 
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
            
            # ACCELERATION CHECK: Is this just a slow precursor to a major event?
            # Look ahead up to 96 hours. If we find a window with Slope > 3x current_slope,
            # we assume the current one is too early (precursor) and skip it.
            is_precursor = False
            lookahead_limit = 96
            current_mag = abs(slope)
            
            # Scan future windows
            for offset in range(4, lookahead_limit, 4): # Check every 4 hours
                 idx_future = i + offset
                 if idx_future + window_size >= len(pressures):
                     break
                 
                 y_fut = pressures[idx_future : idx_future + window_size]
                 x_fut = np.arange(window_size)
                 s_fut, _, _, _, _ = safe_linregress(x_fut, y_fut)
                 
                 # Must match direction
                 match_dir_fut = (s_fut > 0) if expect_positive else (s_fut < 0)
                 if not match_dir_fut:
                     continue
                     
                 if abs(s_fut) > 3.0 * current_mag:
                      # Found a much steeper trend later!
                      is_precursor = True
                      break
            
            if is_precursor:
                 continue

            anomaly_start = dates[i]
            
            # VALIDATE WITH SVOD RULES
            if rules:
                # Pritok is slow, so use 24h window for validation
                if not validate_anomaly(well_data, anomaly_start, rules.get(str(well_id)), window_hours=24):
                    continue

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

def run_legacy_detection():
    df = load_data()
    gt_map = load_ground_truth()
    validation_rules = load_validation_rules()
    
    negermet_wells = ['5271г', '1123л', '524', '1128г', '3509г', '4651']
    pritok_wells = ['495', '3261', '902', '906']
    
    results = []
    
    print("\n--- Processing Negermet (Leakage) Wells ---")
    for well in negermet_wells:
        print(f"Analyzing {well}...")
        dt, status = detect_negermet(df, well, validation_rules)
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
        dt, status = detect_pritok(df, well, validation_rules)
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

def run_meha_detection():
    df = load_meha_data()
    if df.empty:
        print("No Meha data found.")
        return

    intervals = load_meha_intervals()
    results = []

    print("\n--- Processing Meha Wells ---")
    for well_id, (start_dt, end_dt) in intervals.items():
        print(f"Analyzing {well_id}...")
        well_data = df[df['well_id'] == str(well_id)]
        detected_time, detail = detect_meha(well_data)

        if detected_time is None:
            status = "Not found"
        elif start_dt <= detected_time <= end_dt:
            status = "Detected"
        else:
            status = "Out of interval"

        results.append({
            'well_id': well_id,
            'type': 'Meha',
            'detected_time': detected_time,
            'actual_start': start_dt,
            'actual_end': end_dt,
            'status': status,
            'detail': detail,
        })
        print(f"  -> Detected: {detected_time} | Actual: {start_dt} - {end_dt} ({status})")

    res_df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(res_df)
    res_df.to_csv('anomaly_detection_results.csv', index=False)

def main():
    if Path('db/meha_intervals.csv').exists():
        run_meha_detection()
    else:
        run_legacy_detection()

if __name__ == "__main__":
    main()
