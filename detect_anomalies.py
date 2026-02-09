import pandas as pd
import numpy as np
import argparse
import random
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

SALT_PRESSURE_COL = 'Давление на приеме насоса кгс/см²'
SALT_FREQ_COL = 'Выходная частота'


def normalize_well_id(well_id):
    wid = str(well_id).strip()
    if wid.endswith('_МЕХА'):
        wid = wid.split('_', 1)[0]
    return wid

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


def load_salt_data():
    print("Loading Salt data...")
    candidates = [
        Path('db/salt_anomaly_database_interpolated.csv'),
        Path('db/salt_anomaly_database.csv'),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        return pd.DataFrame()

    df = pd.read_csv(src, dtype={'well_id': str}, low_memory=False)
    df['well_id'] = df['well_id'].astype(str).str.strip().str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    return df


def load_salt_intervals():
    print("Loading Salt ground truth intervals...")
    intervals = pd.read_csv('db/salt_intervals.csv', dtype={'well_id': str})
    intervals['well_id'] = intervals['well_id'].astype(str).str.strip().str.lower()
    intervals['start_date'] = pd.to_datetime(intervals['start_date'], errors='coerce')
    intervals['end_date'] = pd.to_datetime(intervals['end_date'], errors='coerce')
    intervals = intervals.dropna(subset=['well_id', 'start_date', 'end_date']).copy()
    if 'interval_idx' not in intervals.columns:
        intervals['interval_idx'] = (
            intervals.groupby('well_id').cumcount() + 1
        )
    return intervals.sort_values(['well_id', 'start_date', 'interval_idx']).reset_index(drop=True)

def median_abs_deviation(series):
    values = np.asarray(series, dtype=float)
    med = np.nanmedian(values)
    return np.nanmedian(np.abs(values - med))

def clip_series(series, mask=None, lower_q=0.01, upper_q=0.99):
    if mask is not None:
        valid = series[mask]
    else:
        valid = series
    valid = valid.dropna()
    if valid.empty:
        return series
    low, high = valid.quantile([lower_q, upper_q])
    if pd.isna(low) or pd.isna(high) or low == high:
        return series
    return series.clip(lower=low, upper=high)

def choose_resample_rule(timestamps):
    deltas = timestamps.sort_values().diff().dropna()
    if deltas.empty:
        return '1h'
    median_delta = deltas.median()
    if median_delta <= pd.Timedelta(minutes=5):
        return '15min'
    if median_delta <= pd.Timedelta(minutes=15):
        return '30min'
    return '1h'

def compute_window_points(resample_rule, window_hours):
    resample_td = pd.to_timedelta(resample_rule)
    points = int(pd.Timedelta(hours=window_hours) / resample_td)
    return max(points, 3)

def rolling_sawtooth(series, window, min_periods=None):
    if min_periods is None:
        min_periods = window
    diff = series.diff()
    sign = np.sign(diff)
    sign_change = (sign != sign.shift(1)) & (sign != 0) & (sign.shift(1) != 0)
    sign_rate = sign_change.rolling(window, min_periods=min_periods).mean()
    mean_abs_diff = diff.abs().rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return sign_rate * (mean_abs_diff / (std + 1e-6))

def normalize_positive(series):
    values = series.dropna()
    if values.empty:
        return series * 0
    mad = median_abs_deviation(values)
    if mad == 0 or np.isnan(mad):
        mad = np.nanstd(values) or 1.0
    median = np.nanmedian(values)
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

def find_first_true_run(mask, run_length):
    run = 0
    run_start = None
    for timestamp, flag in mask.items():
        if flag:
            if run_start is None:
                run_start = timestamp
            run += 1
            if run >= run_length:
                return run_start
        else:
            run = 0
            run_start = None
    return None

def detect_meha(
    well_data,
    interval_start=None,
    window_hours=24,
    score_quantile=0.7,
    temp_weight=0.5,
    min_run_days=5,
    baseline_days=30,
    baseline_k=3.0,
    stop_weight=0.3,
    daily_quantile=0.75,
    min_pump_on_fraction=0.2,
    pump_off_ratio=0.2,
    baseline_trim_quantile=0.6,
):
    well_data = well_data.sort_values('timestamp')
    resample_rule = choose_resample_rule(well_data['timestamp'])
    window_points = compute_window_points(resample_rule, window_hours)
    resampled = well_data.set_index('timestamp').resample(resample_rule).mean(numeric_only=True)
    resampled = resampled.dropna(how='all')
    if len(resampled) < window_points:
        return None, "Not enough data"

    freq_med = np.nanmedian(resampled['frequency'])
    current_med = np.nanmedian(resampled['current'])
    load_med = np.nanmedian(resampled['load_coef'])
    if np.isnan(freq_med):
        freq_med = 0.0
    if np.isnan(current_med):
        current_med = 0.0
    if np.isnan(load_med):
        load_med = 0.0

    freq_thr = max(1.0, freq_med * pump_off_ratio)
    current_thr = max(0.1, current_med * pump_off_ratio)
    load_thr = max(0.05, load_med * pump_off_ratio)
    pump_off = (
        (resampled['frequency'] <= freq_thr)
        | ((resampled['current'] <= current_thr) & (resampled['load_coef'] <= load_thr))
    )

    cleaned = resampled.copy()
    for col in MEHA_SAW_COLS + [MEHA_TEMP_COL]:
        cleaned[col] = clip_series(cleaned[col], mask=~pump_off)
        cleaned.loc[pump_off, col] = np.nan

    min_periods = max(3, int(window_points * 0.7))
    saw_scores = [
        rolling_sawtooth(cleaned[col], window_points, min_periods=min_periods)
        for col in MEHA_SAW_COLS
    ]
    saw_mean = pd.concat(saw_scores, axis=1).mean(axis=1, skipna=True)

    stop_rate = pump_off.astype(float).rolling(window_points, min_periods=min_periods).mean()

    temp = cleaned[MEHA_TEMP_COL]
    temp_slope = (temp - temp.shift(window_points - 1)) / float(window_points - 1)

    score = (
        normalize_positive(saw_mean)
        + temp_weight * normalize_positive(temp_slope)
        + stop_weight * normalize_positive(stop_rate)
    )

    daily_score = score.resample('D').quantile(daily_quantile)
    daily_on = (~pump_off).astype(float).resample('D').mean()
    daily_score = daily_score[daily_on >= min_pump_on_fraction]

    if interval_start is not None:
        baseline_start = interval_start - pd.Timedelta(days=baseline_days)
        baseline = daily_score[(daily_score.index >= baseline_start) & (daily_score.index < interval_start)]
        if baseline.empty:
            baseline = daily_score[daily_score > 0]
        if baseline.empty:
            return None, "No baseline for threshold"
        cutoff = baseline.quantile(baseline_trim_quantile)
        trimmed = baseline[baseline <= cutoff]
        if len(trimmed) >= max(5, int(len(baseline) * 0.3)):
            baseline = trimmed

        base_med = np.nanmedian(baseline)
        base_mad = median_abs_deviation(baseline)
        if base_mad == 0 or np.isnan(base_mad):
            base_mad = np.nanstd(baseline) or 1.0

        noise_ratio = base_mad / (abs(base_med) + 1e-6)
        run_days = min_run_days
        k = baseline_k
        if noise_ratio < 0.3:
            k *= 0.8
            run_days = max(3, min_run_days - 2)
        elif noise_ratio > 0.8:
            k *= 1.2
            run_days = min_run_days + 2

        threshold = base_med + k * base_mad
        search = daily_score[daily_score.index >= interval_start]
        detail_prefix = (
            f"baseline_days={baseline_days}, k={round(k,2)}, run_days={run_days}, "
            f"resample={resample_rule}, trim_q={baseline_trim_quantile}"
        )
    else:
        # No ground truth interval: use an adaptive threshold based on an initial baseline window
        # (avoids using a global quantile threshold which can drift with time and delay detection).
        nonzero = daily_score[daily_score > 0]
        if nonzero.empty:
            return None, "No anomaly score"

        baseline_start = nonzero.index.min()
        baseline_end = baseline_start + pd.Timedelta(days=baseline_days)
        baseline = daily_score[(daily_score.index >= baseline_start) & (daily_score.index < baseline_end)]
        baseline = baseline[baseline > 0]
        if baseline.empty:
            baseline = nonzero.head(min(len(nonzero), baseline_days))
        if baseline.empty:
            return None, "No baseline for threshold"

        cutoff = baseline.quantile(baseline_trim_quantile)
        trimmed = baseline[baseline <= cutoff]
        if len(trimmed) >= max(5, int(len(baseline) * 0.3)):
            baseline = trimmed

        base_med = np.nanmedian(baseline)
        base_mad = median_abs_deviation(baseline)
        if base_mad == 0 or np.isnan(base_mad):
            base_mad = np.nanstd(baseline) or 1.0

        noise_ratio = base_mad / (abs(base_med) + 1e-6)
        run_days = min_run_days
        k = baseline_k
        if noise_ratio < 0.3:
            k *= 0.9
            run_days = max(5, min_run_days - 2)
        elif noise_ratio > 0.8:
            k *= 1.1
            run_days = min_run_days + 2

        threshold = base_med + k * base_mad
        search = daily_score[daily_score.index >= baseline_end]
        detail_prefix = (
            f"auto_baseline_days={baseline_days}, k={round(k,2)}, run_days={run_days}, "
            f"resample={resample_rule}, trim_q={baseline_trim_quantile}"
        )

    high = search >= threshold
    start = find_first_true_run(high, run_days)
    if start is not None:
        return start, f"Detected (first run days: {run_days}, {detail_prefix})"

    start, seg_len = find_longest_true_segment(high)
    if start is None:
        return None, "No sustained anomaly"

    return start, f"Detected (longest segment days: {seg_len}, {detail_prefix})"


def detect_salt_starts(
    well_data,
    slope_hours=4,
    baseline_days=7,
    threshold_q=0.90,
    min_hits=2,
    persistence_hours=12,
    cooldown_days=1,
    max_starts=200,
):
    """
    Blind Salt detection: finds anomaly starts across the full well history.
    Ground-truth intervals are not used inside this detector.
    """
    if well_data.empty:
        return [], "Data not found"

    wd = well_data.sort_values('timestamp').copy()
    wd_res = wd.set_index('timestamp').resample('1h').mean(numeric_only=True)
    wd_res = wd_res.dropna(how='all')
    if wd_res.empty:
        return [], "Not enough data"
    if SALT_PRESSURE_COL not in wd_res.columns or SALT_FREQ_COL not in wd_res.columns:
        return [], "Missing required Salt columns"

    pressure = wd_res[SALT_PRESSURE_COL].interpolate(limit_area='inside')
    frequency = wd_res[SALT_FREQ_COL].interpolate(limit_area='inside')

    lag = max(2, int(slope_hours))
    p_slope = (pressure - pressure.shift(lag)) / float(lag)
    f_slope = (frequency - frequency.shift(lag)) / float(lag)

    aux_slopes = []
    for col in wd_res.columns:
        if col in (SALT_PRESSURE_COL, SALT_FREQ_COL):
            continue
        series = wd_res[col]
        if series.notna().sum() < lag * 4:
            continue
        aux_slopes.append((series - series.shift(lag)) / float(lag))

    if aux_slopes:
        aux_mean_slope = pd.concat(aux_slopes, axis=1).mean(axis=1, skipna=True)
    else:
        aux_mean_slope = pd.Series(0.0, index=wd_res.index)

    # Score emphasizes pressure growth and adverse pressure-frequency coupling.
    coupling = p_slope.clip(lower=0) * f_slope.clip(lower=0)
    score = (
        1.2 * normalize_positive(p_slope)
        + 1.0 * normalize_positive(coupling)
        + 0.4 * normalize_positive(aux_mean_slope)
    )

    win_points = max(24 * int(baseline_days), lag * 12)
    min_points = max(72, win_points // 3)

    score_thr = score.rolling(win_points, min_periods=min_points).quantile(float(threshold_q))
    p_thr = p_slope.rolling(win_points, min_periods=min_points).quantile(0.70).clip(lower=0)
    f_thr = f_slope.rolling(win_points, min_periods=min_points).quantile(0.50).clip(lower=0)
    aux_thr = aux_mean_slope.rolling(win_points, min_periods=min_points).quantile(0.60).clip(lower=0)

    # Fallback thresholds for sparse wells where rolling windows cannot be formed.
    score_global = score.dropna().quantile(float(threshold_q)) if score.notna().any() else np.nan
    p_global = p_slope.dropna().quantile(0.70) if p_slope.notna().any() else np.nan
    f_global = f_slope.dropna().quantile(0.50) if f_slope.notna().any() else np.nan
    aux_global = aux_mean_slope.dropna().quantile(0.60) if aux_mean_slope.notna().any() else np.nan

    if np.isfinite(score_global):
        score_thr = score_thr.fillna(float(score_global))
    if np.isfinite(p_global):
        p_thr = p_thr.fillna(max(0.0, float(p_global)))
    if np.isfinite(f_global):
        f_thr = f_thr.fillna(max(0.0, float(f_global)))
    if np.isfinite(aux_global):
        aux_thr = aux_thr.fillna(max(0.0, float(aux_global)))

    trigger = (
        (score >= score_thr)
        & (p_slope >= p_thr)
        & ((f_slope >= f_thr) | (aux_mean_slope >= aux_thr))
    ).fillna(False)

    starts = []
    cooldown = pd.Timedelta(days=float(cooldown_days))
    def append_starts(mask, required_hits):
        for ts, flag in mask.items():
            if not bool(flag):
                continue
            if starts and ts - starts[-1] < cooldown:
                continue

            win = mask[(mask.index >= ts) & (mask.index < ts + pd.Timedelta(hours=persistence_hours))]
            if int(win.sum()) >= int(required_hits):
                starts.append(ts)
                if len(starts) >= int(max_starts):
                    break

    append_starts(trigger, min_hits)

    # Reserve path for sparse/low-signal wells:
    # sustained pressure growth with a softer score gate.
    if len(starts) < int(max_starts):
        p_soft = p_slope.rolling(win_points, min_periods=min_points).quantile(0.65).clip(lower=0)
        p_soft_global = p_slope.dropna().quantile(0.65) if p_slope.notna().any() else np.nan
        if np.isfinite(p_soft_global):
            p_soft = p_soft.fillna(max(0.0, float(p_soft_global)))

        score_soft = (score_thr * 0.70).fillna(score.quantile(0.70) if score.notna().any() else np.nan)
        soft_trigger = ((p_slope >= p_soft) & (score >= score_soft)).fillna(False)
        append_starts(soft_trigger, max(2, int(min_hits) - 1))

    starts = sorted(set(starts))
    if len(starts) > int(max_starts):
        starts = starts[:int(max_starts)]

    detail = (
        f"Blind Salt (starts={len(starts)}, lag={lag}h, baseline_days={baseline_days}, "
        f"score_q={threshold_q}, min_hits={min_hits}, cooldown_days={cooldown_days})"
    )
    return starts, detail

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

    well_data['intake_pressure'] = well_data['intake_pressure'].ffill().bfill()
    well_data['frequency'] = well_data['frequency'].ffill().bfill()

    # Prepare signal for Ruptures
    # We focus on Intake Pressure
    signal = well_data['intake_pressure'].values
    
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
        
        diff = float(np.mean(post_window) - np.mean(prev_window))
        
        # Negermet usually implies INCREASE in pressure (up-up arrows in svod)
        # But general "sharp change" is the trigger.
        # Based on svod: 5271g -> "Sharp pressure jump" (Up)
        
        # Negermet is expected to be a sustained INCREASE in intake pressure.
        if diff < 0.5:
            continue
            
        # CHECK FREQUENCY STABILITY around this point
        freq_signal = well_data['frequency'].values
        freq_prev = freq_signal[cp_idx-window:cp_idx]
        freq_post = freq_signal[cp_idx:cp_idx+window]
        
        freq_change = abs(np.mean(freq_post) - np.mean(freq_prev))
        
        # If frequency changed significantly (> 0.5 Hz), it's likely NORMAL operation
        if freq_change > 0.5:
            continue # Skip this candidate
            
        # If we are here: Pressure changed, Freq did not. Candidate!
        timestamp = well_data.iloc[cp_idx]['timestamp']

        # Persistence check: in real negermet the pressure jump should remain for hours, not revert quickly.
        before = well_data[(well_data['timestamp'] >= timestamp - pd.Timedelta(hours=1)) & (well_data['timestamp'] < timestamp)]
        after = well_data[(well_data['timestamp'] >= timestamp) & (well_data['timestamp'] < timestamp + pd.Timedelta(hours=6))]
        if len(before) < 5 or len(after) < 5:
            continue
        persist_diff = float(after['intake_pressure'].median() - before['intake_pressure'].median())
        if persist_diff < 8.0:
            continue
        
        # VALIDATE WITH SVOD RULES (e.g. Check Current Direction)
        if rules:
            rule = rules.get(str(well_id))
            if rule is not None:
                # Use a short window (1h) for Negermet validation as jumps are sharp
                if not validate_anomaly(well_data, timestamp, rule, window_hours=1):
                    continue

        candidates.append((timestamp, diff, persist_diff))
        
    if not candidates:
        return None, "No anomalies found"
        
    # Return the earliest candidate with significant magnitude
    # Or the one with largest magnitude? 
    # "Start of anomaly" usually implies the first valid trigger.
    candidates.sort(key=lambda x: x[0])
    dt, jump, shift = candidates[0]
    return dt, f"Detected (jump={jump:.3f}, shift6h={shift:.3f})"

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
    
    allowed_direction = None
    if rules:
        rule = rules.get(str(well_id))
        if rule is not None and rule.get('pressure') in (1, -1):
            allowed_direction = rule['pressure']
    
    # We iterate through the series
    for i in range(len(pressures) - window_size):
        y = pressures[i : i + window_size]
        x = np.arange(window_size)
        
        slope, intercept, r_value, p_value, std_err = safe_linregress(x, y)
        if slope == 0:
            continue
        direction = 1 if slope > 0 else -1
        if allowed_direction is not None and direction != allowed_direction:
            continue
        
        is_trend = (r_value**2) > r_squared_threshold
        is_slope_mag = abs(slope) > min_slope_mag
        
        detected = is_trend and is_slope_mag
        
        if detected:
            # CONFIRMATION CHECK: Look ahead 96 hours (4 days)
            # This filters out transient changes (like 3261 on Oct 15) that reverse later.
            future_step_max = 96
            future_step = min(future_step_max, len(pressures) - i)
            if future_step >= 48:
                y_conf = pressures[i : i + future_step]
                x_conf = np.arange(future_step)
                s_c, i_c, r_c, _, _ = safe_linregress(x_conf, y_conf)

                if s_c == 0:
                    continue
                match_dir_conf = (1 if s_c > 0 else -1) == direction
                if not match_dir_conf:
                    continue

                r2_conf = r_c**2
                if r2_conf <= r_squared_threshold:
                    continue

                # Require a minimum net change over the confirmation window.
                # This helps reject slow drift/oscillation while allowing gradual clean trends (e.g. 1772).
                delta = float(y_conf[-1] - y_conf[0])
                min_delta_96h = 0.2
                delta_thr = min_delta_96h * (future_step / float(future_step_max))
                if direction == 1 and delta < delta_thr:
                    continue
                if direction == -1 and delta > -delta_thr:
                    continue

                # Arch/Convexity Check to filter transient spikes.
                linear_mid = (y_conf[0] + y_conf[-1]) / 2
                actual_mean = np.mean(y_conf)
                convexity = actual_mean - linear_mid

                conv_thresh = 0.1
                if direction == 1:
                    if convexity > conv_thresh:
                        continue
                else:
                    if convexity < -conv_thresh:
                        continue
            
            # ACCELERATION CHECK: Is this just a slow precursor to a major event?
            # Look ahead up to 96 hours. If we find a window with Slope > 3x current_slope,
            # we assume the current one is too early (precursor) and skip it.
            is_precursor = False
            lookahead_limit = 168
            current_mag = abs(slope)
            
            # Scan future windows
            for offset in range(4, lookahead_limit, 4): # Check every 4 hours
                 idx_future = i + offset
                 if idx_future + window_size >= len(pressures):
                     break
                 
                 y_fut = pressures[idx_future : idx_future + window_size]
                 x_fut = np.arange(window_size)
                 s_fut, _, _, _, _ = safe_linregress(x_fut, y_fut)
                 
                 if s_fut == 0:
                     continue
                 match_dir_fut = (1 if s_fut > 0 else -1) == direction
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
                rule = rules.get(str(well_id))
                if rule is not None:
                    # Pritok is slow, so use 24h window for validation
                    if not validate_anomaly(well_data, anomaly_start, rule, window_hours=24):
                        continue

            sign = 'Positive' if direction == 1 else 'Negative'
            return anomaly_start, f"{sign} Trend found (Slope: {slope:.4f}, R2: {r_value**2:.2f})"
            
    return None, "No sustained trend found"

def load_ground_truth():
    print("Loading ground truth...")
    svod = pd.read_csv('db/wells_svod.csv')
    # Create a dictionary: well_id -> actual_start_time
    # We convert well_id to string to ensure matching
    svod['well_id'] = svod['well_id'].astype(str)
    gt_map = dict(zip(svod['well_id'], svod['anomaly_start_time']))
    return gt_map


def load_legacy_ground_truth():
    svod = pd.read_csv('db/wells_svod.csv')
    svod['well_id'] = svod['well_id'].astype(str)
    svod = svod[svod['anomaly_type'].isin(['Негермет', 'Приток'])].copy()
    svod['actual_start'] = pd.to_datetime(svod['anomaly_start_time'], errors='coerce')
    svod['actual_end'] = pd.to_datetime(svod.get('well_stop_time'), errors='coerce')
    return svod


def detect_any_anomaly(
    well_id,
    legacy_df,
    meha_df,
    validation_rules=None,
    meha_intervals=None,
    meha_min_run_days=10,
):
    wid = normalize_well_id(well_id)

    if not meha_df.empty and (meha_df['well_id'] == wid).any():
        well_data = meha_df[meha_df['well_id'] == wid]
        interval_start = None
        if meha_intervals and wid in meha_intervals:
            interval_start = meha_intervals[wid][0]

        run_days = 5 if interval_start is not None else meha_min_run_days

        detected_time, detail = detect_meha(
            well_data,
            interval_start=interval_start,
            min_run_days=run_days,
        )
        if detected_time is not None:
            return 'Meha', detected_time, detail

        neg_dt, neg_detail = detect_negermet(meha_df, wid, rules=validation_rules)
        if neg_dt is not None:
            return 'Negermet', neg_dt, neg_detail

        prit_dt, prit_detail = detect_pritok(meha_df, wid, rules=validation_rules)
        if prit_dt is not None:
            return 'Pritok', prit_dt, prit_detail

        return None, None, 'No anomaly found'

    neg_dt, neg_detail = detect_negermet(legacy_df, wid, rules=validation_rules)
    if neg_dt is not None:
        return 'Negermet', neg_dt, neg_detail

    prit_dt, prit_detail = detect_pritok(legacy_df, wid, rules=validation_rules)
    if prit_dt is not None:
        return 'Pritok', prit_dt, prit_detail

    return None, None, 'No anomaly found'


def run_universal_detection(output_path='anomaly_detection_results.csv', salt_method='classic'):
    legacy_df = load_data()
    meha_df = load_meha_data()
    salt_df = load_salt_data()
    validation_rules = load_validation_rules()

    use_paano = salt_method == 'paano'
    if use_paano:
        from detect_salt_paano import detect_salt_paano_starts, load_salt_intervals as load_salt_intervals_paano

    legacy_gt = load_legacy_ground_truth()
    legacy_wells = legacy_gt['well_id'].unique().tolist()

    meha_intervals = {}
    meha_wells = []
    if Path('db/meha_intervals.csv').exists():
        meha_intervals = load_meha_intervals()
        meha_wells = sorted(meha_intervals.keys())

    salt_intervals = pd.DataFrame()
    if Path('db/salt_intervals.csv').exists():
        salt_intervals = load_salt_intervals()

    wells = legacy_wells + meha_wells

    results = []
    print("\n--- Processing ALL Wells (Universal Detection) ---")

    for wid in wells:
        print(f"Analyzing {wid}...")
        pred_type, detected_time, detail = detect_any_anomaly(
            wid,
            legacy_df,
            meha_df,
            validation_rules=validation_rules,
            meha_intervals=meha_intervals,
        )

        actual_type = None
        actual_start = None
        actual_end = None
        if wid in legacy_wells:
            row = legacy_gt[legacy_gt['well_id'] == wid].iloc[0]
            actual_type = 'Negermet' if row['anomaly_type'] == 'Негермет' else 'Pritok'
            actual_start = row['actual_start']
            actual_end = row['actual_end']
        elif wid in meha_intervals:
            actual_type = 'Meha'
            actual_start, actual_end = meha_intervals[wid]

        if detected_time is None:
            status = 'Not found'
        elif actual_type == 'Meha' and actual_start is not None and actual_end is not None:
            status = 'Detected' if actual_start <= detected_time <= actual_end else 'Out of interval'
        elif actual_type is not None and pred_type is not None and pred_type != actual_type:
            status = f"Wrong type ({pred_type})"
        else:
            status = 'Detected'

        results.append({
            'well_id': wid,
            'interval_idx': np.nan,
            'type': pred_type,
            'detected_time': detected_time,
            'actual_type': actual_type,
            'actual_start': actual_start,
            'actual_end': actual_end,
            'status': status,
            'detail': detail,
        })
        print(f"  -> Type: {pred_type} | Detected: {detected_time} | Status: {status}")

    if not salt_intervals.empty and not salt_df.empty:
        method_label = 'PaAno' if use_paano else 'Classic'
        print(f"\n--- Processing Salt Intervals ({method_label}) ---")
        prestart_tolerance = pd.Timedelta(hours=6)
        early_status_tolerance = pd.Timedelta(hours=6)
        for wid, grp in salt_intervals.groupby('well_id', sort=True):
            wid = str(wid).strip().lower()
            well_data = salt_df[salt_df['well_id'] == wid]
            if use_paano:
                pred_starts, blind_detail = detect_salt_paano_starts(
                    well_data, intervals_df=salt_intervals, well_id=wid)
            else:
                pred_starts, blind_detail = detect_salt_starts(well_data)
            pred_starts = sorted(pred_starts)
            used = [False] * len(pred_starts)

            print(f"Analyzing Salt {wid} (blind predictions: {len(pred_starts)})...")
            for _, row in grp.sort_values(['start_date', 'interval_idx']).iterrows():
                interval_idx = int(row.get('interval_idx', 1))
                start_dt = row['start_date']
                end_dt = row['end_date']

                detected_time = None
                candidates = []
                for i, ts in enumerate(pred_starts):
                    if used[i]:
                        continue
                    if (start_dt - prestart_tolerance) <= ts <= end_dt:
                        # Match to the closest blind candidate to interval start
                        # to avoid systematic late picks from "first-in-interval" logic.
                        dist = abs((ts - start_dt).total_seconds())
                        candidates.append((dist, i, ts))

                if candidates:
                    _, idx, detected_time = min(candidates, key=lambda x: x[0])
                    used[idx] = True

                    # If the closest candidate is too early, but there is a reasonably close
                    # non-early candidate, prefer the non-early one.
                    late_switch_window = pd.Timedelta(hours=54)
                    if detected_time < start_dt - early_status_tolerance:
                        non_early = [
                            (i, ts)
                            for i, ts in enumerate(pred_starts)
                            if (not used[i]) and (start_dt <= ts <= min(end_dt, start_dt + late_switch_window))
                        ]
                        if non_early:
                            repl_idx, repl_ts = min(
                                non_early,
                                key=lambda x: abs((x[1] - start_dt).total_seconds()),
                            )
                            # release old candidate and lock replacement
                            used[idx] = False
                            used[repl_idx] = True
                            detected_time = repl_ts

                if detected_time is None:
                    status = 'Not found'
                elif detected_time < start_dt - early_status_tolerance:
                    status = 'Early detected'
                else:
                    status = 'Detected'
                detail = blind_detail
                if detected_time is None:
                    detail = f"{blind_detail}; no blind start matched this interval"

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
                print(f"  -> Type: Salt | Interval: {interval_idx} | Detected: {detected_time} | Status: {status}")

    res_df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(res_df)
    res_df.to_csv(output_path, index=False)

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
        detected_time, detail = detect_meha(well_data, interval_start=start_dt)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--well-id', type=str, default=None)
    parser.add_argument('--random-well', action='store_true')
    parser.add_argument('--output', type=str, default='anomaly_detection_results.csv')
    parser.add_argument('--salt-method', type=str, default='classic',
                        choices=['classic', 'paano'],
                        help='Salt detection method: classic (rolling stats) or paano (AI)')
    args = parser.parse_args()

    if args.well_id is not None or args.random_well:
        legacy_df = load_data()
        meha_df = load_meha_data()
        validation_rules = load_validation_rules()

        legacy_gt = load_legacy_ground_truth()
        legacy_wells = legacy_gt['well_id'].unique().tolist()
        meha_intervals = load_meha_intervals() if Path('db/meha_intervals.csv').exists() else {}
        meha_wells = sorted(meha_intervals.keys())
        wells = legacy_wells + meha_wells

        wid = normalize_well_id(args.well_id) if args.well_id is not None else random.choice(wells)
        pred_type, detected_time, _detail = detect_any_anomaly(
            wid,
            legacy_df,
            meha_df,
            validation_rules=validation_rules,
            meha_intervals=meha_intervals,
        )

        print(pred_type)
        print(detected_time)
        return

    run_universal_detection(output_path=args.output, salt_method=args.salt_method)

if __name__ == "__main__":
    main()
