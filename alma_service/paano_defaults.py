from __future__ import annotations

# Per-anomaly patch sizes for paano_shared: (short, long) adapted to anomaly speed.
# Актуальные сетки в db/: negermet=5min, pritok=10min+5min, salt=10min+5min.
# Глобальный детектор paano_global эти значения не использует — у него patch=(192, 384)
# на общей 5min-сетке из configs/alma_global_normality_5min.json
# (+ prepare_patch_size_overrides: negermet=52, pritok/salt=384).
PATCH_SIZES = {
    "negermet": (64, 128),   # fast anomaly (hours)
    "pritok":   (96, 192),   # slow anomaly (weeks)
    "salt":     (64, 128),   # slow anomaly (weeks)
}
# Fallback defaults
SHORT_PATCH = 64
LONG_PATCH = 128

REFERENCE_MIN_RATIO = 0.20
REFERENCE_MAX_RATIO = 0.40
REFERENCE_MIN_DAYS = 0.50
MIN_REFERENCE_COVERAGE = 0.80
MIN_TOTAL_COVERAGE = 0.60

PRESTART_TOLERANCE_HOURS = 2.0

DEFAULT_CONFIG = {
    "fusion_weight_short": 0.60,
    "target_far_per_day": 0.50,
    "min_run_points": 3,
    "cooldown_hours": 8.0,
    "ema_alpha": 0.08,
    "gate_mode": "score_ema",
}

TUNE_GRID = {
    "fusion_weight_short": [0.40, 0.60, 0.75],
    "target_far_per_day": [0.25, 0.50, 1.00],
    "min_run_points": [2, 3, 4, 6],
    "cooldown_hours": [4.0, 8.0, 12.0, 24.0],
    "ema_alpha": [0.04, 0.08, 0.12],
    "gate_mode": ["score_ema", "relaxed", "strict"],
}
