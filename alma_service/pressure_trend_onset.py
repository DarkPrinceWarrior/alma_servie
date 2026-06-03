from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alma_service.stop_influence import detect_stop_influence_zones, find_channel, stop_influence_mask

# Параметры по умолчанию — из калибровки и строгого LOO (03.06.2026).
DEFAULT_WINDOW_DAYS = 10.0
DEFAULT_SLOPE_THRESHOLD_PCT_PER_DAY = 0.15
DEFAULT_SCORE_THRESHOLD = 0.0040
DEFAULT_FREQ_STABLE_RANGE_HZ = 2.0
DEFAULT_FREQ_JUMP_THRESHOLD_HZ = 0.5
DEFAULT_FREQ_JUMP_BUFFER_DAYS = 2.0
DEFAULT_MIN_RUN = 3
DEFAULT_RESAMPLE = "12h"


@dataclass(frozen=True)
class PressureTrendFusionConfig:
    window_days: float = DEFAULT_WINDOW_DAYS
    slope_threshold_pct_per_day: float = DEFAULT_SLOPE_THRESHOLD_PCT_PER_DAY
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    freq_stable_range_hz: float = DEFAULT_FREQ_STABLE_RANGE_HZ
    freq_jump_threshold_hz: float = DEFAULT_FREQ_JUMP_THRESHOLD_HZ
    freq_jump_buffer_days: float = DEFAULT_FREQ_JUMP_BUFFER_DAYS
    min_run: int = DEFAULT_MIN_RUN

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PressureTrendFusionConfig":
        payload = payload or {}
        return cls(
            window_days=float(payload.get("window_days", DEFAULT_WINDOW_DAYS)),
            slope_threshold_pct_per_day=float(payload.get("slope_threshold_pct_per_day", DEFAULT_SLOPE_THRESHOLD_PCT_PER_DAY)),
            score_threshold=float(payload.get("score_threshold", DEFAULT_SCORE_THRESHOLD)),
            freq_stable_range_hz=float(payload.get("freq_stable_range_hz", DEFAULT_FREQ_STABLE_RANGE_HZ)),
            freq_jump_threshold_hz=float(payload.get("freq_jump_threshold_hz", DEFAULT_FREQ_JUMP_THRESHOLD_HZ)),
            freq_jump_buffer_days=float(payload.get("freq_jump_buffer_days", DEFAULT_FREQ_JUMP_BUFFER_DAYS)),
            min_run=int(payload.get("min_run", DEFAULT_MIN_RUN)),
        )


def _frequency_jump_buffer_mask(freq12: pd.Series, *, threshold_hz: float, buffer_days: float) -> pd.Series:
    # Окна ±buffer_days вокруг скачков выходной частоты — исключаются, чтобы не считать
    # реакцию давления на смену режима за приток (доводка по результату LOO: скв. 305г, 1995).
    jumps = freq12.index[freq12.diff().abs() > threshold_hz]
    blocked = pd.Series(False, index=freq12.index)
    if len(jumps) == 0:
        return blocked
    buffer = pd.Timedelta(days=buffer_days)
    for jump_ts in jumps:
        blocked |= (freq12.index >= jump_ts - buffer) & (freq12.index <= jump_ts + buffer)
    return blocked


def detect_pressure_trend_fusion_onsets(
    timestamps: np.ndarray,
    pressure: np.ndarray,
    frequency: np.ndarray,
    score: np.ndarray,
    reference_mask: np.ndarray,
    onset_mask: np.ndarray | None,
    config: PressureTrendFusionConfig,
) -> list[pd.Timestamp]:
    """Совмещённый (fusion) трендовый детектор притока: устойчивый наклон давления
    при стабильной частоте И повышенный скор нейросети. Возвращает старты эпизодов."""
    ts = pd.to_datetime(np.asarray(timestamps))
    frame = pd.DataFrame(
        {
            "p": np.asarray(pressure, dtype=float),
            "f": np.asarray(frequency, dtype=float),
            "s": np.asarray(score, dtype=float),
        },
        index=ts,
    ).sort_index()

    ref_bool = np.asarray(reference_mask, dtype=bool)
    if ref_bool.any():
        ref_end_ts = ts[np.flatnonzero(ref_bool)[-1]]
    else:
        ref_end_ts = ts[0]
    onset_bool = np.ones(len(ts), dtype=bool) if onset_mask is None else np.asarray(onset_mask, dtype=bool)
    onset_series = pd.Series(onset_bool, index=ts)

    # Зоны влияния остановок: давление и частота внутри них не участвуют в тренде.
    zones = detect_stop_influence_zones(frame.index, frame["f"], frame["p"])
    if zones:
        inside = stop_influence_mask(frame.index, zones)
        frame.loc[inside, ["p", "f"]] = np.nan

    p12 = frame["p"].resample(DEFAULT_RESAMPLE).median()
    f12 = frame["f"].resample(DEFAULT_RESAMPLE).median()
    s12 = frame["s"].resample(DEFAULT_RESAMPLE).median()
    valid = p12.notna()
    p12, f12, s12 = p12[valid], f12[valid], s12[valid]
    if len(p12) < 6:
        return []

    base_window = p12[p12.index <= ref_end_ts]
    base = float(base_window.median()) if len(base_window) else float(p12.iloc[: max(2, len(p12) // 5)].median())
    if not np.isfinite(base) or base <= 0:
        return []

    freq_blocked = _frequency_jump_buffer_mask(
        f12, threshold_hz=config.freq_jump_threshold_hz, buffer_days=config.freq_jump_buffer_days
    )
    window = pd.Timedelta(days=config.window_days)

    run = 0
    starts: list[pd.Timestamp] = []
    armed = True
    for current in p12.index:
        if current <= ref_end_ts:
            continue
        if bool(freq_blocked.loc[current]):
            run = 0
            continue
        seg_p = p12[(p12.index > current - window) & (p12.index <= current)]
        seg_f = f12[(f12.index > current - window) & (f12.index <= current)]
        seg_s = s12[(s12.index > current - window) & (s12.index <= current)]
        if len(seg_p) < 4:
            run = 0
            continue
        if seg_f.notna().sum() >= 2 and float(seg_f.max() - seg_f.min()) > config.freq_stable_range_hz:
            run = 0
            continue
        x = (seg_p.index - seg_p.index[0]).total_seconds().to_numpy(dtype=float) / 86400.0
        slope_pct = float(np.polyfit(x, seg_p.to_numpy(dtype=float), 1)[0]) / base * 100.0
        score_med = float(seg_s.median()) if seg_s.notna().any() else 0.0
        trend_ok = abs(slope_pct) >= config.slope_threshold_pct_per_day
        score_ok = score_med >= config.score_threshold
        onset_ok = bool(onset_series[(onset_series.index > current - window) & (onset_series.index <= current)].any()) \
            if onset_mask is not None else True
        if trend_ok and score_ok and onset_ok:
            run += 1
            if run >= config.min_run and armed:
                starts.append(pd.Timestamp(current))
                armed = False
        else:
            run = 0
            armed = True
    return starts


def detect_from_prepared(
    prepared: Any,
    score: np.ndarray,
    config: PressureTrendFusionConfig,
) -> list[pd.Timestamp]:
    raw_columns = list(getattr(prepared, "raw_columns", []))
    pressure_col = find_channel(raw_columns, "давление на приеме")
    frequency_col = find_channel(raw_columns, "выходная частота")
    if pressure_col is None or frequency_col is None:
        return []
    raw_matrix = np.asarray(prepared.raw_matrix, dtype=float)
    return detect_pressure_trend_fusion_onsets(
        prepared.timestamps,
        raw_matrix[:, raw_columns.index(pressure_col)],
        raw_matrix[:, raw_columns.index(frequency_col)],
        np.asarray(score, dtype=float),
        prepared.reference_mask,
        getattr(prepared, "onset_allowed_mask", None),
        config,
    )
