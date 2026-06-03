from __future__ import annotations

import numpy as np
import pandas as pd

from alma_service.pressure_trend_onset import (
    PressureTrendFusionConfig,
    detect_pressure_trend_fusion_onsets,
)


def _series(days: int, freq_min: int = 720):
    n = days * (1440 // freq_min)
    ts = pd.date_range("2026-01-01", periods=n, freq=f"{freq_min}min").to_numpy()
    return ts, n


def test_slow_pritok_with_high_score_is_detected() -> None:
    # Медленный рост давления +0.35%/сут при стабильной частоте и повышенном скоре.
    ts, n = _series(40)
    ref = n // 5
    pressure = np.r_[np.full(ref, 50.0), np.linspace(50.0, 50.0 * 1.14, n - ref)]
    frequency = np.full(n, 196.0)
    score = np.r_[np.full(ref, 0.002), np.full(n - ref, 0.006)]
    ref_mask = np.zeros(n, dtype=bool); ref_mask[:ref] = True
    cfg = PressureTrendFusionConfig()
    starts = detect_pressure_trend_fusion_onsets(ts, pressure, frequency, score, ref_mask, None, cfg)
    assert len(starts) >= 1


def test_slow_trend_with_low_score_is_rejected() -> None:
    # Тот же наклон давления, но скор низкий (норма с шумовым трендом) — НЕ детекция.
    ts, n = _series(40)
    ref = n // 5
    pressure = np.r_[np.full(ref, 50.0), np.linspace(50.0, 50.0 * 1.14, n - ref)]
    frequency = np.full(n, 196.0)
    score = np.full(n, 0.0028)  # ниже порога 0.0040
    ref_mask = np.zeros(n, dtype=bool); ref_mask[:ref] = True
    cfg = PressureTrendFusionConfig()
    starts = detect_pressure_trend_fusion_onsets(ts, pressure, frequency, score, ref_mask, None, cfg)
    assert starts == []


def test_high_score_without_trend_is_rejected() -> None:
    # Скор высокий, но давление стабильно (нет тренда) — НЕ детекция (не приток).
    ts, n = _series(40)
    ref = n // 5
    pressure = np.full(n, 50.0)
    frequency = np.full(n, 196.0)
    score = np.r_[np.full(ref, 0.002), np.full(n - ref, 0.006)]
    ref_mask = np.zeros(n, dtype=bool); ref_mask[:ref] = True
    cfg = PressureTrendFusionConfig()
    starts = detect_pressure_trend_fusion_onsets(ts, pressure, frequency, score, ref_mask, None, cfg)
    assert starts == []


def test_frequency_jump_buffer_rejects_regime_reaction() -> None:
    # Скачок частоты в середине -> давление реагирует трендово, скор высокий.
    # Буфер ±2 суток должен отсечь эту реакцию (случай 305г из LOO).
    ts, n = _series(40)
    ref = n // 5
    pressure = np.r_[np.full(ref, 50.0), np.full(n - ref, 50.0)]
    half = n // 2
    pressure[half:] = np.linspace(50.0, 50.0 * 1.10, n - half)  # реакция после скачка
    frequency = np.full(n, 196.0)
    frequency[half:] = 199.0  # скачок частоты +3 Гц
    score = np.r_[np.full(ref, 0.002), np.full(n - ref, 0.006)]
    ref_mask = np.zeros(n, dtype=bool); ref_mask[:ref] = True
    cfg = PressureTrendFusionConfig()
    starts = detect_pressure_trend_fusion_onsets(ts, pressure, frequency, score, ref_mask, None, cfg)
    # старты не должны попадать в буфер ±2 суток вокруг скачка
    jump_ts = pd.Timestamp(ts[half])
    for s in starts:
        assert abs((s - jump_ts).total_seconds()) > 2 * 86400 or s > jump_ts + pd.Timedelta(days=2)
