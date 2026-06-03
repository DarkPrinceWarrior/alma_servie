from __future__ import annotations

import numpy as np

from alma_service.telemetry_status import _is_disconnected_channel, _stale_mask


def test_constant_channel_is_disconnected() -> None:
    # Неподключённый датчик: одно значение на всём ряду.
    assert _is_disconnected_channel(np.full(200, 5.0))


def test_all_nan_channel_is_disconnected() -> None:
    assert _is_disconnected_channel(np.full(200, np.nan))


def test_near_constant_channel_is_disconnected() -> None:
    # 99% точек застыли — неподключённый/мёртвый канал.
    values = np.r_[np.full(199, 5.0), [6.0]]
    assert _is_disconnected_channel(values)


def test_real_signal_is_not_disconnected() -> None:
    assert not _is_disconnected_channel(np.linspace(50.0, 55.0, 200))


def test_slow_pressure_with_repeats_is_not_disconnected() -> None:
    # Медленное реальное давление с повторами из-за дискретизации (75% застыло) —
    # НЕ неподключённый канал, должно участвовать в анализе.
    values = np.repeat(np.linspace(50.0, 55.0, 50), 4)
    assert not _is_disconnected_channel(values)


def test_local_stall_still_flagged() -> None:
    # Локальное залипание реального сигнала (датчик завис на время) — sensor_stuck работает.
    values = np.r_[np.linspace(50.0, 55.0, 100), np.full(50, 55.0)]
    assert not _is_disconnected_channel(values)
    stale = _stale_mask(values, min_run=12)
    assert stale[-1]  # хвост залип
