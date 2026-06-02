from __future__ import annotations

import numpy as np
import pandas as pd

from alma_service.stop_influence import detect_stop_influence_zones, stop_influence_mask


def test_zone_covers_stop_and_pressure_recovery() -> None:
    # 48 часов с шагом 5 минут; остановка 2 часа, давление возвращается к базе ещё ~2 часа
    idx = pd.date_range("2026-01-01", periods=576, freq="5min")
    freq = np.full(576, 180.0)
    pressure = np.full(576, 50.0)
    freq[288:312] = 0.0
    pressure[288:312] = np.linspace(50, 75, 24)
    pressure[312:336] = np.linspace(75, 50.5, 24)
    zones = detect_stop_influence_zones(pd.Series(idx), freq, pressure)
    assert len(zones) == 1
    zone = zones[0]
    assert zone.core_start == idx[288]
    assert zone.core_end == idx[311]
    # конец зоны — где давление вернулось к базе (50 * 1.05 = 52.5), а не где восстановилась частота
    assert zone.end > idx[312]
    assert zone.pressure_recovered
    mask = stop_influence_mask(pd.Series(idx), zones)
    assert mask[300] and mask[320]
    assert not mask[100] and not mask[400]


def test_no_zones_without_stops() -> None:
    idx = pd.date_range("2026-01-01", periods=576, freq="5min")
    zones = detect_stop_influence_zones(pd.Series(idx), np.full(576, 180.0), np.full(576, 50.0))
    assert zones == []


def test_close_stops_merge_into_one_zone() -> None:
    # две остановки с зазором 30 минут — давление между ними не успевает вернуться к базе
    idx = pd.date_range("2026-01-01", periods=576, freq="5min")
    freq = np.full(576, 180.0)
    pressure = np.full(576, 50.0)
    freq[288:300] = 0.0
    pressure[288:300] = np.linspace(50, 70, 12)
    pressure[300:306] = np.linspace(70, 65, 6)
    freq[306:318] = 0.0
    pressure[306:318] = np.linspace(65, 80, 12)
    pressure[318:360] = np.linspace(80, 50.0, 42)
    zones = detect_stop_influence_zones(pd.Series(idx), freq, pressure)
    assert len(zones) == 1
    assert zones[0].core_start == idx[288]
    assert zones[0].core_end == idx[317]