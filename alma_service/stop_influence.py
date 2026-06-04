from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

STOP_FREQUENCY_THRESHOLD_HZ = 1.0
STOP_MIN_SAMPLES = 2
FREQ_RECOVERY_RATIO = 0.95
# Зона остановки тянется, пока давление не вернётся почти к base (0.5%). При 1.05 (5%)
# зона обрывалась на спаде пика и оставляла видимый хвост (46-806: пик до 92.7, обрыв на
# 78.99 при base ~75.4); 1.005 доводит обрезку до возврата к норме (~75.8).
PRESSURE_RECOVERY_RATIO = 1.005
BASE_WINDOW_HOURS = 12.0
MAX_TAIL_HOURS = 36.0


@dataclass(frozen=True)
class StopInfluenceZone:
    start: pd.Timestamp
    end: pd.Timestamp
    core_start: pd.Timestamp
    core_end: pd.Timestamp
    excess_pct: float
    pressure_recovered: bool


def _normalize_column(name: str) -> str:
    return str(name).strip().lower().replace("ё", "е")


def find_channel(columns: Any, *needles: str) -> str | None:
    for column in columns:
        normalized = _normalize_column(column)
        if all(_normalize_column(needle) in normalized for needle in needles):
            return str(column)
    return None


def detect_stop_influence_zones(
    timestamps: pd.Series | np.ndarray,
    frequency: pd.Series | np.ndarray,
    pressure: pd.Series | np.ndarray,
) -> list[StopInfluenceZone]:
    frame = pd.DataFrame(
        {
            "frequency": np.asarray(frequency, dtype=float),
            "pressure": np.asarray(pressure, dtype=float),
        },
        index=pd.to_datetime(np.asarray(timestamps)),
    ).dropna().sort_index()
    if len(frame) < 50:
        return []
    freq = frame["frequency"]
    pressure_series = frame["pressure"]
    stopped = freq < STOP_FREQUENCY_THRESHOLD_HZ
    if not stopped.any():
        return []

    base_window = pd.Timedelta(hours=BASE_WINDOW_HOURS)
    max_tail = pd.Timedelta(hours=MAX_TAIL_HOURS)
    groups = (stopped != stopped.shift()).cumsum()
    zones: list[StopInfluenceZone] = []
    for _, core in freq[stopped].groupby(groups[stopped]):
        if len(core) < STOP_MIN_SAMPLES:
            continue
        core_start, core_end = core.index[0], core.index[-1]
        before_freq = freq.loc[core_start - base_window: core_start]
        before_freq = before_freq[before_freq >= STOP_FREQUENCY_THRESHOLD_HZ]
        before_pressure = pressure_series.loc[core_start - base_window: core_start]
        if before_freq.empty or before_pressure.empty:
            continue
        working_freq = float(before_freq.median())
        base_pressure = float(before_pressure.median())

        normal_before = freq.loc[:core_start]
        normal_before = normal_before[normal_before >= working_freq * FREQ_RECOVERY_RATIO]
        zone_start = normal_before.index[-1] if len(normal_before) else core_start

        after = frame.loc[core_end:]
        recovered_mask = (after["frequency"] >= working_freq * FREQ_RECOVERY_RATIO) & (
            after["pressure"] <= base_pressure * PRESSURE_RECOVERY_RATIO
        )
        recovered_times = after.index[recovered_mask]
        limit = core_end + max_tail
        if len(recovered_times) and recovered_times[0] <= limit:
            zone_end = recovered_times[0]
            pressure_recovered = True
        else:
            zone_end = min(frame.index[-1], limit)
            pressure_recovered = False

        zone_pressure = pressure_series.loc[zone_start:zone_end]
        peak_pressure = float(zone_pressure.max()) if len(zone_pressure) else base_pressure
        excess_pct = (peak_pressure / base_pressure - 1.0) * 100.0 if base_pressure > 0 else 0.0
        zones.append(
            StopInfluenceZone(
                start=zone_start,
                end=zone_end,
                core_start=core_start,
                core_end=core_end,
                excess_pct=excess_pct,
                pressure_recovered=pressure_recovered,
            )
        )

    merged: list[dict[str, Any]] = []
    for zone in sorted(zones, key=lambda z: z.start):
        if merged and zone.start <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], zone.end)
            merged[-1]["core_end"] = max(merged[-1]["core_end"], zone.core_end)
            merged[-1]["excess_pct"] = max(merged[-1]["excess_pct"], zone.excess_pct)
            merged[-1]["pressure_recovered"] = merged[-1]["pressure_recovered"] and zone.pressure_recovered
        else:
            merged.append(
                {
                    "start": zone.start,
                    "end": zone.end,
                    "core_start": zone.core_start,
                    "core_end": zone.core_end,
                    "excess_pct": zone.excess_pct,
                    "pressure_recovered": zone.pressure_recovered,
                }
            )
    return [StopInfluenceZone(**item) for item in merged]


def detect_zones_from_frame(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> list[StopInfluenceZone]:
    frequency_col = find_channel(frame.columns, "выходная частота")
    pressure_col = find_channel(frame.columns, "давление на приеме")
    if frequency_col is None or pressure_col is None or timestamp_column not in frame.columns:
        return []
    return detect_stop_influence_zones(frame[timestamp_column], frame[frequency_col], frame[pressure_col])


def stop_influence_mask(
    timestamps: pd.Series | np.ndarray,
    zones: list[StopInfluenceZone],
) -> np.ndarray:
    ts = pd.to_datetime(np.asarray(timestamps))
    inside = np.zeros(len(ts), dtype=bool)
    for zone in zones:
        inside |= np.asarray((ts >= zone.start) & (ts <= zone.end), dtype=bool)
    return inside