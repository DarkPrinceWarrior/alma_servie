from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alma_service.stop_influence import detect_stop_influence_zones, find_channel, stop_influence_mask

# Универсальный онсет-детектор СОБЫТИЯ негерметичности НКТ: резкая ОТНОСИТЕЛЬНАЯ
# ступень давления на приёме. Отметка ставится в момент скачка, а не заранее — у
# негермета нет предвестника (авария = само событие), поэтому ранняя метка эксперту
# необъяснима. Порог относительный (% от собственного уровня скважины) → самонормируется,
# один и тот же критерий работает на любой скважине и любом режиме давления, новые
# скважины не требуют подбора порогов. Частота в гейт НЕ входит (у негермет-скважин она
# ведёт себя по-разному: стабильна 524, растёт 172г, срывается вниз 3509г), общий признак —
# именно резкая ступень давления. Зоны влияния остановок исключаются (рост давления при
# остановке/перезапуске — гидростатика, не авария).

DEFAULT_RISE_PCT = 12.0          # относительный рост давления для срабатывания
DEFAULT_WINDOW_HOURS = 3.0       # причинное окно «резкости» (ступень за часы)
DEFAULT_MIN_RUN = 2              # подтверждающих подряд точек (анти-спайк)
DEFAULT_RESAMPLE = "15min"       # сетка медианы (баланс точность/шум)
DEFAULT_EXCLUDE_STOPS = True


@dataclass(frozen=True)
class PressureStepConfig:
    rise_pct: float = DEFAULT_RISE_PCT
    window_hours: float = DEFAULT_WINDOW_HOURS
    min_run: int = DEFAULT_MIN_RUN
    resample: str = DEFAULT_RESAMPLE
    exclude_stops: bool = DEFAULT_EXCLUDE_STOPS

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PressureStepConfig":
        payload = payload or {}
        return cls(
            rise_pct=float(payload.get("rise_pct", DEFAULT_RISE_PCT)),
            window_hours=float(payload.get("window_hours", DEFAULT_WINDOW_HOURS)),
            min_run=int(payload.get("min_run", DEFAULT_MIN_RUN)),
            resample=str(payload.get("resample", DEFAULT_RESAMPLE)),
            exclude_stops=bool(payload.get("exclude_stops", DEFAULT_EXCLUDE_STOPS)),
        )


def detect_pressure_step_onsets(
    timestamps: np.ndarray,
    pressure: np.ndarray,
    frequency: np.ndarray,
    reference_mask: np.ndarray,
    onset_mask: np.ndarray | None,
    config: PressureStepConfig,
) -> list[pd.Timestamp]:
    """Резкая относительная ступень давления вверх как онсет негермета. Метка — первая
    точка устойчивого (min_run подряд) превышения относительного роста над порогом,
    считая от минимума давления в причинном окне. Возвращает список меток (по эпизодам)."""
    ts = pd.to_datetime(np.asarray(timestamps))
    frame = pd.DataFrame(
        {"p": np.asarray(pressure, dtype=float), "f": np.asarray(frequency, dtype=float)},
        index=ts,
    ).sort_index()

    if config.exclude_stops:
        zones = detect_stop_influence_zones(frame.index, frame["f"], frame["p"])
        if zones:
            inside = stop_influence_mask(frame.index, zones)
            frame.loc[inside, ["p", "f"]] = np.nan

    p = frame["p"].resample(config.resample).median()
    p = p[p.notna()]
    if len(p) < 4:
        return []

    ref_bool = np.asarray(reference_mask, dtype=bool)
    ref_end_ts = ts[np.flatnonzero(ref_bool)[-1]] if ref_bool.any() else ts[0]
    onset_series = (
        pd.Series(np.asarray(onset_mask, dtype=bool), index=ts)
        if onset_mask is not None
        else None
    )

    window = pd.Timedelta(hours=config.window_hours)
    idx = p.index
    vals = p.to_numpy(dtype=float)

    onsets: list[pd.Timestamp] = []
    run = 0
    run_start: pd.Timestamp | None = None
    armed = True
    for i, current in enumerate(idx):
        if current <= ref_end_ts:
            continue
        if onset_series is not None:
            allowed = onset_series[(onset_series.index > current - window) & (onset_series.index <= current)]
            if not bool(allowed.any()):
                run = 0
                run_start = None
                continue
        seg = vals[(idx > current - window) & (idx <= current)]
        if seg.size < 2:
            run = 0
            run_start = None
            continue
        base = float(np.nanmin(seg))
        if not np.isfinite(base) or base <= 0:
            run = 0
            run_start = None
            continue
        rise_pct = (vals[i] - base) / base * 100.0
        if rise_pct >= config.rise_pct:
            if run == 0:
                run_start = current
            run += 1
            if run >= config.min_run and armed:
                onsets.append(pd.Timestamp(run_start))
                armed = False
        else:
            run = 0
            run_start = None
            armed = True
    return onsets


def detect_step_from_prepared(prepared: Any, config: PressureStepConfig) -> list[pd.Timestamp]:
    raw_columns = list(getattr(prepared, "raw_columns", []))
    pressure_col = find_channel(raw_columns, "давление на приеме")
    frequency_col = find_channel(raw_columns, "выходная частота")
    if pressure_col is None:
        return []
    raw_matrix = np.asarray(prepared.raw_matrix, dtype=float)
    freq = (
        raw_matrix[:, raw_columns.index(frequency_col)]
        if frequency_col is not None
        else np.zeros(raw_matrix.shape[0], dtype=float)
    )
    return detect_pressure_step_onsets(
        prepared.timestamps,
        raw_matrix[:, raw_columns.index(pressure_col)],
        freq,
        prepared.reference_mask,
        getattr(prepared, "onset_allowed_mask", None),
        config,
    )
