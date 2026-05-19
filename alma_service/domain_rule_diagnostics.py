from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alma_service.engineered_features import PreparedWellData


PRESSURE_ALIASES = (
    "Давление на приеме насоса кгс/см²",
    "Давление на приёме насоса кгс/см²",
    "Давление на приеме насоса",
    "Давление на приёме насоса",
    "Давление на выкиде ЭЦН",
)
FREQUENCY_ALIASES = (
    "Выходная частота",
    "Частота",
)


@dataclass(frozen=True)
class DomainRuleWindow:
    pre_hours: float
    post_hours: float


def interval_domain_rule_diagnostics(
    prepared: PreparedWellData | None,
    anomaly_key: str,
    actual_start: Any,
    actual_end: Any = None,
) -> dict[str, Any]:
    out = _empty_output(anomaly_key)
    if prepared is None:
        out["domain_rule_status"] = "missing_prepared_well"
        return out

    start_ts = _to_timestamp(actual_start)
    if start_ts is None:
        out["domain_rule_status"] = "missing_actual_start"
        return out

    times = pd.to_datetime(pd.Series(prepared.timestamps), errors="coerce")
    if times.empty or times.isna().all():
        out["domain_rule_status"] = "missing_timestamps"
        return out

    duration_hours = _interval_duration_hours(start_ts, _to_timestamp(actual_end), anomaly_key)
    window = _diagnostic_window(anomaly_key, duration_hours)
    out["domain_rule_pre_window_hours"] = window.pre_hours
    out["domain_rule_post_window_hours"] = window.post_hours

    pre_start = start_ts - pd.Timedelta(hours=window.pre_hours)
    post_end = start_ts + pd.Timedelta(hours=window.post_hours)
    end_ts = _to_timestamp(actual_end)
    if end_ts is not None and end_ts > start_ts:
        post_end = min(post_end, end_ts)

    pre_mask = (times >= pre_start) & (times < start_ts)
    post_mask = (times >= start_ts) & (times <= post_end)
    out["domain_rule_pre_points"] = int(pre_mask.sum())
    out["domain_rule_post_points"] = int(post_mask.sum())

    pressure = _channel_values(prepared, PRESSURE_ALIASES)
    frequency = _channel_values(prepared, FREQUENCY_ALIASES)
    if pressure is None:
        out["domain_rule_status"] = "missing_pressure_channel"
        return out

    pressure_pre_slope = _slope_per_day(times, pressure, pre_mask)
    pressure_post_slope = _slope_per_day(times, pressure, post_mask)
    pressure_slope_change = _nan_subtract(pressure_post_slope, pressure_pre_slope)
    pressure_delta_pct = _median_delta_pct(pressure, pre_mask, post_mask)

    out["pressure_pre_slope_per_day"] = _float_or_none(pressure_pre_slope)
    out["pressure_post_slope_per_day"] = _float_or_none(pressure_post_slope)
    out["pressure_slope_change_per_day"] = _float_or_none(pressure_slope_change)
    out["pressure_post_vs_pre_median_pct"] = _float_or_none(pressure_delta_pct)
    out["pressure_pre_slope_direction"] = _direction(pressure_pre_slope)
    out["pressure_post_slope_direction"] = _direction(pressure_post_slope)
    out["trend_reversal_score"] = _float_or_none(
        _trend_reversal_score(pressure_pre_slope, pressure_post_slope)
    )

    if frequency is not None:
        frequency_delta_pct = _median_delta_pct(frequency, pre_mask, post_mask)
        out["frequency_post_vs_pre_median_pct"] = _float_or_none(frequency_delta_pct)
        out["frequency_stability_score"] = _float_or_none(_stability_score(frequency_delta_pct))
    else:
        out["frequency_post_vs_pre_median_pct"] = None
        out["frequency_stability_score"] = None

    if out["domain_rule_pre_points"] < 2:
        out["domain_rule_status"] = "insufficient_pre_window"
    elif out["domain_rule_post_points"] < 2:
        out["domain_rule_status"] = "insufficient_post_window"
    else:
        out["domain_rule_status"] = "ok"
    return out


def attach_domain_rule_diagnostics(
    records: list[dict[str, Any]],
    prepared_runs: Mapping[str, PreparedWellData],
    anomaly_key: str,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        well_id = str(record.get("well_id", ""))
        diagnostics = interval_domain_rule_diagnostics(
            prepared_runs.get(well_id),
            anomaly_key,
            record.get("actual_start"),
            record.get("actual_end"),
        )
        enriched.append({**record, **diagnostics})
    return enriched


def _empty_output(anomaly_key: str) -> dict[str, Any]:
    return {
        "domain_rule_anomaly": str(anomaly_key),
        "domain_rule_status": "not_computed",
        "domain_rule_pre_window_hours": None,
        "domain_rule_post_window_hours": None,
        "domain_rule_pre_points": 0,
        "domain_rule_post_points": 0,
        "pressure_pre_slope_per_day": None,
        "pressure_post_slope_per_day": None,
        "pressure_slope_change_per_day": None,
        "pressure_post_vs_pre_median_pct": None,
        "pressure_pre_slope_direction": "unknown",
        "pressure_post_slope_direction": "unknown",
        "trend_reversal_score": None,
        "frequency_post_vs_pre_median_pct": None,
        "frequency_stability_score": None,
    }


def _diagnostic_window(anomaly_key: str, duration_hours: float) -> DomainRuleWindow:
    duration_hours = max(float(duration_hours), 0.0)
    if anomaly_key == "negermet":
        return DomainRuleWindow(
            pre_hours=min(24.0, max(3.0, duration_hours)),
            post_hours=min(6.0, max(1.0, duration_hours * 0.25)),
        )
    return DomainRuleWindow(
        pre_hours=min(14.0 * 24.0, max(24.0, duration_hours * 0.5)),
        post_hours=min(7.0 * 24.0, max(24.0, duration_hours * 0.2)),
    )


def _interval_duration_hours(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None,
    anomaly_key: str,
) -> float:
    if end_ts is not None and end_ts > start_ts:
        return float((end_ts - start_ts).total_seconds() / 3600.0)
    if anomaly_key == "negermet":
        return 24.0
    if anomaly_key == "pritok":
        return 14.0 * 24.0
    return 21.0 * 24.0


def _channel_values(prepared: PreparedWellData, aliases: tuple[str, ...]) -> np.ndarray | None:
    index = _find_channel_index(prepared.raw_columns, aliases)
    if index is None:
        return None
    return np.asarray(prepared.raw_matrix[:, index], dtype=float)


def _find_channel_index(columns: list[str], aliases: tuple[str, ...]) -> int | None:
    normalized_aliases = [_normalize_name(alias) for alias in aliases]
    normalized_columns = [_normalize_name(column) for column in columns]
    for alias in normalized_aliases:
        for idx, column in enumerate(normalized_columns):
            if column == alias:
                return idx
    for alias in normalized_aliases:
        for idx, column in enumerate(normalized_columns):
            if alias and alias in column:
                return idx
    return None


def _normalize_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("ё", "е")
        .replace("²", "2")
        .replace(" ", "")
    )


def _to_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _slope_per_day(times: pd.Series, values: np.ndarray, mask: pd.Series) -> float:
    valid = mask.to_numpy(dtype=bool) & np.isfinite(values) & times.notna().to_numpy(dtype=bool)
    if int(valid.sum()) < 2:
        return float("nan")
    selected_times = times[valid]
    selected_values = values[valid]
    x_days = (
        selected_times - selected_times.iloc[0]
    ).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    if not np.isfinite(x_days).any() or float(np.nanmax(x_days) - np.nanmin(x_days)) <= 0.0:
        return float("nan")
    return float(np.polyfit(x_days, selected_values.astype(float), deg=1)[0])


def _median_delta_pct(values: np.ndarray, pre_mask: pd.Series, post_mask: pd.Series) -> float:
    pre_values = values[pre_mask.to_numpy(dtype=bool)]
    post_values = values[post_mask.to_numpy(dtype=bool)]
    pre_values = pre_values[np.isfinite(pre_values)]
    post_values = post_values[np.isfinite(post_values)]
    if len(pre_values) == 0 or len(post_values) == 0:
        return float("nan")
    pre_median = float(np.nanmedian(pre_values))
    post_median = float(np.nanmedian(post_values))
    denom = abs(pre_median)
    if denom < 1e-9:
        return float("nan")
    return float((post_median - pre_median) / denom * 100.0)


def _trend_reversal_score(pre_slope: float, post_slope: float) -> float:
    if not np.isfinite(pre_slope) or not np.isfinite(post_slope):
        return float("nan")
    denom = abs(pre_slope) + abs(post_slope)
    if denom <= 1e-9 or pre_slope * post_slope >= 0:
        return 0.0
    return float(min(100.0, abs(post_slope - pre_slope) / denom * 100.0))


def _stability_score(delta_pct: float) -> float:
    if not np.isfinite(delta_pct):
        return float("nan")
    return float(max(0.0, 100.0 - min(100.0, abs(delta_pct) * 20.0)))


def _nan_subtract(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return float(left - right)


def _direction(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if np.isclose(value, 0.0, atol=1e-9):
        return "flat"
    return "up" if value > 0 else "down"


def _float_or_none(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value)
