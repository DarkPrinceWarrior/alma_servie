from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alma_service.engineered_features import (
    FREQ_COL,
    OUTPUT_CURRENT_COL,
    PHASE_CURRENT_COLS,
    POWER_COL,
    PRESSURE_COL,
    PreparedWellData,
)


DOMAIN_ACCEPTED = "accepted"
DOMAIN_REJECTED_BAD_DATA = "rejected_bad_data"
DOMAIN_REJECTED_REGIME_EVENT = "rejected_regime_event"
DOMAIN_REJECTED_FREQUENCY_TRANSITION = "rejected_frequency_transition"
DOMAIN_REJECTED_WELL_STOPPED = "rejected_well_stopped"
DOMAIN_PRITOK_CANDIDATE = "pritok_candidate"
DOMAIN_SALT_CANDIDATE = "salt_candidate"
DOMAIN_NEGERMET_CANDIDATE = "negermet_candidate"
DOMAIN_UNCERTAIN = "uncertain"
DOMAIN_NO_DATA = "no_data"

# Остановка скважины: рабочая выходная частота УЭЦН в данных проекта — 170-225 Гц.
# Частота ниже порога означает стоящий насос; рост давления при стоящем насосе —
# гидростатическое восстановление, а не сигнатура аномалии (правило эксперта:
# остановка не считается аномалией; подтверждено случаем 5271г).
WELL_STOPPED_FREQUENCY_THRESHOLD = 1.0
WELL_STOPPED_FRACTION_THRESHOLD = 0.5

TEMPERATURE_COLUMNS = (
    "Температура на приёме насоса",
    "Температура на приеме насоса",
    "Температура масла двигателя",
)
LOAD_COLUMNS = (
    "Коэффициент загрузки ПЭД",
    "Активная выходная мощность",
    POWER_COL,
    OUTPUT_CURRENT_COL,
    *PHASE_CURRENT_COLS,
)
IMBALANCE_COLUMNS = (
    "Дисбаланс токов",
    "Дисбаланс напряжений",
)


@dataclass(frozen=True)
class DomainDecisionConfig:
    pre_hours: float
    post_hours: float
    frequency_transition_pct: float
    pressure_delta_pct: float
    min_points: int = 3
    strong_pressure_delta_pct: float = 5.0
    support_delta_pct: float = 1.0
    min_slope_abs_per_day: float = 0.1


DEFAULT_CONFIGS = {
    "negermet": DomainDecisionConfig(
        pre_hours=2.0,
        post_hours=2.0,
        frequency_transition_pct=3.0,
        pressure_delta_pct=1.0,
        strong_pressure_delta_pct=5.0,
        support_delta_pct=1.0,
        min_slope_abs_per_day=1.0,
    ),
    "pritok": DomainDecisionConfig(
        pre_hours=24.0,
        post_hours=24.0,
        frequency_transition_pct=2.0,
        pressure_delta_pct=1.0,
        strong_pressure_delta_pct=2.0,
        support_delta_pct=1.0,
        min_slope_abs_per_day=0.5,
    ),
    "salt": DomainDecisionConfig(
        pre_hours=72.0,
        post_hours=72.0,
        frequency_transition_pct=2.0,
        pressure_delta_pct=0.5,
        strong_pressure_delta_pct=2.0,
        support_delta_pct=1.0,
        min_slope_abs_per_day=0.1,
    ),
}


def _prepared_from_run(run_or_prepared: Any) -> PreparedWellData | None:
    prepared = getattr(run_or_prepared, "prepared", run_or_prepared)
    if isinstance(prepared, PreparedWellData):
        return prepared
    return None


def _column_index(columns: list[str], column: str) -> int | None:
    try:
        return columns.index(column)
    except ValueError:
        return None


def _column_values(prepared: PreparedWellData, column: str) -> np.ndarray | None:
    idx = _column_index(prepared.raw_columns, column)
    if idx is None:
        return None
    raw = np.asarray(prepared.raw_matrix, dtype=float)
    if raw.ndim != 2 or idx >= raw.shape[1]:
        return None
    return raw[:, idx].astype(float)


def _safe_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _pct_delta(post: float, pre: float) -> float:
    if not np.isfinite(post) or not np.isfinite(pre) or abs(pre) < 1e-9:
        return float("nan")
    return float((post - pre) / abs(pre) * 100.0)


def _slope_per_day(timestamps: np.ndarray, values: np.ndarray) -> float:
    ts = pd.to_datetime(timestamps)
    vals = np.asarray(values, dtype=float)
    valid = np.isfinite(vals)
    if valid.sum() < 2:
        return float("nan")
    x = (ts[valid] - ts[valid][0]).total_seconds().to_numpy(dtype=float) / 86_400.0
    y = vals[valid]
    if np.nanmax(x) <= 0:
        return float("nan")
    return float(np.polyfit(x, y, deg=1)[0])


def _window_masks(
    timestamps: np.ndarray,
    detected_time: pd.Timestamp,
    config: DomainDecisionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.to_datetime(timestamps)
    pre_start = detected_time - pd.Timedelta(hours=float(config.pre_hours))
    post_end = detected_time + pd.Timedelta(hours=float(config.post_hours))
    pre_mask = (ts >= pre_start) & (ts < detected_time)
    post_mask = (ts >= detected_time) & (ts <= post_end)
    return np.asarray(pre_mask, dtype=bool), np.asarray(post_mask, dtype=bool)


def _channel_window_features(
    prepared: PreparedWellData,
    column: str,
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
) -> dict[str, float]:
    values = _column_values(prepared, column)
    if values is None:
        return {
            "pre_median": float("nan"),
            "post_median": float("nan"),
            "post_vs_pre_pct": float("nan"),
            "pre_slope_per_day": float("nan"),
            "post_slope_per_day": float("nan"),
            "slope_change_per_day": float("nan"),
        }

    pre_values = values[pre_mask]
    post_values = values[post_mask]
    pre_median = _safe_median(pre_values)
    post_median = _safe_median(post_values)
    pre_slope = _slope_per_day(prepared.timestamps[pre_mask], pre_values)
    post_slope = _slope_per_day(prepared.timestamps[post_mask], post_values)
    return {
        "pre_median": pre_median,
        "post_median": post_median,
        "post_vs_pre_pct": _pct_delta(post_median, pre_median),
        "pre_slope_per_day": pre_slope,
        "post_slope_per_day": post_slope,
        "slope_change_per_day": (
            float(post_slope - pre_slope)
            if np.isfinite(post_slope) and np.isfinite(pre_slope)
            else float("nan")
        ),
    }


def _group_abs_delta_pct(
    prepared: PreparedWellData,
    columns: tuple[str, ...],
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
) -> float:
    deltas: list[float] = []
    for column in columns:
        features = _channel_window_features(prepared, column, pre_mask, post_mask)
        delta = features["post_vs_pre_pct"]
        if np.isfinite(delta):
            deltas.append(abs(float(delta)))
    if not deltas:
        return float("nan")
    return float(np.median(deltas))


def _flag(row: pd.Series, column: str) -> bool:
    value = row.get(column, False)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _stopped_fraction(
    prepared: PreparedWellData,
    post_mask: np.ndarray,
) -> float:
    frequency = _column_values(prepared, FREQ_COL)
    if frequency is None:
        return float("nan")
    post_values = frequency[post_mask]
    finite = post_values[np.isfinite(post_values)]
    if finite.size == 0:
        return float("nan")
    return float((finite < WELL_STOPPED_FREQUENCY_THRESHOLD).mean())


def _classify_domain_start(
    *,
    anomaly_key: str,
    start_row: pd.Series,
    pressure: dict[str, float],
    frequency_delta_pct: float,
    load_delta_pct: float,
    temperature_delta_pct: float,
    imbalance_delta_pct: float,
    config: DomainDecisionConfig,
    stopped_fraction: float = float("nan"),
) -> tuple[str, str, str]:
    if np.isfinite(stopped_fraction) and stopped_fraction >= WELL_STOPPED_FRACTION_THRESHOLD:
        return (
            DOMAIN_REJECTED_WELL_STOPPED,
            "reject",
            "well_stopped_pressure_recovery_is_not_anomaly",
        )
    start_class = str(start_row.get("start_class", "")).strip().lower()
    bad_data_context = _flag(start_row, "is_bad_data") or start_class == "bad_data"
    regime_context = _flag(start_row, "is_regime_event") or start_class == "regime_event"
    freq_unstable = (
        np.isfinite(frequency_delta_pct)
        and abs(float(frequency_delta_pct)) >= float(config.frequency_transition_pct)
    )
    pressure_delta = pressure["post_vs_pre_pct"]
    post_slope = pressure["post_slope_per_day"]
    slope_change = pressure["slope_change_per_day"]
    pressure_abs_move = np.isfinite(pressure_delta) and abs(float(pressure_delta)) >= float(config.pressure_delta_pct)
    pressure_slope_move = np.isfinite(post_slope) and abs(float(post_slope)) >= float(config.min_slope_abs_per_day)
    pressure_level_up = np.isfinite(pressure_delta) and float(pressure_delta) >= float(config.pressure_delta_pct)
    pressure_level_down = np.isfinite(pressure_delta) and float(pressure_delta) <= -float(config.pressure_delta_pct)
    pressure_up = (
        pressure_level_up
        or (np.isfinite(post_slope) and float(post_slope) > 0 and np.isfinite(slope_change) and float(slope_change) > 0)
    )
    slope_reversal_up = np.isfinite(slope_change) and float(slope_change) > 0 and np.isfinite(post_slope) and float(post_slope) > 0
    strong_pressure_up = np.isfinite(pressure_delta) and float(pressure_delta) >= float(config.strong_pressure_delta_pct)
    load_support = np.isfinite(load_delta_pct) and float(load_delta_pct) >= float(config.support_delta_pct)
    thermal_support = np.isfinite(temperature_delta_pct) and float(temperature_delta_pct) >= float(config.support_delta_pct)
    imbalance_support = np.isfinite(imbalance_delta_pct) and float(imbalance_delta_pct) >= float(config.support_delta_pct)
    any_physical_support = pressure_abs_move or load_support or thermal_support or imbalance_support
    context_suffix = ""
    if bad_data_context:
        context_suffix = "_despite_bad_data_context"
    elif regime_context:
        context_suffix = "_despite_regime_context"

    if anomaly_key == "pritok":
        if freq_unstable:
            return DOMAIN_REJECTED_FREQUENCY_TRANSITION, "reject", "pressure_change_explained_by_frequency"
        if bad_data_context or regime_context:
            if pressure_abs_move:
                return DOMAIN_UNCERTAIN, "uncertain", "pressure_trend_needs_review" + context_suffix
            if bad_data_context:
                return DOMAIN_REJECTED_BAD_DATA, "reject", "bad_data_context_without_pressure_trend"
            return DOMAIN_REJECTED_REGIME_EVENT, "reject", "regime_context_without_pressure_trend"
        if pressure_abs_move:
            return DOMAIN_PRITOK_CANDIDATE, "accept", "pressure_trend_with_stable_frequency"
        if pressure_slope_move:
            return DOMAIN_UNCERTAIN, "uncertain", "local_pressure_slope_without_level_shift"
        return DOMAIN_UNCERTAIN, "uncertain", "weak_pressure_trend_below_pritok_threshold"

    if anomaly_key == "salt":
        support_count = int(load_support) + int(thermal_support) + int(imbalance_support)
        strong_salt_pressure = strong_pressure_up or (
            np.isfinite(post_slope)
            and float(post_slope) >= float(config.min_slope_abs_per_day)
            and np.isfinite(slope_change)
            and float(slope_change) > 0
        )
        expected_frequency_response = freq_unstable and pressure_level_down and (
            not np.isfinite(post_slope) or float(post_slope) <= 0
        )
        weak_reversal_without_level = slope_reversal_up and not pressure_level_up and support_count == 0
        if expected_frequency_response:
            return DOMAIN_REJECTED_FREQUENCY_TRANSITION, "reject", "pressure_drop_explained_by_frequency"
        if pressure_up or slope_reversal_up:
            reason = "pressure_post_slope_or_reversal_up"
            if freq_unstable:
                reason += "_with_frequency_change"
            if weak_reversal_without_level and not strong_salt_pressure:
                return DOMAIN_UNCERTAIN, "uncertain", "weak_salt_reversal_without_level_shift"
            if (bad_data_context or regime_context) and not (strong_salt_pressure or support_count > 0):
                return DOMAIN_UNCERTAIN, "uncertain", "salt_pressure_pattern_needs_review" + context_suffix
            return DOMAIN_SALT_CANDIDATE, "accept", reason + context_suffix
        if freq_unstable:
            return DOMAIN_REJECTED_FREQUENCY_TRANSITION, "reject", "frequency_transition_without_salt_pressure_pattern"
        if bad_data_context:
            return DOMAIN_REJECTED_BAD_DATA, "reject", "bad_data_context_without_salt_pressure_pattern"
        if regime_context:
            return DOMAIN_REJECTED_REGIME_EVENT, "reject", "regime_context_without_salt_pressure_pattern"
        return DOMAIN_UNCERTAIN, "uncertain", "weak_salt_pressure_pattern"

    if anomaly_key == "negermet":
        support_count = int(load_support) + int(thermal_support) + int(imbalance_support)
        if pressure_up and (strong_pressure_up or support_count > 0):
            reason_parts = ["pressure_step_up_pattern"]
            if strong_pressure_up:
                reason_parts.append("strong_pressure")
            if thermal_support:
                reason_parts.append("thermal_support")
            if load_support:
                reason_parts.append("load_support")
            if imbalance_support:
                reason_parts.append("imbalance_support")
            reason = "_".join(reason_parts) + context_suffix
            return DOMAIN_NEGERMET_CANDIDATE, "accept", reason
        if bad_data_context and not any_physical_support:
            return DOMAIN_REJECTED_BAD_DATA, "reject", "bad_data_context_without_negermet_physics"
        if regime_context and not any_physical_support:
            return DOMAIN_REJECTED_REGIME_EVENT, "reject", "regime_context_without_negermet_physics"
        return DOMAIN_UNCERTAIN, "uncertain", "weak_negermet_pressure_step"

    return DOMAIN_UNCERTAIN, "uncertain", "unknown_anomaly_key"


def assess_domain_start(
    *,
    anomaly_key: str,
    prepared: PreparedWellData,
    start_row: pd.Series,
    config: DomainDecisionConfig | None = None,
) -> dict[str, Any]:
    cfg = config or DEFAULT_CONFIGS.get(str(anomaly_key), DomainDecisionConfig(24.0, 24.0, 3.0, 0.5))
    detected_time = pd.Timestamp(start_row["detected_time"])
    pre_mask, post_mask = _window_masks(prepared.timestamps, detected_time, cfg)
    pressure = _channel_window_features(prepared, PRESSURE_COL, pre_mask, post_mask)
    frequency = _channel_window_features(prepared, FREQ_COL, pre_mask, post_mask)
    load_delta = _group_abs_delta_pct(prepared, LOAD_COLUMNS, pre_mask, post_mask)
    temperature_delta = _group_abs_delta_pct(prepared, TEMPERATURE_COLUMNS, pre_mask, post_mask)
    imbalance_delta = _group_abs_delta_pct(prepared, IMBALANCE_COLUMNS, pre_mask, post_mask)
    stopped_fraction = _stopped_fraction(prepared, post_mask)
    verdict, action, reason = _classify_domain_start(
        anomaly_key=str(anomaly_key),
        start_row=start_row,
        pressure=pressure,
        frequency_delta_pct=frequency["post_vs_pre_pct"],
        load_delta_pct=load_delta,
        temperature_delta_pct=temperature_delta,
        imbalance_delta_pct=imbalance_delta,
        config=cfg,
        stopped_fraction=stopped_fraction,
    )
    return {
        "domain_verdict": verdict,
        "domain_action": action,
        "domain_reason": reason,
        "domain_stopped_fraction": stopped_fraction,
        "domain_pre_points": int(pre_mask.sum()),
        "domain_post_points": int(post_mask.sum()),
        "domain_pre_hours": float(cfg.pre_hours),
        "domain_post_hours": float(cfg.post_hours),
        "domain_pressure_pre_median": pressure["pre_median"],
        "domain_pressure_post_median": pressure["post_median"],
        "domain_pressure_post_vs_pre_pct": pressure["post_vs_pre_pct"],
        "domain_pressure_pre_slope_per_day": pressure["pre_slope_per_day"],
        "domain_pressure_post_slope_per_day": pressure["post_slope_per_day"],
        "domain_pressure_slope_change_per_day": pressure["slope_change_per_day"],
        "domain_frequency_post_vs_pre_pct": frequency["post_vs_pre_pct"],
        "domain_load_abs_delta_pct": load_delta,
        "domain_temperature_abs_delta_pct": temperature_delta,
        "domain_imbalance_abs_delta_pct": imbalance_delta,
    }


def attach_domain_decisions_to_starts(
    starts: pd.DataFrame,
    prepared_runs: dict[str, Any],
    *,
    anomaly_key: str,
) -> pd.DataFrame:
    result = starts.copy()
    if result.empty:
        return result

    prepared_lookup = {
        str(getattr(_prepared_from_run(run), "well_id", key)).strip().lower(): _prepared_from_run(run)
        for key, run in prepared_runs.items()
    }
    decisions: list[dict[str, Any]] = []
    for _, row in result.iterrows():
        well_key = str(row.get("well_id", "")).strip().lower()
        prepared = prepared_lookup.get(well_key)
        if prepared is None:
            decisions.append({
                "domain_verdict": DOMAIN_NO_DATA,
                "domain_action": "uncertain",
                "domain_reason": "prepared_well_not_found",
            })
            continue
        decisions.append(
            assess_domain_start(
                anomaly_key=anomaly_key,
                prepared=prepared,
                start_row=row,
            )
        )
    decision_df = pd.DataFrame(decisions, index=result.index)
    for column in decision_df.columns:
        result[column] = decision_df[column]
    return result


def attach_domain_decisions_to_incidents(
    incidents: pd.DataFrame,
    starts: pd.DataFrame,
) -> pd.DataFrame:
    result = incidents.copy()
    if result.empty or starts.empty or "incident_id" not in starts.columns:
        return result

    grouped = starts[starts["incident_id"].fillna("").astype(str) != ""].groupby("incident_id", dropna=False)
    rows: list[dict[str, Any]] = []
    for incident_id, group in grouped:
        first = group.sort_values("detected_time").iloc[0]
        rows.append({
            "incident_id": incident_id,
            "domain_verdict": first.get("domain_verdict", ""),
            "domain_action": first.get("domain_action", ""),
            "domain_reason": first.get("domain_reason", ""),
            "domain_rejected_starts": int((group.get("domain_action", pd.Series(dtype=object)) == "reject").sum()),
            "domain_uncertain_starts": int((group.get("domain_action", pd.Series(dtype=object)) == "uncertain").sum()),
            "domain_accepted_starts": int((group.get("domain_action", pd.Series(dtype=object)) == "accept").sum()),
        })
    if not rows:
        return result
    return result.merge(pd.DataFrame(rows), on="incident_id", how="left")
