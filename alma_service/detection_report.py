from __future__ import annotations

import argparse
import base64
import io
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

from alma_service.anomaly_specs import get_detection_spec
from alma_service.benchmark_metrics import evaluate_predictions
from alma_service.detection_artifacts import (
    DEFAULT_DETECTOR,
    benchmark_summary_path,
    load_json,
    normalize_detector_key,
    predicted_starts_path,
    report_path,
    results_path,
    scores_path,
    summary_path,
)
from alma_service.paths import DB_DIR, REPORTS_DIR, ensure_parent
from alma_service.tabular_io import read_table

PRESSURE_COL = "Давление на приеме насоса кгс/см²"
FREQ_COL = "Выходная частота"
COLOR_THEMES = {
    "negermet": {"accent": "#b42318", "soft": "#fef3f2", "soft_alt": "#fff6f6"},
    "pritok": {"accent": "#c2410c", "soft": "#fff7ed", "soft_alt": "#fffbf5"},
    "salt": {"accent": "#0f766e", "soft": "#f0fdfa", "soft_alt": "#f8fffd"},
}
SPLIT_LABELS = {"train": "Обучающие скважины", "test": "Тестовые скважины", "all": "Все скважины"}
SPLIT_SHORT_LABELS = {"train": "Обучение", "test": "Тест", "all": "Все"}
STATUS_LABELS = {
    "Detected": "Обнаружено",
    "Not found": "Не обнаружено",
    "Not assessed": "Не оценено",
}
DETECTOR_LABELS = {
    "paano_shared": "PaAno Shared Encoder",
    "paano_global": "PaAno Global Encoder",
}
EVENT_CLASS_LABELS = {
    "labelled_anomaly": "размеченная аномалия",
    "anomaly_candidate": "кандидат аномалии",
    "normal_context": "нормальный контекст",
    "regime_event": "режимное событие",
    "bad_data": "проблема качества данных",
}
QUALITY_STATUS_LABELS = {
    "ok": "качество данных в норме",
    "sensor_stuck": "подозрение на залипание датчика",
}
REGIME_STATUS_LABELS = {
    "normal": "без явного режимного перехода",
    "regime_shift": "режимный переход",
    "stop_start": "остановка/пуск",
}
ZONE_STATUS_LABELS = {
    "labelled_anomaly": "размеченная зона аномалии",
    "clean_normal": "подтвержденная нормальная зона",
}
START_CLASS_LABELS = {
    "labelled_anomaly": "старт внутри размеченной аномалии",
    "anomaly_candidate": "кандидат без разметки",
    "regime_event": "режимное событие",
    "bad_data": "плохие данные",
}
INCIDENT_STATE_LABELS = {
    "open": "новый эпизод",
    "continue": "продолжение эпизода",
    "suppressed": "подавлено",
}
SUPPRESSION_REASON_LABELS = {
    "bad_data": "подавлено по качеству данных",
    "regime_event": "подавлено как режимное событие",
}


def _pick_existing_source(spec, source_path: str | None, detector_key: str | None = None) -> Path:
    if source_path:
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")
        return src
    if detector_key == "paano_global":
        from alma_service.global_normality import configured_anomaly_source_path

        src = configured_anomaly_source_path(spec.anomaly_key)
        if not src.exists():
            raise FileNotFoundError(f"Global detector source file not found: {src}")
        return src

    candidates = [DB_DIR / name for name in spec.dataset.source_candidates]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No source dataset found for {spec.anomaly_key}")


def _load_timeseries(path: Path) -> pd.DataFrame:
    df = read_table(
        path,
        dtypes={"well_id": str},
        parse_dates=["timestamp"],
        low_memory=False,
    )
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    return df.dropna(subset=["timestamp"]).sort_values(["well_id", "timestamp"]).reset_index(drop=True)


def _load_intervals(path: Path) -> pd.DataFrame:
    df = read_table(
        path,
        dtypes={"well_id": str},
        parse_dates=["start_date", "end_date", "data_start", "data_end"],
    )
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.strip().str.lower()
    else:
        df["split"] = "train"
    if "interval_idx" not in df.columns:
        df["interval_idx"] = df.groupby("well_id").cumcount() + 1
    df["interval_idx"] = pd.to_numeric(df["interval_idx"], errors="coerce").fillna(1).astype(int)
    return df.dropna(subset=["well_id", "start_date", "end_date"]).sort_values(
        ["well_id", "start_date", "interval_idx"]
    )


def _load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    df = read_table(
        path,
        dtypes={"well_id": str},
        parse_dates=["actual_start", "actual_end", "detected_time", "data_start", "data_end"],
    )
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    if "interval_idx" not in df.columns:
        df["interval_idx"] = df.groupby("well_id").cumcount() + 1
    df["interval_idx"] = pd.to_numeric(df["interval_idx"], errors="coerce").fillna(1).astype(int)
    if "split" not in df.columns:
        df["split"] = "train"
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    if "status" not in df.columns:
        df["status"] = np.where(df["detected_time"].notna(), "Detected", "Not found")
    return df.sort_values(["well_id", "interval_idx"]).reset_index(drop=True)


def _load_scores(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    df = read_table(path, dtypes={"well_id": str}, parse_dates=["timestamp"])
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["timestamp"]).sort_values(["well_id", "timestamp"]).reset_index(drop=True)
    return {well_id: grp for well_id, grp in df.groupby("well_id")}


def _load_predicted_starts(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = read_table(path, dtypes={"well_id": str}, parse_dates=["detected_time"])
    if df.empty:
        return df
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["detected_time"] = pd.to_datetime(df["detected_time"], errors="coerce")
    return df


def _score_column(scores_df: pd.DataFrame | None) -> str | None:
    if scores_df is None or scores_df.empty:
        return None
    if "score" in scores_df.columns:
        return "score"
    if "paano_score" in scores_df.columns:
        return "paano_score"
    for column in scores_df.columns:
        if column.endswith("_score"):
            return column
    return None


def _score_unavailable_reason(scores_df: pd.DataFrame | None) -> str | None:
    if scores_df is None or scores_df.empty or "score_valid" not in scores_df.columns:
        return None
    valid_raw = scores_df["score_valid"]
    if valid_raw.dtype == object:
        valid = valid_raw.fillna("true").astype(str).str.lower().isin({"1", "true", "yes"})
    else:
        valid = valid_raw.fillna(True).astype(bool)
    if bool(valid.any()):
        return None
    if "score_unavailable_reason" not in scores_df.columns:
        return "unknown"
    reasons = (
        scores_df["score_unavailable_reason"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    reasons = reasons[reasons != ""]
    if reasons.empty:
        return "unknown"
    return str(reasons.mode().iloc[0])


def _score_unavailable_text(reason: str | None) -> str:
    if reason == "not_enough_points":
        return (
            "Отклонение от нормы не рассчитано: недостаточно точек в ряду или reference "
            "для выбранного окна PaAno. Нулевой score здесь не означает норму."
        )
    if reason == "unlabeled_no_reference":
        return (
            "Отклонение от нормы не рассчитано: у скважины нет размеченного интервала, "
            "поэтому начало её собственного ряда не используется как эталон нормы. "
            "Нулевой score здесь не означает норму."
        )
    if reason == "population_reference_unavailable":
        return (
            "Отклонение от нормы не рассчитано: локальный reference слишком короткий, "
            "а population memory bank недоступен. Нулевой score здесь не означает норму."
        )
    return "Отклонение от нормы не рассчитано. Нулевой score здесь не означает норму."


def _input_contract(scores_df: pd.DataFrame | None) -> str:
    if scores_df is None or scores_df.empty or "input_contract" not in scores_df.columns:
        return "real_window"
    values = scores_df["input_contract"].dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return "real_window"
    return str(values.mode().iloc[0])


def _input_contract_text(input_contract: str) -> str:
    if input_contract == "real_long_local_memory":
        return (
            "Контракт входа: обычное long-окно без искусственного дополнения; "
            "memory bank построен из локальной размеченной нормы этой скважины."
        )
    if input_contract == "padded_long_local_memory":
        return (
            "Контракт входа: long-окно дополнено edge-hold; memory bank построен "
            "из локальной размеченной нормы этой скважины."
        )
    if input_contract == "real_long_population_memory":
        return (
            "Контракт входа: обычное long-окно без искусственного дополнения; "
            "memory bank взят из общего population-pool подтвержденной нормы."
        )
    if input_contract == "padded_long_population_memory":
        return (
            "Контракт входа: long-окно дополнено edge-hold; memory bank взят "
            "из общего population-pool подтвержденной нормы. Это fallback для "
            "короткой истории или blind-скважины."
        )
    if input_contract == "no_population_reference":
        return (
            "Контракт входа: скважина не оценивалась, потому что локальный "
            "reference короткий, а population memory bank недоступен."
        )
    if input_contract == "no_local_reference":
        return (
            "Контракт входа: скважина без разметки не оценивалась локальным reference. "
            "Для blind-инференса нужен population memory bank или внешний подтвержденный "
            "эталон нормы."
        )
    if input_contract == "edge_hold_padded":
        return (
            "Контракт входа: ряд/эталон были дополнены методом edge-hold "
            "(удержание первого валидного значения в начале ряда). Score рассчитан "
            "только для реальных timestamp после отбрасывания искусственного префикса."
        )
    return "Контракт входа: обычное окно без искусственного дополнения."


def _input_contract_badge(scores_df: pd.DataFrame | None) -> str:
    input_contract = _input_contract(scores_df)
    warn_contracts = {
        "edge_hold_padded",
        "padded_long_local_memory",
        "padded_long_population_memory",
        "real_long_population_memory",
        "no_local_reference",
        "no_population_reference",
    }
    css_class = "warn" if input_contract in warn_contracts else "info"
    return f'<p class="contract-note {css_class}">{escape(_input_contract_text(input_contract))}</p>'


def _truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _label_value(value: Any, mapping: dict[str, str]) -> str | None:
    if pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return mapping.get(raw, raw)


def _prediction_for_detection(
    pred_df: pd.DataFrame,
    well_id: str,
    detected_time: Any,
) -> pd.Series | None:
    if pred_df.empty or pd.isna(detected_time):
        return None
    subset = pred_df[pred_df["well_id"] == str(well_id).strip().lower()].copy()
    if subset.empty:
        return None
    subset = subset.dropna(subset=["detected_time"])
    if subset.empty:
        return None
    detected_ts = pd.Timestamp(detected_time)
    exact = subset[subset["detected_time"] == detected_ts]
    if not exact.empty:
        return exact.iloc[0]
    deltas = (subset["detected_time"] - detected_ts).abs()
    if deltas.empty or deltas.min() > pd.Timedelta(seconds=1):
        return None
    return subset.loc[deltas.idxmin()]


def _domain_decision_html(pred_row: pd.Series | None) -> str:
    if pred_row is None:
        return ""
    actionable = _truthy(pred_row.get("actionable_alert", False))
    css_class = "info" if actionable else "warn"
    action_text = (
        "Доменная проверка: алерт принят."
        if actionable
        else "Доменная проверка: алерт не считается уверенной детекцией."
    )
    details: list[str] = []
    field_specs = [
        ("event_class", "класс события", EVENT_CLASS_LABELS),
        ("start_class", "тип старта", START_CLASS_LABELS),
        ("quality_status", "качество", QUALITY_STATUS_LABELS),
        ("regime_status", "режим", REGIME_STATUS_LABELS),
        ("zone_status", "зона", ZONE_STATUS_LABELS),
        ("incident_state", "эпизод", INCIDENT_STATE_LABELS),
    ]
    for column, label, mapping in field_specs:
        if column not in pred_row:
            continue
        value = _label_value(pred_row.get(column), mapping)
        if value:
            details.append(f"{label}: {value}")
    reason = _label_value(pred_row.get("suppression_reason", ""), SUPPRESSION_REASON_LABELS)
    if reason:
        details.append(f"причина: {reason}")
    suffix = f" {'; '.join(details)}." if details else ""
    return f'<p class="domain-note {css_class}">{escape(action_text + suffix)}</p>'


def _domain_summary_html(pred_df: pd.DataFrame) -> str:
    if pred_df.empty or "actionable_alert" not in pred_df.columns:
        return ""
    accepted = int(pred_df["actionable_alert"].map(_truthy).sum())
    total = int(len(pred_df))
    uncertain = total - accepted
    event_counts: list[str] = []
    if "event_class" in pred_df.columns:
        events = pred_df["event_class"].dropna().astype(str).str.strip()
        for event_class, count in events[events != ""].value_counts().sort_index().items():
            label = EVENT_CLASS_LABELS.get(event_class, event_class)
            event_counts.append(f"{label}: {int(count)}")
    css_class = "info" if accepted == total else "warn"
    parts = [f"Доменная проверка: принято {accepted} из {total} стартов"]
    if uncertain:
        parts.append(f"неуверенных/подавленных: {uncertain}")
    if event_counts:
        parts.append("; ".join(event_counts))
    return f'<p class="domain-note {css_class}">{escape("; ".join(parts))}.</p>'


def _format_dt(value: Any, fmt: str = "%Y-%m-%d %H:%M") -> str:
    ts = pd.to_datetime(value, errors="coerce")
    return ts.strftime(fmt) if pd.notna(ts) else "—"


def _format_float(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(numeric):
        return "—"
    return f"{numeric:.{digits}f}{suffix}"


def _detector_label(detector_key: str) -> str:
    return DETECTOR_LABELS.get(detector_key, detector_key)


def _status_label(status: Any) -> str:
    return STATUS_LABELS.get(str(status), str(status))


def _status_pill_class(status: Any) -> str:
    value = str(status)
    if value == "Detected":
        return "ok"
    if value == "Not assessed":
        return "warn"
    return "miss"


def _fig_to_b64(fig: plt.Figure, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _pick_top_channel(
    well_id: str,
    well_df: pd.DataFrame,
    fi_data: dict[str, dict[str, float]],
) -> str | None:
    """Pick the most important FI v2 channel for a well, skipping pressure if it's #1."""
    well_fi = fi_data.get(well_id)
    if not well_fi:
        return None

    sorted_channels = sorted(well_fi.items(), key=lambda x: x[1], reverse=True)
    for ch_name, _score in sorted_channels:
        if ch_name != PRESSURE_COL and ch_name in well_df.columns:
            return ch_name
    return None


def _load_feature_importance_data(spec, detector_key: str) -> dict[str, dict[str, float]]:
    """Load FI v2 summary and return dict[well_id -> dict[channel -> final_score]]."""
    import json
    fi_summary_path = DB_DIR / f"{spec.dataset.output_prefix}_{detector_key}_fi_summary.json"
    if fi_summary_path.exists():
        data = json.loads(fi_summary_path.read_text(encoding="utf-8"))
        if data.get("version") != 2:
            raise RuntimeError(
                f"Legacy feature importance format in {fi_summary_path}. "
                "Regenerate feature importance report to create version=2 summary."
            )
        wells = data.get("wells")
        if not isinstance(wells, dict):
            raise RuntimeError(f"Invalid feature importance v2 summary: {fi_summary_path}")
        parsed: dict[str, dict[str, float]] = {}
        for well_id, well_payload in wells.items():
            channels = well_payload.get("channels") if isinstance(well_payload, dict) else None
            if not isinstance(channels, dict):
                continue
            parsed[str(well_id)] = {
                str(ch_name): float(ch_payload["final_score"])
                for ch_name, ch_payload in channels.items()
                if isinstance(ch_payload, dict)
                and isinstance(ch_payload.get("final_score"), (int, float))
            }
        print(f"  Feature importance загружен из {fi_summary_path.name}")
        return parsed
    return {}


def _create_plot_html(
    well_df: pd.DataFrame,
    result_row: pd.Series,
    scores_df: pd.DataFrame | None,
    accent: str,
    fi_data: dict[str, dict[str, float]],
) -> str | None:
    well_id = str(result_row["well_id"])
    score_col = _score_column(scores_df)
    score_unavailable_reason = _score_unavailable_reason(scores_df)
    has_scores = score_col is not None
    score_components_available = has_scores and score_unavailable_reason is None
    has_paano_tail = score_components_available and scores_df is not None and "paano_tail_score" in scores_df.columns
    has_pressure_trend = score_components_available and scores_df is not None and "pressure_trend_score" in scores_df.columns
    has_negermet_signature = score_components_available and scores_df is not None and "negermet_signature_score" in scores_df.columns
    has_salt_trend = score_components_available and scores_df is not None and "salt_deposition_score" in scores_df.columns
    has_salt_shift = score_components_available and scores_df is not None and "salt_distribution_shift_score" in scores_df.columns

    x_min = result_row["data_start"] if pd.notna(result_row.get("data_start")) else well_df["timestamp"].min()
    x_max = result_row["data_end"] if pd.notna(result_row.get("data_end")) else well_df["timestamp"].max()
    if pd.notna(x_min) and pd.notna(x_max):
        well_df = well_df[(well_df["timestamp"] >= x_min) & (well_df["timestamp"] <= x_max)].copy()
    if well_df.empty:
        return None

    top_channel = _pick_top_channel(well_id, well_df, fi_data)

    # Panels: 1) Score, 2) Pressure, 3) Top channel
    panels: list[tuple[str, str | None]] = []
    if has_scores:
        panels.append(("Отклонение от нормы (score)", None))
    has_pressure = PRESSURE_COL in well_df.columns
    if has_pressure:
        panels.append((PRESSURE_COL, PRESSURE_COL))
    if top_channel:
        panels.append((top_channel, top_channel))

    if not panels:
        return None

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.8 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    ts_pd = pd.to_datetime(well_df["timestamp"])

    def _draw_zones(ax, show_labels: bool = False):
        ymin, ymax = ax.get_ylim()
        actual_start = pd.Timestamp(result_row["actual_start"])
        actual_end = pd.Timestamp(result_row["actual_end"])
        ax.axvspan(actual_start, actual_end, color="red", alpha=0.10, zorder=0)
        ax.axvline(actual_start, color="#16a34a", linewidth=1.2, linestyle="-")
        ax.axvline(actual_end, color="#dc2626", linewidth=1.2, linestyle="--")
        if show_labels:
            ax.annotate(
                f"Факт начало\n{actual_start.strftime('%Y-%m-%d %H:%M')}",
                xy=(actual_start, ymax), xytext=(5, -5),
                textcoords="offset points", fontsize=6.5, color="#16a34a",
                fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#16a34a", alpha=0.85),
            )
            ax.annotate(
                f"Факт конец\n{actual_end.strftime('%Y-%m-%d %H:%M')}",
                xy=(actual_end, ymax), xytext=(-5, -5),
                textcoords="offset points", fontsize=6.5, color="#dc2626",
                fontweight="bold", va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dc2626", alpha=0.85),
            )
        if pd.notna(result_row["detected_time"]):
            det_ts = pd.Timestamp(result_row["detected_time"])
            ax.axvline(det_ts, color="#7c3aed", linewidth=1.5, linestyle="-.")
            if show_labels:
                ax.annotate(
                    f"Обнаружено\n{det_ts.strftime('%Y-%m-%d %H:%M')}",
                    xy=(det_ts, ymax), xytext=(5, -25),
                    textcoords="offset points", fontsize=6.5, color="#7c3aed",
                    fontweight="bold", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#7c3aed", alpha=0.85),
                )

    for panel_idx, (label, col_name) in enumerate(panels):
        ax = axes[panel_idx]
        is_score = col_name is None

        if is_score and scores_df is not None and score_col is not None:
            score_view = scores_df.copy()
            if pd.notna(x_min) and pd.notna(x_max):
                score_view = score_view[(score_view["timestamp"] >= x_min) & (score_view["timestamp"] <= x_max)]
            if score_unavailable_reason is not None:
                ax.text(
                    0.5,
                    0.5,
                    _score_unavailable_text(score_unavailable_reason),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#92400e",
                    bbox=dict(boxstyle="round,pad=0.45", fc="#fffbeb", ec="#f59e0b", alpha=0.95),
                )
                ax.set_yticks([])
            elif not score_view.empty:
                sts = pd.to_datetime(score_view["timestamp"])
                vals = score_view[score_col].values
                ax.fill_between(sts, 0, vals, color=accent, alpha=0.18)
                ax.plot(sts, vals, color=accent, linewidth=0.7, label="Итоговый score")
                if "paano_tail_score" in score_view.columns:
                    ax.plot(
                        sts,
                        score_view["paano_tail_score"].values,
                        color="#0f766e",
                        linewidth=0.55,
                        alpha=0.75,
                        label="PaAno tail-score",
                    )
                if "pressure_trend_score" in score_view.columns:
                    ax.plot(
                        sts,
                        score_view["pressure_trend_score"].values,
                        color="#ea580c",
                        linewidth=0.65,
                        alpha=0.78,
                        label="Pressure trend",
                    )
                if "negermet_signature_score" in score_view.columns:
                    ax.plot(
                        sts,
                        score_view["negermet_signature_score"].values,
                        color="#b45309",
                        linewidth=0.65,
                        alpha=0.78,
                        label="Negermet signature",
                    )
                if "salt_deposition_score" in score_view.columns:
                    ax.plot(
                        sts,
                        score_view["salt_deposition_score"].values,
                        color="#0891b2",
                        linewidth=0.65,
                        alpha=0.78,
                        label="Salt deposition trend",
                    )
                if "salt_distribution_shift_score" in score_view.columns:
                    ax.plot(
                        sts,
                        score_view["salt_distribution_shift_score"].values,
                        color="#be123c",
                        linewidth=0.62,
                        alpha=0.72,
                        label="Salt KS shift",
                    )
        elif col_name is not None and col_name in well_df.columns:
            vals = pd.to_numeric(well_df[col_name], errors="coerce")
            ax.plot(ts_pd, vals, color="#2563eb", linewidth=0.6, alpha=0.85)

        ax.set_ylabel(label, fontsize=8)
        _draw_zones(ax, show_labels=(panel_idx == 0))
        ax.grid(True, alpha=0.3)

    axes[0].set_title(
        f"Скважина {well_id} | интервал {int(result_row['interval_idx'])}",
        fontsize=11, fontweight="bold",
    )

    # Legend on top panel
    zone_patch = mpatches.Patch(facecolor="red", alpha=0.10, edgecolor="none", label="Зона аномалии")
    start_line = plt.Line2D([0], [0], color="#16a34a", linewidth=1.2, label="Начало (факт)")
    end_line = plt.Line2D([0], [0], color="#dc2626", linewidth=1.2, linestyle="--", label="Конец (факт)")
    detect_line = plt.Line2D([0], [0], color="#7c3aed", linewidth=1.5, linestyle="-.", label="Обнаружено")
    legend_handles = [zone_patch, start_line, end_line]
    if pd.notna(result_row["detected_time"]):
        legend_handles.append(detect_line)
    if has_paano_tail:
        legend_handles.append(
            plt.Line2D([0], [0], color="#0f766e", linewidth=0.75, label="PaAno tail-score")
        )
    if has_pressure_trend:
        legend_handles.append(
            plt.Line2D([0], [0], color="#ea580c", linewidth=0.85, label="Pressure trend")
        )
    if has_negermet_signature:
        legend_handles.append(
            plt.Line2D([0], [0], color="#b45309", linewidth=0.85, label="Negermet signature")
        )
    if has_salt_trend:
        legend_handles.append(
            plt.Line2D([0], [0], color="#0891b2", linewidth=0.85, label="Salt deposition trend")
        )
    if has_salt_shift:
        legend_handles.append(
            plt.Line2D([0], [0], color="#be123c", linewidth=0.85, label="Salt KS shift")
        )
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.9)

    axes[-1].set_xlabel("Время")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()

    b64 = _fig_to_b64(fig, dpi=110)
    return f'<img src="data:image/png;base64,{b64}" alt="График {well_id}">'


def _create_unlabeled_plot_html(
    well_df: pd.DataFrame,
    well_id: str,
    scores_df: pd.DataFrame | None,
    predicted_starts: list[pd.Timestamp],
    accent: str,
    fi_data: dict[str, dict[str, float]],
) -> str | None:
    """Create a plot for a well with no labeled anomaly interval."""
    score_col = _score_column(scores_df)
    score_unavailable_reason = _score_unavailable_reason(scores_df)
    has_scores = score_col is not None
    if well_df.empty and not has_scores:
        return None

    top_channel = _pick_top_channel(well_id, well_df, fi_data)

    panels: list[tuple[str, str | None]] = []
    if has_scores:
        panels.append(("\u041e\u0442\u043a\u043b\u043e\u043d\u0435\u043d\u0438\u0435 \u043e\u0442 \u043d\u043e\u0440\u043c\u044b (score)", None))
    if PRESSURE_COL in well_df.columns:
        panels.append((PRESSURE_COL, PRESSURE_COL))
    if top_channel:
        panels.append((top_channel, top_channel))
    if not panels:
        return None

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.8 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    ts_pd = pd.to_datetime(well_df["timestamp"]) if not well_df.empty else pd.Series(dtype="datetime64[ns]")

    x_min = well_df["timestamp"].min() if not well_df.empty else None
    x_max = well_df["timestamp"].max() if not well_df.empty else None

    for panel_idx, (label, col_name) in enumerate(panels):
        ax = axes[panel_idx]
        is_score = col_name is None

        if is_score and scores_df is not None and score_col is not None:
            score_view = scores_df.copy()
            if x_min is not None and x_max is not None:
                score_view = score_view[
                    (score_view["timestamp"] >= x_min) & (score_view["timestamp"] <= x_max)
                ]
            if score_unavailable_reason is not None:
                ax.text(
                    0.5,
                    0.5,
                    _score_unavailable_text(score_unavailable_reason),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#92400e",
                    bbox=dict(boxstyle="round,pad=0.45", fc="#fffbeb", ec="#f59e0b", alpha=0.95),
                )
                ax.set_yticks([])
            elif not score_view.empty:
                sts = pd.to_datetime(score_view["timestamp"])
                vals = score_view[score_col].values
                ax.fill_between(sts, 0, vals, color=accent, alpha=0.18)
                ax.plot(sts, vals, color=accent, linewidth=0.7)
        elif col_name is not None and col_name in well_df.columns:
            vals = pd.to_numeric(well_df[col_name], errors="coerce")
            ax.plot(ts_pd, vals, color="#2563eb", linewidth=0.6, alpha=0.85)

        for det_ts in predicted_starts:
            ax.axvline(det_ts, color="#7c3aed", linewidth=1.5, linestyle="-.")
            if panel_idx == 0:
                ymin, ymax = ax.get_ylim()
                ax.annotate(
                    f"\u041e\u0431\u043d\u0430\u0440\u0443\u0436\u0435\u043d\u043e\n{det_ts.strftime('%Y-%m-%d %H:%M')}",
                    xy=(det_ts, ymax), xytext=(5, -5),
                    textcoords="offset points", fontsize=6.5, color="#7c3aed",
                    fontweight="bold", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#7c3aed", alpha=0.85),
                )

        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(
        f"\u0421\u043a\u0432\u0430\u0436\u0438\u043d\u0430 {well_id} (\u0431\u0435\u0437 \u0440\u0430\u0437\u043c\u0435\u0442\u043a\u0438)",
        fontsize=11, fontweight="bold",
    )
    detect_line = plt.Line2D(
        [0], [0], color="#7c3aed", linewidth=1.5, linestyle="-.",
        label="\u041e\u0431\u043d\u0430\u0440\u0443\u0436\u0435\u043d\u043e",
    )
    axes[0].legend(handles=[detect_line], loc="upper right", fontsize=7, framealpha=0.9)
    axes[-1].set_xlabel("\u0412\u0440\u0435\u043c\u044f")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()

    b64 = _fig_to_b64(fig, dpi=110)
    return f'<img src="data:image/png;base64,{b64}" alt="\u0413\u0440\u0430\u0444\u0438\u043a {well_id}">'


def _resolve_detector(spec, detector: str | None) -> str:
    if detector:
        return normalize_detector_key(detector)
    benchmark = load_json(benchmark_summary_path(spec))
    selected = benchmark.get("selected_default_detector")
    if selected:
        try:
            return normalize_detector_key(selected)
        except ValueError:
            pass
    return DEFAULT_DETECTOR


def generate_report(
    anomaly_key: str,
    detector: str | None = None,
    output_path: str | None = None,
    source_path: str | None = None,
    results_path_override: str | None = None,
    scores_path_override: str | None = None,
) -> Path:
    spec = get_detection_spec(anomaly_key)
    theme = COLOR_THEMES[anomaly_key]
    detector_key = _resolve_detector(spec, detector)

    source = _pick_existing_source(spec, source_path, detector_key=detector_key)
    results_path_value = Path(results_path_override) if results_path_override else results_path(spec, detector_key)
    scores_path_value = Path(scores_path_override) if scores_path_override else scores_path(spec, detector_key)
    output = ensure_parent(
        Path(output_path) if output_path else report_path(spec, detector_key)
    )

    data_df = _load_timeseries(source)
    intervals_df = _load_intervals(spec.dataset.intervals_path)
    results_df = _load_results(results_path_value)
    scores_by_well = _load_scores(scores_path_value)
    fi_data = _load_feature_importance_data(spec, detector_key)
    summary_payload = load_json(summary_path(spec, detector_key))
    predictions_path_value = predicted_starts_path(spec, detector_key)
    pred_df = _load_predicted_starts(predictions_path_value)
    if not summary_payload:
        if not pred_df.empty:
            split_summaries = {}
            for split_name in ["all", "train", "test"]:
                subset = intervals_df if split_name == "all" else intervals_df[intervals_df["split"] == split_name]
                if subset.empty:
                    continue
                split_summaries[split_name], _ = evaluate_predictions(subset, pred_df, None)
            summary_payload = {"detector": detector_key, "splits": split_summaries}

    cards = []
    for split_name in ["all", "train", "test"]:
        split_payload = summary_payload.get("splits", {}).get(split_name)
        if not split_payload:
            continue
        cards.append(
            f"""
            <section class="card">
              <h3>{SPLIT_LABELS[split_name]}</h3>
              <div class="metrics">
                <span><b>Интервалы:</b> {escape(str(split_payload.get('interval_count', 0)))}</span>
                <span><b>Найдено:</b> {f"{split_payload.get('hit_count', 0)}/{split_payload.get('interval_count', 0)}"}</span>
                <span><b>Доля найденных:</b> {_format_float(100.0 * float(split_payload.get('hit_rate', 0.0)), 1, '%')}</span>
                <span><b>Оценено моделью:</b> {f"{split_payload.get('assessed_interval_count', split_payload.get('interval_count', 0))}/{split_payload.get('interval_count', 0)}"}</span>
                <span><b>Покрытие оценки:</b> {_format_float(100.0 * float(split_payload.get('coverage_rate', 1.0)), 1, '%')}</span>
                <span><b>Доля найденных среди оценённых:</b> {_format_float(100.0 * float(split_payload.get('hit_rate_on_assessed', split_payload.get('hit_rate', 0.0))), 1, '%')}</span>
                <span><b>P90 задержка:</b> {_format_float(split_payload.get('p90_abs_delay_hours'), 1, ' ч')}</span>
                <span><b>Ложные срабатывания в сутки:</b> {_format_float(split_payload.get('false_alarms_per_day'), 3)}</span>
                <span><b>Episode FAR:</b> {_format_float(split_payload.get('episode_far'), 3)}</span>
                <span><b>Повторы внутри интервалов:</b> {escape(str(split_payload.get('duplicate_starts_inside_interval', 0)))}</span>
                <span><b>Алертов на найденный интервал:</b> {_format_float(split_payload.get('alerts_per_detected_interval'), 2)}</span>
              </div>
            </section>
            """
        )

    sections = []
    for idx, result_row in results_df.iterrows():
        well_id = str(result_row["well_id"])
        print(f"  График {idx + 1}/{len(results_df)}: скв. {well_id}, интервал {int(result_row['interval_idx'])}")
        well_scores = scores_by_well.get(well_id)
        score_unavailable_reason = _score_unavailable_reason(well_scores)
        score_warning_html = (
            f'<p class="score-warning">{escape(_score_unavailable_text(score_unavailable_reason))}</p>'
            if score_unavailable_reason is not None
            else ""
        )
        input_contract_html = _input_contract_badge(well_scores)
        well_ts = data_df[data_df["well_id"] == well_id].copy()
        plot_html = _create_plot_html(
            well_df=well_ts,
            result_row=result_row,
            scores_df=well_scores,
            accent=theme["accent"],
            fi_data=fi_data,
        )
        status_text = _status_label(result_row["status"])
        status_pill_class = _status_pill_class(result_row["status"])
        split_text = SPLIT_SHORT_LABELS.get(str(result_row["split"]), str(result_row["split"]))
        domain_decision_html = _domain_decision_html(
            _prediction_for_detection(pred_df, well_id, result_row["detected_time"])
        )
        sections.append(
            f"""
            <article class="interval-card">
              <header>
                <div>
                  <h3>Скважина {escape(well_id)} / интервал {int(result_row['interval_idx'])}</h3>
                  <p class="meta">{escape(split_text)} | {escape(status_text)}</p>
                </div>
                <div class="pill {status_pill_class}">{escape(status_text)}</div>
              </header>
              <div class="interval-meta">
                <span><b>Фактическое начало:</b> {_format_dt(result_row['actual_start'])}</span>
                <span><b>Фактическое окончание:</b> {_format_dt(result_row['actual_end'])}</span>
                <span><b>Время обнаружения:</b> {_format_dt(result_row['detected_time'])}</span>
                <span><b>Задержка:</b> {_format_float(result_row.get('delay_hours'), 2, ' ч')}</span>
              </div>
              <p class="note">Зелёная линия — начало аномалии, красная зона — длительность, фиолетовая пунктирная — момент обнаружения.</p>
              {input_contract_html}
              {domain_decision_html}
              {score_warning_html}
              <div class="plot-wrap">{plot_html if plot_html else '<p>Нет данных для графика</p>'}</div>
            </article>
            """
        )

    # --- Unlabeled wells (no interval, but have scores/predicted_starts) ---
    labeled_wells = set(results_df["well_id"].unique()) if not results_df.empty else set()
    scored_wells = set(scores_by_well.keys())
    unlabeled_wells = sorted(
        (scored_wells | set(pred_df["well_id"].unique() if not pred_df.empty else [])) - labeled_wells
    )

    if unlabeled_wells:
        sections.append('<h2 style="margin-top:32px;">\u0421\u043a\u0432\u0430\u0436\u0438\u043d\u044b \u0431\u0435\u0437 \u0440\u0430\u0437\u043c\u0435\u0442\u043a\u0438 (\u0441\u043b\u0435\u043f\u043e\u0439 \u0442\u0435\u0441\u0442)</h2>')
        for well_id in unlabeled_wells:
            well_preds = []
            well_pred_df = pd.DataFrame()
            if not pred_df.empty:
                well_pred_df = pred_df[pred_df["well_id"] == well_id]
                well_preds = [pd.Timestamp(t) for t in well_pred_df["detected_time"].dropna()]

            well_ts = data_df[data_df["well_id"] == well_id].copy()
            well_scores = scores_by_well.get(well_id)
            score_unavailable_reason = _score_unavailable_reason(well_scores)
            score_warning_html = (
                f'<p class="score-warning">{escape(_score_unavailable_text(score_unavailable_reason))}</p>'
                if score_unavailable_reason is not None
                else ""
            )
            input_contract_html = _input_contract_badge(well_scores)
            print(f"  \u0413\u0440\u0430\u0444\u0438\u043a (\u0431\u0435\u0437 \u0440\u0430\u0437\u043c\u0435\u0442\u043a\u0438): \u0441\u043a\u0432. {well_id}, \u0434\u0435\u0442\u0435\u043a\u0446\u0438\u0439: {len(well_preds)}")

            plot_html = _create_unlabeled_plot_html(
                well_df=well_ts,
                well_id=well_id,
                scores_df=well_scores,
                predicted_starts=well_preds,
                accent=theme["accent"],
                fi_data=fi_data,
            )

            starts_text = ", ".join(_format_dt(t) for t in well_preds) if well_preds else "\u041d\u0435 \u043e\u0431\u043d\u0430\u0440\u0443\u0436\u0435\u043d\u043e"
            n_det = len(well_preds)
            pill_label = f"{n_det} {'\u0434\u0435\u0442\u0435\u043a\u0446\u0438\u044f' if n_det == 1 else '\u0434\u0435\u0442\u0435\u043a\u0446\u0438\u0439'}"
            domain_decision_html = _domain_summary_html(well_pred_df)
            sections.append(
                f"""
                <article class="interval-card">
                  <header>
                    <div>
                      <h3>\u0421\u043a\u0432\u0430\u0436\u0438\u043d\u0430 {escape(well_id)}</h3>
                      <p class="meta">\u0422\u0435\u0441\u0442 (\u0431\u0435\u0437 \u0440\u0430\u0437\u043c\u0435\u0442\u043a\u0438)</p>
                    </div>
                    <div class="pill {'ok' if well_preds else 'miss'}">{escape(pill_label)}</div>
                  </header>
                  <div class="interval-meta">
                    <span><b>\u041e\u0431\u043d\u0430\u0440\u0443\u0436\u0435\u043d\u043d\u044b\u0435 \u0430\u043d\u043e\u043c\u0430\u043b\u0438\u0438:</b> {escape(starts_text)}</span>
                  </div>
                  <p class="note">\u0424\u0438\u043e\u043b\u0435\u0442\u043e\u0432\u0430\u044f \u043f\u0443\u043d\u043a\u0442\u0438\u0440\u043d\u0430\u044f \u043b\u0438\u043d\u0438\u044f \u2014 \u043c\u043e\u043c\u0435\u043d\u0442 \u043e\u0431\u043d\u0430\u0440\u0443\u0436\u0435\u043d\u0438\u044f \u0430\u043b\u0433\u043e\u0440\u0438\u0442\u043c\u043e\u043c. \u0420\u0430\u0437\u043c\u0435\u0442\u043a\u0430 \u0430\u043d\u043e\u043c\u0430\u043b\u0438\u0438 \u043e\u0442\u0441\u0443\u0442\u0441\u0442\u0432\u0443\u0435\u0442.</p>
                  {input_contract_html}
                  {domain_decision_html}
                  {score_warning_html}
                  <div class="plot-wrap">{plot_html if plot_html else '<p>\u041d\u0435\u0442 \u0434\u0430\u043d\u043d\u044b\u0445 \u0434\u043b\u044f \u0433\u0440\u0430\u0444\u0438\u043a\u0430</p>'}</div>
                </article>
                """
            )

    html = f"""
    <!doctype html>
    <html lang="ru">
      <head>
        <meta charset="utf-8">
        <title>{escape(spec.display_name)}: отчёт по детекции</title>
        <style>
          :root {{
            --accent: {theme['accent']};
            --soft: {theme['soft']};
            --soft-alt: {theme['soft_alt']};
            --text: #101828;
            --muted: #475467;
            --border: #d0d5dd;
          }}
          body {{
            margin: 0;
            font-family: "Segoe UI", Arial, sans-serif;
            color: var(--text);
            background: linear-gradient(180deg, var(--soft), #ffffff 40%);
          }}
          .page {{
            max-width: 1320px;
            margin: 0 auto;
            padding: 28px 24px 40px;
          }}
          h1 {{ margin: 0 0 6px; font-size: 28px; }}
          .subtitle {{ color: var(--muted); margin: 0 0 18px; }}
          .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
            margin: 18px 0 26px;
          }}
          .card, .interval-card {{
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: 0 10px 28px rgba(16, 24, 40, 0.06);
          }}
          .card {{
            padding: 18px 20px;
          }}
          .metrics {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px 14px;
            color: var(--muted);
          }}
          .interval-card {{
            padding: 18px 20px;
            margin-bottom: 18px;
            background: linear-gradient(180deg, #fff, var(--soft-alt));
          }}
          .interval-card header {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
          }}
          .interval-card h3 {{ margin: 0; font-size: 18px; }}
          .meta {{ margin: 4px 0 0; color: var(--muted); }}
          .interval-meta {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 8px 14px;
            margin: 14px 0 16px;
            color: var(--muted);
          }}
          .note {{
            margin: 0 0 12px;
            color: var(--muted);
            font-size: 13px;
          }}
          .score-warning {{
            margin: 0 0 12px;
            padding: 10px 12px;
            border: 1px solid #f59e0b;
            border-radius: 12px;
            background: #fffbeb;
            color: #92400e;
            font-weight: 600;
            font-size: 13px;
          }}
          .contract-note {{
            margin: 0 0 12px;
            padding: 9px 12px;
            border-radius: 12px;
            font-size: 13px;
            line-height: 1.45;
          }}
          .contract-note.info {{
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: #1e3a8a;
          }}
          .contract-note.warn {{
            border: 1px solid #f59e0b;
            background: #fffbeb;
            color: #92400e;
            font-weight: 600;
          }}
          .domain-note {{
            margin: 0 0 12px;
            padding: 10px 12px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 600;
          }}
          .domain-note.info {{
            border: 1px solid #99f6e4;
            background: #f0fdfa;
            color: #115e59;
          }}
          .domain-note.warn {{
            border: 1px solid #fdba74;
            background: #fff7ed;
            color: #9a3412;
          }}
          .pill {{
            border-radius: 999px;
            padding: 8px 12px;
            font-weight: 700;
            background: #e4e7ec;
          }}
          .pill.ok {{ color: #027a48; background: #ecfdf3; }}
          .pill.miss {{ color: #b42318; background: #fef3f2; }}
          .pill.warn {{ color: #92400e; background: #fffbeb; }}
          .plot-wrap {{
            width: 100%;
            margin-top: 12px;
          }}
          .plot-wrap img {{
            width: 100%;
            border-radius: 14px;
            border: 1px solid var(--border);
          }}
        </style>
      </head>
      <body>
        <main class="page">
          <h1>{escape(spec.display_name)}</h1>
          <p class="subtitle">Итоговый отчёт по аномалиям. Используемый детектор: <b>{escape(_detector_label(detector_key))}</b>.</p>
          <div class="cards">
            {''.join(cards)}
          </div>
          {''.join(sections)}
        </main>
      </body>
    </html>
    """

    output.write_text(html, encoding="utf-8")
    print(f"Отчёт сгенерирован: {output}")
    return output


def run_cli(default_anomaly: str) -> None:
    parser = argparse.ArgumentParser(description="Generate HTML report for anomaly detection results.")
    parser.add_argument("--detector", default=None, help="Detector key, default is benchmark-selected paano_shared.")
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument("--source", default=None, help="Override source dataset path")
    parser.add_argument("--results", default=None, help="Override results Parquet path")
    parser.add_argument("--scores", default=None, help="Override scores Parquet path")
    args = parser.parse_args()

    generate_report(
        anomaly_key=default_anomaly,
        detector=args.detector,
        output_path=args.output,
        source_path=args.source,
        results_path_override=args.results,
        scores_path_override=args.scores,
    )
