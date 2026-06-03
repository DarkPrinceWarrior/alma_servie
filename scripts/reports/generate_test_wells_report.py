from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANOMALY_LABELS = {
    "negermet": "Негерметичность",
    "pritok": "Приток",
    "salt": "Солеотложение",
}
ANOMALY_ORDER = ("negermet", "pritok", "salt")

SMOOTH_WINDOW_HOURS = 24.0
TREND_EDGE_DAYS = 3.0
FREQUENCY_STABLE_RANGE_HZ = 2.0
DISPLAY_TARGET_POINTS = 2400

COLOR_PRESSURE = "#1d4ed8"
COLOR_SMOOTHED = "#0f766e"
COLOR_FREQUENCY = "#7c3aed"
COLOR_CURRENT = "#ea580c"
COLOR_LOAD = "#ca8a04"
COLOR_TEMP_INTAKE = "#dc2626"
COLOR_TEMP_OIL = "#9f1239"
COLOR_CRITICAL = "#b91c1c"
COLOR_FIRST_ALERT = "#d97706"

_PLOTLY_EMBEDDED = {"done": False}


def normalize_column(name: str) -> str:
    return str(name).strip().lower().replace("ё", "е")


def find_column(df: pd.DataFrame, *needles: str) -> str | None:
    for column in df.columns:
        normalized = normalize_column(column)
        if all(normalize_column(needle) in normalized for needle in needles):
            return str(column)
    return None


def load_batch_summary(batch_dir: Path) -> dict:
    path = batch_dir / "_batch_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Не найден {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_source(well_dir: Path, anomaly: str) -> pd.DataFrame:
    path = well_dir / anomaly / "source.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def load_preds(well_dir: Path, anomaly: str) -> pd.DataFrame:
    path = well_dir / anomaly / "predicted_starts.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "detected_time" in df.columns:
        df["detected_time"] = pd.to_datetime(df["detected_time"])
    return df


def load_well_summary(well_dir: Path, anomaly: str) -> dict:
    path = well_dir / anomaly / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def critical_detections(preds: pd.DataFrame) -> list[pd.Timestamp]:
    if preds.empty or "event_class" not in preds.columns:
        return []
    event = preds["event_class"].fillna("").astype(str).str.lower()
    rows = preds.loc[event != "early_warning"]
    if "actionable_alert" in rows.columns:
        rows = rows[rows["actionable_alert"].fillna(False).astype(bool)]
    return sorted(pd.to_datetime(rows["detected_time"]).tolist())


def actionable_alerts(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty or "actionable_alert" not in preds.columns:
        return pd.DataFrame()
    rows = preds[preds["actionable_alert"].fillna(False).astype(bool)].copy()
    return rows.sort_values("detected_time") if not rows.empty else rows


def smooth_series(timestamps: pd.Series, values: pd.Series) -> pd.Series:
    indexed = pd.Series(np.asarray(values, dtype=float), index=pd.to_datetime(timestamps))
    return indexed.rolling(f"{int(SMOOTH_WINDOW_HOURS)}h", min_periods=12).median()


def downsample(df: pd.DataFrame, target: int = DISPLAY_TARGET_POINTS) -> pd.DataFrame:
    if len(df) <= target:
        return df
    indexed = df.set_index("timestamp").select_dtypes(include=[np.number])
    duration_min = (indexed.index[-1] - indexed.index[0]).total_seconds() / 60.0
    step = max(5, int(np.ceil(duration_min / target)))
    return indexed.resample(f"{step}min").median().reset_index()


def trend_summary(source: pd.DataFrame) -> dict:
    pressure_col = find_column(source, "давление на приеме")
    frequency_col = find_column(source, "выходная частота")
    out: dict[str, object] = {"pressure_col": pressure_col, "frequency_col": frequency_col}
    if pressure_col is None or source.empty:
        return out
    smoothed = smooth_series(source["timestamp"], source[pressure_col]).dropna()
    if smoothed.empty:
        return out
    edge = pd.Timedelta(days=TREND_EDGE_DAYS)
    head = smoothed[smoothed.index <= smoothed.index[0] + edge]
    tail = smoothed[smoothed.index >= smoothed.index[-1] - edge]
    p_start = float(head.median())
    p_end = float(tail.median())
    delta_pct = (p_end / p_start - 1.0) * 100.0 if p_start > 0 else 0.0
    duration_days = (smoothed.index[-1] - smoothed.index[0]).total_seconds() / 86400.0
    out.update(
        {
            "p_start": p_start,
            "p_end": p_end,
            "delta_pct": delta_pct,
            "rate_pct_per_day": delta_pct / duration_days if duration_days > 0 else 0.0,
            "duration_days": duration_days,
        }
    )
    if frequency_col is not None:
        freq = source[frequency_col].dropna()
        working = freq[freq >= 1.0]
        if len(working):
            out["freq_median"] = float(working.median())
            out["freq_range"] = float(working.max() - working.min())
            out["freq_stable"] = bool(out["freq_range"] <= FREQUENCY_STABLE_RANGE_HZ)
    return out


def trend_direction_text(delta_pct: float) -> str:
    if delta_pct >= 3.0:
        return "трендово растёт"
    if delta_pct <= -3.0:
        return "трендово снижается"
    return "стабильно"


def build_explanation(
    anomaly: str,
    source: pd.DataFrame,
    preds: pd.DataFrame,
    summary: dict,
) -> str:
    trend = trend_summary(source)
    criticals = critical_detections(preds)
    alerts = actionable_alerts(preds)
    n_alerts = len(alerts)
    score_max = summary.get("score_max")

    parts: list[str] = []
    if trend.get("p_start") is not None:
        direction = trend_direction_text(float(trend["delta_pct"]))
        parts.append(
            f"Сглаженное давление на приёме {direction}: "
            f"{trend['p_start']:.1f} → {trend['p_end']:.1f} кгс/см² "
            f"({trend['delta_pct']:+.1f}% за {trend['duration_days']:.0f} суток, "
            f"{trend['rate_pct_per_day']:+.2f}%/сутки)."
        )
    if trend.get("freq_median") is not None:
        freq_note = "стабильна" if trend.get("freq_stable") else "менялась"
        parts.append(
            f"Выходная частота {freq_note}: медиана {trend['freq_median']:.1f} Гц, "
            f"размах {trend['freq_range']:.1f} Гц."
        )

    if n_alerts > 0:
        first_alert = pd.Timestamp(alerts["detected_time"].iloc[0])
        verdict_head = "<strong>Решение детектора: аномалия (приток) обнаружена.</strong>"
        reason_bits: list[str] = [
            f"Действующих алертов: {n_alerts}, первый — {first_alert.strftime('%d.%m.%Y %H:%M')}."
        ]
        if criticals:
            crit_list = ", ".join(ts.strftime("%d.%m.%Y %H:%M") for ts in criticals[:4])
            reason_bits.append(f"Подтверждённые детекции (красные линии): {crit_list}.")
        if trend.get("p_start") is not None and abs(float(trend["delta_pct"])) >= 3.0:
            freq_part = (
                "при стабильной выходной частоте"
                if trend.get("freq_stable")
                else "при этом частота менялась — детектор учёл это"
            )
            reason_bits.append(
                f"Причина: трендовое изменение давления ({trend['delta_pct']:+.1f}%) {freq_part} — "
                "классический признак притока; поведение скважины не похоже на банк нормальной работы."
            )
        else:
            reason_bits.append(
                "Причина: многоканальное представление скважины устойчиво отклоняется от банка "
                "нормальной популяции (детектор видит не только давление, но и токи, загрузку, температуры)."
            )
        verdict_body = " ".join(reason_bits)
        css_class = "вывод-детекция"
    else:
        verdict_head = "<strong>Решение детектора: аномалии нет.</strong>"
        reason_bits = []
        if trend.get("p_start") is not None:
            if abs(float(trend["delta_pct"])) < 3.0:
                reason_bits.append(
                    f"Причина: сглаженное давление стабильно ({trend['delta_pct']:+.1f}% за весь период) — "
                    "трендового роста или снижения, характерного для притока, нет."
                )
            else:
                freq_explained = (
                    " Изменение давления сопровождалось изменением частоты — это реакция на смену режима, не приток."
                    if not trend.get("freq_stable")
                    else ""
                )
                reason_bits.append(
                    f"Причина: давление изменилось на {trend['delta_pct']:+.1f}%, но поведение скважины "
                    f"остаётся в пределах банка нормальной работы.{freq_explained}"
                )
        if isinstance(score_max, (int, float)):
            reason_bits.append(f"Максимальный скор детектора за период: {float(score_max):.4f} — порог не достигнут.")
        verdict_body = " ".join(reason_bits)
        css_class = "вывод-норма"

    return (
        f"<div class='{css_class}'>"
        f"<p>{verdict_head}</p>"
        f"<p>{' '.join(parts)}</p>"
        f"<p>{verdict_body}</p>"
        "</div>"
    )


def build_well_figure(
    source: pd.DataFrame,
    preds: pd.DataFrame,
    anomaly: str,
) -> go.Figure | None:
    pressure_col = find_column(source, "давление на приеме")
    if source.empty or pressure_col is None:
        return None
    frequency_col = find_column(source, "выходная частота")
    current_col = find_column(source, "ток на фазе а")
    load_col = find_column(source, "коэффициент загрузки")
    temp_intake_col = find_column(source, "температура на приеме")
    temp_oil_col = find_column(source, "температура масла")

    smoothed_full = smooth_series(source["timestamp"], source[pressure_col])
    display = downsample(source)
    smoothed_display = smooth_series(display["timestamp"], display[find_column(display, "давление на приеме")])

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.34, 0.24, 0.16, 0.26],
        vertical_spacing=0.045,
        subplot_titles=(
            "Давление на приёме насоса",
            "Давление на приёме — сглаженное (тренд, медиана за 24 часа)",
            "Выходная частота",
            "Токи, загрузка ПЭД и температуры",
        ),
        specs=[[{}], [{}], [{}], [{"secondary_y": True}]],
    )

    x_display = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    pressure_display = display[find_column(display, "давление на приеме")]

    fig.add_trace(
        go.Scatter(
            x=x_display, y=[None if pd.isna(v) else round(float(v), 2) for v in pressure_display],
            mode="lines", name="Давление на приёме",
            line={"color": COLOR_PRESSURE, "width": 1.4},
            hovertemplate="%{x|%d.%m.%Y %H:%M}<br>Давление: %{y:.2f} кгс/см²<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_display,
            y=[None if pd.isna(v) else round(float(v), 2) for v in smoothed_display.to_numpy()],
            mode="lines", name="Давление сглаженное",
            line={"color": COLOR_SMOOTHED, "width": 2.4},
            hovertemplate="%{x|%d.%m.%Y %H:%M}<br>Сглаженное: %{y:.2f} кгс/см²<extra></extra>",
        ),
        row=2, col=1,
    )
    if frequency_col:
        freq_display = display[find_column(display, "выходная частота")]
        fig.add_trace(
            go.Scatter(
                x=x_display, y=[None if pd.isna(v) else round(float(v), 1) for v in freq_display],
                mode="lines", name="Выходная частота",
                line={"color": COLOR_FREQUENCY, "width": 1.3},
                hovertemplate="%{x|%d.%m.%Y %H:%M}<br>Частота: %{y:.1f} Гц<extra></extra>",
            ),
            row=3, col=1,
        )
    for column_key, name, color, unit, secondary in (
        ("ток на фазе а", "Ток фазы А", COLOR_CURRENT, "А", False),
        ("коэффициент загрузки", "Загрузка ПЭД", COLOR_LOAD, "%", False),
        ("температура на приеме", "Температура на приёме", COLOR_TEMP_INTAKE, "°C", True),
        ("температура масла", "Температура масла", COLOR_TEMP_OIL, "°C", True),
    ):
        column = find_column(display, column_key)
        if column is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=x_display, y=[None if pd.isna(v) else round(float(v), 1) for v in display[column]],
                mode="lines", name=name,
                line={"color": color, "width": 1.2, "dash": "dot" if secondary else "solid"},
                hovertemplate=f"%{{x|%d.%m.%Y %H:%M}}<br>{name}: %{{y:.1f}} {unit}<extra></extra>",
            ),
            row=4, col=1, secondary_y=secondary,
        )

    # Вертикальные линии: подтверждённые детекции (красные) и первый действующий алерт (оранжевая)
    shapes = []
    annotations = []
    criticals = critical_detections(preds)
    alerts = actionable_alerts(preds)
    if not alerts.empty:
        first_alert = pd.Timestamp(alerts["detected_time"].iloc[0])
        shapes.append(
            {
                "type": "line", "xref": "x", "yref": "paper",
                "x0": first_alert.strftime("%Y-%m-%d %H:%M"), "x1": first_alert.strftime("%Y-%m-%d %H:%M"),
                "y0": 0, "y1": 1, "line": {"color": COLOR_FIRST_ALERT, "width": 1.8, "dash": "dash"},
            }
        )
        annotations.append(
            {
                "x": first_alert.strftime("%Y-%m-%d %H:%M"), "y": 1.05, "xref": "x", "yref": "paper",
                "text": "Первый действующий алерт", "showarrow": False,
                "font": {"size": 11, "color": COLOR_FIRST_ALERT}, "xanchor": "left",
            }
        )
    for idx, ts in enumerate(criticals):
        shapes.append(
            {
                "type": "line", "xref": "x", "yref": "paper",
                "x0": ts.strftime("%Y-%m-%d %H:%M"), "x1": ts.strftime("%Y-%m-%d %H:%M"),
                "y0": 0, "y1": 1, "line": {"color": COLOR_CRITICAL, "width": 2.0},
            }
        )
        if idx < 4:
            annotations.append(
                {
                    "x": ts.strftime("%Y-%m-%d %H:%M"), "y": 0.99, "xref": "x", "yref": "paper",
                    "text": "Детекция", "showarrow": False, "textangle": -90,
                    "font": {"size": 10, "color": COLOR_CRITICAL}, "xanchor": "right", "yanchor": "top",
                }
            )

    axis_style = {
        "showgrid": True, "gridcolor": "#eef2f7", "linecolor": "#cbd5e1",
        "ticks": "outside", "tickcolor": "#cbd5e1", "zeroline": False,
    }
    fig.update_layout(
        height=760,
        margin={"l": 64, "r": 56, "t": 40, "b": 40},
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        hovermode="x unified",
        font={"family": "'Segoe UI', 'PT Sans', Arial, sans-serif", "size": 12, "color": "#1e293b"},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.1, "x": 0, "font": {"size": 11}},
        shapes=shapes,
        annotations=list(fig.layout.annotations) + annotations,
    )
    fig.update_xaxes(**axis_style, tickformat="%d.%m.%y")
    fig.update_yaxes(**axis_style)
    fig.update_yaxes(title_text="кгс/см²", title_font={"size": 11, "color": "#64748b"}, row=1, col=1)
    fig.update_yaxes(title_text="кгс/см²", title_font={"size": 11, "color": "#64748b"}, row=2, col=1)
    fig.update_yaxes(title_text="Гц", title_font={"size": 11, "color": "#64748b"}, row=3, col=1)
    fig.update_yaxes(title_text="А / %", title_font={"size": 11, "color": "#64748b"}, row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="°C", title_font={"size": 11, "color": "#64748b"}, row=4, col=1, secondary_y=True)
    for annotation in fig.layout.annotations[:4]:
        annotation.font = {"size": 13, "color": "#334155"}
        annotation.x = 0.0
        annotation.xanchor = "left"
    return fig


def render_well_card(well_id: str, well_dir: Path, anomaly: str) -> str:
    source = load_source(well_dir, anomaly)
    preds = load_preds(well_dir, anomaly)
    summary = load_well_summary(well_dir, anomaly)

    alerts = actionable_alerts(preds)
    n_alerts = len(alerts)
    if n_alerts > 0:
        status_badge = f"<span class='бейдж' style='background:#b91c1c'>Аномалия обнаружена · алертов: {n_alerts}</span>"
    else:
        status_badge = "<span class='бейдж' style='background:#059669'>Аномалий нет</span>"

    period_text = ""
    if not source.empty:
        start = source["timestamp"].iloc[0]
        end = source["timestamp"].iloc[-1]
        period_text = f"Период данных: {start.strftime('%d.%m.%Y')} — {end.strftime('%d.%m.%Y')}"

    figure = build_well_figure(source, preds, anomaly)
    if figure is None:
        chart_html = "<div class='нет-данных'>Нет данных по давлению для этой скважины.</div>"
    else:
        include_js = "inline" if not _PLOTLY_EMBEDDED["done"] else False
        _PLOTLY_EMBEDDED["done"] = True
        chart_html = figure.to_html(
            full_html=False,
            include_plotlyjs=include_js,
            div_id=f"график-{well_id}-{anomaly}",
            config={"responsive": True, "displayModeBar": False, "doubleClick": "reset"},
        )

    explanation_html = build_explanation(anomaly, source, preds, summary)

    return (
        f"<section class='карточка-скважины' id='скважина-{escape(well_id)}'>"
        f"<div class='заголовок'><h2>Скважина {escape(well_id)} — {escape(ANOMALY_LABELS[anomaly])}</h2>{status_badge}</div>"
        f"<div class='подпись'>{escape(period_text)}</div>"
        f"{chart_html}"
        f"{explanation_html}"
        "</section>"
    )


REPORT_CSS = """
<style>
* { box-sizing: border-box; }
body {
  margin: 0; background: #f1f5f9; color: #0f172a;
  font-family: 'Segoe UI', 'PT Sans', Arial, sans-serif; font-size: 15px; line-height: 1.55;
  padding: 28px 36px 80px 36px; max-width: 1280px; margin: 0 auto;
}
.шапка h1 { font-size: 24px; margin: 0 0 6px 0; }
.шапка .подзаголовок { color: #64748b; font-size: 14px; margin-bottom: 26px; }
.карточка-скважины {
  background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
  padding: 20px 24px; margin-bottom: 28px;
}
.карточка-скважины .заголовок { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-bottom: 4px; }
.карточка-скважины .заголовок h2 { margin: 0; font-size: 19px; }
.карточка-скважины .подпись { color: #64748b; font-size: 13px; margin-bottom: 14px; }
.бейдж {
  display: inline-block; padding: 3px 13px; border-radius: 999px; font-size: 12.5px; font-weight: 600;
  color: #ffffff; white-space: nowrap;
}
.вывод-детекция, .вывод-норма {
  margin-top: 14px; padding: 14px 18px; border-radius: 10px; font-size: 14px;
}
.вывод-детекция { background: #fef2f2; border-left: 4px solid #b91c1c; }
.вывод-норма { background: #f0fdf4; border-left: 4px solid #059669; }
.вывод-детекция p, .вывод-норма p { margin: 5px 0; }
.нет-данных { padding: 36px; text-align: center; color: #64748b; background: #f8fafc; border-radius: 10px; }
.легенда { display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 22px; font-size: 13.5px; }
.легенда .элемент { display: flex; align-items: center; gap: 8px; }
.легенда .линия { width: 30px; height: 0; display: inline-block; }
footer { color: #64748b; font-size: 12.5px; margin-top: 50px; border-top: 1px solid #e2e8f0; padding-top: 14px; }
</style>
"""


def render_report(batch_dir: Path, anomalies: tuple[str, ...]) -> str:
    batch = load_batch_summary(batch_dir)
    wells = batch.get("wells", [])
    generated_at = datetime.now().strftime("%d.%m.%Y %H:%M")
    anomaly_titles = ", ".join(ANOMALY_LABELS[a] for a in anomalies)

    legend_html = (
        "<div class='легенда'>"
        f"<div class='элемент'><span class='линия' style='border-top:2px solid {COLOR_CRITICAL}'></span> Детекция (подтверждённый старт аномалии)</div>"
        f"<div class='элемент'><span class='линия' style='border-top:2px dashed {COLOR_FIRST_ALERT}'></span> Первый действующий алерт</div>"
        f"<div class='элемент'><span class='линия' style='border-top:3px solid {COLOR_SMOOTHED}'></span> Сглаженное давление (тренд)</div>"
        "</div>"
    )

    header_html = (
        "<div class='шапка'>"
        f"<h1>Тестовые скважины — результаты детекции ({escape(anomaly_titles)})</h1>"
        f"<div class='подзаголовок'>Скважин: {len(wells)} · Детектор: paano_global · Сформировано: {generated_at}</div>"
        f"{legend_html}"
        "</div>"
    )

    cards: list[str] = []
    for well_summary in wells:
        well_id = str(well_summary.get("well_id", ""))
        well_dir = batch_dir / well_id
        for anomaly in anomalies:
            cards.append(render_well_card(well_id, well_dir, anomaly))

    return (
        "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Тестовые скважины — результаты детекции</title>"
        + REPORT_CSS
        + "</head><body>"
        + header_html
        + "".join(cards)
        + "<footer>Отчёт сформирован автоматически. Все графики работают без подключения к сети. "
        "Сглаженное давление — скользящая медиана за 24 часа: по ней видно трендовое увеличение или снижение без шума.</footer>"
        + "</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Сводный HTML по batch test_wells (карточки в стиле отчёта по физике).")
    parser.add_argument("--batch-dir", default="artifacts/test_wells")
    parser.add_argument(
        "--output",
        default="artifacts/reports/test_wells/test_wells_paano_global_report.html",
    )
    parser.add_argument("--anomalies", default=",".join(ANOMALY_ORDER))
    parser.add_argument("--no-disclaimer", action="store_true", help="Совместимость: блок «Внимание» больше не выводится.")
    parser.add_argument("--no-frequency-table", action="store_true", help="Совместимость: таблица частоты больше не выводится.")
    parser.add_argument("--note", default=None, help="Совместимость: заметка больше не выводится.")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        raise FileNotFoundError(batch_dir)

    anomalies = tuple(a.strip().lower() for a in args.anomalies.split(",") if a.strip())
    unknown = [a for a in anomalies if a not in ANOMALY_ORDER]
    if unknown:
        raise ValueError(f"Неизвестные типы аномалий: {unknown}; поддерживаются: {ANOMALY_ORDER}")

    html = render_report(batch_dir, anomalies)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Отчёт: {output_path} ({size_mb:.1f} МБ)")


if __name__ == "__main__":
    main()
