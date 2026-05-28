"""Сводный HTML по batch test_wells — простой формат.

Для каждой скв. × тип аномалии — один график: «Давление на приеме насоса»
с вертикальными линиями на найденных стартах детектора. Никаких сводных
таблиц и сложных subplots. Plotly bundled (offline), zero-English.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANOMALY_LABELS = {
    "negermet": "Негерметичность",
    "pritok": "Приток",
    "salt": "Солеотложение",
}
ANOMALY_ORDER = ("negermet", "pritok", "salt")
PRESSURE_COLUMN = "Давление на приеме насоса кгс/см²"

COLOR_PRESSURE = "#1e40af"
COLOR_CRITICAL = "#dc2626"
COLOR_EARLY = "#f59e0b"
COLOR_BG = "#fafafa"


def load_batch_summary(batch_dir: Path) -> dict:
    path = batch_dir / "_batch_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Не найден {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_source(well_dir: Path, anomaly: str) -> pd.DataFrame:
    path = well_dir / anomaly / "source.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=["timestamp", PRESSURE_COLUMN])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_preds(well_dir: Path, anomaly: str) -> pd.DataFrame:
    path = well_dir / anomaly / "predicted_starts.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def extract_critical(preds: pd.DataFrame) -> list[pd.Timestamp]:
    if preds.empty:
        return []
    event = preds.get("event_class", pd.Series([""] * len(preds))).fillna("").astype(str).str.lower()
    times = pd.to_datetime(preds["detected_time"])
    return sorted(times[event != "early_warning"].tolist())


_PLOTLY_EMBEDDED = {"done": False}


def render_chart(well_id: str, anomaly: str, well_dir: Path) -> str:
    source = load_source(well_dir, anomaly)
    preds = load_preds(well_dir, anomaly)
    if source.empty or PRESSURE_COLUMN not in source.columns:
        return f"<p class='no-data'>Скв. {escape(well_id)} — {escape(ANOMALY_LABELS[anomaly])}: нет данных по давлению.</p>"

    critical = extract_critical(preds)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=source["timestamp"],
            y=source[PRESSURE_COLUMN],
            mode="lines",
            line={"color": COLOR_PRESSURE, "width": 1.6, "shape": "spline", "smoothing": 0.4},
            name="Давление на приёме",
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>P = %{y:.2f} кгс/см²<extra></extra>",
        )
    )

    shapes = []
    for ts in critical:
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": ts,
                "x1": ts,
                "y0": 0,
                "y1": 1,
                "line": {"color": COLOR_CRITICAL, "width": 2.0},
            }
        )

    annotations = []
    if critical:
        y_max = float(source[PRESSURE_COLUMN].max())
        for ts in critical:
            annotations.append(
                {
                    "x": ts,
                    "y": y_max,
                    "yref": "y",
                    "xref": "x",
                    "showarrow": False,
                    "text": "▼",
                    "font": {"color": COLOR_CRITICAL, "size": 14},
                    "yshift": 6,
                }
            )

    fig.update_layout(
        title={
            "text": f"<b>Скв. {well_id}</b> — {ANOMALY_LABELS[anomaly]}",
            "font": {"size": 16, "color": "#0f172a"},
            "x": 0.02,
            "xanchor": "left",
        },
        height=340,
        margin={"l": 70, "r": 30, "t": 60, "b": 50},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
        shapes=shapes,
        annotations=annotations,
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "size": 12, "color": "#1e293b"},
        xaxis={
            "title": None,
            "showgrid": True,
            "gridcolor": "#f1f5f9",
            "linecolor": "#cbd5e1",
            "ticks": "outside",
            "tickcolor": "#cbd5e1",
        },
        yaxis={
            "title": {"text": "P, кгс/см²", "font": {"size": 12, "color": "#64748b"}},
            "showgrid": True,
            "gridcolor": "#f1f5f9",
            "linecolor": "#cbd5e1",
            "ticks": "outside",
            "tickcolor": "#cbd5e1",
            "zeroline": False,
        },
        showlegend=False,
    )

    include_js = "inline" if not _PLOTLY_EMBEDDED["done"] else False
    _PLOTLY_EMBEDDED["done"] = True
    div_id = f"chart_{well_id}_{anomaly}"
    n_crit = len(critical)
    caption = f"Аномалий обнаружено: <strong>{n_crit}</strong>"
    chart_html = fig.to_html(full_html=False, include_plotlyjs=include_js, div_id=div_id)
    return f"<div class='chart-block'><div class='chart-caption'>{caption}</div>{chart_html}</div>"


HEAD_CSS = """
<style>
body { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; margin: 0; padding: 28px 40px; background: #f8fafc; color: #0f172a; }
h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 18px; margin: 24px 0 8px; color: #1e293b; letter-spacing: -0.01em; }
.header { background: white; padding: 20px 26px; border-radius: 14px; margin-bottom: 22px; box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04); }
.subtitle { color: #64748b; font-size: 13px; }
.well-block { background: white; padding: 18px 24px 22px; border-radius: 14px; margin-bottom: 22px; box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04); }
.chart-block { margin: 6px 0 18px; }
.chart-caption { color: #475569; font-size: 13px; margin: 8px 0 0 72px; }
.chart-caption strong { color: #b91c1c; }
.legend-row { display: flex; gap: 22px; font-size: 12px; color: #475569; margin-top: 12px; flex-wrap: wrap; }
.legend-line { display: inline-block; width: 22px; height: 2px; vertical-align: middle; margin-right: 8px; border-radius: 2px; }
.legend-line.critical { background: #dc2626; height: 3px; }
.legend-line.pressure { background: #1e40af; height: 3px; }
.disclaimer { background: #fef9c3; border-left: 4px solid #facc15; padding: 12px 16px; border-radius: 8px; color: #713f12; font-size: 13px; margin-top: 14px; line-height: 1.5; }
.no-data { color: #94a3b8; font-style: italic; padding: 14px; }
</style>
"""

LEGEND_BLOCK = """
<div class='legend-row'>
  <span><span class='legend-line pressure'></span>Давление на приёме насоса</span>
  <span><span class='legend-line critical'></span>Обнаруженная аномалия</span>
</div>
"""

LOCAL_REFERENCE_DISCLAIMER_BLOCK = """
<div class='disclaimer'>
<strong>Внимание.</strong> Скважины без экспертной разметки. Эталон нормы построен
из первых 20 % точек ряда (≈10 суток на 49-дневном ряду). Если начальный участок
содержал отклонения, рассчитанные старты могут быть смещены или приглушены —
эксперт должен визуально подтвердить штатность начального участка.
</div>
"""

POPULATION_MEMORY_BANK_DISCLAIMER_BLOCK = """
<div class='disclaimer'>
<strong>Внимание.</strong> Скважины без экспертной разметки. Отклонение от нормы
рассчитано через один global-normality PaAno encoder и общий population memory
bank: нормальные окна train-скважин одной физической сущности — скважин УЭЦН.
Первые 20 % ряда остаются локальным blind-контекстом для нормировки и порога,
но PaAno memory bank берётся из population reference.
</div>
"""


def render_disclaimer(batch: dict) -> str:
    if bool(batch.get("use_population_memory_bank")):
        return POPULATION_MEMORY_BANK_DISCLAIMER_BLOCK
    return LOCAL_REFERENCE_DISCLAIMER_BLOCK


def render_report(batch_dir: Path, anomalies: tuple[str, ...]) -> str:
    batch = load_batch_summary(batch_dir)
    n_wells = len(batch.get("wells", []))
    detector = batch.get("detector_choice", "—")
    elapsed = batch.get("elapsed_seconds")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    subtitle_parts = [
        f"Скважин: {n_wells}",
        f"Детектор: <strong>{escape(str(detector))}</strong>",
        f"Типов аномалий: {len(anomalies)}",
    ]
    if elapsed is not None:
        subtitle_parts.append(f"Время прогона: {elapsed:.1f} с")
    subtitle_parts.append(f"Сгенерировано: {generated_at}")

    header_html = (
        "<div class='header'>"
        "<h1>Прогон тестовых скважин — отчёт детекции</h1>"
        f"<div class='subtitle'>{' · '.join(subtitle_parts)}</div>"
        + LEGEND_BLOCK
        + render_disclaimer(batch)
        + "</div>"
    )

    sections: list[str] = []
    for well_summary in batch.get("wells", []):
        well_id = str(well_summary.get("well_id", ""))
        well_dir = batch_dir / well_id
        charts = [render_chart(well_id, a, well_dir) for a in anomalies]
        sections.append(
            f"<section class='well-block'>"
            f"<h2>Скважина {escape(well_id)}</h2>"
            + "".join(charts)
            + "</section>"
        )

    body = (
        "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>"
        "<title>Тестовые скважины — отчёт детекции</title>"
        + HEAD_CSS + "</head><body>"
        + header_html
        + "".join(sections)
        + "</body></html>"
    )
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description="Сводный HTML по batch test_wells (простой формат).")
    parser.add_argument("--batch-dir", default="artifacts/test_wells")
    parser.add_argument(
        "--output",
        default="artifacts/reports/test_wells/test_wells_paano_global_report.html",
    )
    parser.add_argument("--anomalies", default=",".join(ANOMALY_ORDER))
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        raise FileNotFoundError(batch_dir)
    anomalies = tuple(a.strip().lower() for a in args.anomalies.split(",") if a.strip())
    unknown = [a for a in anomalies if a not in ANOMALY_ORDER]
    if unknown:
        raise ValueError(f"Неизвестные аномалии: {unknown}")

    html = render_report(batch_dir, anomalies)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"HTML saved to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
