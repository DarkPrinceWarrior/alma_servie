"""Generate per-well anomaly PDF reports based on rule detections."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Matplotlib defaults for readable Russian plots
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.figsize": (12, 7),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DETECTIONS_PATH = PROJECT_ROOT / "reports" / "anomalies" / "anomaly_analysis.parquet"
MERGED_PATH = PROJECT_ROOT / "data" / "processed" / "merged_hourly.parquet"
OUTPUT_DIR = HERE / "well_pipeline_anom"

CONTEXT_HOURS = 12

METRIC_GROUPS = [
    (
        "Электропараметры",
        [
            "Выходная частота",
            "Ток фазы A",
        ],
    ),
    (
        "Давление",
        [
            "Давление на приеме насоса",
            "Устьевое давление",
        ],
    ),
    (
        "Производительность",
        [
            "Объемный дебит жидкости, м3/сут",
        ],
    ),
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DETECTIONS_PATH.exists():
        raise FileNotFoundError(
            "Не найден файл с детекциями. Запустите `PYTHONPATH=src python -m pipeline anomalies`."
        )
    if not MERGED_PATH.exists():
        raise FileNotFoundError(
            "Не найдены объединённые данные (`data/processed/merged_hourly.parquet`). "
            "Выполните шаги clean/align перед генерацией PDF."
        )

    detections = pd.read_parquet(DETECTIONS_PATH)
    detections["start"] = pd.to_datetime(detections["start"])
    detections["end"] = pd.to_datetime(detections["end"])
    detections["well_number"] = detections["well_number"].astype(str)
    detections.sort_values(["well_number", "start", "end"], inplace=True)

    merged = pd.read_parquet(MERGED_PATH)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged["well_number"] = merged["well_number"].astype(str)
    merged.sort_values(["well_number", "timestamp"], inplace=True)

    return detections, merged


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_cover_page(well: str, well_detections: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(11.7, 8.3))
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.suptitle(f"Скважина {well}: сводка по найденным аномалиям", fontsize=18, fontweight="bold")

    lines = []
    for idx, row in enumerate(well_detections.itertuples(), start=1):
        duration = row.duration_hours
        label = getattr(row, "rule_label", row.rule_name)
        notes = getattr(row, "notes", "") or "Комментарий отсутствует."
        lines.append(
            f"{idx}. {label}\n"
            f"   Период: {row.start:%d.%m.%Y %H:%M} – {row.end:%d.%m.%Y %H:%M} "
            f"(длительность {duration:.1f} ч)\n"
            f"   Описание: {notes}"
        )

    text = "\n\n".join(lines) if lines else "Для скважины не найдено аномалий."
    ax.text(0.02, 0.95, text, ha="left", va="top", fontsize=11, linespacing=1.4)
    return fig


def plot_detection(
    well_data: pd.DataFrame,
    detection: pd.Series,
) -> plt.Figure | None:
    start = detection["start"]
    end = detection["end"]
    window_start = start - pd.Timedelta(hours=CONTEXT_HOURS)
    window_end = end + pd.Timedelta(hours=CONTEXT_HOURS)

    window_df = well_data[(well_data["timestamp"] >= window_start) & (well_data["timestamp"] <= window_end)].copy()
    if window_df.empty:
        return None

    fig, axes = plt.subplots(len(METRIC_GROUPS), 1, figsize=(12, 3.2 * len(METRIC_GROUPS)), sharex=True)
    if len(METRIC_GROUPS) == 1:
        axes = [axes]

    title_rule = detection.get("rule_label", detection["rule_name"])
    fig.suptitle(
        f"{title_rule}\n"
        f"{start:%d.%m.%Y %H:%M} – {end:%d.%м.%Y %H:%M} "
        f"(длительность {detection['duration_hours']:.1f} ч, score={detection.get('score', float('nan')):.2f})",
        fontsize=15,
        fontweight="bold",
    )

    interval_df = window_df[(window_df["timestamp"] >= start) & (window_df["timestamp"] <= end)]

    for ax, (group_name, metrics) in zip(axes, METRIC_GROUPS):
        plotted = False
        for metric in metrics:
            if metric not in window_df.columns:
                continue
            series = window_df[["timestamp", metric]].dropna()
            if series.empty:
                continue
            plotted = True
            ax.plot(series["timestamp"], series[metric], label=metric, linewidth=1.4)

            interval_series = interval_df[["timestamp", metric]].dropna()
            if not interval_series.empty:
                ax.plot(
                    interval_series["timestamp"],
                    interval_series[metric],
                    linewidth=2.0,
                    color=ax.lines[-1].get_color(),
                )
        ax.axvspan(start, end, color="#ff9896", alpha=0.2)
        ax.axvline(start, color="#d62728", linestyle="--", linewidth=1)
        ax.axvline(end, color="#d62728", linestyle="--", linewidth=1)
        ax.set_ylabel(group_name)
        if plotted:
            ax.legend(loc="upper left", fontsize=9)
        else:
            ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes, color="gray")

    axes[-1].set_xlabel("Дата и время")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
    fig.autofmt_xdate(rotation=20)

    notes = detection.get("notes", "") or "Комментарий отсутствует."
    focus = detection.get("focus_metrics", "") or ""
    footer_lines = [
        f"Правило: {title_rule}",
        f"Описание: {detection.get('rule_description', '—') or '—'}",
        f"Фокусные метрики: {focus or '—'}",
        f"Комментарий: {notes}",
    ]

    # Include aggregated metrics if present (keys ending with '__pct_mean' or '__delta_mean').
    metric_summaries: list[str] = []
    for col in detection.index:
        if col.endswith("__pct_mean") or col.endswith("__delta_mean"):
            value = detection[col]
            if not isinstance(value, (int, float)) or math.isnan(value):
                continue
            label = col.replace("__", ": ").replace("_", " ")
            metric_summaries.append(f"{label} = {value:.2f}")
    if metric_summaries:
        footer_lines.append("Агрегаты: " + "; ".join(metric_summaries[:8]))

    fig.text(0.01, 0.01, "\n".join(footer_lines), ha="left", va="bottom", fontsize=9)
    return fig


def generate_pdfs() -> None:
    detections, merged = load_inputs()
    ensure_output_dir()

    for well, well_detections in detections.groupby("well_number"):
        well_data = merged[merged["well_number"] == well]
        if well_data.empty:
            continue

        pdf_path = OUTPUT_DIR / f"скважина_{well}_pipeline_аномалии.pdf"
        with pdf_backend.PdfPages(pdf_path) as pdf:
            cover = build_cover_page(well, well_detections)
            pdf.savefig(cover)
            plt.close(cover)

            for _, detection in well_detections.iterrows():
                fig = plot_detection(well_data, detection)
                if fig is None:
                    continue
                pdf.savefig(fig)
                plt.close(fig)

        print(f"PDF сформирован: {pdf_path}")


if __name__ == "__main__":
    generate_pdfs()
