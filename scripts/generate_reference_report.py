from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.anomalies import (
    StepwiseResult,
    build_detection_context,
    evaluate_stepwise_for_intervals,
    load_svod_sheet,
    load_well_series,
    parse_reference_intervals,
)
from pipeline.config import DEFAULT_CONFIG_PATH, load_config


METRICS_TO_PLOT = [
    ("Intake_Pressure", "Давление на приеме"),
    ("Current", "Ток"),
    ("Motor_Temperature", "Температура масла ПЭД"),
    ("Frequency", "Частота"),
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HTML report comparing reference anomalies with detected segments."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to pipeline configuration.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/anomalies/reference_wells_report.html"),
        help="Path to resulting HTML report.",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=Path("reports/anomalies/anomaly_analysis.parquet"),
        help="Path to anomaly detections parquet.",
    )
    parser.add_argument(
        "--wells",
        type=str,
        default=None,
        help="Comma-separated list of wells to include. По умолчанию — все листы кроме svod.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_well_series_subset(xl: pd.ExcelFile, wells: List[str]) -> Dict[str, Dict[str, pd.Series]]:
    data: Dict[str, Dict[str, pd.Series]] = {}
    requested = set(wells)
    for sheet_name in xl.sheet_names:
        if sheet_name == "svod" or sheet_name not in requested:
            continue
        df = xl.parse(sheet_name)
        metric_frames: Dict[str, pd.Series] = {}
        for column in df.columns:
            if not str(column).startswith("timestamp_"):
                continue
            metric = column.replace("timestamp_", "")
            if metric not in df.columns:
                continue
            timestamps = pd.to_datetime(df[column], errors="coerce", dayfirst=True)
            values = pd.to_numeric(df[metric], errors="coerce")
            mask = timestamps.notna() & values.notna()
            if not mask.any():
                continue
            series = pd.Series(values[mask].values, index=pd.Index(timestamps[mask].values).tz_localize(None))
            series = series.sort_index()
            series = series[~series.index.duplicated(keep="last")]
            metric_frames[metric] = series
        if not metric_frames:
            continue
        start = max(series.index.min() for series in metric_frames.values())
        end = min(series.index.max() for series in metric_frames.values())
        if pd.isna(start) or pd.isna(end) or start >= end:
            continue
        aligned: Dict[str, pd.Series] = {}
        for metric, series in metric_frames.items():
            clipped = series[(series.index >= start) & (series.index <= end)]
            if clipped.empty:
                continue
            aligned[metric] = clipped
        if aligned:
            data[sheet_name] = aligned
    return data


def load_reference_anomalies(
    config: Dict,
    wells: List[str],
    xl: pd.ExcelFile,
) -> Tuple[
    Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
    Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
    Dict[str, Dict[str, pd.Series]],
]:
    svod_sheet = config["anomalies"].get("svod_sheet", "svod")

    svod = load_svod_sheet(xl, svod_sheet)
    alignment_cfg = config.get("alignment", {})
    base_frequency = alignment_cfg.get("frequency", "15T")
    base_aggregation = alignment_cfg.get("base_aggregation", "mean")
    pressure_metrics_cfg = alignment_cfg.get("pressure_fast_metrics", ["Intake_Pressure"])
    if isinstance(pressure_metrics_cfg, str):
        pressure_metrics = [pressure_metrics_cfg]
    else:
        pressure_metrics = list(pressure_metrics_cfg)

    full_well_series = load_well_series(
        xl,
        svod_sheet,
        base_frequency=base_frequency,
        pressure_metrics=pressure_metrics,
        base_aggregation=base_aggregation,
    )
    base_well_data = {
        well: series.base for well, series in full_well_series.items() if series.base is not None and not series.base.empty
    }
    raw_well_series = load_well_series_subset(xl, wells)
    intervals = parse_reference_intervals(
        svod=svod,
        well_data=base_well_data,
        anomaly_label=config["anomalies"].get("anomaly_cause", "Негерметичность НКТ"),
        normal_label=config["anomalies"].get("normal_cause", "Нормальная работа при изменении частоты"),
    )

    anomaly_mapping: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {well: [] for well in wells}
    normal_mapping: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {well: [] for well in wells}
    for interval in intervals:
        if interval.well not in anomaly_mapping:
            continue
        if interval.label == "anomaly":
            anomaly_mapping[interval.well].append((interval.start, interval.end))
        elif interval.label == "normal":
            normal_mapping[interval.well].append((interval.start, interval.end))
    return anomaly_mapping, normal_mapping, raw_well_series


def load_detected_anomalies(detections_path: Path, wells: List[str]) -> Dict[str, List[pd.Timestamp]]:
    detections_df = pd.read_parquet(detections_path)
    detections_df["well"] = detections_df["well"].astype(str)
    detections_df = detections_df[detections_df["segment_type"] == "anomaly"]

    mapping: Dict[str, List[pd.Timestamp]] = {well: [] for well in wells}
    for _, row in detections_df.iterrows():
        well = row["well"]
        if well not in mapping:
            continue
        start = pd.to_datetime(row["start"])
        mapping[well].append(start)
    return mapping


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _format_timestamp(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return ts.strftime("%d.%m.%Y %H:%M")


def _format_number(value: Optional[float], precision: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{precision}f}"


def _describe_triggers(result: StepwiseResult) -> str:
    if not result.triggers:
        return "—"
    parts: List[str] = []
    delta = result.triggers.get("pressure_delta")
    if delta is not None and not pd.isna(delta):
        parts.append(f"ΔP={delta:.3f}")
    slope = result.triggers.get("pressure_slope")
    if slope is not None and not pd.isna(slope):
        parts.append(f"slope={slope:.3f}")
    current_delta = result.triggers.get("current_delta")
    if current_delta is not None and not pd.isna(current_delta):
        parts.append(f"ΔI={current_delta:.3f}")
    temp_delta = result.triggers.get("temperature_delta")
    if temp_delta is not None and not pd.isna(temp_delta):
        parts.append(f"ΔT={temp_delta:.3f}")
    for key in ("residual", "ewma", "spike"):
        if result.triggers.get(key):
            parts.append(key)
    lag = result.triggers.get("notification_lag_minutes")
    if lag is not None and not pd.isna(lag):
        parts.append(f"lag={lag:.1f} мин")
    return ", ".join(parts) if parts else "—"


def render_stepwise_section(results: Sequence[StepwiseResult]) -> str:
    if not results:
        return ""

    rows: List[str] = []
    ordered = sorted(
        results,
        key=lambda r: (
            r.well,
            r.interval_start if r.interval_start is not None else pd.Timestamp.min,
        ),
    )
    for res in ordered:
        rows.append(
            "<tr>"
            f"<td>{res.well}</td>"
            f"<td>{_format_timestamp(res.interval_start)}</td>"
            f"<td>{_format_timestamp(res.detection_start)}</td>"
            f"<td>{_format_timestamp(res.notification_time)}</td>"
            f"<td>{_format_number(res.delay_minutes, precision=1)}</td>"
            f"<td>{res.evaluated_points}</td>"
            f"<td>{_describe_triggers(res)}</td>"
            "</tr>"
        )

    table = (
        "<section>"
        "<h2>Пошаговая проверка holdout-скважин</h2>"
        "<p>Для скважин из holdout-набора данные подавались пошагово с шагом базовой частоты. "
        "Фиксируется момент первой сработки детектора и отставание от референсного интервала.</p>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<thead>"
        "<tr>"
        "<th>Скважина</th>"
        "<th>Старт референса</th>"
        "<th>Старт детекции</th>"
        "<th>Момент уведомления</th>"
        "<th>Задержка, мин</th>"
        "<th>Точек рассмотрено</th>"
        "<th>Активные признаки</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )
    return table


def build_well_figure(
    well: str,
    well_series: Dict[str, pd.Series],
    reference_windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    normal_windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    detected_events: List[pd.Timestamp],
) -> go.Figure | None:
    if not well_series:
        return None

    rows = len(METRICS_TO_PLOT)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[label for _, label in METRICS_TO_PLOT],
    )

    color_map = {
        "Intake_Pressure": "#1f77b4",
        "Current": "#2ca02c",
        "Motor_Temperature": "#9467bd",
        "Frequency": "#d62728",
    }

    primary_series: Optional[pd.Series] = None

    for idx, (metric, label) in enumerate(METRICS_TO_PLOT, start=1):
        series = well_series.get(metric)
        if series is None or series.empty:
            continue
        series = pd.to_numeric(series, errors="coerce").dropna()
        if primary_series is None:
            primary_series = series
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=label,
                line=dict(color=color_map.get(metric, "#1f77b4"), width=1.2),
                showlegend=True,
            ),
            row=idx,
            col=1,
        )

    if primary_series is not None and not primary_series.empty:
        y_top = float(primary_series.max())
        y_bottom = float(primary_series.min())
        span = y_top - y_bottom
        if span == 0:
            span = max(abs(y_top), 1.0)
        def _y_pos(base: float) -> float:
            value = y_top - base * span
            return max(min(value, y_top), y_bottom)
        yref_value = "y"
    else:
        y_top = None
        def _y_pos(base: float) -> float:
            return 1.04 + 0.05 * base
        yref_value = "paper"

    for window_idx, (start, end) in enumerate(reference_windows):
        vertical_offset = (window_idx % 3) * 0.05
        fig.add_vline(x=start, line=dict(color="goldenrod", dash="dot", width=1))
        fig.add_annotation(
            x=start,
            y=_y_pos(0.05 + vertical_offset),
            xref="x",
            yref=yref_value,
            text=f"Начало аномалии<br>{start:%d.%m %H:%M}",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="goldenrod",
            ax=-80,
            ay=0,
            font=dict(size=9, color="goldenrod"),
            bgcolor="rgba(255, 215, 0, 0.15)",
            bordercolor="goldenrod",
            borderwidth=0.5,
        )
        fig.add_vline(x=end, line=dict(color="goldenrod", dash="dot", width=1))
        fig.add_annotation(
            x=end,
            y=_y_pos(0.1 + vertical_offset),
            xref="x",
            yref=yref_value,
            text=f"Остановка скважины<br>{end:%d.%m %H:%M}",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="goldenrod",
            ax=80,
            ay=0,
            font=dict(size=9, color="goldenrod"),
            bgcolor="rgba(255, 215, 0, 0.15)",
            bordercolor="goldenrod",
            borderwidth=0.5,
        )
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="gold",
            opacity=0.25,
            line_width=0,
            row="all",
            col=1,
        )

    for window_idx, (start, end) in enumerate(normal_windows):
        base_offset = 0.18 + (window_idx % 3) * 0.06
        fig.add_vline(x=start, line=dict(color="#2ecc71", dash="dot", width=1))
        fig.add_annotation(
            x=start,
            y=_y_pos(base_offset),
            xref="x",
            yref=yref_value,
            text=f"Начало нормы<br>{start:%d.%m %H:%M}",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#2ecc71",
            ax=-80,
            ay=0,
            font=dict(size=8, color="#1f7a3a"),
            bgcolor="rgba(46, 204, 113, 0.15)",
            bordercolor="#1f7a3a",
            borderwidth=0.5,
        )
        fig.add_vline(x=end, line=dict(color="#2ecc71", dash="dot", width=1))
        fig.add_annotation(
            x=end,
            y=_y_pos(base_offset + 0.02),
            xref="x",
            yref=yref_value,
            text=f"Конец нормы<br>{end:%d.%m %H:%M}",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#2ecc71",
            ax=80,
            ay=0,
            font=dict(size=8, color="#1f7a3a"),
            bgcolor="rgba(46, 204, 113, 0.15)",
            bordercolor="#1f7a3a",
            borderwidth=0.5,
        )
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(46, 204, 113, 0.2)",
            opacity=0.2,
            line_width=0,
            row="all",
            col=1,
        )

    for event_idx, event_time in enumerate(detected_events):
        det_offset = (event_idx % 3) * 0.05
        fig.add_vline(x=event_time, line=dict(color="#ff5733", dash="dot", width=1))
        fig.add_annotation(
            x=event_time,
            y=_y_pos(0.25 + det_offset),
            xref="x",
            yref=yref_value,
            text=f"Детектор: старт<br>{event_time:%d.%m %H:%M}",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#ff5733",
            ax=-80,
            ay=0,
            font=dict(size=9, color="#ff5733"),
            bgcolor="rgba(255, 87, 51, 0.15)",
            bordercolor="#ff5733",
            borderwidth=0.5,
        )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="gold"),
            name="Референсная аномалия",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="#2ecc71"),
            name="Референсная норма",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="#ff5733"),
            name="Аномалия (детектор)",
        )
    )

    fig.update_layout(
        height=300 * rows,
        width=1200,
        title=f"Скважина {well}: сравнение аномалий",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    fig.update_xaxes(title_text="Дата/время", row=rows, col=1)

    return fig


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    workbook_path = Path(config["anomalies"]["source_workbook"])
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")
    xl = pd.ExcelFile(workbook_path)

    if args.wells:
        wells = [w.strip() for w in args.wells.split(",") if w.strip()]
    else:
        svod_sheet = config["anomalies"].get("svod_sheet", "svod")
        wells = sorted(sheet for sheet in xl.sheet_names if sheet != svod_sheet)

    if not wells:
        raise ValueError("Не удалось определить список скважин для отчёта.")

    context = build_detection_context(config, xl, use_streaming_calibration=True)
    reference_mapping, normal_mapping, well_data = load_reference_anomalies(config, wells, xl)
    detection_mapping = load_detected_anomalies(args.detections, wells)

    stepwise_targets = sorted(set(context.holdout_wells) & set(wells))
    if stepwise_targets:
        holdout_intervals = [
            interval
            for interval in context.reference_intervals
            if interval.label == "anomaly" and interval.well in stepwise_targets
        ]
        stepwise_results = evaluate_stepwise_for_intervals(context, holdout_intervals)
    else:
        stepwise_results = []

    ensure_output_dir(args.output)

    figures: List[go.Figure] = []
    for well in wells:
        signals = well_data.get(well)
        if not signals:
            continue
        fig = build_well_figure(
            well=well,
            well_series=signals,
            reference_windows=reference_mapping.get(well, []),
            normal_windows=normal_mapping.get(well, []),
            detected_events=detection_mapping.get(well, []),
        )
        if fig is not None:
            figures.append(fig)

    if not figures:
        raise ValueError("Не удалось построить ни одного графика — проверьте входные данные.")

    html_sections: List[str] = []
    stepwise_section = render_stepwise_section(stepwise_results)
    if stepwise_section:
        html_sections.append(stepwise_section)
    for idx, fig in enumerate(figures):
        include_js = "cdn" if idx == 0 else False
        html_sections.append(
            pio.to_html(fig, include_plotlyjs=include_js, full_html=False, auto_play=False)
        )

    html_content = "<html><head><meta charset='utf-8'></head><body>" + "<hr>".join(html_sections) + "</body></html>"
    ensure_output_dir(args.output)
    args.output.write_text(html_content, encoding="utf-8")

    print(f"HTML report saved to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
