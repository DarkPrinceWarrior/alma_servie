from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

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
STATUS_LABELS = {"Detected": "Обнаружено", "Not found": "Не обнаружено"}
DETECTOR_LABELS = {
    "pca_spe": "PCA/SPE",
    "fused": "Комбинированный детектор",
    "lof": "LOF",
    "iforest": "Isolation Forest",
    "paano_feat": "PaAno + признаки",
}


def _pick_existing_source(spec, source_path: str | None) -> Path:
    if source_path:
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")
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


def _pick_plot_columns(well_df: pd.DataFrame) -> list[str]:
    preferred = [col for col in [PRESSURE_COL, FREQ_COL] if col in well_df.columns]
    if len(preferred) >= 2:
        return preferred[:2]

    numeric_cols = []
    for col in well_df.columns:
        if col in {"well_id", "timestamp"} or col in preferred:
            continue
        series = pd.to_numeric(well_df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(col)
    return (preferred + numeric_cols)[:2]


def _create_plot_html(
    well_df: pd.DataFrame,
    result_row: pd.Series,
    scores_df: pd.DataFrame | None,
    *,
    include_plotlyjs: bool,
) -> str | None:
    plot_cols = _pick_plot_columns(well_df)
    score_col = _score_column(scores_df)
    has_scores = score_col is not None
    if not plot_cols and not has_scores:
        return None

    x_min = result_row["data_start"] if pd.notna(result_row.get("data_start")) else well_df["timestamp"].min()
    x_max = result_row["data_end"] if pd.notna(result_row.get("data_end")) else well_df["timestamp"].max()
    if pd.notna(x_min) and pd.notna(x_max):
        well_df = well_df[(well_df["timestamp"] >= x_min) & (well_df["timestamp"] <= x_max)]
    if well_df.empty:
        return None

    subplot_titles = list(plot_cols)
    if has_scores:
        subplot_titles.append(str(score_col))
    fig = make_subplots(
        rows=max(len(subplot_titles), 1),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles or ["Signals"],
    )

    line_colors = ["#2563eb", "#ea580c", "#0f766e", "#7c3aed"]
    current_row = 1
    for idx, col in enumerate(plot_cols):
        line_df = well_df[["timestamp", col]].copy()
        line_df[col] = pd.to_numeric(line_df[col], errors="coerce")
        line_df = line_df.dropna(subset=[col])
        if not line_df.empty:
            fig.add_trace(
                go.Scattergl(
                    x=line_df["timestamp"],
                    y=line_df[col],
                    mode="lines",
                    name=col,
                    line={"color": line_colors[idx % len(line_colors)], "width": 1.1},
                    showlegend=False,
                ),
                row=current_row,
                col=1,
            )
        current_row += 1

    if has_scores and scores_df is not None:
        score_view = scores_df.copy()
        if pd.notna(x_min) and pd.notna(x_max):
            score_view = score_view[(score_view["timestamp"] >= x_min) & (score_view["timestamp"] <= x_max)]
        if not score_view.empty:
            fig.add_trace(
                go.Scatter(
                    x=score_view["timestamp"],
                    y=score_view[score_col],
                    mode="lines",
                    line={"color": "#7c3aed", "width": 1.2},
                    fill="tozeroy",
                    fillcolor="rgba(124,58,237,0.14)",
                    name=str(score_col),
                    showlegend=False,
                ),
                row=current_row,
                col=1,
            )

    for row_idx in range(1, max(len(subplot_titles), 1) + 1):
        fig.add_vrect(
            x0=result_row["actual_start"],
            x1=result_row["actual_end"],
            fillcolor="rgba(220,38,38,0.10)",
            line_width=0,
            row=row_idx,
            col=1,
        )
        fig.add_vline(
            x=result_row["actual_start"],
            line_color="#16a34a",
            line_width=1,
            row=row_idx,
            col=1,
        )
        if pd.notna(result_row["detected_time"]):
            fig.add_vline(
                x=result_row["detected_time"],
                line_color="#7c3aed",
                line_width=1.25,
                line_dash="dash",
                row=row_idx,
                col=1,
            )

    fig.update_layout(
        height=max(360 * max(len(subplot_titles), 1), 420),
        margin={"l": 44, "r": 24, "t": 56, "b": 36},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        title={
            "text": (
                f"Скважина {result_row['well_id']} | интервал {int(result_row['interval_idx'])}"
            ),
            "x": 0.01,
        },
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
    )


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

    source = _pick_existing_source(spec, source_path)
    results_path_value = Path(results_path_override) if results_path_override else results_path(spec, detector_key)
    scores_path_value = Path(scores_path_override) if scores_path_override else scores_path(spec, detector_key)
    output = ensure_parent(
        Path(output_path) if output_path else report_path(spec, detector_key)
    )

    data_df = _load_timeseries(source)
    intervals_df = _load_intervals(spec.dataset.intervals_path)
    results_df = _load_results(results_path_value)
    scores_by_well = _load_scores(scores_path_value)
    summary_payload = load_json(summary_path(spec, detector_key))
    if not summary_payload:
        predictions_path = predicted_starts_path(spec, detector_key)
        pred_df = (
            read_table(predictions_path, dtypes={"well_id": str}, parse_dates=["detected_time"])
            if predictions_path.exists()
            else pd.DataFrame()
        )
        if not pred_df.empty:
            pred_df["well_id"] = pred_df["well_id"].astype(str).str.strip().str.lower()
            pred_df["detected_time"] = pd.to_datetime(pred_df["detected_time"], errors="coerce")
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
                <span><b>P90 задержка:</b> {_format_float(split_payload.get('p90_abs_delay_hours'), 1, ' ч')}</span>
                <span><b>Ложные срабатывания в сутки:</b> {_format_float(split_payload.get('false_alarms_per_day'), 3)}</span>
              </div>
            </section>
            """
        )

    sections = []
    for idx, result_row in results_df.iterrows():
        well_id = str(result_row["well_id"])
        print(f"  График {idx + 1}/{len(results_df)}: скв. {well_id}, интервал {int(result_row['interval_idx'])}")
        well_ts = data_df[data_df["well_id"] == well_id].copy()
        plot_html = _create_plot_html(
            well_df=well_ts,
            result_row=result_row,
            scores_df=scores_by_well.get(well_id),
            include_plotlyjs=(idx == 0),
        )
        status_text = _status_label(result_row["status"])
        split_text = SPLIT_SHORT_LABELS.get(str(result_row["split"]), str(result_row["split"]))
        sections.append(
            f"""
            <article class="interval-card">
              <header>
                <div>
                  <h3>Скважина {escape(well_id)} / интервал {int(result_row['interval_idx'])}</h3>
                  <p class="meta">{escape(split_text)} | {escape(status_text)}</p>
                </div>
                <div class="pill {'ok' if result_row['status'] == 'Detected' else 'miss'}">{escape(status_text)}</div>
              </header>
              <div class="interval-meta">
                <span><b>Фактическое начало:</b> {_format_dt(result_row['actual_start'])}</span>
                <span><b>Фактическое окончание:</b> {_format_dt(result_row['actual_end'])}</span>
                <span><b>Время обнаружения:</b> {_format_dt(result_row['detected_time'])}</span>
                <span><b>Задержка:</b> {_format_float(result_row.get('delay_hours'), 2, ' ч')}</span>
              </div>
              <p class="note">Зелёная линия показывает начало интервала аномалии, красная зона показывает её длительность, фиолетовая пунктирная линия показывает момент обнаружения.</p>
              <div class="plot-wrap">{plot_html}</div>
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
          .pill {{
            border-radius: 999px;
            padding: 8px 12px;
            font-weight: 700;
            background: #e4e7ec;
          }}
          .pill.ok {{ color: #027a48; background: #ecfdf3; }}
          .pill.miss {{ color: #b42318; background: #fef3f2; }}
          .plot-wrap {{
            width: 100%;
            margin-top: 12px;
          }}
          .plot-wrap img {{
            width: 100%;
            border-radius: 14px;
            border: 1px solid var(--border);
          }}
          .plot-wrap .plotly-graph-div {{
            width: 100% !important;
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
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
    parser.add_argument("--detector", default=None, help="Detector key, default is benchmark-selected or fused.")
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
