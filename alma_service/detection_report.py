from __future__ import annotations

import argparse
import base64
import io
from html import escape
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

plt.rcParams["font.size"] = 10

PRESSURE_COL = "Давление на приеме насоса кгс/см²"
FREQ_COL = "Выходная частота"
COLOR_THEMES = {
    "negermet": {"accent": "#b42318", "soft": "#fef3f2", "soft_alt": "#fff6f6"},
    "pritok": {"accent": "#c2410c", "soft": "#fff7ed", "soft_alt": "#fffbf5"},
    "salt": {"accent": "#0f766e", "soft": "#f0fdfa", "soft_alt": "#f8fffd"},
}
SPLIT_LABELS = {"train": "Train", "test": "Test", "all": "All"}


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
    df = pd.read_csv(path, dtype={"well_id": str}, low_memory=False)
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values(["well_id", "timestamp"]).reset_index(drop=True)


def _load_intervals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"well_id": str})
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["data_start"] = pd.to_datetime(df.get("data_start"), errors="coerce")
    df["data_end"] = pd.to_datetime(df.get("data_end"), errors="coerce")
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
    df = pd.read_csv(path, dtype={"well_id": str})
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    if "interval_idx" not in df.columns:
        df["interval_idx"] = df.groupby("well_id").cumcount() + 1
    df["interval_idx"] = pd.to_numeric(df["interval_idx"], errors="coerce").fillna(1).astype(int)
    for col in ["actual_start", "actual_end", "detected_time", "data_start", "data_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "split" not in df.columns:
        df["split"] = "train"
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    if "status" not in df.columns:
        df["status"] = np.where(df["detected_time"].notna(), "Detected", "Not found")
    return df.sort_values(["well_id", "interval_idx"]).reset_index(drop=True)


def _load_scores(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"well_id": str})
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
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


def _create_plot_base64(
    well_df: pd.DataFrame,
    result_row: pd.Series,
    scores_df: pd.DataFrame | None,
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

    n_rows = max(len(plot_cols), 1) + (1 if has_scores else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4.0 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    line_colors = ["tab:blue", "tab:orange", "tab:green"]
    for idx, col in enumerate(plot_cols):
        ax = axes[idx]
        line_df = well_df[["timestamp", col]].copy()
        line_df[col] = pd.to_numeric(line_df[col], errors="coerce")
        line_df = line_df.dropna(subset=[col])
        if line_df.empty:
            ax.text(0.5, 0.5, f"Нет данных: {col}", transform=ax.transAxes, ha="center", va="center")
        else:
            ax.plot(line_df["timestamp"], line_df[col], color=line_colors[idx % len(line_colors)], linewidth=0.7)
        ax.axvspan(result_row["actual_start"], result_row["actual_end"], color="tab:red", alpha=0.12)
        ax.axvline(result_row["actual_start"], color="tab:green", linewidth=1.0)
        if pd.notna(result_row["detected_time"]):
            ax.axvline(result_row["detected_time"], color="#7c3aed", linestyle="--", linewidth=1.2)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.25)
        if pd.notna(x_min) and pd.notna(x_max) and x_min < x_max:
            ax.set_xlim(x_min, x_max)

    if has_scores and scores_df is not None:
        score_ax = axes[-1]
        score_view = scores_df.copy()
        if pd.notna(x_min) and pd.notna(x_max):
            score_view = score_view[(score_view["timestamp"] >= x_min) & (score_view["timestamp"] <= x_max)]
        if not score_view.empty:
            score_ax.fill_between(score_view["timestamp"], 0, score_view[score_col], color="#7c3aed", alpha=0.22)
            score_ax.plot(score_view["timestamp"], score_view[score_col], color="#7c3aed", linewidth=0.7)
        score_ax.axvspan(result_row["actual_start"], result_row["actual_end"], color="tab:red", alpha=0.12)
        score_ax.axvline(result_row["actual_start"], color="tab:green", linewidth=1.0)
        if pd.notna(result_row["detected_time"]):
            score_ax.axvline(result_row["detected_time"], color="#7c3aed", linestyle="--", linewidth=1.2)
        score_ax.set_ylabel(score_col)
        score_ax.grid(True, alpha=0.25)
        if pd.notna(x_min) and pd.notna(x_max) and x_min < x_max:
            score_ax.set_xlim(x_min, x_max)

    axes[0].set_title(
        f"Скважина {result_row['well_id']} | интервал {int(result_row['interval_idx'])} | split={result_row['split']}",
        fontsize=12,
        fontweight="bold",
    )
    axes[-1].set_xlabel("Время")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


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
        pred_df = pd.read_csv(predictions_path, dtype={"well_id": str}) if predictions_path.exists() else pd.DataFrame()
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
                <span><b>Hit</b> {f"{split_payload.get('hit_count', 0)}/{split_payload.get('interval_count', 0)}"}</span>
                <span><b>Hit rate</b> {_format_float(100.0 * float(split_payload.get('hit_rate', 0.0)), 1, '%')}</span>
                <span><b>P90 delay ratio</b> {_format_float(split_payload.get('p90_delay_ratio'), 3)}</span>
                <span><b>P90 |delay|</b> {_format_float(split_payload.get('p90_abs_delay_hours'), 1, ' h')}</span>
                <span><b>FAR/day</b> {_format_float(split_payload.get('false_alarms_per_day'), 3)}</span>
                <span><b>Starts/interval</b> {_format_float(split_payload.get('avg_starts_per_interval'), 2)}</span>
              </div>
            </section>
            """
        )

    sections = []
    for idx, result_row in results_df.iterrows():
        well_id = str(result_row["well_id"])
        print(f"  График {idx + 1}/{len(results_df)}: скв. {well_id}, интервал {int(result_row['interval_idx'])}")
        well_ts = data_df[data_df["well_id"] == well_id].copy()
        plot_b64 = _create_plot_base64(
            well_df=well_ts,
            result_row=result_row,
            scores_df=scores_by_well.get(well_id),
        )
        plot_html = f'<img src="data:image/png;base64,{plot_b64}" alt="{well_id}">' if plot_b64 else ""
        detail_text = escape(str(result_row.get("detail", "—")))
        sections.append(
            f"""
            <article class="interval-card">
              <header>
                <div>
                  <h3>{escape(well_id)} / interval {int(result_row['interval_idx'])}</h3>
                  <p class="meta">split={escape(str(result_row['split']))} | status={escape(str(result_row['status']))}</p>
                </div>
                <div class="pill {'ok' if result_row['status'] == 'Detected' else 'miss'}">{escape(str(result_row['status']))}</div>
              </header>
              <div class="interval-meta">
                <span><b>Actual start:</b> {_format_dt(result_row['actual_start'])}</span>
                <span><b>Actual end:</b> {_format_dt(result_row['actual_end'])}</span>
                <span><b>Detected:</b> {_format_dt(result_row['detected_time'])}</span>
                <span><b>Delay:</b> {_format_float(result_row.get('delay_hours'), 2, ' h')}</span>
                <span><b>Channels:</b> {escape(str(result_row.get('n_channels', '—')))}</span>
                <span><b>Features:</b> {escape(str(result_row.get('n_features', '—')))}</span>
                <span><b>Pred starts:</b> {escape(str(result_row.get('n_predicted_starts', '—')))}</span>
              </div>
              <pre>{detail_text}</pre>
              {plot_html}
            </article>
            """
        )

    html = f"""
    <!doctype html>
    <html lang="ru">
      <head>
        <meta charset="utf-8">
        <title>{escape(spec.display_name)} report [{escape(detector_key)}]</title>
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
          .pill {{
            border-radius: 999px;
            padding: 8px 12px;
            font-weight: 700;
            background: #e4e7ec;
          }}
          .pill.ok {{ color: #027a48; background: #ecfdf3; }}
          .pill.miss {{ color: #b42318; background: #fef3f2; }}
          img {{
            width: 100%;
            border-radius: 14px;
            border: 1px solid var(--border);
            margin-top: 12px;
          }}
          pre {{
            white-space: pre-wrap;
            word-break: break-word;
            padding: 12px 14px;
            background: rgba(16, 24, 40, 0.04);
            border-radius: 12px;
            color: var(--muted);
            font-size: 12px;
          }}
        </style>
      </head>
      <body>
        <main class="page">
          <h1>{escape(spec.display_name)}</h1>
          <p class="subtitle">Detector: <b>{escape(detector_key)}</b> | source: {escape(str(source.name))}</p>
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
    parser.add_argument("--results", default=None, help="Override results CSV path")
    parser.add_argument("--scores", default=None, help="Override scores CSV path")
    args = parser.parse_args()

    generate_report(
        anomaly_key=default_anomaly,
        detector=args.detector,
        output_path=args.output,
        source_path=args.source,
        results_path_override=args.results,
        scores_path_override=args.scores,
    )
