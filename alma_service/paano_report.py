from __future__ import annotations

import argparse
import base64
import io
import json
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
from alma_service.paano_defaults import (
    DEFAULT_CONFIG,
    LONG_PATCH,
    PRESTART_TOLERANCE_HOURS,
    REFERENCE_MAX_RATIO,
    REFERENCE_MIN_DAYS,
    REFERENCE_MIN_RATIO,
    SHORT_PATCH,
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
    for col in ["actual_start", "actual_end", "detected_time"]:
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fallback_summary(results: pd.DataFrame) -> dict[str, Any]:
    if results.empty:
        return {
            "interval_count": 0,
            "hit_count": 0,
            "hit_rate": 0.0,
            "median_abs_delay_hours": float("nan"),
            "p90_abs_delay_hours": float("nan"),
        }
    detected = results[results["status"] == "Detected"].copy()
    delays = detected["delay_hours"].dropna().astype(float).abs()
    return {
        "interval_count": int(len(results)),
        "hit_count": int(len(detected)),
        "hit_rate": float(len(detected) / len(results)) if len(results) else 0.0,
        "median_abs_delay_hours": float(np.median(delays)) if len(delays) else float("nan"),
        "p90_abs_delay_hours": float(np.quantile(delays, 0.90)) if len(delays) else float("nan"),
    }


def _format_dt(value: pd.Timestamp | str | float | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
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
    if not plot_cols:
        return None

    x_min = result_row["data_start"] if pd.notna(result_row["data_start"]) else well_df["timestamp"].min()
    x_max = result_row["data_end"] if pd.notna(result_row["data_end"]) else well_df["timestamp"].max()
    if pd.notna(x_min) and pd.notna(x_max):
        well_df = well_df[(well_df["timestamp"] >= x_min) & (well_df["timestamp"] <= x_max)]
    if well_df.empty:
        return None

    has_scores = scores_df is not None and not scores_df.empty
    n_rows = len(plot_cols) + (1 if has_scores else 0)
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
        ax.axvspan(result_row["start_date"], result_row["end_date"], color="tab:red", alpha=0.12)
        ax.axvline(result_row["start_date"], color="tab:green", linewidth=1.0)
        if pd.notna(result_row["detected_time"]):
            ax.axvline(result_row["detected_time"], color="#7c3aed", linestyle="--", linewidth=1.2)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.25)
        if pd.notna(x_min) and pd.notna(x_max) and x_min < x_max:
            ax.set_xlim(x_min, x_max)

    if has_scores:
        score_ax = axes[-1]
        score_view = scores_df.copy()
        if pd.notna(x_min) and pd.notna(x_max):
            score_view = score_view[(score_view["timestamp"] >= x_min) & (score_view["timestamp"] <= x_max)]
        if not score_view.empty:
            score_ax.fill_between(score_view["timestamp"], 0, score_view["paano_score"], color="#7c3aed", alpha=0.22)
            score_ax.plot(score_view["timestamp"], score_view["paano_score"], color="#7c3aed", linewidth=0.7)
        score_ax.axvspan(result_row["start_date"], result_row["end_date"], color="tab:red", alpha=0.12)
        score_ax.axvline(result_row["start_date"], color="tab:green", linewidth=1.0)
        if pd.notna(result_row["detected_time"]):
            score_ax.axvline(result_row["detected_time"], color="#7c3aed", linestyle="--", linewidth=1.2)
        score_ax.set_ylabel("PaAno score")
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


def generate_report(
    anomaly_key: str,
    output_path: str | None = None,
    source_path: str | None = None,
    results_path: str | None = None,
    scores_path: str | None = None,
) -> Path:
    spec = get_detection_spec(anomaly_key)
    theme = COLOR_THEMES[anomaly_key]

    source = _pick_existing_source(spec, source_path)
    results = Path(results_path) if results_path else spec.results_path
    scores = Path(scores_path) if scores_path else spec.scores_path
    output = ensure_parent(Path(output_path) if output_path else REPORTS_DIR / f"{spec.dataset.output_prefix}_paano_report.html")

    data_df = _load_timeseries(source)
    intervals_df = _load_intervals(spec.dataset.intervals_path)
    results_df = _load_results(results)
    scores_map = _load_scores(scores)

    merged = intervals_df.merge(
        results_df,
        on=["well_id", "interval_idx"],
        how="left",
        suffixes=("", "_res"),
    )
    merged["split"] = merged["split_res"].fillna(merged["split"]).astype(str).str.strip().str.lower()
    merged["detected_time"] = pd.to_datetime(merged["detected_time"], errors="coerce")
    default_status = pd.Series(
        np.where(merged["detected_time"].notna(), "Detected", "Not found"),
        index=merged.index,
    )
    merged["status"] = merged["status"].where(merged["status"].notna(), default_status)
    if "delay_hours" not in merged.columns:
        merged["delay_hours"] = np.nan
    if "n_channels" not in merged.columns:
        merged["n_channels"] = np.nan
    if "n_predicted_starts" not in merged.columns:
        merged["n_predicted_starts"] = np.nan
    if "detail" not in merged.columns:
        merged["detail"] = ""
    merged = merged.sort_values(["well_id", "interval_idx"]).reset_index(drop=True)

    summary_payload = _load_json(results.with_suffix(".summary.json"))
    split_summaries = summary_payload.get("splits", {})
    for split_name in ["train", "test", "all"]:
        if split_name not in split_summaries:
            subset = merged if split_name == "all" else merged[merged["split"] == split_name]
            split_summaries[split_name] = _fallback_summary(subset)

    config_payload = summary_payload.get("config", {})
    if not config_payload and spec.config_path.exists():
        raw_payload = _load_json(spec.config_path)
        config_payload = raw_payload.get("config", raw_payload)
    if not config_payload:
        config_payload = DEFAULT_CONFIG

    cards_html = []
    for split_name in ["train", "test", "all"]:
        summary = split_summaries.get(split_name, {})
        if int(summary.get("interval_count", 0) or 0) == 0:
            continue
        cards_html.append(
            f"""
            <div class="metric-card">
                <div class="metric-label">{SPLIT_LABELS[split_name]}</div>
                <div class="metric-main">{int(summary.get('hit_count', 0))}/{int(summary.get('interval_count', 0))}</div>
                <div class="metric-sub">Hit rate: {_format_float(100.0 * float(summary.get('hit_rate', 0.0)), 1, '%')}</div>
                <div class="metric-sub">P90 |delay|: {_format_float(summary.get('p90_abs_delay_hours'), 1, ' h')}</div>
                <div class="metric-sub">FAR/day: {_format_float(summary.get('false_alarms_per_day'), 3)}</div>
            </div>
            """
        )

    table_rows = []
    for _, row in merged.iterrows():
        table_rows.append(
            f"""
            <tr>
                <td>{escape(str(row['well_id']))}</td>
                <td>{escape(str(row['split']))}</td>
                <td>{int(row['interval_idx'])}</td>
                <td>{_format_dt(row['data_start'])}</td>
                <td>{_format_dt(row['data_end'])}</td>
                <td>{_format_dt(row['start_date'])}</td>
                <td>{_format_dt(row['end_date'])}</td>
                <td>{_format_dt(row['detected_time'])}</td>
                <td>{escape(str(row['status']))}</td>
                <td>{_format_float(row.get('delay_hours'), 1, ' h')}</td>
                <td>{_format_float(row.get('n_channels'), 0)}</td>
                <td>{_format_float(row.get('n_predicted_starts'), 0)}</td>
            </tr>
            """
        )

    plot_blocks = []
    total = len(merged)
    for idx, (_, row) in enumerate(merged.iterrows(), 1):
        print(f"  График {idx}/{total}: скв. {row['well_id']}, интервал {int(row['interval_idx'])}")
        well_df = data_df[data_df["well_id"] == row["well_id"]]
        plot_base64 = _create_plot_base64(well_df, row, scores_map.get(row["well_id"]))
        detail = escape(str(row.get("detail", "")))
        detail_html = f"<div class='detail-box'>{detail}</div>" if detail.strip() else ""
        if plot_base64 is None:
            plot_blocks.append(
                f"""
                <section class="plot-card">
                    <h3>Скважина {escape(str(row['well_id']))}, интервал {int(row['interval_idx'])}</h3>
                    <p>Нет данных для визуализации.</p>
                    {detail_html}
                </section>
                """
            )
            continue
        plot_blocks.append(
            f"""
            <section class="plot-card">
                <h3>Скважина {escape(str(row['well_id']))}, интервал {int(row['interval_idx'])}</h3>
                <div class="plot-meta">
                    <span>split={escape(str(row['split']))}</span>
                    <span>status={escape(str(row['status']))}</span>
                    <span>delay={_format_float(row.get('delay_hours'), 1, ' h')}</span>
                    <span>channels={_format_float(row.get('n_channels'), 0)}</span>
                    <span>starts={_format_float(row.get('n_predicted_starts'), 0)}</span>
                </div>
                <img src="data:image/png;base64,{plot_base64}" alt="plot" />
                {detail_html}
            </section>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>PaAno report: {escape(spec.display_name)}</title>
    <style>
        body {{
            font-family: "Segoe UI", Arial, sans-serif;
            margin: 24px;
            color: #12202f;
            background: #f8fafc;
        }}
        h1 {{
            margin: 0 0 12px;
            color: #0f172a;
        }}
        h2 {{
            margin-top: 30px;
            color: #0f172a;
        }}
        h3 {{
            margin: 0 0 12px;
            color: #0f172a;
        }}
        p {{
            line-height: 1.55;
        }}
        .hero {{
            background: linear-gradient(135deg, {theme['soft']} 0%, #ffffff 100%);
            border: 1px solid {theme['accent']}33;
            border-left: 5px solid {theme['accent']};
            border-radius: 12px;
            padding: 18px 20px;
        }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin: 18px 0 8px;
        }}
        .metric-card {{
            background: #ffffff;
            border: 1px solid #dbe2ea;
            border-top: 4px solid {theme['accent']};
            border-radius: 10px;
            padding: 14px 16px;
        }}
        .metric-label {{
            font-size: 12px;
            text-transform: uppercase;
            color: #64748b;
            letter-spacing: 0.06em;
        }}
        .metric-main {{
            margin-top: 6px;
            font-size: 28px;
            font-weight: 700;
            color: #0f172a;
        }}
        .metric-sub {{
            margin-top: 4px;
            color: #334155;
            font-size: 13px;
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            margin-top: 16px;
        }}
        .config-box {{
            background: #ffffff;
            border: 1px solid #dbe2ea;
            border-radius: 10px;
            padding: 14px 16px;
        }}
        .config-box code {{
            font-size: 13px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            border: 1px solid #dbe2ea;
        }}
        th, td {{
            border: 1px solid #e2e8f0;
            padding: 8px 10px;
            text-align: left;
            font-size: 13px;
        }}
        th {{
            background: {theme['soft']};
            color: #0f172a;
        }}
        tr:nth-child(even) {{
            background: {theme['soft_alt']};
        }}
        .plot-card {{
            background: #ffffff;
            border: 1px solid #dbe2ea;
            border-radius: 12px;
            padding: 16px;
            margin-top: 18px;
        }}
        .plot-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px 16px;
            margin-bottom: 12px;
            color: #475569;
            font-size: 13px;
        }}
        .detail-box {{
            margin-top: 12px;
            padding: 10px 12px;
            background: {theme['soft']};
            border-radius: 8px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <section class="hero">
        <h1>PaAno: {escape(spec.display_name)}</h1>
        <p>
            Единый blind pipeline для всех аномалий: по каждой скважине автоматически выбирается
            нормальный префикс, используются только реально доступные каналы этой скважины, затем
            считается двухмасштабный PaAno score на патчах {SHORT_PATCH} и {LONG_PATCH}. Параметры
            onset-детектора подбираются только по train-скважинам и затем без label leakage
            применяются к train и test.
        </p>
        <p>
            Ограничения и правила: без backfill на этапе builder, предстартовый допуск для попадания
            в интервал {_format_float(PRESTART_TOLERANCE_HOURS, 1, ' h')}, reference-окно в диапазоне
            {_format_float(100.0 * REFERENCE_MIN_RATIO, 0, '%')}–{_format_float(100.0 * REFERENCE_MAX_RATIO, 0, '%')}
            ряда, минимум {_format_float(REFERENCE_MIN_DAYS, 1, ' days')}.
        </p>
        <div class="cards">
            {''.join(cards_html)}
        </div>
        <div class="config-grid">
            <div class="config-box">
                <b>Текущая конфигурация</b><br />
                <code>w_short={_format_float(config_payload.get('fusion_weight_short'), 2)}</code><br />
                <code>target_far/day={_format_float(config_payload.get('target_far_per_day'), 2)}</code><br />
                <code>min_run={_format_float(config_payload.get('min_run_points'), 0)}</code><br />
                <code>cooldown={_format_float(config_payload.get('cooldown_hours'), 0, ' h')}</code><br />
                <code>ema={_format_float(config_payload.get('ema_alpha'), 2)}</code><br />
                <code>gate={escape(str(config_payload.get('gate_mode', DEFAULT_CONFIG['gate_mode'])))}</code>
            </div>
            <div class="config-box">
                <b>Файлы</b><br />
                <code>source={escape(str(source))}</code><br />
                <code>results={escape(str(results))}</code><br />
                <code>scores={escape(str(scores))}</code>
            </div>
        </div>
    </section>

    <h2>Сводная таблица</h2>
    <table>
        <tr>
            <th>Скважина</th>
            <th>Split</th>
            <th>Интервал</th>
            <th>Data start</th>
            <th>Data end</th>
            <th>Факт. старт</th>
            <th>Факт. конец</th>
            <th>Detected</th>
            <th>Status</th>
            <th>Delay</th>
            <th>Каналы</th>
            <th>Старты</th>
        </tr>
        {''.join(table_rows)}
    </table>

    <h2>Графики</h2>
    {''.join(plot_blocks)}
</body>
</html>
"""

    output.write_text(html, encoding="utf-8")
    print(f"Отчёт сгенерирован: {output}")
    return output


def run_cli(default_anomaly: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate unified PaAno HTML report.")
    if default_anomaly is None:
        parser.add_argument("anomaly", choices=["negermet", "pritok", "salt"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--scores", type=str, default=None)
    args = parser.parse_args()

    anomaly_key = default_anomaly or args.anomaly
    generate_report(
        anomaly_key=anomaly_key,
        output_path=args.output,
        source_path=args.source,
        results_path=args.results,
        scores_path=args.scores,
    )
