from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _safe_get(d: dict, key: str, default=None):
    v = d.get(key, default)
    if v is None:
        return default
    return v


def _format_metric(v, fmt="{:.3f}", default="—"):
    if v is None:
        return default
    try:
        if isinstance(v, (int, float)) and not (np.isfinite(v) if isinstance(v, float) else True):
            return default
        return fmt.format(v)
    except Exception:
        return default


def build_summary_html(metrics: dict, event_class: int, event_name: str) -> str:
    splits = metrics.get("splits", {})
    rows_html = []
    for split_name in ("train", "val", "test", "all"):
        if split_name not in splits:
            continue
        s = splits[split_name]
        rows_html.append(
            "<tr>"
            f"<td><b>{split_name}</b></td>"
            f"<td>{_safe_get(s, 'interval_count', 0)}</td>"
            f"<td>{_format_metric(s.get('hit_rate'))}</td>"
            f"<td>{_format_metric(s.get('false_alarms_per_day'))}</td>"
            f"<td>{_format_metric(s.get('avg_starts_per_interval'), '{:.2f}')}</td>"
            f"<td>{_format_metric(s.get('first_alert_delay_median_hours'), '{:.2f}')}</td>"
            f"<td>{_format_metric(s.get('first_alert_delay_p90_hours'), '{:.2f}')}</td>"
            f"<td>{_safe_get(s, 'observed_days_total', 0):.1f}</td>"
            "</tr>"
        )
    cfg = metrics.get("config", {})
    cfg_html = "".join(
        f"<li><b>{k}</b>: {v}</li>" for k, v in cfg.items()
    )
    return f"""
    <div style="font-family: system-ui, -apple-system, sans-serif; margin: 1.5rem;">
      <h1>3W PaAno — класс {event_class}: {event_name}</h1>
      <h2>Метрики</h2>
      <table border="1" cellpadding="6" style="border-collapse: collapse; font-size: 14px;">
        <thead>
          <tr>
            <th>Сплит</th><th>Интервалов</th><th>Hit-rate</th><th>FAR/сутки</th>
            <th>Срабатываний/интервал</th><th>Median delay, ч</th><th>P90 delay, ч</th>
            <th>Наблюдалось, дни</th>
          </tr>
        </thead>
        <tbody>{"".join(rows_html)}</tbody>
      </table>
      <h3>Выбранная конфигурация порогов</h3>
      <ul>{cfg_html}</ul>
    </div>
    """


def plot_one_instance(
    instance_id: str,
    scores_sub: pd.DataFrame,
    interval_row: pd.Series | None,
    starts_sub: pd.DataFrame,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=("PaAno score", "EMA + reference mask"),
        vertical_spacing=0.08,
    )
    ts = scores_sub["timestamp"].to_numpy()
    fig.add_trace(
        go.Scatter(x=ts, y=scores_sub["score"], mode="lines", name="score", line=dict(color="#1f77b4")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=ts, y=scores_sub["paano_short"], mode="lines", name="paano_short",
                   line=dict(color="#ff7f0e", width=1, dash="dot"), opacity=0.55),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=ts, y=scores_sub["paano_long"], mode="lines", name="paano_long",
                   line=dict(color="#2ca02c", width=1, dash="dot"), opacity=0.55),
        row=1, col=1,
    )
    score_max = float(scores_sub["score"].max()) if len(scores_sub) else 1.0
    if interval_row is not None:
        start = pd.Timestamp(interval_row["start_date"])
        fig.add_trace(
            go.Scatter(x=[start, start], y=[0, score_max], mode="lines",
                       line=dict(color="red", width=2, dash="dash"),
                       name="undesirable_start", showlegend=True),
            row=1, col=1,
        )
        if pd.notna(interval_row.get("transient_start_ts")):
            ts_t = pd.Timestamp(interval_row["transient_start_ts"])
            fig.add_trace(
                go.Scatter(x=[ts_t, ts_t], y=[0, score_max], mode="lines",
                           line=dict(color="orange", width=1, dash="dot"),
                           name="transient", showlegend=True),
                row=1, col=1,
            )
        if pd.notna(interval_row.get("event_start_ts")):
            ts_e = pd.Timestamp(interval_row["event_start_ts"])
            fig.add_trace(
                go.Scatter(x=[ts_e, ts_e], y=[0, score_max], mode="lines",
                           line=dict(color="purple", width=1, dash="dot"),
                           name="event", showlegend=True),
                row=1, col=1,
            )
    for i_st, st in starts_sub.iterrows():
        ts_d = pd.Timestamp(st["detected_time"])
        fig.add_trace(
            go.Scatter(x=[ts_d, ts_d], y=[0, score_max], mode="lines",
                       line=dict(color="green", width=1),
                       name="detected" if i_st == starts_sub.index[0] else "detected_repeat",
                       showlegend=(i_st == starts_sub.index[0])),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=ts, y=scores_sub["reference_mask"].astype(int),
            mode="lines", name="reference_mask",
            line=dict(color="#888", width=1),
            fill="tozeroy", opacity=0.4,
        ),
        row=2, col=1,
    )
    fig.update_layout(
        height=520,
        title=f"Instance {instance_id}",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline HTML report for one 3W event class.")
    parser.add_argument("--config", default="configs/3w_paano.json")
    parser.add_argument("--event-class", type=int, required=True)
    parser.add_argument("--max-instances", type=int, default=40)
    args = parser.parse_args()

    cfg = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    event_labels = {int(k): v for k, v in cfg["dataset"]["event_labels"].items()}
    event_name = event_labels.get(args.event_class, f"CLASS_{args.event_class}")
    scores_path = PROJECT_ROOT / "artifacts" / "3w" / "scores" / f"class_{args.event_class}_scores.parquet"
    starts_path = PROJECT_ROOT / "artifacts" / "3w" / "scores" / f"class_{args.event_class}_predicted_starts.parquet"
    intervals_path = PROJECT_ROOT / cfg["dataset"]["intervals_path"]
    metrics_path = PROJECT_ROOT / "artifacts" / "3w" / "metrics" / f"class_{args.event_class}_metrics.json"
    out_dir = PROJECT_ROOT / "artifacts" / "3w" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"3w_class_{args.event_class}_{event_name}.html"

    scores_df = pd.read_parquet(scores_path, engine="pyarrow")
    intervals_df = pd.read_parquet(intervals_path, engine="pyarrow")
    if "folder_label" in intervals_df.columns:
        intervals_df = intervals_df[intervals_df["folder_label"].astype(int) == args.event_class]
    starts_df = pd.read_parquet(starts_path, engine="pyarrow") if starts_path.exists() else pd.DataFrame(
        columns=["well_id", "detected_time", "split"]
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {"splits": {}, "config": {}}

    instance_order = (
        ["test"] * 1000 + ["val"] * 1000 + ["train"] * 1000
    )
    ordered_ids: list[str] = []
    for split_priority in ("test", "val", "train"):
        ids = intervals_df[intervals_df["split"].eq(split_priority)]["well_id"].tolist()
        ordered_ids.extend([i for i in ids if i not in ordered_ids])
    if args.max_instances > 0:
        ordered_ids = ordered_ids[: args.max_instances]

    figures_html: list[str] = []
    for i, iid in enumerate(ordered_ids):
        sub = scores_df[scores_df["well_id"].eq(iid)].sort_values("timestamp")
        if sub.empty:
            continue
        interval_row = intervals_df[intervals_df["well_id"].eq(iid)]
        interval_row = interval_row.iloc[0] if len(interval_row) else None
        starts_sub = starts_df[starts_df["well_id"].eq(iid)]
        fig = plot_one_instance(iid, sub, interval_row, starts_sub)
        figures_html.append(
            fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False))
        )

    html_summary = build_summary_html(metrics, args.event_class, event_name)
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>3W PaAno — класс {args.event_class}: {event_name}</title>
</head>
<body>
  {html_summary}
  <div style="margin: 1.5rem;">
    <h2>Графики по инстансам ({len(figures_html)} из {len(ordered_ids)})</h2>
    {''.join(figures_html)}
  </div>
</body>
</html>"""
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"[report] {out_path}  size={out_path.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
