"use client";

import dynamic from "next/dynamic";
import type { Data, Layout, Shape } from "plotly.js";
import { useMemo } from "react";
import type { WellSeriesResponse } from "@/lib/api/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  series: WellSeriesResponse;
}

const ANOMALY_HIGHLIGHT = "rgba(220, 38, 38, 0.12)";
const ANOMALY_BORDER = "rgba(220, 38, 38, 0.8)";
const PRED_ONSET = "rgba(239, 68, 68, 0.9)";

export function WellReportChart({ series }: Props) {
  const { scoreTraces, telemetryTraces, shapes } = useMemo(() => {
    const tr: Data[] = [];

    if (series.score.length > 0) {
      tr.push({
        type: "scattergl",
        mode: "lines",
        name: "score",
        x: series.score.map((p) => p.t),
        y: series.score.map((p) => p.v),
        line: { color: "#111827", width: 1.6 },
        yaxis: "y",
      });
    }
    if (series.paano_short.length > 0) {
      tr.push({
        type: "scattergl",
        mode: "lines",
        name: "paano_short",
        x: series.paano_short.map((p) => p.t),
        y: series.paano_short.map((p) => p.v),
        line: { color: "#0891b2", width: 1 },
        yaxis: "y",
      });
    }
    if (series.paano_long.length > 0) {
      tr.push({
        type: "scattergl",
        mode: "lines",
        name: "paano_long",
        x: series.paano_long.map((p) => p.t),
        y: series.paano_long.map((p) => p.v),
        line: { color: "#a855f7", width: 1 },
        yaxis: "y",
      });
    }

    const tel: Data[] = series.telemetry.map((ch) => ({
      type: "scattergl",
      mode: "lines",
      name: ch.name,
      x: ch.points.map((p) => p.t),
      y: ch.points.map((p) => p.v),
      line: { width: 1 },
      yaxis: "y2",
      visible: "legendonly",
    }));

    const anomalyShapes: Partial<Shape>[] = series.intervals.map((iv) => ({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: iv.start,
      x1: iv.end,
      y0: 0,
      y1: 1,
      fillcolor: ANOMALY_HIGHLIGHT,
      line: { color: ANOMALY_BORDER, width: 2 },
      layer: "below",
    }));

    const onsetShapes: Partial<Shape>[] = series.predicted_starts.map((ps) => ({
      type: "line",
      xref: "x",
      yref: "paper",
      x0: ps.t,
      x1: ps.t,
      y0: 0,
      y1: 1,
      line: { color: PRED_ONSET, width: 1.5, dash: "dot" },
    }));

    return {
      scoreTraces: tr,
      telemetryTraces: tel,
      shapes: [...anomalyShapes, ...onsetShapes],
    };
  }, [series]);

  const data = [...scoreTraces, ...telemetryTraces];

  const layout: Partial<Layout> = {
    autosize: true,
    height: 560,
    margin: { l: 60, r: 60, t: 30, b: 50 },
    showlegend: true,
    legend: { orientation: "h", y: -0.15 },
    xaxis: { title: { text: "Время" } },
    yaxis: {
      title: { text: "score / paano" },
      side: "left",
      zeroline: true,
    },
    yaxis2: {
      title: { text: "телеметрия" },
      overlaying: "y",
      side: "right",
      showgrid: false,
    },
    shapes,
    hovermode: "x unified",
  };

  return (
    <div className="w-full rounded-md border bg-card p-2">
      <Plot
        data={data}
        layout={layout}
        config={{ responsive: true, displaylogo: false }}
        useResizeHandler
        style={{ width: "100%", height: "560px" }}
      />
      <div className="px-2 py-1 text-xs text-muted-foreground">
        {series.n_points_downsampled} / {series.n_points_raw} точек ·{" "}
        {series.telemetry.length} каналов · {series.intervals.length} интервалов
        · {series.predicted_starts.length} предсказанных onset'ов
      </div>
    </div>
  );
}
