"use client";

import dynamic from "next/dynamic";
import type { Data, Layout, Shape } from "plotly.js";
import { useMemo } from "react";
import type {
  FeatureImportanceResponse,
  WellSeriesResponse,
} from "@/lib/api/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  fi: FeatureImportanceResponse;
  series?: WellSeriesResponse | null;
}

const POS_COLOR = "#ef4444";
const NEG_COLOR = "#3b82f6";
const ANOMALY_FILL = "rgba(220, 38, 38, 0.1)";
const ANOMALY_BORDER = "rgba(220, 38, 38, 0.8)";
const ONSET_COLOR = "#a855f7";

const TOP_POSITIVE = 3;
const TOP_NEGATIVE = 4;

export function FeatureImportanceChart({ fi, series }: Props) {
  // 1) Bar chart
  const {
    data: barData,
    layout: barLayout,
    count,
  } = useMemo(() => {
    const items = fi.items;
    const sorted = [...items].sort((a, b) => a.importance - b.importance);
    const colors = sorted.map((it) =>
      it.importance >= 0 ? POS_COLOR : NEG_COLOR,
    );
    const trace: Data = {
      type: "bar",
      orientation: "h",
      x: sorted.map((it) => it.importance),
      y: sorted.map((it) => it.feature),
      marker: { color: colors },
      hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
    };
    const layout: Partial<Layout> = {
      autosize: true,
      height: Math.max(400, sorted.length * 24 + 100),
      margin: { l: 300, r: 40, t: 24, b: 40 },
      xaxis: {
        title: { text: "Важность (%)" },
        zeroline: true,
        zerolinecolor: "#9ca3af",
      },
      yaxis: { automargin: true },
      bargap: 0.3,
    };
    return { data: [trace], layout, count: sorted.length };
  }, [fi]);

  // 2) Time-series top-positive / top-negative channels
  const timeseries = useMemo(() => {
    if (!series || series.telemetry.length === 0) return null;

    const positives = fi.items.filter((it) => it.importance > 0);
    const negatives = fi.items.filter((it) => it.importance < 0);

    const topPos = positives
      .slice()
      .sort((a, b) => b.importance - a.importance)
      .slice(0, TOP_POSITIVE);
    const topNeg = negatives
      .slice()
      .sort((a, b) => a.importance - b.importance)
      .slice(0, TOP_NEGATIVE);

    const byName = new Map(series.telemetry.map((c) => [c.name, c]));
    const pick = (items: typeof fi.items, color: string): Data[] =>
      items
        .map((it) => byName.get(it.feature))
        .filter(
          (c): c is NonNullable<typeof c> =>
            c !== undefined && c.points.length > 0,
        )
        .map((c) => ({
          type: "scattergl",
          mode: "lines",
          name: c.name,
          x: c.points.map((p) => p.t),
          y: c.points.map((p) => p.v),
          line: { width: 1.2, color },
        }));

    const posTraces: Data[] = pick(topPos, POS_COLOR);
    const negTraces: Data[] = pick(topNeg, NEG_COLOR);

    if (posTraces.length === 0 && negTraces.length === 0) return null;

    const scoreTrace: Data[] =
      series.score.length > 0
        ? [
            {
              type: "scattergl",
              mode: "lines",
              name: "Отклонение от нормы",
              x: series.score.map((p) => p.t),
              y: series.score.map((p) => p.v),
              line: { color: "#111827", width: 1.6 },
              yaxis: "y",
            },
          ]
        : [];

    const telemetryTraces: Data[] = [...posTraces, ...negTraces].map((t) => ({
      ...t,
      yaxis: "y2",
    }));

    const firstResult = series.results[0];
    const startDt = firstResult?.actual_start ?? series.intervals[0]?.start;
    const endDt = firstResult?.actual_end ?? series.intervals[0]?.end;
    const detected = firstResult?.detected_time;

    const shapes: Partial<Shape>[] = [];
    if (startDt && endDt) {
      shapes.push({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: startDt,
        x1: endDt,
        y0: 0,
        y1: 1,
        fillcolor: ANOMALY_FILL,
        line: { color: ANOMALY_BORDER, width: 1 },
        layer: "below",
      });
    }
    if (detected) {
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: detected,
        x1: detected,
        y0: 0,
        y1: 1,
        line: { color: ONSET_COLOR, width: 2, dash: "dashdot" },
      });
    }

    const layout: Partial<Layout> = {
      autosize: true,
      height: 460,
      margin: { l: 60, r: 60, t: 30, b: 70 },
      xaxis: { title: { text: "Время" } },
      yaxis: { title: { text: "Отклонение" }, side: "left" },
      yaxis2: {
        title: { text: "Каналы" },
        overlaying: "y",
        side: "right",
        showgrid: false,
      },
      legend: { orientation: "h", y: -0.2 },
      hovermode: "x unified",
      shapes,
    };

    return {
      data: [...scoreTrace, ...telemetryTraces],
      layout,
      positiveCount: posTraces.length,
      negativeCount: negTraces.length,
    };
  }, [fi, series]);

  if (count === 0) {
    return (
      <div className="rounded-md border bg-card p-6 text-sm text-muted-foreground">
        Feature importance для скважины «{fi.well_id}» не найдена.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="rounded-md border bg-card p-2">
        <p className="px-2 pt-2 text-sm font-medium">
          Важность каналов (permutation importance, %)
        </p>
        <Plot
          data={barData}
          layout={barLayout}
          config={{ responsive: true, displaylogo: false }}
          useResizeHandler
          style={{ width: "100%", height: `${barLayout.height}px` }}
        />
      </div>

      {timeseries && (
        <div className="rounded-md border bg-card p-2">
          <p className="px-2 pt-2 text-sm font-medium">
            Top-каналы во времени ({timeseries.positiveCount} полезных ·{" "}
            {timeseries.negativeCount} «шумных»)
          </p>
          <Plot
            data={timeseries.data}
            layout={timeseries.layout}
            config={{ responsive: true, displaylogo: false }}
            useResizeHandler
            style={{ width: "100%", height: "460px" }}
          />
        </div>
      )}
    </div>
  );
}
