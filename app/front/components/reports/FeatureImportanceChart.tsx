"use client";

import dynamic from "next/dynamic";
import type { Annotations, Data, Layout, Shape } from "plotly.js";
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

const ACCENT = "#7c3aed";
const NEUTRAL = "#bdc3c7";
const ANOMALY_FILL = "rgba(239, 68, 68, 0.10)";
const START_COLOR = "#16a34a";
const END_COLOR = "#dc2626";
const DETECT_COLOR = "#7c3aed";

const TOP_BAR = 15;
const TOP_POSITIVE = 3;
const TOP_NEGATIVE = 4;

function fmt(ts: string): string {
  const d = new Date(ts);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function domainsFor(n: number): [number, number][] {
  const gap = 0.04;
  const panelH = (1 - gap * (n - 1)) / n;
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const top = 1 - i * (panelH + gap);
    const bottom = top - panelH;
    out.push([Math.max(0, bottom), Math.min(1, top)]);
  }
  return out;
}

export function FeatureImportanceChart({ fi, series }: Props) {
  const {
    data: barData,
    layout: barLayout,
    count,
  } = useMemo(() => {
    const sorted = [...fi.items]
      .sort((a, b) => b.importance - a.importance)
      .slice(0, TOP_BAR);
    const forPlot = sorted.slice().reverse();
    const colors = forPlot.map((it) => (it.importance > 0 ? ACCENT : NEUTRAL));
    const trace: Data = {
      type: "bar",
      orientation: "h",
      x: forPlot.map((it) => it.importance),
      y: forPlot.map((it) => it.feature),
      marker: { color: colors },
      hovertemplate: "%{y}: %{x:+.1f}%<extra></extra>",
    };
    const layout: Partial<Layout> = {
      autosize: true,
      height: Math.max(400, forPlot.length * 28 + 100),
      margin: { l: 300, r: 40, t: 60, b: 60 },
      title: {
        text: `Скважина ${fi.well_id} — влияние каналов на обнаружение аномалии`,
        font: { size: 14 },
      },
      xaxis: {
        title: {
          text: "Изменение способности обнаружить аномалию при отключении канала, %",
        },
        zeroline: true,
        zerolinecolor: "#111827",
        zerolinewidth: 1,
      },
      yaxis: { automargin: true },
      bargap: 0.3,
      showlegend: true,
      legend: { orientation: "h", y: -0.15 },
    };
    const dummyPos: Data = {
      type: "bar",
      orientation: "h",
      x: [0],
      y: [""],
      marker: { color: ACCENT },
      name: "Канал помогает обнаружить аномалию",
      showlegend: true,
      hoverinfo: "skip",
    };
    const dummyNeg: Data = {
      type: "bar",
      orientation: "h",
      x: [0],
      y: [""],
      marker: { color: NEUTRAL },
      name: "Канал не помогает / мешает",
      showlegend: true,
      hoverinfo: "skip",
    };
    return { data: [trace, dummyPos, dummyNeg], layout, count: sorted.length };
  }, [fi]);

  const timeseries = useMemo(() => {
    if (!series || series.telemetry.length === 0 || series.score.length === 0)
      return null;

    const positives = fi.items
      .filter((it) => it.importance > 0)
      .sort((a, b) => b.importance - a.importance)
      .slice(0, TOP_POSITIVE);
    const negatives = fi.items
      .filter((it) => it.importance <= 0)
      .sort((a, b) => a.importance - b.importance)
      .slice(0, TOP_NEGATIVE);

    const byName = new Map(series.telemetry.map((c) => [c.name, c]));
    const available = [...positives, ...negatives].filter((it) =>
      byName.has(it.feature),
    );
    if (available.length === 0) return null;

    type Panel =
      | { label: string; kind: "score" }
      | { label: string; kind: "channel"; feat: string };

    const panels: Panel[] = [
      { label: "Отклонение от нормы<br>(все каналы)", kind: "score" },
      ...available.map<Panel>((it) => {
        const pct = it.importance;
        const tag =
          pct > 0.5
            ? `влияние: +${pct.toFixed(1)}%`
            : pct < -0.5
              ? `влияние: ${pct.toFixed(1)}%`
              : "влияние: ~0%";
        return {
          label: `${it.feature}<br><span style='font-size:9px'>(${tag})</span>`,
          kind: "channel",
          feat: it.feature,
        };
      }),
    ];

    const domains = domainsFor(panels.length);
    const traces: Data[] = [];

    panels.forEach((p, idx) => {
      const axisKey = idx === 0 ? "y" : `y${idx + 1}`;
      if (p.kind === "score") {
        traces.push({
          type: "scattergl",
          mode: "lines",
          name: "Отклонение от нормы",
          x: series.score.map((pt) => pt.t),
          y: series.score.map((pt) => pt.v),
          line: { color: ACCENT, width: 0.9 },
          fill: "tozeroy",
          fillcolor: "rgba(124, 58, 237, 0.18)",
          yaxis: axisKey,
          showlegend: false,
        });
      } else {
        const ch = byName.get(p.feat);
        if (ch) {
          traces.push({
            type: "scattergl",
            mode: "lines",
            name: p.feat,
            x: ch.points.map((pt) => pt.t),
            y: ch.points.map((pt) => pt.v),
            line: { color: "#2563eb", width: 0.8 },
            yaxis: axisKey,
            showlegend: false,
          });
        }
      }
    });

    const firstResult = series.results[0];
    const startDt = firstResult?.actual_start ?? series.intervals[0]?.start;
    const endDt = firstResult?.actual_end ?? series.intervals[0]?.end;
    const detected = firstResult?.detected_time;

    const shapes: Partial<Shape>[] = [];
    const annotations: Partial<Annotations>[] = [];
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
        line: { width: 0 },
        layer: "below",
      });
    }
    if (startDt) {
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: startDt,
        x1: startDt,
        y0: 0,
        y1: 1,
        line: { color: START_COLOR, width: 1.2 },
      });
      annotations.push({
        xref: "x",
        yref: "paper",
        x: startDt,
        y: 1,
        xanchor: "left",
        yanchor: "top",
        text: `Факт начало<br>${fmt(startDt)}`,
        font: { size: 10, color: START_COLOR },
        bgcolor: "rgba(255,255,255,0.9)",
        bordercolor: START_COLOR,
        borderwidth: 1,
        borderpad: 3,
        showarrow: false,
      });
    }
    if (endDt) {
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: endDt,
        x1: endDt,
        y0: 0,
        y1: 1,
        line: { color: END_COLOR, width: 1.2, dash: "dash" },
      });
      annotations.push({
        xref: "x",
        yref: "paper",
        x: endDt,
        y: 1,
        xanchor: "right",
        yanchor: "top",
        text: `Факт конец<br>${fmt(endDt)}`,
        font: { size: 10, color: END_COLOR },
        bgcolor: "rgba(255,255,255,0.9)",
        bordercolor: END_COLOR,
        borderwidth: 1,
        borderpad: 3,
        showarrow: false,
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
        line: { color: DETECT_COLOR, width: 1.6, dash: "dashdot" },
      });
      annotations.push({
        xref: "x",
        yref: "paper",
        x: detected,
        y: 0.95,
        xanchor: "left",
        yanchor: "top",
        text: `Обнаружено<br>${fmt(detected)}`,
        font: { size: 10, color: DETECT_COLOR },
        bgcolor: "rgba(255,255,255,0.9)",
        bordercolor: DETECT_COLOR,
        borderwidth: 1,
        borderpad: 3,
        showarrow: false,
      });
    }

    const layout: Partial<Layout> = {
      autosize: true,
      height: 240 * panels.length + 80,
      margin: { l: 100, r: 40, t: 70, b: 60 },
      title: {
        text: `Скважина ${fi.well_id} — показания наиболее значимых каналов`,
        font: { size: 14 },
      },
      xaxis: { title: { text: "Время" }, anchor: `y${panels.length}` as never },
      hovermode: "x unified",
      shapes,
      annotations,
      showlegend: false,
      plot_bgcolor: "#ffffff",
    };

    panels.forEach((panel, idx) => {
      const axisKey = idx === 0 ? "yaxis" : `yaxis${idx + 1}`;
      const [bottom, top] = domains[idx];
      (layout as Record<string, unknown>)[axisKey] = {
        title: { text: panel.label, font: { size: 9 } },
        domain: [bottom, top],
        showgrid: true,
        gridcolor: "rgba(0,0,0,0.08)",
        zeroline: panel.kind === "score",
      };
    });

    return {
      data: traces,
      layout,
      positiveCount: positives.length,
      negativeCount: negatives.length,
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
          <p className="px-2 pt-2 text-xs text-muted-foreground">
            {timeseries.positiveCount} top-каналов помогают ·{" "}
            {timeseries.negativeCount} мешают
          </p>
          <Plot
            data={timeseries.data}
            layout={timeseries.layout}
            config={{ responsive: true, displaylogo: false }}
            useResizeHandler
            style={{ width: "100%", height: `${timeseries.layout.height}px` }}
          />
        </div>
      )}
    </div>
  );
}
