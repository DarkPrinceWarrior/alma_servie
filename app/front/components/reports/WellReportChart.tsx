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
  series: WellSeriesResponse;
  fi?: FeatureImportanceResponse | null;
}

const ACCENT = "#7c3aed";
const ANOMALY_FILL = "rgba(239, 68, 68, 0.10)";
const ANOMALY_BORDER = "rgba(239, 68, 68, 0.55)";
const START_COLOR = "#16a34a";
const END_COLOR = "#dc2626";
const DETECT_COLOR = "#7c3aed";
const CHANNEL_COLOR = "#2563eb";

const PRESSURE_CHANNEL = "Давление на приеме насоса кгс/см²";
const FREQ_CHANNEL = "Выходная частота";

function pickTopChannel(
  telemetryNames: string[],
  fi: FeatureImportanceResponse | null | undefined,
): string | null {
  if (fi?.items && fi.items.length > 0) {
    const sorted = [...fi.items].sort((a, b) => b.importance - a.importance);
    for (const it of sorted) {
      if (it.feature === PRESSURE_CHANNEL) continue;
      if (telemetryNames.includes(it.feature)) return it.feature;
    }
  }
  const skip = new Set([PRESSURE_CHANNEL, FREQ_CHANNEL]);
  return telemetryNames.find((n) => !skip.has(n)) ?? null;
}

function fmt(ts: string): string {
  const d = new Date(ts);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function domainsFor(n: number): [number, number][] {
  const gap = 0.06;
  const panelH = (1 - gap * (n - 1)) / n;
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const top = 1 - i * (panelH + gap);
    const bottom = top - panelH;
    out.push([Math.max(0, bottom), Math.min(1, top)]);
  }
  return out;
}

export function WellReportChart({ series, fi }: Props) {
  const {
    traces,
    shapes,
    annotations,
    layout,
  }: {
    traces: Data[];
    shapes: Partial<Shape>[];
    annotations: Partial<Annotations>[];
    layout: Partial<Layout>;
  } = useMemo(() => {
    const telNames = series.telemetry.map((c) => c.name);
    const hasScore = series.score.length > 0;
    const hasPressure = telNames.includes(PRESSURE_CHANNEL);
    const top = pickTopChannel(telNames, fi);
    const panels: {
      label: string;
      kind: "score" | "channel";
      channel?: string;
    }[] = [];
    if (hasScore)
      panels.push({ label: "Отклонение от нормы (score)", kind: "score" });
    if (hasPressure)
      panels.push({
        label: PRESSURE_CHANNEL,
        kind: "channel",
        channel: PRESSURE_CHANNEL,
      });
    if (top && top !== PRESSURE_CHANNEL)
      panels.push({ label: top, kind: "channel", channel: top });

    const domains = domainsFor(panels.length);
    const traces: Data[] = [];

    panels.forEach((panel, idx) => {
      const axisKey = idx === 0 ? "y" : `y${idx + 1}`;
      if (panel.kind === "score") {
        traces.push({
          type: "scattergl",
          mode: "lines",
          name: "Отклонение от нормы",
          x: series.score.map((p) => p.t),
          y: series.score.map((p) => p.v),
          line: { color: ACCENT, width: 0.9 },
          fill: "tozeroy",
          fillcolor: "rgba(124, 58, 237, 0.18)",
          yaxis: axisKey,
          showlegend: idx === 0,
        });
      } else if (panel.channel) {
        const ch = series.telemetry.find((c) => c.name === panel.channel);
        if (ch) {
          traces.push({
            type: "scattergl",
            mode: "lines",
            name: panel.channel,
            x: ch.points.map((p) => p.t),
            y: ch.points.map((p) => p.v),
            line: { color: CHANNEL_COLOR, width: 0.9 },
            yaxis: axisKey,
            showlegend: false,
          });
        }
      }
    });

    const firstResult = series.results[0];
    const fallbackInterval = series.intervals[0];
    const actualStart =
      firstResult?.actual_start ?? fallbackInterval?.start ?? null;
    const actualEnd = firstResult?.actual_end ?? fallbackInterval?.end ?? null;
    const detected = firstResult?.detected_time ?? null;

    const shapes: Partial<Shape>[] = [];
    const annotations: Partial<Annotations>[] = [];

    if (actualStart && actualEnd) {
      shapes.push({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: actualStart,
        x1: actualEnd,
        y0: 0,
        y1: 1,
        fillcolor: ANOMALY_FILL,
        line: { color: ANOMALY_BORDER, width: 0 },
        layer: "below",
      });
    }
    if (actualStart) {
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: actualStart,
        x1: actualStart,
        y0: 0,
        y1: 1,
        line: { color: START_COLOR, width: 1.4 },
      });
      annotations.push({
        xref: "x",
        yref: "paper",
        x: actualStart,
        y: 1,
        xanchor: "left",
        yanchor: "top",
        text: `Факт начало<br>${fmt(actualStart)}`,
        font: { size: 10, color: START_COLOR },
        bgcolor: "rgba(255,255,255,0.9)",
        bordercolor: START_COLOR,
        borderwidth: 1,
        borderpad: 3,
        ax: 5,
        ay: -2,
        showarrow: false,
      });
    }
    if (actualEnd) {
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: actualEnd,
        x1: actualEnd,
        y0: 0,
        y1: 1,
        line: { color: END_COLOR, width: 1.4, dash: "dash" },
      });
      annotations.push({
        xref: "x",
        yref: "paper",
        x: actualEnd,
        y: 1,
        xanchor: "right",
        yanchor: "top",
        text: `Факт конец<br>${fmt(actualEnd)}`,
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
        line: { color: DETECT_COLOR, width: 1.8, dash: "dashdot" },
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

    const intervalIdx =
      firstResult?.interval_idx ?? fallbackInterval?.interval_idx ?? 1;
    const layout: Partial<Layout> = {
      autosize: true,
      height: 260 * Math.max(panels.length, 1) + 80,
      margin: { l: 80, r: 40, t: 60, b: 60 },
      title: {
        text: `Скважина ${series.well_id} · интервал ${intervalIdx}`,
        font: { size: 15 },
      },
      hovermode: "x unified",
      showlegend: false,
      xaxis: { title: { text: "Время" }, anchor: `y${panels.length}` as never },
      shapes,
      annotations,
      plot_bgcolor: "#ffffff",
    };

    panels.forEach((panel, idx) => {
      const axisKey = idx === 0 ? "yaxis" : `yaxis${idx + 1}`;
      const [bottom, top] = domains[idx];
      (layout as Record<string, unknown>)[axisKey] = {
        title: { text: panel.label, font: { size: 11 } },
        domain: [bottom, top],
        showgrid: true,
        gridcolor: "rgba(0,0,0,0.08)",
        zeroline: panel.kind === "score",
      };
    });

    return { traces, shapes, annotations, layout };
  }, [series, fi]);

  const firstResult = series.results[0];
  const n = traces.length;

  return (
    <div className="w-full space-y-2 rounded-md border bg-card p-2">
      {n === 0 ? (
        <div className="p-6 text-sm text-muted-foreground">
          Нет данных для графика.
        </div>
      ) : (
        <Plot
          data={traces}
          layout={layout}
          config={{ responsive: true, displaylogo: false }}
          useResizeHandler
          style={{ width: "100%", height: `${layout.height}px` }}
        />
      )}
      <div className="flex flex-wrap gap-4 px-2 py-1 text-xs text-muted-foreground">
        <span>
          {series.n_points_downsampled} / {series.n_points_raw} точек score
        </span>
        <span>{series.telemetry.length} каналов в телеметрии</span>
        <span>{series.intervals.length} размеченных интервалов</span>
        {firstResult?.status && (
          <span className="font-medium text-foreground">
            Статус: {firstResult.status}
          </span>
        )}
      </div>
    </div>
  );
}
