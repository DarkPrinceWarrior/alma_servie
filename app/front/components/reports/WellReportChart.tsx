"use client";

import dynamic from "next/dynamic";
import type { Data, Layout, Shape } from "plotly.js";
import { useMemo } from "react";
import type {
  FeatureImportanceResponse,
  WellSeriesResponse,
} from "@/lib/api/types";
import { useI18n } from "@/lib/i18n";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  series: WellSeriesResponse;
  fi?: FeatureImportanceResponse | null;
}

const ANOMALY_FILL = "rgba(220, 38, 38, 0.12)";
const ANOMALY_BORDER = "rgba(220, 38, 38, 0.8)";
const ONSET_COLOR = "#a855f7";
const START_COLOR = "#16a34a";
const END_COLOR = "#dc2626";

const PRESSURE_CHANNEL = "Давление на приеме насоса кгс/см²";

const STATUS_KEY: Record<string, string> = {
  Detected: "status.detected",
  detected: "status.detected",
  "Not found": "status.notFound",
  "not found": "status.notFound",
  "Not detected": "status.notFound",
  Missed: "status.missed",
};

function pickDefaultChannels(
  telemetryNames: string[],
  fi: FeatureImportanceResponse | null | undefined,
): Set<string> {
  const defaults = new Set<string>();
  if (telemetryNames.includes(PRESSURE_CHANNEL)) defaults.add(PRESSURE_CHANNEL);

  if (fi?.items && fi.items.length > 0) {
    const top = fi.items
      .filter((it) => it.feature !== PRESSURE_CHANNEL && it.importance > 0)
      .sort((a, b) => b.importance - a.importance);
    if (top[0] && telemetryNames.includes(top[0].feature)) {
      defaults.add(top[0].feature);
    }
  }
  return defaults;
}

export function WellReportChart({ series, fi }: Props) {
  const { t, ch: chLabel, plural } = useI18n();
  const { scoreTraces, telemetryTraces, shapes } = useMemo(() => {
    const tr: Data[] = [];
    if (series.score.length > 0) {
      tr.push({
        type: "scattergl",
        mode: "lines",
        name: t("chart.scoreDeviation"),
        x: series.score.map((p) => p.t),
        y: series.score.map((p) => p.v),
        line: { color: "#111827", width: 1.6 },
        yaxis: "y",
      });
    }

    const telemetryNames = series.telemetry.map((t) => t.name);
    const defaults = pickDefaultChannels(telemetryNames, fi);

    const tel: Data[] = series.telemetry.map((ch) => ({
      type: "scattergl",
      mode: "lines",
      name: chLabel(ch.name),
      x: ch.points.map((p) => p.t),
      y: ch.points.map((p) => p.v),
      line: { width: 1 },
      yaxis: "y2",
      visible: defaults.has(ch.name) ? true : "legendonly",
    }));

    const firstResult = series.results[0];
    const fallbackInterval = series.intervals[0];
    const actualStart =
      firstResult?.actual_start ?? fallbackInterval?.start ?? null;
    const actualEnd = firstResult?.actual_end ?? fallbackInterval?.end ?? null;
    const detected = firstResult?.detected_time ?? null;
    // Размеченные скв. — время обнаружения из результата; слепые (без результата) —
    // из предсказанных стартов (выбранная дата детекции, без зоны/факт. начала).
    const onsetTimes: string[] = detected
      ? [detected]
      : series.predicted_starts.map((p) => p.t);

    const s: Partial<Shape>[] = [];

    // Красная зона фактической аномалии (actual_start → actual_end)
    if (actualStart && actualEnd) {
      s.push({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: actualStart,
        x1: actualEnd,
        y0: 0,
        y1: 1,
        fillcolor: ANOMALY_FILL,
        line: { color: ANOMALY_BORDER, width: 1.5 },
        layer: "below",
      });
    }
    // Зелёная вертикальная линия — фактическое начало
    if (actualStart) {
      s.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: actualStart,
        x1: actualStart,
        y0: 0,
        y1: 1,
        line: { color: START_COLOR, width: 1.5 },
      });
    }
    // Красная пунктирная — фактическое окончание
    if (actualEnd) {
      s.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: actualEnd,
        x1: actualEnd,
        y0: 0,
        y1: 1,
        line: { color: END_COLOR, width: 1.5, dash: "dot" },
      });
    }
    // Фиолетовая штрихпунктирная — время обнаружения (предполагаемая дата)
    for (const t of onsetTimes) {
      s.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: t,
        x1: t,
        y0: 0,
        y1: 1,
        line: { color: ONSET_COLOR, width: 2, dash: "dashdot" },
      });
    }

    return { scoreTraces: tr, telemetryTraces: tel, shapes: s };
  }, [series, fi, t, chLabel]);

  const data = [...scoreTraces, ...telemetryTraces];

  const layout: Partial<Layout> = {
    autosize: true,
    height: 560,
    margin: { l: 60, r: 60, t: 30, b: 70 },
    showlegend: true,
    legend: { orientation: "h", y: -0.18 },
    xaxis: { title: { text: t("chart.time") } },
    yaxis: {
      title: { text: t("chart.scoreDeviation") },
      side: "left",
      zeroline: true,
    },
    yaxis2: {
      title: { text: t("chart.telemetryChannels") },
      overlaying: "y",
      side: "right",
      showgrid: false,
    },
    shapes,
    hovermode: "x unified",
  };

  const firstResult = series.results[0];

  return (
    <div className="w-full space-y-2 rounded-md border bg-card p-2">
      <div className="flex flex-wrap items-center gap-4 px-2 text-xs text-muted-foreground">
        <LegendDot color="#111827" label={t("chart.legend.score")} />
        <LegendDot color={ANOMALY_BORDER} label={t("chart.legend.zone")} filled />
        <LegendDot color={START_COLOR} label={t("chart.legend.actualStart")} />
        <LegendDot color={END_COLOR} label={t("chart.legend.actualEnd")} dashed />
        <LegendDot color={ONSET_COLOR} label={t("chart.legend.detected")} dashed />
      </div>
      <Plot
        data={data}
        layout={layout}
        config={{ responsive: true, displaylogo: false }}
        useResizeHandler
        style={{ width: "100%", height: "560px" }}
      />
      <div className="flex flex-wrap gap-4 px-2 py-1 text-xs text-muted-foreground">
        <span>{t("chart.channelsCount", { n: series.telemetry.length })}</span>
        <span>
          {series.intervals.length}{" "}
          {plural(series.intervals.length, "labeledInterval")}
        </span>
        {firstResult?.status && (
          <span className="font-medium text-foreground">
            {t("chart.statusPrefix")}
            {STATUS_KEY[firstResult.status]
              ? t(STATUS_KEY[firstResult.status])
              : firstResult.status}
          </span>
        )}
      </div>
    </div>
  );
}

function LegendDot({
  color,
  label,
  dashed,
  filled,
}: {
  color: string;
  label: string;
  dashed?: boolean;
  filled?: boolean;
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-block h-2 w-4"
        style={{
          background: filled ? `${color}22` : "transparent",
          borderTop: dashed
            ? `2px dashed ${color}`
            : filled
              ? `1px solid ${color}`
              : `2px solid ${color}`,
          ...(filled
            ? {
                borderLeft: `1px solid ${color}`,
                borderRight: `1px solid ${color}`,
              }
            : {}),
        }}
      />
      {label}
    </span>
  );
}
