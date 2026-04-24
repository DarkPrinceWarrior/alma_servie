"use client";

import dynamic from "next/dynamic";
import type { Data, Layout } from "plotly.js";
import { useMemo } from "react";
import type { FeatureImportanceResponse } from "@/lib/api/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  fi: FeatureImportanceResponse;
}

const POS_COLOR = "#ef4444";
const NEG_COLOR = "#3b82f6";

export function FeatureImportanceChart({ fi }: Props) {
  const { data, layout, count } = useMemo(() => {
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
      margin: { l: 280, r: 40, t: 24, b: 40 },
      xaxis: {
        title: { text: "importance" },
        zeroline: true,
        zerolinecolor: "#9ca3af",
      },
      yaxis: { automargin: true },
      bargap: 0.3,
    };

    return { data: [trace], layout, count: sorted.length };
  }, [fi]);

  if (count === 0) {
    return (
      <div className="rounded-md border bg-card p-6 text-sm text-muted-foreground">
        Feature importance для скважины «{fi.well_id}» не найдена.
      </div>
    );
  }

  return (
    <div className="w-full rounded-md border bg-card p-2">
      <Plot
        data={data}
        layout={layout}
        config={{ responsive: true, displaylogo: false }}
        useResizeHandler
        style={{ width: "100%", height: `${layout.height}px` }}
      />
    </div>
  );
}
