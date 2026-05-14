"use client";

import { ArrowLeft } from "lucide-react";
import dynamic from "next/dynamic";
import Link from "next/link";
import type { Data, Layout, Shape } from "plotly.js";
import { type ChangeEvent, type FormEvent, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { type AnomalyType, uploads } from "@/lib/api";
import type { UploadResult } from "@/lib/api/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const ANOMALIES: { value: AnomalyType; label: string }[] = [
  { value: "negermet", label: "Негерметичность" },
  { value: "pritok", label: "Приток" },
  { value: "salt", label: "Солеотложение" },
];

type Phase = "idle" | "uploading" | "running" | "done" | "error";

function fmtDt(s: string | null | undefined): string {
  if (!s) return "—";
  return s.replace("T", " ").slice(0, 16);
}

export default function UploadPage() {
  const [anomaly, setAnomaly] = useState<AnomalyType>("negermet");
  const [wellId, setWellId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadResult | null>(null);

  function onFileChange(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    if (f && !wellId) {
      setWellId(f.name.replace(/\.(xlsx|xls)$/i, ""));
    }
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file || !wellId.trim()) return;
    setError(null);
    setResult(null);
    setPhase("uploading");
    try {
      const run = await uploads.createUpload(anomaly, wellId.trim(), file);
      setPhase("running");
      let status = run.status;
      for (
        let i = 0;
        i < 80 && status !== "succeeded" && status !== "failed";
        i++
      ) {
        await new Promise((r) => setTimeout(r, 3000));
        status = (await uploads.getUpload(run.id)).status;
      }
      if (status !== "succeeded") {
        const r = await uploads.getUpload(run.id);
        throw new Error(
          r.error_message || `Прогон завершился со статусом: ${status}`,
        );
      }
      setResult(await uploads.getUploadResult(run.id));
      setPhase("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка загрузки");
      setPhase("error");
    }
  }

  const busy = phase === "uploading" || phase === "running";

  const chart = useMemo<{ data: Data[]; layout: Partial<Layout> }>(() => {
    if (!result || result.score_series.length === 0) {
      return { data: [], layout: {} };
    }
    const x = result.score_series.map((p) => p.t);
    const y = result.score_series.map((p) => p.score);
    const shapes: Partial<Shape>[] = result.detected_starts.map((ts) => ({
      type: "line",
      x0: ts,
      x1: ts,
      yref: "paper",
      y0: 0,
      y1: 1,
      line: { color: "#a855f7", width: 2, dash: "dash" },
    }));
    return {
      data: [
        {
          x,
          y,
          type: "scatter",
          mode: "lines",
          name: "Отклонение от нормы",
          line: { color: "#4b4ce6", width: 1.5 },
        },
      ],
      layout: {
        height: 360,
        margin: { l: 50, r: 20, t: 20, b: 40 },
        xaxis: { title: { text: "Время" } },
        yaxis: { title: { text: "Score" } },
        shapes,
        showlegend: false,
        paper_bgcolor: "white",
        plot_bgcolor: "white",
      },
    };
  }, [result]);

  return (
    <div className="mx-auto max-w-[1100px] px-10 py-6 space-y-5">
      <div className="flex items-center gap-3">
        <Link
          href="/"
          className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          Главная
        </Link>
        <span className="text-muted-foreground">/</span>
        <span className="text-sm font-medium">Загрузка скважины</span>
      </div>

      <h1 className="font-display text-[23.04px] font-medium tracking-[-0.576px] text-[#222226]">
        Детекция по загруженному Excel
      </h1>
      <p className="text-sm text-[#797979]">
        Формат файла — как у отдельной скважины в сырых данных. Скважина
        неразмеченная: система прогонит инференс и покажет обнаруженные старты
        аномалии.
      </p>

      <Card>
        <CardContent className="py-5">
          <form onSubmit={onSubmit} className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <label htmlFor="anomaly" className="text-sm font-medium">
                Класс аномалии
              </label>
              <select
                id="anomaly"
                value={anomaly}
                onChange={(e) => setAnomaly(e.target.value as AnomalyType)}
                className="h-10 rounded-md border border-[#e5e5e5] bg-white px-3 text-sm"
              >
                {ANOMALIES.map((a) => (
                  <option key={a.value} value={a.value}>
                    {a.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-2">
              <label htmlFor="well" className="text-sm font-medium">
                Идентификатор скважины
              </label>
              <Input
                id="well"
                value={wellId}
                onChange={(e) => setWellId(e.target.value)}
                placeholder="например, 5042"
                required
              />
            </div>
            <div className="flex flex-col gap-2">
              <label htmlFor="file" className="text-sm font-medium">
                Excel-файл скважины (.xlsx)
              </label>
              <input
                id="file"
                type="file"
                accept=".xlsx,.xls"
                onChange={onFileChange}
                required
                className="text-sm"
              />
            </div>
            {error && <p className="text-sm text-destructive">{error}</p>}
            <Button
              type="submit"
              disabled={busy || !file || !wellId.trim()}
              className="mt-1 w-fit"
            >
              {phase === "uploading"
                ? "Загрузка файла…"
                : phase === "running"
                  ? "Идёт инференс…"
                  : "Загрузить и запустить детекцию"}
            </Button>
          </form>
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardContent className="py-5 space-y-4">
            <div className="flex flex-wrap items-center gap-8">
              <Metric
                label="Обнаружено стартов"
                value={String(result.n_detected)}
                accent={result.n_detected > 0}
              />
              <Metric label="Точек ряда" value={String(result.n_points)} />
              <Metric label="Начало данных" value={fmtDt(result.time_start)} />
              <Metric label="Конец данных" value={fmtDt(result.time_end)} />
              <Metric
                label="Score (мин/медиана/макс)"
                value={`${result.score_min?.toFixed(4) ?? "—"} / ${
                  result.score_median?.toFixed(4) ?? "—"
                } / ${result.score_max?.toFixed(4) ?? "—"}`}
              />
            </div>

            {result.detected_starts.length > 0 ? (
              <div>
                <p className="text-sm font-medium text-[#222226]">
                  Обнаруженные старты аномалии:
                </p>
                <ul className="mt-1 flex flex-wrap gap-2">
                  {result.detected_starts.map((ts) => (
                    <li
                      key={ts}
                      className="rounded-full bg-[rgba(149,45,45,0.1)] px-3 py-1 text-sm font-medium text-[#c43232]"
                    >
                      {fmtDt(ts)}
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <p className="text-sm text-[#16a34a]">
                Аномалий не обнаружено — скважина в норме.
              </p>
            )}

            {chart.data.length > 0 && (
              <div className="rounded-md border border-[#e5e5e5]">
                <Plot
                  data={chart.data}
                  layout={chart.layout}
                  config={{ displaylogo: false, responsive: true }}
                  style={{ width: "100%" }}
                  useResizeHandler
                />
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function Metric({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="flex flex-col">
      <span
        className={`text-base font-semibold leading-tight ${
          accent ? "text-[#c43232]" : "text-[#222226]"
        }`}
      >
        {value}
      </span>
      <span className="mt-0.5 text-xs text-muted-foreground">{label}</span>
    </div>
  );
}
