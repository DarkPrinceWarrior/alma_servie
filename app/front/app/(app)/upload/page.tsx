"use client";

import {
  AlertTriangle,
  ArrowLeft,
  CalendarRange,
  CheckCircle2,
  FileSpreadsheet,
  Loader2,
  Upload,
} from "lucide-react";
import dynamic from "next/dynamic";
import Link from "next/link";
import type { Data, Layout, Shape } from "plotly.js";
import {
  type ChangeEvent,
  type FormEvent,
  useMemo,
  useRef,
  useState,
} from "react";
import { Card, CardContent } from "@/components/ui/card";
import { type AnomalyType, uploads } from "@/lib/api";
import type { UploadAnomalyResult, UploadResultBundle } from "@/lib/api/types";
import { cn } from "@/lib/utils";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const LABEL: Record<AnomalyType, string> = {
  negermet: "Негерметичность",
  pritok: "Приток",
  salt: "Солеотложение",
};
const ACCENT: Record<AnomalyType, string> = {
  negermet: "#c43232",
  pritok: "#2f6fb5",
  salt: "#d2a232",
};

type Phase = "idle" | "uploading" | "running" | "done" | "error";

function fmtDt(s: string | null | undefined): string {
  if (!s) return "—";
  return s.replace("T", " ").slice(0, 16);
}

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [bundle, setBundle] = useState<UploadResultBundle | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function onFileChange(e: ChangeEvent<HTMLInputElement>) {
    setFile(e.target.files?.[0] ?? null);
  }

  function wellIdFromFile(f: File): string {
    const base = f.name.replace(/\.(xlsx|xls)$/i, "");
    const cleaned = base.replace(/[^\w\-.]+/g, "_").slice(0, 64);
    return cleaned || "well";
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setBundle(null);
    setElapsed(0);
    setPhase("uploading");

    const startedAt = Date.now();
    const ticker = setInterval(
      () => setElapsed(Math.floor((Date.now() - startedAt) / 1000)),
      1000,
    );

    try {
      const run = await uploads.createUpload(wellIdFromFile(file), file);
      setPhase("running");

      let current: UploadResultBundle | null = null;
      for (let i = 0; i < 200; i++) {
        await new Promise((r) => setTimeout(r, 3000));
        current = await uploads.getUploadResult(run.id);
        setBundle(current);
        if (current.status === "succeeded" || current.status === "failed") {
          break;
        }
      }

      if (!current) throw new Error("Не удалось получить результат прогона.");
      if (current.status === "failed" && current.n_done === 0) {
        throw new Error(
          "Прогон завершился с ошибкой — ни один класс аномалий не обработан.",
        );
      }
      setPhase("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка загрузки");
      setPhase("error");
    } finally {
      clearInterval(ticker);
    }
  }

  const busy = phase === "uploading" || phase === "running";
  const nDone = bundle?.n_done ?? 0;
  const nTotal = bundle?.n_total ?? 3;
  const progressPct =
    phase === "uploading" ? 6 : Math.round((nDone / nTotal) * 100);

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
        Формат файла — как у отдельной скважины в сырых данных. Класс аномалии
        указывать не нужно: система прогонит инференс сразу по трём классам —
        негерметичность, приток и солеотложение — и покажет обнаруженные старты.
      </p>

      <Card>
        <CardContent className="py-5">
          <form onSubmit={onSubmit} className="flex flex-col gap-4">
            <div className="flex flex-wrap items-center gap-3">
              <input
                ref={fileInputRef}
                id="file"
                type="file"
                accept=".xlsx,.xls"
                onChange={onFileChange}
                className="hidden"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={busy}
                className={cn(
                  "inline-flex items-center gap-2 rounded-[12px] border px-4 py-2 text-sm font-medium transition-colors",
                  busy
                    ? "cursor-not-allowed border-[#e5e5e5] bg-[#f3f3f3] text-[#aaa]"
                    : "border-[#4b4ce6] bg-white text-[#4b4ce6] hover:bg-[rgba(75,76,230,0.06)]",
                )}
              >
                <FileSpreadsheet className="h-4 w-4" />
                Выбрать Excel-файл
              </button>
              {file ? (
                <span className="inline-flex items-center gap-1.5 text-sm text-[#222226]">
                  <FileSpreadsheet className="h-4 w-4 text-[#16a34a]" />
                  {file.name}
                </span>
              ) : (
                <span className="text-sm text-[#aaa]">Файл не выбран</span>
              )}
            </div>

            {error && <p className="text-sm text-destructive">{error}</p>}

            <button
              type="submit"
              disabled={busy || !file}
              className={cn(
                "mt-1 inline-flex w-fit items-center gap-2 rounded-[12px] px-4 py-2 text-sm font-medium transition-colors",
                busy || !file
                  ? "cursor-not-allowed bg-[rgba(75,76,230,0.4)] text-white"
                  : "bg-[#4b4ce6] text-white hover:bg-[#3f40d1]",
              )}
            >
              {busy ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Upload className="h-4 w-4" />
              )}
              {phase === "uploading"
                ? "Загрузка файла…"
                : phase === "running"
                  ? "Идёт инференс…"
                  : "Загрузить и запустить детекцию"}
            </button>
          </form>
        </CardContent>
      </Card>

      {(busy || phase === "done") && (
        <ProgressPanel
          phase={phase}
          nDone={nDone}
          nTotal={nTotal}
          progressPct={progressPct}
          elapsed={elapsed}
          results={bundle?.results ?? []}
        />
      )}

      {bundle?.results
        .filter((r) => r.status !== "pending")
        .map((r) => (
          <AnomalyResultCard key={r.anomaly} result={r} />
        ))}
    </div>
  );
}

function ProgressPanel({
  phase,
  nDone,
  nTotal,
  progressPct,
  elapsed,
  results,
}: {
  phase: Phase;
  nDone: number;
  nTotal: number;
  progressPct: number;
  elapsed: number;
  results: UploadAnomalyResult[];
}) {
  const order: AnomalyType[] = ["negermet", "pritok", "salt"];
  const statusOf = (a: AnomalyType) =>
    results.find((r) => r.anomaly === a)?.status ?? "pending";

  return (
    <Card>
      <CardContent className="py-5 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-[#222226]">
            {phase === "uploading"
              ? "Загрузка файла на сервер…"
              : phase === "done"
                ? `Готово — обработано ${nDone} из ${nTotal} классов аномалий`
                : `Идёт инференс — обработано ${nDone} из ${nTotal} классов аномалий`}
          </span>
          <span className="text-sm tabular-nums text-[#797979]">
            {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}
          </span>
        </div>

        <div className="h-2 w-full overflow-hidden rounded-full bg-[#ececef]">
          <div
            className={cn(
              "h-full rounded-full bg-[#4b4ce6] transition-all duration-500",
              phase !== "done" && "animate-pulse",
            )}
            style={{ width: `${Math.max(progressPct, 4)}%` }}
          />
        </div>

        <div className="flex flex-wrap gap-2">
          {order.map((a) => {
            const st = statusOf(a);
            return (
              <span
                key={a}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium",
                  st === "succeeded" &&
                    "bg-[rgba(22,163,74,0.1)] text-[#16a34a]",
                  st === "failed" && "bg-[rgba(196,50,50,0.1)] text-[#c43232]",
                  st === "pending" && "bg-[rgba(34,34,38,0.05)] text-[#797979]",
                )}
              >
                {st === "succeeded" && <CheckCircle2 className="h-3.5 w-3.5" />}
                {st === "failed" && <AlertTriangle className="h-3.5 w-3.5" />}
                {st === "pending" && (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                )}
                {LABEL[a]}
              </span>
            );
          })}
        </div>

        {phase !== "done" && (
          <p className="text-xs text-[#aaa]">
            Инференс идёт на GPU последовательно по трём классам. Это может
            занять несколько минут — страницу можно не обновлять.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

const PRESSURE_CHANNEL = "Давление на приеме насоса кгс/см²";
const DEFAULT_TELEMETRY = new Set([
  PRESSURE_CHANNEL,
  "Коэффициент загрузки ПЭД",
  "Выходная частота",
]);

const CHANNEL_PALETTE = [
  "#2f6fb5",
  "#16a34a",
  "#d2772f",
  "#9333ea",
  "#0891b2",
  "#dc2626",
  "#65a30d",
  "#db2777",
  "#0f766e",
  "#b45309",
  "#4f46e5",
  "#be123c",
];

function channelColor(idx: number): string {
  return CHANNEL_PALETTE[idx % CHANNEL_PALETTE.length];
}

function AnomalyResultCard({ result }: { result: UploadAnomalyResult }) {
  const accent = ACCENT[result.anomaly];

  const [visible, setVisible] = useState<Set<string>>(
    () =>
      new Set(
        result.telemetry
          .map((ch) => ch.name)
          .filter((name) => DEFAULT_TELEMETRY.has(name)),
      ),
  );

  function toggleChannel(name: string) {
    setVisible((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  const chart = useMemo<{ data: Data[]; layout: Partial<Layout> }>(() => {
    if (result.score_series.length === 0) return { data: [], layout: {} };

    const scoreTrace: Data = {
      type: "scattergl",
      mode: "lines",
      name: "Отклонение от нормы",
      x: result.score_series.map((p) => p.t),
      y: result.score_series.map((p) => p.score),
      line: { color: accent, width: 1.6 },
      yaxis: "y",
    };

    const telemetryTraces: Data[] = result.telemetry.map((ch, idx) => ({
      type: "scattergl",
      mode: "lines",
      name: ch.name,
      x: ch.points.map((p) => p.t),
      y: ch.points.map((p) => p.v),
      line: { width: 1, color: channelColor(idx) },
      yaxis: "y2",
      visible: visible.has(ch.name),
    }));

    const shapes: Partial<Shape>[] = result.detected_starts.map((ts) => ({
      type: "line",
      xref: "x",
      yref: "paper",
      x0: ts,
      x1: ts,
      y0: 0,
      y1: 1,
      line: { color: "#a855f7", width: 2, dash: "dashdot" },
    }));

    return {
      data: [scoreTrace, ...telemetryTraces],
      layout: {
        height: 380,
        margin: { l: 60, r: 60, t: 16, b: 44 },
        xaxis: { title: { text: "Время" } },
        yaxis: {
          title: { text: "Отклонение от нормы" },
          side: "left",
          zeroline: true,
        },
        yaxis2: {
          title: { text: "Каналы телеметрии" },
          overlaying: "y",
          side: "right",
          showgrid: false,
        },
        shapes,
        showlegend: false,
        hovermode: "x unified",
        paper_bgcolor: "white",
        plot_bgcolor: "white",
      },
    };
  }, [result, accent, visible]);

  return (
    <Card>
      <CardContent className="py-5 space-y-4">
        <h2
          className="font-display text-[19.2px] font-medium tracking-[-0.192px]"
          style={{ color: accent }}
        >
          {LABEL[result.anomaly]}
        </h2>

        {result.status === "failed" ? (
          <p className="text-sm text-[#797979]">
            {result.error || "Недостаточно данных для этого класса аномалии."}
          </p>
        ) : (
          <>
            <div className="flex flex-wrap items-stretch gap-3">
              <div className="flex min-w-[180px] flex-col justify-center rounded-[12px] border border-[#e5e5e5] bg-[#fafafa] px-5 py-3.5">
                <span
                  className={cn(
                    "text-[32px] font-semibold leading-none",
                    (result.n_detected ?? 0) > 0
                      ? "text-[#c43232]"
                      : "text-[#16a34a]",
                  )}
                >
                  {result.n_detected ?? 0}
                </span>
                <span className="mt-1.5 text-xs text-muted-foreground">
                  Обнаружено стартов аномалии
                </span>
              </div>

              <div className="flex flex-1 items-center gap-3 rounded-[12px] border border-[#e5e5e5] bg-[#fafafa] px-5 py-3.5">
                <CalendarRange className="h-6 w-6 shrink-0 text-[#4b4ce6]" />
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground">
                    Период данных
                  </span>
                  <span className="text-sm font-medium tabular-nums text-[#222226]">
                    {fmtDt(result.time_start)}
                    <span className="mx-2 text-[#aaa]">→</span>
                    {fmtDt(result.time_end)}
                  </span>
                </div>
              </div>
            </div>

            {result.detected_starts.length > 0 ? (
              <div>
                <p className="text-sm font-medium text-[#222226]">
                  Обнаруженные старты аномалии:
                </p>
                <ul className="mt-2 flex flex-wrap gap-2">
                  {result.detected_starts.map((ts) => (
                    <li
                      key={ts}
                      className="rounded-full bg-[rgba(149,45,45,0.1)] px-3 py-1 text-sm font-medium tabular-nums text-[#c43232]"
                    >
                      {fmtDt(ts)}
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <p className="text-sm text-[#16a34a]">
                Аномалий этого класса не обнаружено — скважина в норме.
              </p>
            )}

            {chart.data.length > 0 && (
              <div className="space-y-2 rounded-md border border-[#e5e5e5] p-2">
                <div className="flex flex-wrap items-center gap-4 px-1 text-xs text-muted-foreground">
                  <span className="inline-flex items-center gap-1.5">
                    <span
                      className="inline-block h-0.5 w-4"
                      style={{ background: accent }}
                    />
                    Отклонение от нормы (score)
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span
                      className="inline-block h-0.5 w-4"
                      style={{ borderTop: "2px dashed #a855f7" }}
                    />
                    Обнаруженный старт аномалии
                  </span>
                </div>
                <Plot
                  data={chart.data}
                  layout={chart.layout}
                  config={{ displaylogo: false, responsive: true }}
                  style={{ width: "100%" }}
                  useResizeHandler
                />
                {result.telemetry.length > 0 && (
                  <div className="border-t border-[#eee] px-1 pt-2">
                    <p className="mb-2 text-xs font-medium text-[#797979]">
                      Каналы телеметрии — нажмите, чтобы показать на графике
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {result.telemetry.map((ch, idx) => {
                        const on = visible.has(ch.name);
                        const color = channelColor(idx);
                        return (
                          <button
                            key={ch.name}
                            type="button"
                            onClick={() => toggleChannel(ch.name)}
                            className={cn(
                              "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors",
                              on
                                ? "bg-white font-medium"
                                : "border-[#e5e5e5] bg-white text-[#9a9a9a] hover:border-[#cfcfcf]",
                            )}
                            style={
                              on ? { color, borderColor: color } : undefined
                            }
                          >
                            <span
                              className="h-2 w-2 rounded-full"
                              style={{ background: on ? color : "#cfcfcf" }}
                            />
                            {ch.name}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
