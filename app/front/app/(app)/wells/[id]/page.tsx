"use client";

import { ArrowLeft, HelpCircle } from "lucide-react";
import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { type AnomalyType, reports, wells } from "@/lib/api";
import { featureImportanceHtmlUrl, reportHtmlUrl } from "@/lib/api/reports";
import type { AnomalyReportAvailability, WellDetail } from "@/lib/api/types";
import { cn } from "@/lib/utils";

const ANOMALY_LABEL: Record<AnomalyType, string> = {
  negermet: "Негермет",
  pritok: "Приток",
  salt: "Соли",
};

const ANOMALY_ACCENT: Record<AnomalyType, string> = {
  negermet: "text-amber-500",
  pritok: "text-sky-500",
  salt: "text-rose-500",
};

type Tab = "report" | "feature_importance";

function fmtDt(s: string | null | undefined): string {
  if (!s) return "—";
  const d = new Date(s);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function hoursBetween(a: string, b: string): string {
  const ms = new Date(b).getTime() - new Date(a).getTime();
  if (!Number.isFinite(ms)) return "—";
  return `${(ms / 3_600_000).toFixed(2)}ч`;
}

export default function WellPage() {
  const params = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const wellId = decodeURIComponent(params.id);
  const anomaly = (searchParams.get("anomaly") ?? "negermet") as AnomalyType;

  const [well, setWell] = useState<WellDetail | null>(null);
  const [availability, setAvailability] =
    useState<AnomalyReportAvailability | null>(null);
  const [tab, setTab] = useState<Tab>("report");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setError(null);
    Promise.all([
      wells.getWell(anomaly, wellId),
      reports.getAvailability(anomaly),
    ])
      .then(([w, a]) => {
        if (!active) return;
        setWell(w);
        setAvailability(a);
      })
      .catch((err) => {
        if (!active) return;
        setError(
          err instanceof Error ? err.message : "Не удалось загрузить данные",
        );
      });
    return () => {
      active = false;
    };
  }, [anomaly, wellId]);

  const detector = availability?.best_detector ?? null;
  const detectorRow = availability?.detectors.find(
    (d) => d.detector === detector,
  );
  const hasReport = !!detectorRow?.has_report;
  const hasFI = !!detectorRow?.has_feature_importance;

  const activeTab: Tab =
    tab === "feature_importance" && hasFI ? "feature_importance" : "report";
  const iframeSrc =
    detector && hasReport
      ? activeTab === "feature_importance"
        ? featureImportanceHtmlUrl(anomaly, detector)
        : reportHtmlUrl(anomaly, detector)
      : null;

  const firstInterval = well?.intervals[0];
  const actualStart = firstInterval?.start_date ?? null;
  const actualEnd = firstInterval?.end_date ?? null;
  const detectedAt = actualStart;
  const delay =
    actualStart && detectedAt ? hoursBetween(actualStart, detectedAt) : "—";

  return (
    <div className="mx-auto max-w-[1440px] px-10 py-6 space-y-5">
      <div className="flex items-center gap-3">
        <Link
          href="/"
          className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          Главная
        </Link>
        <span className="text-muted-foreground">/</span>
        <span className="text-sm font-medium">{wellId}</span>
      </div>

      <Card>
        <CardContent className="flex items-start justify-between gap-8 py-5">
          <div className="flex items-center gap-4">
            <span className="text-2xl font-semibold">Аномалии:</span>
            <AnomalyChip anomaly={anomaly} />
          </div>

          <div className="flex items-center gap-10">
            <Metric label="Фактическое начало" value={fmtDt(actualStart)} />
            <Metric label="Фактическое окончание" value={fmtDt(actualEnd)} />
            <Metric label="Время обнаружения" value={fmtDt(detectedAt)} />
            <Metric label="Задержка" value={delay} />
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TabButton
            active={activeTab === "report"}
            onClick={() => setTab("report")}
            disabled={!hasReport}
          >
            Основной отчёт
          </TabButton>
          <TabButton
            active={activeTab === "feature_importance"}
            onClick={() => setTab("feature_importance")}
            disabled={!hasFI}
          >
            Feature importance
          </TabButton>
        </div>
        <div className="text-xs text-muted-foreground">
          Детектор: <span className="text-foreground">{detector ?? "—"}</span>
        </div>
      </div>

      {error && <p className="text-sm text-destructive">Ошибка: {error}</p>}

      {iframeSrc ? (
        <iframe
          key={iframeSrc}
          src={iframeSrc}
          title={`${anomaly} report`}
          className="h-[80vh] w-full rounded-md border bg-white"
        />
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center gap-3 py-16 text-center">
            <p className="text-muted-foreground">
              Отчёт для аномалии «{ANOMALY_LABEL[anomaly]}» ещё не сгенерирован.
            </p>
            <Link href="/">
              <Button variant="outline">Вернуться на главную</Button>
            </Link>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function AnomalyChip({ anomaly }: { anomaly: AnomalyType }) {
  return (
    <span
      className={cn("inline-flex items-center gap-1", ANOMALY_ACCENT[anomaly])}
    >
      <span className="text-2xl font-semibold">{ANOMALY_LABEL[anomaly]}</span>
      <HelpCircle className="h-4 w-4 opacity-60" />
    </span>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-sm font-semibold leading-none">{value}</span>
      <span className="mt-1 text-xs text-muted-foreground">{label}</span>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  disabled,
  children,
}: {
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "rounded-md border px-4 py-2 text-sm transition-colors",
        active
          ? "bg-primary text-primary-foreground border-primary"
          : "bg-background text-foreground hover:bg-accent",
        disabled && "opacity-40 cursor-not-allowed hover:bg-background",
      )}
    >
      {children}
    </button>
  );
}
