"use client";

import {
  ArrowLeft,
  CalendarCheck2,
  CalendarX2,
  Clock,
  Timer,
} from "lucide-react";
import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { FeatureImportanceChart } from "@/components/reports/FeatureImportanceChart";
import { WellReportChart } from "@/components/reports/WellReportChart";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { type AnomalyType, reports, wells } from "@/lib/api";
import type {
  AnomalyReportAvailability,
  FeatureImportanceResponse,
  WellDetail,
  WellSeriesResponse,
} from "@/lib/api/types";
import { cn } from "@/lib/utils";

const ANOMALY_LABEL: Record<AnomalyType, string> = {
  negermet: "Негермет",
  pritok: "Приток",
};

const ANOMALY_ACCENT: Record<AnomalyType, string> = {
  negermet: "text-amber-500",
  pritok: "text-sky-500",
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
  const [series, setSeries] = useState<WellSeriesResponse | null>(null);
  const [fi, setFi] = useState<FeatureImportanceResponse | null>(null);
  const [seriesLoading, setSeriesLoading] = useState(false);
  const [fiLoading, setFiLoading] = useState(false);

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

  useEffect(() => {
    if (!detector || !hasReport) {
      setSeries(null);
      return;
    }
    let active = true;
    setSeriesLoading(true);
    reports
      .getWellSeries(anomaly, detector, wellId, 2000)
      .then((s) => {
        if (active) setSeries(s);
      })
      .catch((err) => {
        if (active)
          setError(
            err instanceof Error ? err.message : "Ошибка загрузки графика",
          );
      })
      .finally(() => {
        if (active) setSeriesLoading(false);
      });
    return () => {
      active = false;
    };
  }, [anomaly, detector, wellId, hasReport]);

  useEffect(() => {
    if (!detector || !hasFI) return;
    if (fi?.well_id === wellId && fi.detector === detector) return;
    let active = true;
    setFiLoading(true);
    reports
      .getFeatureImportance(anomaly, detector, wellId)
      .then((r) => {
        if (active) setFi(r);
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : "Ошибка FI");
      })
      .finally(() => {
        if (active) setFiLoading(false);
      });
    return () => {
      active = false;
    };
  }, [anomaly, detector, wellId, hasFI, fi]);

  const activeTab: Tab =
    tab === "feature_importance" && hasFI ? "feature_importance" : "report";

  const firstResult = series?.results[0] ?? null;
  const firstInterval = well?.intervals[0];
  // Слепая скв. без аномалии: вырожденный интервал (начало == конец) — фактического
  // старта нет, показываем «—» вместо служебной даты.
  const degenerateInterval =
    !!firstInterval && firstInterval.start_date === firstInterval.end_date;
  const actualStart =
    firstResult?.actual_start ??
    (degenerateInterval ? null : (firstInterval?.start_date ?? null));
  const actualEnd =
    firstResult?.actual_end ??
    (degenerateInterval ? null : (firstInterval?.end_date ?? null));
  const detectedAt = firstResult?.detected_time ?? null;
  const delay =
    firstResult?.delay_hours !== null && firstResult?.delay_hours !== undefined
      ? `${firstResult.delay_hours.toFixed(2)}ч`
      : actualStart && detectedAt
        ? hoursBetween(actualStart, detectedAt)
        : "—";

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
        <CardContent className="flex flex-col gap-5 py-5">
          <div className="flex items-center gap-4">
            <span className="text-2xl font-semibold">Аномалии:</span>
            <AnomalyChip anomaly={anomaly} />
          </div>

          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <MetricCard
              icon={<CalendarCheck2 className="h-5 w-5" />}
              label="Фактическое начало"
              value={fmtDt(actualStart)}
              color="#16a34a"
            />
            <MetricCard
              icon={<CalendarX2 className="h-5 w-5" />}
              label="Фактическое окончание"
              value={fmtDt(actualEnd)}
              color="#dc2626"
            />
            <MetricCard
              icon={<Clock className="h-5 w-5" />}
              label="Время обнаружения"
              value={fmtDt(detectedAt)}
              color="#a855f7"
            />
            <MetricCard
              icon={<Timer className="h-5 w-5" />}
              label="Задержка"
              value={delay}
              color="#4b4ce6"
            />
          </div>
        </CardContent>
      </Card>

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
          Важность признаков
        </TabButton>
      </div>

      {error && <p className="text-sm text-destructive">Ошибка: {error}</p>}

      {activeTab === "report" ? (
        <>
          {seriesLoading && !series && (
            <LoadingCard text="Загружаем временные ряды…" />
          )}
          {series && <WellReportChart series={series} fi={fi} />}
          {!seriesLoading && !series && !hasReport && (
            <NoReportCard anomaly={anomaly} />
          )}
        </>
      ) : (
        <>
          {fiLoading && !fi && (
            <LoadingCard text="Загружаем feature importance…" />
          )}
          {fi && <FeatureImportanceChart fi={fi} series={series} />}
        </>
      )}
    </div>
  );
}

function AnomalyChip({ anomaly }: { anomaly: AnomalyType }) {
  return (
    <span className={cn("text-2xl font-semibold", ANOMALY_ACCENT[anomaly])}>
      {ANOMALY_LABEL[anomaly]}
    </span>
  );
}

function MetricCard({
  icon,
  label,
  value,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-[12px] border border-[#e5e5e5] bg-[#fafafa] px-4 py-3">
      <span
        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full"
        style={{ background: `${color}1a`, color }}
      >
        {icon}
      </span>
      <div className="flex flex-col gap-0.5 min-w-0">
        <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
          {label}
        </span>
        <span className="truncate text-sm font-semibold tabular-nums text-[#222226]">
          {value}
        </span>
      </div>
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

function LoadingCard({ text }: { text: string }) {
  return (
    <Card>
      <CardContent className="py-10 text-center text-sm text-muted-foreground">
        {text}
      </CardContent>
    </Card>
  );
}

function NoReportCard({ anomaly }: { anomaly: AnomalyType }) {
  return (
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
  );
}
