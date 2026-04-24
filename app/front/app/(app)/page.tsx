"use client";

import { Box, ChevronRight, Search, Star } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Input } from "@/components/ui/input";
import { type AnomalyType, detections, reports, wells } from "@/lib/api";
import type { AnomalyReportAvailability, WellSummary } from "@/lib/api/types";
import { cn } from "@/lib/utils";

const ANOMALIES: AnomalyType[] = ["negermet", "pritok", "salt"];
const LABEL: Record<AnomalyType, string> = {
  negermet: "Негермет",
  pritok: "Приток",
  salt: "Соли",
};

interface WellRow extends WellSummary {}

export default function HomePage() {
  const [wellsByAnomaly, setWellsByAnomaly] = useState<
    Record<AnomalyType, WellSummary[]>
  >({
    negermet: [],
    pritok: [],
    salt: [],
  });
  const [availability, setAvailability] = useState<
    Record<AnomalyType, AnomalyReportAvailability | null>
  >({ negermet: null, pritok: null, salt: null });
  const [search, setSearch] = useState("");
  const [training, setTraining] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);

  const loadAll = useCallback(async () => {
    const results = await Promise.all(
      ANOMALIES.map(async (a) => {
        const [w, av] = await Promise.all([
          wells.listWells(a).catch(() => [] as WellSummary[]),
          reports
            .getAvailability(a)
            .catch(() => null as AnomalyReportAvailability | null),
        ]);
        return [a, w, av] as const;
      }),
    );
    const wn: Record<AnomalyType, WellSummary[]> = {
      negermet: [],
      pritok: [],
      salt: [],
    };
    const an: Record<AnomalyType, AnomalyReportAvailability | null> = {
      negermet: null,
      pritok: null,
      salt: null,
    };
    for (const [a, w, av] of results) {
      wn[a] = w;
      an[a] = av;
    }
    setWellsByAnomaly(wn);
    setAvailability(an);
  }, []);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  const allReady = ANOMALIES.every((a) => availability[a]?.has_any_report);
  const totalWells = ANOMALIES.reduce(
    (acc, a) => acc + wellsByAnomaly[a].length,
    0,
  );

  const allRows = useMemo<WellRow[]>(() => {
    const rows = ANOMALIES.flatMap((a) => wellsByAnomaly[a]);
    const q = search.trim().toLowerCase();
    return q ? rows.filter((r) => r.well_id.toLowerCase().includes(q)) : rows;
  }, [wellsByAnomaly, search]);

  async function onTrain() {
    setTraining(true);
    setTrainError(null);
    try {
      const jobs = ANOMALIES.filter(
        (a) => !availability[a]?.has_any_report,
      ).map(async (a) => {
        const best = availability[a]?.best_detector ?? "paano_shared";
        return detections.launchDetection(a, best as never);
      });
      await Promise.all(jobs);
      await loadAll();
    } catch (err) {
      setTrainError(
        err instanceof Error ? err.message : "Ошибка запуска обучения",
      );
    } finally {
      setTraining(false);
    }
  }

  const midpoint = Math.ceil(allRows.length / 2);
  const leftCol = allRows.slice(0, midpoint);
  const rightCol = allRows.slice(midpoint);

  return (
    <div className="min-h-screen bg-[#f9f9f9]">
      <div className="mx-auto flex max-w-[1440px] flex-col gap-6 px-10 pt-4 pb-10">
        <h1 className="font-display text-[23.04px] font-medium leading-[1.3] tracking-[-0.576px] text-[#222226]">
          Главная
        </h1>

        <section className="flex gap-2">
          <KpiCard
            icon={<Box className="h-6 w-6 text-[#424247]" />}
            label="Количество скважин"
            value={totalWells}
            iconBg="bg-[rgba(34,34,38,0.05)]"
            labelColor="text-[#222226]"
          />
          {ANOMALIES.map((a) => (
            <KpiCard
              key={a}
              icon={<Star className="h-6 w-6 fill-[#d2a232] text-[#d2a232]" />}
              label={LABEL[a]}
              value={wellsByAnomaly[a].length}
              iconBg="bg-[rgba(230,179,58,0.17)]"
              labelColor="text-[#d2a232]"
            />
          ))}
        </section>

        <section className="flex items-center gap-6 rounded-[32px] border border-[#e5e5e5] bg-white p-4">
          <div className="flex-1 px-4 py-2">
            <p className="text-sm text-[#424247]">
              {allReady
                ? "Модель обучена — отчёты и графики доступны на страницах скважин."
                : "Данных ещё нет. Запустите обучение чтобы подготовить отчёты."}
            </p>
            {trainError && (
              <p className="mt-2 text-sm text-[#c43232]">{trainError}</p>
            )}
          </div>
          <div className="flex flex-col items-center gap-2">
            <button
              type="button"
              onClick={onTrain}
              disabled={training || allReady}
              className={cn(
                "rounded-[16px] px-6 py-3 text-base font-medium transition-colors",
                allReady || training
                  ? "bg-[rgba(75,76,230,0.06)] text-[rgba(75,76,230,0.22)] cursor-not-allowed"
                  : "bg-[#4b4ce6] text-white hover:bg-[#3f40d1]",
              )}
            >
              {training
                ? "Обучаем..."
                : allReady
                  ? "Модель обучена"
                  : "Обучить модель"}
            </button>
            {!allReady && !training && (
              <p className="text-sm text-[#c43232]">~30 мин на обучение</p>
            )}
          </div>
        </section>

        <section className="flex flex-col gap-8 rounded-[16px] bg-white p-4">
          <div className="flex flex-col gap-6">
            <h2 className="font-display text-[23.04px] font-medium leading-[1.3] tracking-[-0.576px] text-[#222226]">
              Все скважины
            </h2>
            <div className="relative">
              <Search className="absolute left-4 top-1/2 h-6 w-6 -translate-y-1/2 text-[#797979]" />
              <Input
                placeholder="Имя"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="h-12 rounded-[8px] border-0 bg-[#f3f3f3] pl-12 text-base placeholder:text-[rgba(34,34,38,0.22)] focus-visible:ring-1 focus-visible:ring-[#c1c1c1]"
              />
            </div>
          </div>

          <div className="relative grid grid-cols-2">
            <div className="absolute left-1/2 top-0 bottom-0 w-px bg-[#e5e5e5]" />
            <WellColumn rows={leftCol} />
            <WellColumn rows={rightCol} />
          </div>
          {allRows.length === 0 && (
            <p className="text-sm text-[#797979]">Скважины не найдены.</p>
          )}
        </section>
      </div>
    </div>
  );
}

function KpiCard({
  icon,
  label,
  value,
  iconBg,
  labelColor,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
  iconBg: string;
  labelColor: string;
}) {
  return (
    <div className="flex flex-1 flex-col gap-4 rounded-[32px] bg-[rgba(34,34,38,0.02)] p-4 backdrop-blur-[21px]">
      <div className="flex items-center gap-2">
        <div
          className={cn(
            "flex h-11 w-11 items-center justify-center rounded-full",
            iconBg,
          )}
        >
          {icon}
        </div>
        <p
          className={cn(
            "flex-1 truncate text-base font-semibold leading-[1.5]",
            labelColor,
          )}
        >
          {label}
        </p>
      </div>
      <div className="flex items-end gap-0.5 pl-2">
        <span className="font-display text-[33.18px] font-medium leading-[1.5] tracking-[-0.796px] text-[#222226]">
          {value}
        </span>
        <span className="py-2 text-[14.3px] font-medium leading-[1.5] tracking-[-0.214px] text-[rgba(34,34,38,0.22)]">
          шт
        </span>
      </div>
    </div>
  );
}

function WellColumn({ rows }: { rows: WellRow[] }) {
  return (
    <ul>
      {rows.map((r, idx) => (
        <li
          key={`${r.anomaly}:${r.well_id}`}
          className={cn(idx !== rows.length - 1 && "border-b border-[#e5e5e5]")}
        >
          <Link
            href={`/wells/${encodeURIComponent(r.well_id)}?anomaly=${r.anomaly}`}
            className="group flex items-center justify-between px-4 py-2 transition-colors hover:bg-[#f9f9f9]"
          >
            <div className="flex w-[445px] items-center gap-2">
              <p className="w-[200px] font-display text-[19.2px] font-medium leading-[1.35] tracking-[-0.192px] text-[#222226]">
                {r.well_id}
              </p>
              <div className="flex items-center gap-2">
                <span className="flex items-center justify-center rounded-full bg-[rgba(149,45,45,0.1)] px-2 py-1 text-[11.11px] font-medium leading-[1.5] tracking-[-0.089px] text-[#c43232]">
                  {r.n_intervals}
                </span>
                <span className="text-base font-medium leading-[1.5] tracking-[-0.16px] text-[#c43232]">
                  Аномалий
                </span>
              </div>
            </div>
            <div className="flex h-10 w-10 items-center justify-center rounded-[16px] bg-[rgba(34,34,38,0.05)] backdrop-blur-[21px] transition-colors group-hover:bg-[rgba(34,34,38,0.1)]">
              <ChevronRight className="h-5 w-5 text-[#222226]" />
            </div>
          </Link>
        </li>
      ))}
    </ul>
  );
}
