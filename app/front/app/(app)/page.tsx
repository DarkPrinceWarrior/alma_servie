"use client";

import {
  AlertTriangle,
  Droplets,
  FlaskConical,
  Hammer,
  Search,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { type AnomalyType, detections, reports, wells } from "@/lib/api";
import type { AnomalyReportAvailability, WellSummary } from "@/lib/api/types";
import { cn } from "@/lib/utils";

type AnomalyConf = {
  anomaly: AnomalyType;
  label: string;
  icon: typeof AlertTriangle;
  accent: string;
};

const ANOMALY_ORDER: AnomalyConf[] = [
  {
    anomaly: "negermet",
    label: "Негермет",
    icon: AlertTriangle,
    accent: "text-amber-500",
  },
  {
    anomaly: "pritok",
    label: "Приток",
    icon: Droplets,
    accent: "text-sky-500",
  },
  {
    anomaly: "salt",
    label: "Соли",
    icon: FlaskConical,
    accent: "text-rose-500",
  },
];

const ANOMALY_BY_CODE: Record<AnomalyType, AnomalyConf> = Object.fromEntries(
  ANOMALY_ORDER.map((c) => [c.anomaly, c]),
) as Record<AnomalyType, AnomalyConf>;

interface WellRow extends WellSummary {
  anomalyLabel: string;
}

export default function HomePage() {
  const [wellsByAnomaly, setWellsByAnomaly] = useState<
    Record<AnomalyType, WellSummary[] | null>
  >({ negermet: null, pritok: null, salt: null });
  const [availability, setAvailability] = useState<
    Record<AnomalyType, AnomalyReportAvailability | null>
  >({ negermet: null, pritok: null, salt: null });
  const [search, setSearch] = useState("");
  const [training, setTraining] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);

  const loadAll = useCallback(async () => {
    const results = await Promise.all(
      ANOMALY_ORDER.map(async ({ anomaly }) => {
        const [w, a] = await Promise.all([
          wells.listWells(anomaly).catch(() => [] as WellSummary[]),
          reports
            .getAvailability(anomaly)
            .catch(() => null as AnomalyReportAvailability | null),
        ]);
        return [anomaly, w, a] as const;
      }),
    );
    const wellsNext: Record<AnomalyType, WellSummary[] | null> = {
      negermet: null,
      pritok: null,
      salt: null,
    };
    const availNext: Record<AnomalyType, AnomalyReportAvailability | null> = {
      negermet: null,
      pritok: null,
      salt: null,
    };
    for (const [anomaly, w, a] of results) {
      wellsNext[anomaly] = w;
      availNext[anomaly] = a;
    }
    setWellsByAnomaly(wellsNext);
    setAvailability(availNext);
  }, []);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  const allReady = ANOMALY_ORDER.every(
    ({ anomaly }) => availability[anomaly]?.has_any_report,
  );

  async function onTrain() {
    setTraining(true);
    setTrainError(null);
    try {
      const jobs = ANOMALY_ORDER.filter(
        ({ anomaly }) => !availability[anomaly]?.has_any_report,
      ).map(async ({ anomaly }) => {
        const best = availability[anomaly]?.best_detector ?? "paano_shared";
        return detections.launchDetection(anomaly, best as never);
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

  const totalWells = useMemo(
    () =>
      ANOMALY_ORDER.reduce(
        (acc, { anomaly }) => acc + (wellsByAnomaly[anomaly]?.length ?? 0),
        0,
      ),
    [wellsByAnomaly],
  );

  const allRows = useMemo<WellRow[]>(() => {
    const rows: WellRow[] = [];
    for (const { anomaly, label } of ANOMALY_ORDER) {
      const list = wellsByAnomaly[anomaly] ?? [];
      for (const w of list) rows.push({ ...w, anomalyLabel: label });
    }
    const q = search.trim().toLowerCase();
    return q
      ? rows.filter(
          (r) =>
            r.well_id.toLowerCase().includes(q) ||
            r.anomalyLabel.toLowerCase().includes(q),
        )
      : rows;
  }, [wellsByAnomaly, search]);

  const midpoint = Math.ceil(allRows.length / 2);
  const leftCol = allRows.slice(0, midpoint);
  const rightCol = allRows.slice(midpoint);

  return (
    <div className="mx-auto max-w-[1440px] px-10 py-6 space-y-6">
      <h1 className="text-[30px] font-semibold leading-none">Главная</h1>

      <section className="grid grid-cols-4 gap-2">
        <KpiCard
          icon={Hammer}
          label="Количество скважин"
          value={totalWells}
          hint="шт"
          accent="text-zinc-600"
        />
        {ANOMALY_ORDER.map(({ anomaly, label, icon, accent }) => (
          <KpiCard
            key={anomaly}
            icon={icon}
            label={label}
            value={wellsByAnomaly[anomaly]?.length ?? 0}
            hint="шт"
            accent={accent}
          />
        ))}
      </section>

      <Card>
        <CardContent className="flex items-center justify-between gap-6 py-8">
          <div className="flex-1">
            <p className="text-sm text-muted-foreground">
              {allReady
                ? "Модель обучена по всем классам аномалий — отчёты и графики доступны на страницах скважин."
                : "Модель ещё не обучена по всем классам. Запустите обучение чтобы подготовить отчёты."}
            </p>
            {trainError && (
              <p className="mt-2 text-sm text-destructive">{trainError}</p>
            )}
          </div>
          <Button
            size="lg"
            disabled={training || allReady}
            onClick={onTrain}
            className="min-w-[200px]"
          >
            {training
              ? "Обучаем..."
              : allReady
                ? "Модель обучена"
                : "Обучить модель"}
          </Button>
        </CardContent>
      </Card>

      <section className="space-y-4">
        <h2 className="text-2xl font-semibold">Все скважины</h2>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Поиск скважины"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9 h-12"
          />
        </div>

        <div className="grid grid-cols-2 divide-x border rounded-md">
          <WellColumn rows={leftCol} />
          <WellColumn rows={rightCol} />
        </div>
        {allRows.length === 0 && (
          <p className="text-sm text-muted-foreground">Скважины не найдены.</p>
        )}
      </section>
    </div>
  );
}

function KpiCard({
  icon: Icon,
  label,
  value,
  hint,
  accent,
}: {
  icon: typeof AlertTriangle;
  label: string;
  value: number;
  hint: string;
  accent: string;
}) {
  return (
    <Card>
      <CardContent className="flex flex-col gap-3 py-6">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Icon className={cn("h-4 w-4", accent)} />
          {label}
        </div>
        <div className="flex items-end gap-1">
          <span className="text-[46px] leading-none font-semibold">
            {value}
          </span>
          <span className="pb-2 text-sm text-muted-foreground">{hint}</span>
        </div>
      </CardContent>
    </Card>
  );
}

function WellColumn({ rows }: { rows: WellRow[] }) {
  return (
    <ul className="divide-y">
      {rows.map((r) => {
        const conf = ANOMALY_BY_CODE[r.anomaly as AnomalyType];
        return (
          <li key={`${r.anomaly}:${r.well_id}`}>
            <Link
              href={`/wells/${encodeURIComponent(r.well_id)}?anomaly=${r.anomaly}`}
              className="flex items-center justify-between px-4 py-4 hover:bg-accent"
            >
              <div className="flex items-center gap-10">
                <span className="font-medium">{r.well_id}</span>
                <span className={cn("text-sm", conf.accent)}>{conf.label}</span>
              </div>
              <span className="text-muted-foreground">›</span>
            </Link>
          </li>
        );
      })}
    </ul>
  );
}
