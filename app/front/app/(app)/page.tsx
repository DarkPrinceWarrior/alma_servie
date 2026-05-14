"use client";

import { Box, ChevronRight, Star } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { type AnomalyType, detections, reports, wells } from "@/lib/api";
import type { AnomalyReportAvailability, WellSummary } from "@/lib/api/types";
import { cn } from "@/lib/utils";

const ANOMALIES: AnomalyType[] = ["negermet", "pritok", "salt"];
const LABEL: Record<AnomalyType, string> = {
  negermet: "Негерметичность",
  pritok: "Приток",
  salt: "Солеотложение",
};
const ACCENT: Record<AnomalyType, string> = {
  negermet: "text-[#c43232]",
  pritok: "text-[#2f6fb5]",
  salt: "text-[#d2a232]",
};

export default function HomePage() {
  const [testWells, setTestWells] = useState<Record<AnomalyType, WellSummary[]>>(
    { negermet: [], pritok: [], salt: [] },
  );
  const [availability, setAvailability] = useState<
    Record<AnomalyType, AnomalyReportAvailability | null>
  >({ negermet: null, pritok: null, salt: null });
  const [running, setRunning] = useState<Record<AnomalyType, boolean>>({
    negermet: false,
    pritok: false,
    salt: false,
  });
  const [error, setError] = useState<string | null>(null);

  const loadAll = useCallback(async () => {
    const results = await Promise.all(
      ANOMALIES.map(async (a) => {
        const [w, av] = await Promise.all([
          wells.listWells(a).catch(() => [] as WellSummary[]),
          reports
            .getAvailability(a)
            .catch(() => null as AnomalyReportAvailability | null),
        ]);
        return [a, w.filter((x) => x.split === "test"), av] as const;
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
    setTestWells(wn);
    setAvailability(an);
  }, []);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  const totalTest = ANOMALIES.reduce(
    (acc, a) => acc + testWells[a].length,
    0,
  );

  async function onRunInference(a: AnomalyType) {
    setRunning((s) => ({ ...s, [a]: true }));
    setError(null);
    try {
      const detector = availability[a]?.best_detector ?? "paano_shared";
      await detections.launchDetection(a, detector as never);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка запуска инференса");
    } finally {
      setRunning((s) => ({ ...s, [a]: false }));
    }
  }

  return (
    <div className="min-h-screen bg-[#f9f9f9]">
      <div className="mx-auto flex max-w-[1440px] flex-col gap-6 px-10 pt-4 pb-10">
        <h1 className="font-display text-[23.04px] font-medium leading-[1.3] tracking-[-0.576px] text-[#222226]">
          Главная — пример на тестовых скважинах
        </h1>

        <section className="flex gap-2">
          <KpiCard
            icon={<Box className="h-6 w-6 text-[#424247]" />}
            label="Тестовых скважин"
            value={totalTest}
            iconBg="bg-[rgba(34,34,38,0.05)]"
            labelColor="text-[#222226]"
          />
          {ANOMALIES.map((a) => (
            <KpiCard
              key={a}
              icon={<Star className="h-6 w-6 fill-[#d2a232] text-[#d2a232]" />}
              label={LABEL[a]}
              value={testWells[a].length}
              iconBg="bg-[rgba(230,179,58,0.17)]"
              labelColor="text-[#d2a232]"
            />
          ))}
        </section>

        {error && <p className="text-sm text-[#c43232]">{error}</p>}

        {ANOMALIES.map((a) => (
          <AnomalySection
            key={a}
            anomaly={a}
            wells={testWells[a]}
            ready={availability[a]?.has_any_report ?? false}
            running={running[a]}
            onRun={() => onRunInference(a)}
          />
        ))}
      </div>
    </div>
  );
}

function AnomalySection({
  anomaly,
  wells: rows,
  ready,
  running,
  onRun,
}: {
  anomaly: AnomalyType;
  wells: WellSummary[];
  ready: boolean;
  running: boolean;
  onRun: () => void;
}) {
  return (
    <section className="flex flex-col gap-4 rounded-[16px] bg-white p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2
            className={cn(
              "font-display text-[23.04px] font-medium leading-[1.3] tracking-[-0.576px]",
              ACCENT[anomaly],
            )}
          >
            {LABEL[anomaly]}
          </h2>
          <span className="rounded-full bg-[rgba(34,34,38,0.05)] px-2 py-1 text-[11.11px] font-medium text-[#424247]">
            {rows.length} тестовых
          </span>
        </div>
        <button
          type="button"
          onClick={onRun}
          disabled={running}
          className={cn(
            "rounded-[12px] px-4 py-2 text-sm font-medium transition-colors",
            running
              ? "bg-[rgba(75,76,230,0.06)] text-[rgba(75,76,230,0.4)] cursor-not-allowed"
              : "bg-[#4b4ce6] text-white hover:bg-[#3f40d1]",
          )}
        >
          {running
            ? "Инференс запущен…"
            : ready
              ? "Перезапустить инференс"
              : "Запустить инференс"}
        </button>
      </div>

      {rows.length === 0 ? (
        <p className="text-sm text-[#797979]">Тестовых скважин нет.</p>
      ) : (
        <ul>
          {rows.map((r, idx) => (
            <li
              key={`${r.anomaly}:${r.well_id}`}
              className={cn(
                idx !== rows.length - 1 && "border-b border-[#e5e5e5]",
              )}
            >
              <Link
                href={`/wells/${encodeURIComponent(r.well_id)}?anomaly=${r.anomaly}`}
                className="group flex items-center justify-between px-4 py-2 transition-colors hover:bg-[#f9f9f9]"
              >
                <div className="flex items-center gap-3">
                  <p className="w-[200px] font-display text-[19.2px] font-medium leading-[1.35] tracking-[-0.192px] text-[#222226]">
                    {r.well_id}
                  </p>
                  <span className="flex items-center justify-center rounded-full bg-[rgba(149,45,45,0.1)] px-2 py-1 text-[11.11px] font-medium leading-[1.5] tracking-[-0.089px] text-[#c43232]">
                    {r.n_intervals}
                  </span>
                  <span className="text-base font-medium leading-[1.5] tracking-[-0.16px] text-[#c43232]">
                    {r.n_intervals === 1 ? "интервал" : "интервалов"}
                  </span>
                  {!ready && (
                    <span className="text-sm text-[#797979]">
                      отчёты не готовы — запустите инференс
                    </span>
                  )}
                </div>
                <div className="flex h-10 w-10 items-center justify-center rounded-[16px] bg-[rgba(34,34,38,0.05)] backdrop-blur-[21px] transition-colors group-hover:bg-[rgba(34,34,38,0.1)]">
                  <ChevronRight className="h-5 w-5 text-[#222226]" />
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </section>
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
