"use client";

import { ChevronRight } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { type AnomalyType, reports, wells } from "@/lib/api";
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

const KPI_COLOR: Record<AnomalyType | "total", string> = {
  total: "#424247",
  negermet: "#c43232",
  pritok: "#2f6fb5",
  salt: "#d2a232",
};

export default function HomePage() {
  const [testWells, setTestWells] = useState<
    Record<AnomalyType, WellSummary[]>
  >({ negermet: [], pritok: [], salt: [] });
  const [availability, setAvailability] = useState<
    Record<AnomalyType, AnomalyReportAvailability | null>
  >({ negermet: null, pritok: null, salt: null });

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

  const totalTest = ANOMALIES.reduce((acc, a) => acc + testWells[a].length, 0);

  return (
    <div className="min-h-screen bg-[#f9f9f9]">
      <div className="mx-auto flex max-w-[1440px] flex-col gap-6 px-10 pt-4 pb-10">
        <h1 className="font-display text-[23.04px] font-medium leading-[1.3] tracking-[-0.576px] text-[#222226]">
          Главная — пример на тестовых скважинах
        </h1>

        <section className="flex gap-2">
          <KpiCard
            label="Тестовых скважин"
            value={totalTest}
            color={KPI_COLOR.total}
          />
          {ANOMALIES.map((a) => (
            <KpiCard
              key={a}
              label={LABEL[a]}
              value={testWells[a].length}
              color={KPI_COLOR[a]}
            />
          ))}
        </section>

        {ANOMALIES.map((a) => (
          <AnomalySection
            key={a}
            anomaly={a}
            wells={testWells[a]}
            ready={availability[a]?.has_any_report ?? false}
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
}: {
  anomaly: AnomalyType;
  wells: WellSummary[];
  ready: boolean;
}) {
  return (
    <section className="flex flex-col gap-4 rounded-[16px] bg-white p-4">
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
                      отчёты ещё не готовы
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
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="flex flex-1 items-center justify-between gap-3 rounded-[20px] border border-[#e5e5e5] bg-white px-4 py-3">
      <p
        className="truncate text-sm font-semibold leading-[1.4]"
        style={{ color }}
      >
        {label}
      </p>
      <span
        className="flex h-12 min-w-[48px] items-center justify-center rounded-full px-3 font-display text-[22px] font-medium tabular-nums"
        style={{ background: `${color}1a`, color }}
      >
        {value}
      </span>
    </div>
  );
}
