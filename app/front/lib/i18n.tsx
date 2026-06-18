"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { cn } from "@/lib/utils";

export type Locale = "ru" | "en";

type Entry = { ru: string; en: string };

// Полный словарь интерфейса. Ключи сгруппированы по областям.
const DICT: Record<string, Entry> = {
  // app shell / nav
  "app.loading": { ru: "Загрузка...", en: "Loading..." },
  "app.redirecting": { ru: "Перенаправление...", en: "Redirecting..." },
  "nav.brand": { ru: "Alma · Аномалии", en: "Alma · Anomalies" },
  "nav.upload": { ru: "Загрузить скважину", en: "Upload a well" },
  "nav.logout": { ru: "Выйти", en: "Sign out" },

  // login
  "login.failed": { ru: "Не удалось войти", en: "Sign-in failed" },
  "login.title": { ru: "Вход в Alma", en: "Sign in to Alma" },
  "login.subtitle": {
    ru: "Система детекции аномалий",
    en: "Anomaly detection system",
  },
  "login.email": { ru: "Email", en: "Email" },
  "login.password": { ru: "Пароль", en: "Password" },
  "login.submit": { ru: "Войти", en: "Sign in" },
  "login.submitting": { ru: "Вход...", en: "Signing in..." },

  // anomaly labels
  "anomaly.negermet": { ru: "Негерметичность", en: "Tubing leak" },
  "anomaly.negermet.short": { ru: "Негермет", en: "Tubing leak" },
  "anomaly.pritok": { ru: "Приток", en: "Inflow" },

  // dashboard
  "home.title": {
    ru: "Главная — детекция аномалий (последняя модель: PaAno Global)",
    en: "Home — anomaly detection (latest model: PaAno Global)",
  },
  "home.test.title": { ru: "Тест", en: "Test" },
  "home.test.subtitle": {
    ru: "Слепые скважины — модель видит их впервые",
    en: "Blind wells — the model sees them for the first time",
  },
  "home.test.total": { ru: "Тестовых скважин", en: "Test wells" },
  "home.test.word": { ru: "тестовых", en: "test" },
  "home.train.title": { ru: "Обучение", en: "Training" },
  "home.train.subtitle": {
    ru: "Размеченные скважины, использованные при обучении и калибровке",
    en: "Labeled wells used for training and calibration",
  },
  "home.train.total": { ru: "Обучающих скважин", en: "Training wells" },
  "home.train.word": { ru: "обучающих", en: "training" },
  "home.noWells": { ru: "Скважин нет.", en: "No wells." },
  "home.reportsNotReady": {
    ru: "отчёты ещё не готовы",
    en: "reports not ready yet",
  },

  // breadcrumbs / common
  "breadcrumb.home": { ru: "Главная", en: "Home" },
  "common.errorPrefix": { ru: "Ошибка: ", en: "Error: " },

  // well detail
  "well.anomalies": { ru: "Аномалии:", en: "Anomalies:" },
  "well.actualStart": { ru: "Фактическое начало", en: "Actual onset" },
  "well.actualEnd": { ru: "Фактическое окончание", en: "Actual end" },
  "well.detectedTime": { ru: "Время обнаружения", en: "Detection time" },
  "well.delay": { ru: "Задержка", en: "Delay" },
  "well.tab.report": { ru: "Основной отчёт", en: "Main report" },
  "well.tab.fi": { ru: "Важность признаков", en: "Feature importance" },
  "well.err.load": { ru: "Не удалось загрузить данные", en: "Failed to load data" },
  "well.err.chart": {
    ru: "Ошибка загрузки графика",
    en: "Chart loading error",
  },
  "well.err.fi": { ru: "Ошибка FI", en: "FI error" },
  "well.loadingSeries": {
    ru: "Загружаем временные ряды…",
    en: "Loading time series…",
  },
  "well.loadingFi": {
    ru: "Загружаем feature importance…",
    en: "Loading feature importance…",
  },
  "well.noReport": {
    ru: "Отчёт для аномалии «{label}» ещё не сгенерирован.",
    en: "The report for the «{label}» anomaly has not been generated yet.",
  },
  "well.backHome": { ru: "Вернуться на главную", en: "Back to home" },

  // charts (shared)
  "chart.time": { ru: "Время", en: "Time" },
  "chart.scoreDeviation": { ru: "Отклонение от нормы", en: "Deviation from normal" },
  "chart.telemetryChannels": { ru: "Каналы телеметрии", en: "Telemetry channels" },
  "chart.legend.score": {
    ru: "Отклонение от нормы (score)",
    en: "Deviation from normal (score)",
  },
  "chart.legend.zone": { ru: "Зона аномалии", en: "Anomaly zone" },
  "chart.legend.actualStart": { ru: "Фактическое начало", en: "Actual onset" },
  "chart.legend.actualEnd": { ru: "Фактическое окончание", en: "Actual end" },
  "chart.legend.detected": { ru: "Время обнаружения", en: "Detection time" },
  "chart.channelsCount": { ru: "{n} каналов", en: "{n} channels" },
  "chart.statusPrefix": { ru: "Статус: ", en: "Status: " },
  "status.detected": { ru: "Обнаружено", en: "Detected" },
  "status.notFound": { ru: "Не обнаружено", en: "Not detected" },
  "status.missed": { ru: "Пропуск", en: "Missed" },

  // feature importance chart
  "fi.importancePct": { ru: "Важность (%)", en: "Importance (%)" },
  "fi.deviation": { ru: "Отклонение", en: "Deviation" },
  "fi.channels": { ru: "Каналы", en: "Channels" },
  "fi.notFound": {
    ru: "Важность признаков для скважины «{id}» не найдена.",
    en: "Feature importance for well «{id}» not found.",
  },
  "fi.panelTitle": {
    ru: "Важность каналов (важность по перестановке, %)",
    en: "Channel importance (permutation importance, %)",
  },
  "fi.topChannels": {
    ru: "Top-каналы во времени ({p} полезных · {n} «шумных»)",
    en: "Top channels over time ({p} useful · {n} noisy)",
  },

  // upload page
  "upload.title": { ru: "Загрузка скважины", en: "Well upload" },
  "upload.cardTitle": {
    ru: "Детекция по загруженному Excel",
    en: "Detection from an uploaded Excel file",
  },
  "upload.instructions": {
    ru: "Формат файла — как у отдельной скважины в сырых данных. Класс аномалии указывать не нужно: система прогонит инференс сразу по двум классам — негерметичность и приток — и покажет обнаруженные старты.",
    en: "The file format is the same as a single well in the raw data. You don't need to specify the anomaly class: the system runs inference over both classes — tubing leak and inflow — and shows the detected onsets.",
  },
  "upload.chooseFile": { ru: "Выбрать Excel-файл", en: "Choose Excel file" },
  "upload.noFile": { ru: "Файл не выбран", en: "No file selected" },
  "upload.btn.uploading": { ru: "Загрузка файла…", en: "Uploading file…" },
  "upload.btn.running": { ru: "Идёт инференс…", en: "Running inference…" },
  "upload.btn.run": {
    ru: "Загрузить и запустить детекцию",
    en: "Upload and run detection",
  },
  "upload.history.empty": {
    ru: "Загрузок пока нет. Загрузите Excel выше — результаты сохранятся здесь и будут доступны после перезагрузки страницы.",
    en: "No uploads yet. Upload an Excel file above — results are saved here and remain available after a page reload.",
  },
  "upload.history.title": { ru: "История загрузок", en: "Upload history" },
  "upload.history.selected": { ru: "выбрано {n}", en: "{n} selected" },
  "upload.history.deleteSelected": {
    ru: "Удалить выбранные",
    en: "Delete selected",
  },
  "upload.history.clearAll": { ru: "Очистить всё", en: "Clear all" },
  "upload.history.noAnomalies": {
    ru: "аномалий не обнаружено",
    en: "no anomalies detected",
  },
  "upload.history.opening": { ru: "Открываю…", en: "Opening…" },
  "upload.history.opened": { ru: "Открыто", en: "Opened" },
  "upload.history.open": { ru: "Открыть", en: "Open" },
  "upload.history.fromHistory": {
    ru: "Открыто из истории · статусы:",
    en: "Opened from history · statuses:",
  },
  "upload.status.done": { ru: "Готово", en: "Done" },
  "upload.status.failed": { ru: "Сбой", en: "Failed" },
  "upload.status.running": { ru: "В работе", en: "In progress" },
  "upload.progress.uploading": {
    ru: "Загрузка файла на сервер…",
    en: "Uploading file to the server…",
  },
  "upload.progress.done": {
    ru: "Готово — обработано {nDone} из {nTotal} классов аномалий",
    en: "Done — {nDone} of {nTotal} anomaly classes processed",
  },
  "upload.progress.running": {
    ru: "Идёт инференс — обработано {nDone} из {nTotal} классов аномалий",
    en: "Inference running — {nDone} of {nTotal} anomaly classes processed",
  },
  "upload.progress.note": {
    ru: "Инференс идёт на GPU последовательно по двум классам. Это может занять несколько минут — страницу можно не обновлять.",
    en: "Inference runs on the GPU sequentially over two classes. This may take a few minutes — you don't need to refresh the page.",
  },
  "upload.confirm.deleteSelected": {
    ru: "Удалить выбранные загрузки ({n})? Действие необратимо.",
    en: "Delete the selected uploads ({n})? This action cannot be undone.",
  },
  "upload.confirm.clearAll": {
    ru: "Очистить всю историю ({n} загрузок)? Действие необратимо.",
    en: "Clear the entire history ({n} uploads)? This action cannot be undone.",
  },
  "upload.err.loadHistory": {
    ru: "Не удалось загрузить историю",
    en: "Failed to load history",
  },
  "upload.err.delete": { ru: "Не удалось удалить", en: "Failed to delete" },
  "upload.err.clear": { ru: "Не удалось очистить", en: "Failed to clear" },
  "upload.err.open": {
    ru: "Не удалось открыть результат",
    en: "Failed to open the result",
  },
  "upload.err.getResult": {
    ru: "Не удалось получить результат прогона.",
    en: "Failed to fetch the run result.",
  },
  "upload.err.runFailed": {
    ru: "Прогон завершился с ошибкой — ни один класс аномалий не обработан.",
    en: "The run failed — no anomaly class was processed.",
  },
  "upload.err.chartLoad": { ru: "Ошибка загрузки", en: "Loading error" },

  // upload result card
  "result.notEnoughData": {
    ru: "Недостаточно данных для этого класса аномалии.",
    en: "Not enough data for this anomaly class.",
  },
  "result.period": { ru: "Период:", en: "Period:" },
  "result.detectedStarts": {
    ru: "Обнаруженные старты аномалии",
    en: "Detected anomaly onsets",
  },
  "result.noAnomaly": {
    ru: "Аномалий этого класса не обнаружено — скважина в норме.",
    en: "No anomalies of this class detected — the well is normal.",
  },
  "result.detectedMarker": {
    ru: "Обнаруженный старт аномалии",
    en: "Detected anomaly onset",
  },
  "result.paramsHint": {
    ru: "Параметры — нажмите, чтобы показать/скрыть на графике",
    en: "Parameters — click to show/hide on the chart",
  },
};

// Названия каналов телеметрии (данные из parquet, RU) -> EN.
const CHANNELS_EN: Record<string, string> = {
  "Cos Ф": "Power factor (cos φ)",
  "Активная выходная мощность": "Active output power",
  "Вибрация Y": "Vibration Y",
  "Вибрация Z": "Vibration Z",
  "Вибрация Х": "Vibration X",
  "Вибрация ХY": "Vibration XY",
  "Вибрация ХYZ": "Vibration XYZ",
  "Выходная частота": "Output frequency",
  "Выходной ток ПЧ": "VFD output current",
  "Давление на приеме насоса кгс/см²": "Pump intake pressure, kgf/cm²",
  "Дисбаланс напряжений": "Voltage imbalance",
  "Дисбаланс токов": "Current imbalance",
  "Коэффициент загрузки ПЭД": "Motor load factor",
  "Линейное напряжение по фазе АB": "Line voltage AB",
  "Линейное напряжение по фазе ВC": "Line voltage BC",
  "Линейное напряжение по фазе СA": "Line voltage CA",
  "Напряжение в звене постоянного тока ПЧ": "VFD DC-link voltage",
  "Полная выходная мощность": "Apparent output power",
  "Температура масла двигателя": "Motor oil temperature",
  "Температура на приёме насоса": "Pump intake temperature",
  "Ток на фазе А": "Phase A current",
  "Ток на фазе В": "Phase B current",
  "Ток на фазе С": "Phase C current",
  "Фазное напряжение Ua": "Phase voltage Ua",
  "Фазное напряжение Ub": "Phase voltage Ub",
  "Фазное напряжение Uc": "Phase voltage Uc",
};

// плюрализация (RU 3 формы / EN 2 формы) — возвращает слово/фразу ПОСЛЕ числа
const PLURALS = {
  interval: {
    ru: ["интервал", "интервала", "интервалов"],
    en: ["interval", "intervals"],
  },
  labeledInterval: {
    ru: ["размеченный интервал", "размеченных интервала", "размеченных интервалов"],
    en: ["labeled interval", "labeled intervals"],
  },
  start: {
    ru: ["обнаруженный старт", "обнаруженных старта", "обнаруженных стартов"],
    en: ["detected onset", "detected onsets"],
  },
} as const;

function ruPluralIdx(n: number): 0 | 1 | 2 {
  const m10 = n % 10;
  const m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return 0;
  if (m10 >= 2 && m10 <= 4 && (m100 < 10 || m100 >= 20)) return 1;
  return 2;
}

interface I18nCtx {
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: (key: string, params?: Record<string, string | number>) => string;
  ch: (name: string) => string;
  plural: (n: number, kind: keyof typeof PLURALS) => string;
}

const I18nContext = createContext<I18nCtx | null>(null);

export function LocaleProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("ru");

  useEffect(() => {
    const saved = window.localStorage.getItem("alma_locale");
    if (saved === "ru" || saved === "en") setLocaleState(saved);
  }, []);

  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  const setLocale = useCallback((l: Locale) => {
    setLocaleState(l);
    window.localStorage.setItem("alma_locale", l);
  }, []);

  const value = useMemo<I18nCtx>(
    () => ({
      locale,
      setLocale,
      t: (key, params) => {
        const entry = DICT[key];
        let s = entry ? entry[locale] : key;
        if (params) {
          for (const k of Object.keys(params)) {
            s = s.replaceAll(`{${k}}`, String(params[k]));
          }
        }
        return s;
      },
      ch: (name) => (locale === "en" ? (CHANNELS_EN[name] ?? name) : name),
      plural: (n, kind) => {
        const forms = PLURALS[kind][locale];
        const idx = locale === "ru" ? ruPluralIdx(n) : n === 1 ? 0 : 1;
        return forms[idx] ?? "";
      },
    }),
    [locale, setLocale],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nCtx {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used within LocaleProvider");
  return ctx;
}

export function LangToggle({ className }: { className?: string }) {
  const { locale, setLocale } = useI18n();
  return (
    <div
      className={cn(
        "inline-flex overflow-hidden rounded-[8px] border border-[#e5e5e5] bg-white text-xs",
        className,
      )}
    >
      {(["ru", "en"] as Locale[]).map((l) => (
        <button
          key={l}
          type="button"
          onClick={() => setLocale(l)}
          className={cn(
            "px-2.5 py-1.5 font-semibold uppercase transition-colors",
            locale === l
              ? "bg-[#4b4ce6] text-white"
              : "text-[#797979] hover:bg-[#f3f3f3]",
          )}
        >
          {l}
        </button>
      ))}
    </div>
  );
}
