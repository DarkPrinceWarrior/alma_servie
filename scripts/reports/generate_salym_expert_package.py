from __future__ import annotations

import argparse
import html
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from alma_service.paths import SALYM_PREPARED_DIR
from alma_service.salym_raw_pipeline import SELECTED_PARAM_MAP


PLOTLY_BUNDLE_PATH = Path(go.__file__).resolve().parent.parent / "package_data" / "plotly.min.js"
PLOTLY_CONFIG: dict[str, Any] = {
    "displaylogo": False,
    "locale": "ru",
    "responsive": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toImage"],
}
MAX_STARTS_ON_CHART = 50


ANOMALY_ORDER = ["negermet", "pritok", "salt"]
ANOMALY_LABELS = {
    "negermet": "Негерметичность",
    "pritok": "Приток",
    "salt": "Солеотложение",
}
ANOMALY_SHORT_LABELS = {
    "negermet": "Негермет",
    "pritok": "Приток",
    "salt": "Соли",
}
STATUS_LABELS = {
    "Detected": "Есть детекция",
    "Not found": "Детекция не найдена",
    "Not detected": "Детекция не найдена",
    "Skipped": "Пропущено",
}
STATUS_CLASS = {
    "Detected": "ok",
    "Not found": "empty",
    "Not detected": "empty",
    "Skipped": "skip",
}
PARAM_UNITS = {
    "ESP.IntakePressure": "кгс/см²",
    "ESP.Frequency": "Гц",
    "ESP.Motor.Load": "%",
    "ESP.Motor.Temperature": "°C",
    "ESP.IntakeTemperature": "°C",
    "ESP.Motor.CurrentU": "А",
    "ESP.Motor.CurrentV": "А",
    "ESP.Motor.CurrentW": "А",
    "ESP.Motor.VoltageAB": "В",
    "ESP.Motor.VoltageBC": "В",
    "ESP.Motor.VoltageCA": "В",
    "ESP.Motor.CurrentUnbalance": "%",
    "ESP.Motor.VibrationX": "мм/с",
    "ESP.Motor.VibrationY": "мм/с",
    "ESP.Motor.VibrationZ": "мм/с",
}
CHART_CANDIDATES = [
    "ESP.IntakePressure",
    "ESP.Frequency",
    "ESP.Motor.Load",
    "ESP.Motor.CurrentU",
    "ESP.Motor.VoltageAB",
    "ESP.Motor.CurrentUnbalance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Russian Salym expert review package.")
    parser.add_argument(
        "--input-dir",
        default="artifacts/results/salym_screening_20260507_4h",
        help="Directory with per-anomaly Salym screening outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output package directory. Default: <input-dir>/expert_package_ru.",
    )
    parser.add_argument(
        "--salym-root",
        default=str(SALYM_PREPARED_DIR),
        help="Prepared Salym directory with raw_wells for per-well plots.",
    )
    return parser.parse_args()


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def read_anomaly_results(input_dir: Path) -> pd.DataFrame:
    frames = []
    for anomaly in ANOMALY_ORDER:
        parquet_path = input_dir / anomaly / ("salym_" + anomaly + "_screening_results.parquet")
        csv_path = input_dir / anomaly / ("salym_" + anomaly + "_screening_results.csv")
        if parquet_path.exists():
            frame = read_frame(parquet_path)
        elif csv_path.exists():
            frame = read_frame(csv_path)
        else:
            raise FileNotFoundError("Missing screening results for " + anomaly)
        frame = frame.copy()
        frame["anomaly"] = anomaly
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    long["well_id"] = long["well_id"].astype(str)
    return long.sort_values(["well_id", "anomaly"]).reset_index(drop=True)


def read_predicted_starts(input_dir: Path) -> pd.DataFrame:
    frames = []
    for anomaly in ANOMALY_ORDER:
        parquet_path = input_dir / anomaly / ("salym_" + anomaly + "_predicted_starts.parquet")
        csv_path = input_dir / anomaly / ("salym_" + anomaly + "_predicted_starts.csv")
        if parquet_path.exists():
            frame = read_frame(parquet_path)
        elif csv_path.exists():
            frame = read_frame(csv_path)
        else:
            continue
        frame = frame.copy()
        frame["anomaly"] = anomaly
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["anomaly", "well_id", "detected_time", "score_at_start"])
    starts = pd.concat(frames, ignore_index=True)
    starts["well_id"] = starts["well_id"].astype(str)
    starts["detected_time"] = pd.to_datetime(starts["detected_time"], errors="coerce")
    return starts.sort_values(["well_id", "anomaly", "detected_time"]).reset_index(drop=True)


def read_errors(input_dir: Path) -> pd.DataFrame:
    frames = []
    for anomaly in ANOMALY_ORDER:
        path = input_dir / anomaly / ("salym_" + anomaly + "_screening_errors.csv")
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame.copy()
        frame["anomaly"] = anomaly
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def status_ru(value: Any) -> str:
    return STATUS_LABELS.get(str(value), str(value))


def bool_ru(value: Any) -> str:
    if bool(value):
        return "Да"
    return "Нет"


def dt_ru(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d %H:%M")


def dt_value(value: Any) -> Any:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return ts.to_pydatetime()


def float_value(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(number):
        return np.nan
    return number


def score_ru(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    if abs(number) >= 10000 or (abs(number) > 0 and abs(number) < 0.001):
        return "{:.3e}".format(number)
    return "{:.4f}".format(number)


def percent_ru(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return "{:.1f}%".format(number * 100.0)


def build_wide_summary(long: pd.DataFrame) -> pd.DataFrame:
    wells = sorted(long["well_id"].unique())
    wide = pd.DataFrame({"Скважина": wells}).set_index("Скважина")

    for anomaly in ANOMALY_ORDER:
        label = ANOMALY_LABELS[anomaly]
        sub = long[long["anomaly"] == anomaly].set_index("well_id")
        wide[label + ": статус"] = sub["status"].map(status_ru)
        wide[label + ": есть детекция"] = sub["has_detection"].map(bool_ru)
        wide[label + ": первая детекция"] = sub["first_detected_time"].map(dt_value)
        wide[label + ": последняя детекция"] = sub["last_detected_time"].map(dt_value)
        wide[label + ": число стартов"] = sub["n_detected_starts"].fillna(0).astype(int)
        wide[label + ": максимальная оценка"] = sub["max_score"].map(float_value)
        wide[label + ": медианная оценка"] = sub["median_score"].map(float_value)
        wide[label + ": точек ряда"] = sub["n_points"].fillna(0).astype(int)
        wide[label + ": сырых каналов"] = sub["n_raw_channels"].fillna(0).astype(int)
        wide[label + ": точек в базовом участке"] = sub["reference_points"].fillna(0).astype(int)
        wide[label + ": доля исключенных данных"] = sub["masked_fraction"].map(float_value)

    detection_cols = [ANOMALY_LABELS[item] + ": есть детекция" for item in ANOMALY_ORDER]
    start_cols = [ANOMALY_LABELS[item] + ": число стартов" for item in ANOMALY_ORDER]
    status_cols = [ANOMALY_LABELS[item] + ": статус" for item in ANOMALY_ORDER]

    wide["Классов с детекцией"] = wide[detection_cols].eq("Да").sum(axis=1)
    wide["Классы с детекцией"] = wide.apply(detected_classes_text, axis=1)
    wide["Всего стартов детекции"] = wide[start_cols].sum(axis=1).astype(int)
    wide["Есть пропуск/ошибка"] = wide[status_cols].eq("Пропущено").any(axis=1).map(bool_ru)

    priority = [
        "Классов с детекцией",
        "Классы с детекцией",
        "Всего стартов детекции",
        "Есть пропуск/ошибка",
    ]
    columns = priority + [column for column in wide.columns if column not in priority]
    wide = wide[columns].reset_index()
    return wide.sort_values(
        ["Классов с детекцией", "Всего стартов детекции", "Скважина"],
        ascending=[False, False, True],
    )


def detected_classes_text(row: pd.Series) -> str:
    labels = []
    for anomaly in ANOMALY_ORDER:
        label = ANOMALY_LABELS[anomaly]
        if row.get(label + ": есть детекция") == "Да":
            labels.append(label)
    return ", ".join(labels)


def build_long_ru(long: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Скважина"] = long["well_id"]
    out["Класс аномалии"] = long["anomaly"].map(ANOMALY_LABELS)
    out["Статус"] = long["status"].map(status_ru)
    out["Есть детекция"] = long["has_detection"].map(bool_ru)
    out["Первая детекция"] = long["first_detected_time"].map(dt_value)
    out["Последняя детекция"] = long["last_detected_time"].map(dt_value)
    out["Число стартов"] = long["n_detected_starts"].fillna(0).astype(int)
    out["Максимальная оценка"] = long["max_score"].map(float_value)
    out["Медианная оценка"] = long["median_score"].map(float_value)
    out["Точек ряда"] = long["n_points"].fillna(0).astype(int)
    out["Сырых каналов"] = long["n_raw_channels"].fillna(0).astype(int)
    out["Расчетных показателей"] = long["n_features"].fillna(0).astype(int)
    out["Точек в базовом участке"] = long["reference_points"].fillna(0).astype(int)
    out["Доля исключенных данных"] = long["masked_fraction"].map(float_value)
    out["Начало ряда"] = long["input_start"].map(dt_value)
    out["Конец ряда"] = long["input_end"].map(dt_value)
    return out


def build_starts_ru(starts: pd.DataFrame) -> pd.DataFrame:
    if starts.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["Скважина"] = starts["well_id"]
    out["Класс аномалии"] = starts["anomaly"].map(ANOMALY_LABELS)
    out["Номер старта"] = starts["start_idx"].fillna(0).astype(int)
    out["Время детекции"] = starts["detected_time"].map(dt_value)
    out["Оценка в момент старта"] = starts["score_at_start"].map(float_value)
    return out


def build_errors_ru(errors: pd.DataFrame) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame(columns=["Скважина", "Класс аномалии", "Причина пропуска"])
    out = pd.DataFrame()
    out["Скважина"] = errors.get("well_id", "")
    out["Класс аномалии"] = errors.get("anomaly", "").map(ANOMALY_LABELS)
    reason_col = None
    for candidate in ["error", "message", "reason"]:
        if candidate in errors.columns:
            reason_col = candidate
            break
    out["Причина пропуска"] = errors[reason_col].map(translate_error) if reason_col else ""
    return out


def translate_error(value: Any) -> str:
    text = str(value)
    translations = {
        "Cannot build full 15-channel Salym frame": (
            "Не удалось собрать полный ряд по 15 обязательным параметрам. "
            "Для этой скважины не хватает одного или нескольких параметров либо после очистки качества данных "
            "по ним осталось недостаточно точек."
        ),
        "Not enough usable data after engineered preprocessing": (
            "После очистки и подготовки осталось недостаточно пригодных данных для надежной проверки."
        ),
    }
    return translations.get(text, text)


def description_rows() -> pd.DataFrame:
    rows = [
        ("Назначение файла", "Файл предназначен для первичной экспертной проверки кандидатов аномалий по 547 скважинам Salym."),
        ("Статус данных", "Скважины не размечены. Результат является подсказкой для проверки, а не подтвержденным фактом осложнения."),
        ("Что означает детекция", "Алгоритм нашел участок временного ряда, похожий на один из трех классов: негерметичность, приток или солеотложение."),
        ("Что означает отсутствие детекции", "На выбранном временном ряду алгоритм не нашел устойчивого признака соответствующего класса аномалии."),
        ("Что означает число стартов", "Сколько отдельных начал подозрительных участков найдено за весь анализируемый период. Большое число требует просмотра временной шкалы и первичных данных."),
        ("Что означает максимальная оценка", "Наибольшая сила сигнала по классу аномалии внутри ряда. Значения нужны для ранжирования внутри одного класса, а не для прямого сравнения разных классов между собой."),
        ("Что означает медианная оценка", "Типичный уровень сигнала по ряду. Полезна для понимания, был ли ряд почти всегда спокойным или часто находился в подозрительном состоянии."),
        ("Что означает базовый участок", "Начальная часть ряда, которую алгоритм использовал как условно нормальное поведение данной скважины."),
        ("Что означает доля исключенных данных", "Доля точек, которые алгоритм не использовал как надежную базу из-за пропусков, неустойчивого режима или подозрительных участков."),
        ("Лист «Сводка»", "Одна строка на скважину. Рядом показаны статусы, первые детекции и количество стартов по трем классам аномалий."),
        ("Лист «По классам»", "Одна строка на пару скважина-класс аномалии. Удобен, если нужно фильтровать отдельно негерметичность, приток или солеотложение."),
        ("Лист «Старты»", "Все найденные начала подозрительных участков. Используйте его, если по скважине нужно проверить не только первую детекцию, но и последующие повторные события."),
        ("Лист «Пропуски»", "Скважины и классы, где обработка была пропущена или завершилась ошибкой."),
        ("HTML-отчет", "Откройте index.html. Там есть общий список скважин, поиск, переходы на страницы скважин и временные шкалы детекций."),
    ]
    return pd.DataFrame(rows, columns=["Раздел", "Пояснение"])


def format_excel(writer: pd.ExcelWriter, sheets: dict[str, pd.DataFrame]) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    header_fill = PatternFill("solid", fgColor="F6EFD8")
    header_font = Font(bold=True)

    for sheet_name, frame in sheets.items():
        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for col_idx, column in enumerate(frame.columns):
            excel_col = col_idx + 1
            letter = get_column_letter(excel_col)
            column_text = str(column)
            lower = column_text.lower()
            header = worksheet.cell(row=1, column=excel_col)
            header.value = column_text
            header.fill = header_fill
            header.font = header_font
            header.alignment = Alignment(wrap_text=True, vertical="top")
            width = min(max(len(column_text) + 4, 14), 44)
            number_format = None
            if "детекция" in lower or "начало" in lower or "конец" in lower or "время" in lower:
                number_format = "yyyy-mm-dd hh:mm"
                width = max(width, 20)
            elif "доля" in lower:
                number_format = "0.0%"
                width = max(width, 18)
            elif (
                "число" in lower
                or "точек" in lower
                or "каналов" in lower
                or "показателей" in lower
                or "классов" in lower
                or "номер" in lower
            ):
                number_format = "0"
                width = max(width, 14)
            elif "оценка" in lower:
                number_format = "0.0000"
                width = max(width, 18)
            worksheet.column_dimensions[letter].width = width
            if number_format:
                for row_idx in range(2, len(frame) + 2):
                    worksheet.cell(row=row_idx, column=excel_col).number_format = number_format


def write_tables(output_dir: Path, wide_ru: pd.DataFrame, long_ru: pd.DataFrame, starts_ru: pd.DataFrame, errors_ru: pd.DataFrame) -> None:
    sheets = {
        "Описание": description_rows(),
        "Сводка": wide_ru,
        "По классам": long_ru,
        "Старты": starts_ru,
        "Пропуски": errors_ru,
    }
    with pd.ExcelWriter(output_dir / "salym_эксперт_547_скважин.xlsx", engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
        format_excel(writer, sheets)


def safe_name(value: str, used: set[str]) -> str:
    base = re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", value.strip())
    base = base.strip("_") or "well"
    candidate = base
    index = 2
    while candidate in used:
        candidate = base + "_" + str(index)
        index += 1
    used.add(candidate)
    return candidate


def html_escape(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def format_int(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "0"


def css_text() -> str:
    return """
:root {
  --bg: #f5f2ea;
  --paper: #fffdf7;
  --ink: #18212f;
  --muted: #667085;
  --line: #ddd6c7;
  --negermet: #b42318;
  --pritok: #c26a00;
  --salt: #087f74;
  --ok-bg: #e9f7ef;
  --ok-fg: #17663a;
  --empty-bg: #eef2f6;
  --empty-fg: #475467;
  --skip-bg: #fff2cc;
  --skip-fg: #875a00;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: radial-gradient(circle at top left, #fff8df 0, #f5f2ea 360px, #ece7dc 100%);
  color: var(--ink);
  font-family: Georgia, "Times New Roman", serif;
}
a { color: #0b5c7c; text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 1440px; margin: 0 auto; padding: 32px; }
.hero {
  background: linear-gradient(135deg, #172033, #27435f);
  color: #fff;
  border-radius: 24px;
  padding: 32px;
  box-shadow: 0 20px 50px rgba(24, 33, 47, 0.18);
}
.hero h1 { margin: 0 0 12px; font-size: 34px; line-height: 1.1; }
.hero p { margin: 8px 0 0; max-width: 980px; color: #dce7f3; font-size: 17px; }
.section { margin-top: 28px; }
.grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }
.card {
  background: rgba(255, 253, 247, 0.96);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 18px;
  box-shadow: 0 12px 30px rgba(24, 33, 47, 0.08);
}
.metric-title { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-value { font-size: 32px; font-weight: 700; margin-top: 8px; }
.note { border-left: 5px solid #0b5c7c; background: #eef7fb; padding: 16px 18px; border-radius: 14px; }
.tools { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.search {
  width: min(460px, 100%);
  border: 1px solid var(--line);
  border-radius: 14px;
  background: #fff;
  padding: 12px 14px;
  font-size: 15px;
}
.table-box {
  overflow-x: auto;
  background: rgba(255, 253, 247, 0.96);
  border: 1px solid var(--line);
  border-radius: 20px;
  box-shadow: 0 12px 30px rgba(24, 33, 47, 0.08);
}
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 11px 12px; border-bottom: 1px solid #ece6d9; text-align: left; vertical-align: top; }
th { position: sticky; top: 0; background: #fbf7ec; z-index: 1; color: #344054; }
tr:hover td { background: #fffaf0; }
.badge { display: inline-flex; align-items: center; border-radius: 999px; padding: 5px 9px; font-weight: 700; font-size: 12px; white-space: nowrap; }
.badge.ok { color: var(--ok-fg); background: var(--ok-bg); }
.badge.empty { color: var(--empty-fg); background: var(--empty-bg); }
.badge.skip { color: var(--skip-fg); background: var(--skip-bg); }
.pill { display: inline-block; border-radius: 999px; padding: 5px 9px; background: #eef2f6; color: #344054; margin: 2px 4px 2px 0; font-size: 12px; }
.lane-title { font-weight: 700; margin-bottom: 8px; }
.timeline {
  width: 100%;
  min-height: 290px;
  border: 1px solid var(--line);
  border-radius: 18px;
  background: #fffdf7;
  padding: 8px;
  overflow: hidden;
}
.timeline .plotly-graph-div { width: 100% !important; }
.axis-labels { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; margin-top: 8px; }
.chart-stack { display: flex; flex-direction: column; gap: 22px; }
.chart-card {
  background: #fffdf7;
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 18px;
  overflow: hidden;
  box-shadow: 0 8px 22px rgba(24, 33, 47, 0.06);
}
.chart-card .plotly-graph-div { width: 100% !important; }
.chart-note {
  margin: 10px 4px 0;
  color: var(--muted);
  font-size: 13px;
  font-style: italic;
}
.chart-empty {
  min-height: 200px;
  color: var(--muted);
  padding: 28px 24px;
  border: 1px dashed var(--line);
  border-radius: 14px;
  background: #fbf6e8;
}
.chart-empty strong {
  display: block;
  color: var(--ink);
  font-size: 20px;
  margin-bottom: 10px;
}
.well-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
.anomaly-card h2 { margin: 0 0 10px; font-size: 22px; }
.kv { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; margin-top: 12px; }
.kv div:nth-child(odd) { color: var(--muted); }
.small { color: var(--muted); font-size: 13px; }
.footer { color: var(--muted); margin-top: 30px; font-size: 13px; }
.negermet { color: var(--negermet); }
.pritok { color: var(--pritok); }
.salt { color: var(--salt); }
@media (max-width: 980px) {
  .wrap { padding: 18px; }
  .grid, .well-grid { grid-template-columns: 1fr; }
  .hero h1 { font-size: 27px; }
}
"""


def anomaly_color(anomaly: str) -> str:
    if anomaly == "negermet":
        return "#b42318"
    if anomaly == "pritok":
        return "#c26a00"
    return "#087f74"


def status_badge(status: Any) -> str:
    text = status_ru(status)
    klass = STATUS_CLASS.get(str(status), "empty")
    return '<span class="badge ' + klass + '">' + html_escape(text) + "</span>"


def load_salym_series(salym_root: Path, well_id: str, param_key: str) -> pd.Series | None:
    output_name = SELECTED_PARAM_MAP.get(param_key, param_key)
    path = salym_root / "raw_wells" / well_id / (param_key + ".parquet")
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=["timestamp", "value", "quality_flag"])
    frame = frame[frame["quality_flag"].eq("ok")]
    frame = frame.dropna(subset=["timestamp", "value"])
    if frame.empty:
        return None
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "value"])
    if frame.empty:
        return None
    series = (
        frame[["timestamp", "value"]]
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")["value"]
        .sort_index()
    )
    series.name = output_name
    return series


def downsample_series(series: pd.Series, limit: int = 520) -> pd.Series:
    if len(series) <= limit:
        return series
    step = max(int(np.ceil(len(series) / limit)), 1)
    return series.iloc[::step]


def series_variability(series: pd.Series | None) -> float:
    if series is None or series.empty:
        return 0.0
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return 0.0
    if values.nunique(dropna=True) <= 1:
        return 0.0
    q1 = float(values.quantile(0.01))
    q99 = float(values.quantile(0.99))
    return max(q99 - q1, float(values.max() - values.min()), 0.0)


def format_axis_value(value: float) -> str:
    if not np.isfinite(value):
        return ""
    abs_value = abs(value)
    if abs_value >= 1000:
        return "{:.0f}".format(value)
    if abs_value >= 100:
        return "{:.1f}".format(value)
    if abs_value >= 10:
        return "{:.2f}".format(value)
    return "{:.3f}".format(value)


def build_time_ticks(start: pd.Timestamp, end: pd.Timestamp, count: int = 6) -> list[pd.Timestamp]:
    if start == end:
        return [start]
    ticks = pd.date_range(start, end, periods=count)
    return [pd.Timestamp(item) for item in ticks]


def _layout_common() -> dict[str, Any]:
    return dict(
        plot_bgcolor="#fffdf7",
        paper_bgcolor="#fffdf7",
        font=dict(family="Georgia, 'Times New Roman', serif", size=13, color="#18212f"),
        margin=dict(l=72, r=24, t=64, b=72),
        hoverlabel=dict(bgcolor="#fffdf7", bordercolor="#1f4e79", font=dict(size=13, color="#18212f")),
    )


def _empty_chart_block(title: str, message: str) -> str:
    return (
        '<div class="chart-empty"><strong>'
        + html_escape(title)
        + "</strong>"
        + html_escape(message)
        + "</div>"
    )


def parameter_chart_plotly(
    *,
    div_id: str,
    title: str,
    unit: str,
    series: pd.Series | None,
    well_starts: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> str:
    if series is None or series.empty:
        return _empty_chart_block(title, "Нет пригодного ряда для этого параметра.")

    full_values = pd.to_numeric(series, errors="coerce").dropna()
    if full_values.empty:
        return _empty_chart_block(title, "Нет числовых значений для этого параметра.")

    if series_variability(series) <= 0:
        constant_value = format_axis_value(float(full_values.iloc[0]))
        suffix = (" " + unit) if unit else ""
        return _empty_chart_block(
            title,
            "Параметр постоянный: " + constant_value + suffix + ". Для визуальной проверки изменения режима он неинформативен.",
        )

    plot_series = downsample_series(series.dropna(), limit=2200)

    if start is None or pd.isna(start):
        start = pd.Timestamp(plot_series.index.min())
    if end is None or pd.isna(end):
        end = pd.Timestamp(plot_series.index.max())
    if start == end:
        end = start + pd.Timedelta(hours=1)

    unit_text = unit or "без единиц"
    fig = go.Figure()
    hover_template = (
        "<b>" + title + "</b><br>"
        + "Дата: %{x|%Y-%m-%d %H:%M}<br>"
        + "Значение: %{y:.4f} " + unit_text
        + "<extra></extra>"
    )
    fig.add_trace(
        go.Scattergl(
            x=plot_series.index,
            y=plot_series.values.astype(float),
            mode="lines",
            name="Параметр",
            line=dict(color="#1f4e79", width=1.7),
            hovertemplate=hover_template,
            connectgaps=False,
        )
    )

    truncation_notes: list[str] = []
    for anomaly in ANOMALY_ORDER:
        sub = well_starts[well_starts["anomaly"] == anomaly]
        if sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color=anomaly_color(anomaly), width=2.0, dash="dash"),
                    name="Старт: " + ANOMALY_LABELS[anomaly] + " (нет стартов)",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
            continue
        sub_sorted = sub.sort_values("detected_time")
        total = len(sub_sorted)
        sub_display = sub_sorted.head(MAX_STARTS_ON_CHART)
        color = anomaly_color(anomaly)
        for _, item in sub_display.iterrows():
            ts = pd.to_datetime(item.get("detected_time"), errors="coerce")
            if pd.isna(ts):
                continue
            fig.add_shape(
                type="line",
                x0=ts,
                x1=ts,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color=color, width=1.4, dash="dash"),
                opacity=0.55,
                layer="above",
            )
        legend_label = "Старт: " + ANOMALY_LABELS[anomaly]
        if total > MAX_STARTS_ON_CHART:
            legend_label = legend_label + " (показаны " + str(MAX_STARTS_ON_CHART) + " из " + str(total) + ")"
            truncation_notes.append(
                ANOMALY_LABELS[anomaly] + ": показаны первые "
                + str(MAX_STARTS_ON_CHART) + " стартов из " + str(total)
            )
        else:
            legend_label = legend_label + " (" + str(total) + ")"
        marker_x = list(pd.to_datetime(sub_display["detected_time"], errors="coerce"))
        fig.add_trace(
            go.Scatter(
                x=marker_x,
                y=[float(full_values.min())] * len(marker_x),
                mode="markers",
                marker=dict(color=color, symbol="line-ns-open", size=14, line=dict(color=color, width=2)),
                name=legend_label,
                hovertemplate=(
                    "<b>" + ANOMALY_LABELS[anomaly] + "</b><br>"
                    + "Старт: %{x|%Y-%m-%d %H:%M}<extra></extra>"
                ),
                showlegend=True,
            )
        )

    layout = _layout_common()
    layout.update(
        title=dict(
            text="<b>" + html_escape(title) + "</b>",
            x=0.0,
            xanchor="left",
            font=dict(size=18, color="#18212f"),
        ),
        xaxis=dict(
            title="Дата/время",
            range=[start, end],
            showgrid=True,
            gridcolor="#efe8dc",
            zeroline=False,
            rangeslider=dict(visible=False),
            tickformat="%Y-%m",
        ),
        yaxis=dict(
            title="Значение, " + unit_text,
            showgrid=True,
            gridcolor="#efe8dc",
            zeroline=False,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.32,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,253,247,0.9)",
            bordercolor="#ddd6c7",
            borderwidth=1,
        ),
        height=470,
    )
    fig.update_layout(**layout)

    chart_html = pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config=PLOTLY_CONFIG,
    )
    note_html = ""
    if truncation_notes:
        note_html = (
            '<p class="chart-note">'
            + html_escape("; ".join(truncation_notes))
            + '. Полный список стартов находится в Excel на листе «Старты».</p>'
        )
    return chart_html + note_html


def parameter_charts_html(salym_root: Path, well_id: str, long: pd.DataFrame, starts: pd.DataFrame) -> str:
    rows = long[long["well_id"] == well_id]
    start_values = pd.to_datetime(rows["input_start"], errors="coerce").dropna()
    end_values = pd.to_datetime(rows["input_end"], errors="coerce").dropna()
    start = start_values.min() if not start_values.empty else None
    end = end_values.max() if not end_values.empty else None
    well_starts = starts[starts["well_id"] == well_id]
    loaded = []
    for param_key in CHART_CANDIDATES:
        series = load_salym_series(salym_root, well_id, param_key)
        loaded.append((param_key, SELECTED_PARAM_MAP.get(param_key, param_key), series, series_variability(series)))

    selected = []
    pressure_items = [item for item in loaded if item[0] == "ESP.IntakePressure"]
    if pressure_items:
        selected.append(pressure_items[0])
    informative = [item for item in loaded if item[0] != "ESP.IntakePressure" and item[3] > 0]
    informative.sort(key=lambda item: item[3], reverse=True)
    selected.extend(informative[:5])
    if len(selected) < 4:
        for item in loaded:
            if item not in selected:
                selected.append(item)
            if len(selected) >= 4:
                break

    safe_well = re.sub(r"[^0-9A-Za-z_-]+", "_", str(well_id))
    charts = []
    for index, (param_key, title, series, _) in enumerate(selected):
        chart_id = "chart_" + safe_well + "_" + str(index)
        charts.append(
            '<div class="chart-card">'
            + parameter_chart_plotly(
                div_id=chart_id,
                title=title,
                unit=PARAM_UNITS.get(param_key, ""),
                series=series,
                well_starts=well_starts,
                start=start,
                end=end,
            )
            + "</div>"
        )
    return "\n".join(charts)


def timeline_plotly(div_id: str, well_id: str, long: pd.DataFrame, starts: pd.DataFrame) -> str:
    rows = long[long["well_id"] == well_id]
    times: list[pd.Timestamp] = []
    for column in ["input_start", "input_end"]:
        series = pd.to_datetime(rows[column], errors="coerce")
        times.extend([item for item in series if pd.notna(item)])
    well_starts = starts[starts["well_id"] == well_id].copy()
    if not well_starts.empty:
        times.extend([item for item in well_starts["detected_time"] if pd.notna(item)])
    if not times:
        return '<p class="chart-note">Нет временных данных для шкалы.</p>'

    start = min(times)
    end = max(times)
    if start == end:
        end = start + pd.Timedelta(hours=1)

    fig = go.Figure()
    category_labels = [ANOMALY_LABELS[a] for a in ANOMALY_ORDER]
    for anomaly in ANOMALY_ORDER:
        label = ANOMALY_LABELS[anomaly]
        color = anomaly_color(anomaly)
        sub = well_starts[well_starts["anomaly"] == anomaly]
        if sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=[start],
                    y=[label],
                    mode="markers",
                    marker=dict(color=color, size=10, opacity=0.0),
                    name=label + " — нет стартов",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
            continue
        ts = pd.to_datetime(sub["detected_time"], errors="coerce").dropna()
        fig.add_trace(
            go.Scatter(
                x=list(ts),
                y=[label] * len(ts),
                mode="markers",
                marker=dict(color=color, size=11, opacity=0.78, line=dict(color=color, width=1)),
                name=label + " (" + str(len(ts)) + ")",
                hovertemplate=(
                    "<b>" + label + "</b><br>%{x|%Y-%m-%d %H:%M}<extra></extra>"
                ),
            )
        )

    layout = _layout_common()
    layout.update(
        xaxis=dict(
            title="Дата/время",
            range=[start, end],
            showgrid=True,
            gridcolor="#efe8dc",
            zeroline=False,
            tickformat="%Y-%m",
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(reversed(category_labels)),
            showgrid=False,
            zeroline=False,
            automargin=True,
        ),
        height=290,
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, x=0.0),
        margin=dict(l=140, r=24, t=24, b=72),
    )
    fig.update_layout(**layout)
    return pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config=PLOTLY_CONFIG,
    )


def well_file_map(wide_ru: pd.DataFrame) -> dict[str, str]:
    used = set()
    mapping = {}
    for well_id in wide_ru["Скважина"].astype(str):
        mapping[well_id] = safe_name(well_id, used) + ".html"
    return mapping


def write_index(output_dir: Path, wide_ru: pd.DataFrame, file_map: dict[str, str], generated_at: str) -> None:
    detected_counts = wide_ru["Классов с детекцией"].value_counts().to_dict()
    rows = []
    for _, item in wide_ru.iterrows():
        well_id = str(item["Скважина"])
        cells = [
            '<tr data-search="' + html_escape(well_id.lower()) + " " + html_escape(str(item["Классы с детекцией"]).lower()) + '">',
            '<td><a href="wells/' + html_escape(file_map[well_id]) + '">' + html_escape(well_id) + "</a></td>",
            "<td>" + html_escape(item["Классов с детекцией"]) + "</td>",
            "<td>" + html_escape(item["Всего стартов детекции"]) + "</td>",
        ]
        for anomaly in ANOMALY_ORDER:
            label = ANOMALY_LABELS[anomaly]
            cells.append("<td>" + html_escape(item[label + ": статус"]) + "</td>")
            cells.append("<td>" + html_escape(item[label + ": первая детекция"]) + "</td>")
            cells.append("<td>" + html_escape(item[label + ": число стартов"]) + "</td>")
        cells.append("</tr>")
        rows.append("".join(cells))

    html_text = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Salym: экспертная проверка 547 скважин</title>
  <link rel="stylesheet" href="assets/styles.css">
</head>
<body>
<main class="wrap">
  <section class="hero">
    <h1>Salym: экспертная проверка детекций по 547 скважинам</h1>
    <p>Пакет показывает кандидаты трех классов аномалий: негерметичность, приток и солеотложение. Данные не размечены, поэтому результат является списком кандидатов для экспертной проверки, а не подтвержденными событиями.</p>
    <p>Сформировано: """ + html_escape(generated_at) + """</p>
  </section>

  <section class="section grid">
    <div class="card"><div class="metric-title">Скважин</div><div class="metric-value">""" + str(len(wide_ru)) + """</div></div>
    <div class="card"><div class="metric-title">Детекция по 3 классам</div><div class="metric-value">""" + str(detected_counts.get(3, 0)) + """</div></div>
    <div class="card"><div class="metric-title">Детекция по 2 классам</div><div class="metric-value">""" + str(detected_counts.get(2, 0)) + """</div></div>
    <div class="card"><div class="metric-title">Без детекций</div><div class="metric-value">""" + str(detected_counts.get(0, 0)) + """</div></div>
  </section>

  <section class="section note">
    <strong>Как читать:</strong> откройте строку скважины, чтобы увидеть отдельную страницу с тремя карточками аномалий и временной шкалой стартов. Большое число стартов означает, что детектор многократно видел кандидаты на длинном четырехлетнем ряду.
    Подробное объяснение полей находится в файле <a href="Пояснительная_записка.html">Пояснительная_записка.html</a>.
  </section>

  <section class="section tools">
    <input class="search" id="searchBox" type="search" placeholder="Поиск по скважине или классу аномалии">
    <span class="small">Подробная таблица находится в файле <b>salym_эксперт_547_скважин.xlsx</b>.</span>
  </section>

  <section class="section table-box">
    <table id="wellTable">
      <thead>
        <tr>
          <th>Скважина</th>
          <th>Классов с детекцией</th>
          <th>Всего стартов</th>
          <th>Негермет: статус</th>
          <th>Негермет: первая детекция</th>
          <th>Негермет: стартов</th>
          <th>Приток: статус</th>
          <th>Приток: первая детекция</th>
          <th>Приток: стартов</th>
          <th>Соли: статус</th>
          <th>Соли: первая детекция</th>
          <th>Соли: стартов</th>
        </tr>
      </thead>
      <tbody>
""" + "\n".join(rows) + """
      </tbody>
    </table>
  </section>

  <div class="footer">Экспертный пакет Salym. Отчет статический, внешние библиотеки не используются.</div>
</main>
<script>
var searchBox = document.getElementById('searchBox');
var rows = document.querySelectorAll('#wellTable tbody tr');
searchBox.addEventListener('input', function () {
  var value = searchBox.value.toLowerCase();
  for (var i = 0; i < rows.length; i += 1) {
    var row = rows[i];
    var text = row.getAttribute('data-search');
    row.style.display = text.indexOf(value) >= 0 ? '' : 'none';
  }
});
</script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def anomaly_card(anomaly: str, row: pd.Series) -> str:
    label = ANOMALY_LABELS[anomaly]
    return """
<article class="card anomaly-card">
  <h2 class=\"""" + anomaly + """\">""" + html_escape(label) + """</h2>
  """ + status_badge(row.get("status")) + """
  <div class="kv">
    <div>Первая детекция</div><div>""" + html_escape(dt_ru(row.get("first_detected_time"))) + """</div>
    <div>Последняя детекция</div><div>""" + html_escape(dt_ru(row.get("last_detected_time"))) + """</div>
    <div>Число стартов</div><div>""" + format_int(row.get("n_detected_starts")) + """</div>
    <div>Максимальная оценка</div><div>""" + html_escape(score_ru(row.get("max_score"))) + """</div>
    <div>Медианная оценка</div><div>""" + html_escape(score_ru(row.get("median_score"))) + """</div>
    <div>Сырых каналов</div><div>""" + format_int(row.get("n_raw_channels")) + """</div>
    <div>Точек ряда</div><div>""" + format_int(row.get("n_points")) + """</div>
    <div>Доля исключённых данных</div><div>""" + html_escape(percent_ru(row.get("masked_fraction"))) + """</div>
  </div>
</article>
"""


def starts_table(well_id: str, starts: pd.DataFrame) -> str:
    sub = starts[starts["well_id"] == well_id].copy()
    if sub.empty:
        return "<p class=\"small\">По этой скважине стартов детекции нет.</p>"
    rows = []
    for _, item in sub.head(120).iterrows():
        rows.append(
            "<tr><td>"
            + html_escape(ANOMALY_LABELS.get(str(item["anomaly"]), item["anomaly"]))
            + "</td><td>"
            + html_escape(format_int(item.get("start_idx")))
            + "</td><td>"
            + html_escape(dt_ru(item.get("detected_time")))
            + "</td><td>"
            + html_escape(score_ru(item.get("score_at_start")))
            + "</td></tr>"
        )
    suffix = ""
    if len(sub) > 120:
        suffix = "<p class=\"small\">Показаны первые 120 стартов из " + str(len(sub)) + ". Полный список есть в Excel-файле.</p>"
    return """
<div class="table-box">
  <table>
    <thead><tr><th>Класс аномалии</th><th>Номер старта</th><th>Время детекции</th><th>Оценка в момент старта</th></tr></thead>
    <tbody>""" + "".join(rows) + """</tbody>
  </table>
</div>
""" + suffix


def write_well_pages(
    output_dir: Path,
    long: pd.DataFrame,
    starts: pd.DataFrame,
    wide_ru: pd.DataFrame,
    file_map: dict[str, str],
    generated_at: str,
    salym_root: Path,
) -> None:
    wells_dir = output_dir / "wells"
    wells_dir.mkdir(exist_ok=True)
    grouped = {well_id: group for well_id, group in long.groupby("well_id")}
    wide_by_well = wide_ru.set_index("Скважина")

    for well_id in wide_ru["Скважина"].astype(str):
        group = grouped.get(well_id, pd.DataFrame())
        rows = {row["anomaly"]: row for _, row in group.iterrows()}
        cards = []
        for anomaly in ANOMALY_ORDER:
            if anomaly in rows:
                cards.append(anomaly_card(anomaly, rows[anomaly]))
        summary = wide_by_well.loc[well_id]
        page = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>""" + html_escape(well_id) + """ — Salym экспертная проверка</title>
  <link rel="stylesheet" href="../assets/styles.css">
  <script src="../assets/plotly.min.js"></script>
</head>
<body>
<main class="wrap">
  <section class="hero">
    <p><a href="../index.html" style="color:#dce7f3">← Назад к списку скважин</a></p>
    <h1>Скважина """ + html_escape(well_id) + """</h1>
    <p>Классы с детекцией: """ + html_escape(summary.get("Классы с детекцией", "")) + """. Всего стартов: """ + html_escape(summary.get("Всего стартов детекции", "")) + """.</p>
    <p>Сформировано: """ + html_escape(generated_at) + """</p>
  </section>

  <section class="section well-grid">
""" + "\n".join(cards) + """
  </section>

  <section class="section card">
    <h2>Временная шкала стартов</h2>
    <p class="small">Каждая точка — найденный старт подозрительного участка. Наведите мышь на точку, чтобы увидеть точное время. Цвета: красный — негерметичность, оранжевый — приток, зелёный — солеотложение.</p>
    <div class="timeline">""" + timeline_plotly("timeline_" + re.sub(r"[^0-9A-Za-z_-]+", "_", str(well_id)), well_id, long, starts) + """</div>
  </section>

  <section class="section card">
    <h2>Графики ключевых параметров</h2>
    <p class="small">Графики интерактивные: можно увеличить участок выделением, прокрутить колесом, навести мышь на любую точку и увидеть значение. Синяя линия — параметр во времени; пунктирные вертикальные линии — найденные старты: красный — негерметичность, оранжевый — приток, зелёный — солеотложение. Если стартов больше 50, на графике показаны первые 50; полный список — в Excel на листе «Старты». Если параметр постоянный или нулевой, вместо графика выведено текстовое пояснение.</p>
    <div class="chart-stack">
""" + parameter_charts_html(salym_root, well_id, long, starts) + """
    </div>
  </section>

  <section class="section">
    <h2>Старты детекций</h2>
""" + starts_table(well_id, starts) + """
  </section>
</main>
</body>
</html>
"""
        (wells_dir / file_map[well_id]).write_text(page, encoding="utf-8")


def cleanup_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.iterdir():
        if path.is_dir():
            if path.name == "tables":
                shutil.rmtree(path)
            continue
        if path.suffix.lower() == ".csv":
            path.unlink()
            continue
        if path.name.lower().startswith("readme"):
            path.unlink()


def write_assets(output_dir: Path) -> None:
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    (assets_dir / "styles.css").write_text(css_text(), encoding="utf-8")
    target_plotly = assets_dir / "plotly.min.js"
    if not PLOTLY_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "Не найден локальный plotly.min.js по пути " + str(PLOTLY_BUNDLE_PATH)
            + ". Установите plotly в окружение."
        )
    if (
        not target_plotly.exists()
        or target_plotly.stat().st_size != PLOTLY_BUNDLE_PATH.stat().st_size
    ):
        shutil.copyfile(PLOTLY_BUNDLE_PATH, target_plotly)


def write_explanation_html(output_dir: Path) -> None:
    page = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Пояснительная записка к экспертному пакету Salym</title>
  <link rel="stylesheet" href="assets/styles.css">
</head>
<body>
<main class="wrap">
  <section class="hero">
    <h1>Пояснительная записка</h1>
    <p>Эта записка объясняет, что находится в экспертном пакете Salym и как читать результаты без знания внутреннего устройства алгоритма.</p>
  </section>

  <section class="section card">
    <h2>Назначение пакета</h2>
    <p>Пакет подготовлен для первичной проверки кандидатов аномалий по 547 неразмеченным скважинам Salym. Для каждой скважины показаны три возможных класса: негерметичность, приток и солеотложение.</p>
    <p><strong>Важно:</strong> детекция означает кандидат на экспертную проверку, а не подтвержденное осложнение. Окончательное решение принимает эксперт по графикам, истории эксплуатации и технологическому контексту.</p>
  </section>

  <section class="section card">
    <h2>Файлы в папке</h2>
    <p><strong>index.html</strong> — главный файл. Откройте его в браузере, чтобы увидеть общий список скважин и перейти на страницу конкретной скважины.</p>
    <p><strong>wells</strong> — папка с отдельными страницами по каждой скважине. На странице есть три карточки аномалий, временная шкала и таблица найденных стартов.</p>
    <p><strong>salym_эксперт_547_скважин.xlsx</strong> — единый табличный файл. Отдельных табличных дублей в пакете нет, чтобы не путать эксперта.</p>
  </section>

  <section class="section card">
    <h2>Листы Excel-файла</h2>
    <p><strong>Описание</strong> — краткое пояснение назначения файла и смысла основных полей.</p>
    <p><strong>Сводка</strong> — главная таблица: одна строка соответствует одной скважине, рядом указаны результаты по трем классам аномалий.</p>
    <p><strong>По классам</strong> — более подробная таблица: одна строка соответствует одной паре «скважина — класс аномалии».</p>
    <p><strong>Старты</strong> — все найденные моменты начала подозрительных участков. Этот лист нужен, если по скважине найдено несколько повторных событий.</p>
    <p><strong>Пропуски</strong> — случаи, где обработка не была выполнена или была пропущена.</p>
  </section>

  <section class="section card">
    <h2>Как понимать поля</h2>
    <p><strong>Статус</strong> показывает, была ли найдена детекция по выбранному классу аномалии.</p>
    <p><strong>Первая детекция</strong> — первый момент времени, где алгоритм увидел устойчивый подозрительный участок.</p>
    <p><strong>Последняя детекция</strong> — последний найденный старт подозрительного участка по этому классу.</p>
    <p><strong>Число стартов</strong> — сколько отдельных начал подозрительных участков найдено за весь период. Большое число не означает много аварий автоматически, но показывает, что скважину нужно смотреть внимательнее.</p>
    <p><strong>Максимальная оценка</strong> — наибольшая выраженность признака внутри данного класса аномалии. Ее корректно сравнивать прежде всего внутри одного класса.</p>
    <p><strong>Медианная оценка</strong> — типичный уровень сигнала по ряду.</p>
    <p><strong>Точек в базовом участке</strong> — сколько точек использовано как условно нормальное поведение скважины.</p>
    <p><strong>Доля исключенных данных</strong> — какая часть ряда не использовалась как надежная база из-за пропусков, нестабильного режима или подозрительных участков.</p>
  </section>
</main>
</body>
</html>
"""
    (output_dir / "Пояснительная_записка.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "expert_package_ru"
    salym_root = Path(args.salym_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_output_dir(output_dir)

    long = read_anomaly_results(input_dir)
    starts = read_predicted_starts(input_dir)
    errors = read_errors(input_dir)
    wide_ru = build_wide_summary(long)
    long_ru = build_long_ru(long)
    starts_ru = build_starts_ru(starts)
    errors_ru = build_errors_ru(errors)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_map = well_file_map(wide_ru)

    write_tables(output_dir, wide_ru, long_ru, starts_ru, errors_ru)
    write_assets(output_dir)
    write_explanation_html(output_dir)
    write_index(output_dir, wide_ru, file_map, generated_at)
    write_well_pages(output_dir, long, starts, wide_ru, file_map, generated_at, salym_root)

    print("Пакет сформирован:", output_dir)
    print("Скважин:", len(wide_ru))
    print("HTML страниц скважин:", len(file_map))
    print("Excel:", output_dir / "salym_эксперт_547_скважин.xlsx")
    print("Index:", output_dir / "index.html")


if __name__ == "__main__":
    main()
