from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.paths import DB_DIR, RAW_CACHE_DIR, REFERENCE_DATA_DIR, REPORTS_DIR, ensure_parent

MD_PATH = PROJECT_ROOT / "docs" / "физика_аномалий_по_сводной_и_экспертным_комментариям.md"
SUMMARY_XLSX_PATH = REFERENCE_DATA_DIR / "Сводная информация_объединённая.xlsx"
DEFAULT_OUTPUT_PATH = REPORTS_DIR / "physics" / "физика_аномалий_визуальный_отчёт.html"

PRESSURE_COL = "Давление на приеме насоса кгс/см²"
FREQUENCY_COL = "Выходная частота"
CURRENT_COL = "Ток на фазе А"
LOAD_COL = "Коэффициент загрузки ПЭД"
TEMP_INTAKE_COL = "Температура на приёме насоса"
TEMP_OIL_COL = "Температура масла двигателя"

ANOMALY_SOURCES = ("negermet", "pritok", "salt")

CLASS_BY_MD_TYPE = {
    "негерметичность нкт": "negermet",
    "приток": "pritok",
    "соли": "salt",
    "нормальная работа": "norm",
    "нормальное поведение (изменение частоты)": "norm_freq",
    "нормальное изменение частоты": "norm_freq",
}
CLASS_LABELS = {
    "negermet": "Негерметичность НКТ",
    "pritok": "Приток",
    "salt": "Солеотложение",
    "norm": "Нормальная работа",
    "norm_freq": "Нормальное изменение частоты",
}
CLASS_COLORS = {
    "negermet": "#dc2626",
    "pritok": "#2563eb",
    "salt": "#d97706",
    "norm": "#059669",
    "norm_freq": "#0d9488",
}
SPLIT_LABELS = {
    "train": "Обучающая",
    "test": "Отложенная проверочная",
    "norm": "Источник нормы",
    "excluded": "Исключена из набора",
}
SPLIT_COLORS = {
    "train": "#475569",
    "test": "#7c3aed",
    "norm": "#059669",
    "excluded": "#9f1239",
}

COLOR_PRESSURE = "#1d4ed8"
COLOR_FREQUENCY = "#7c3aed"
COLOR_CURRENT = "#ea580c"
COLOR_LOAD = "#ca8a04"
COLOR_TEMP_INTAKE = "#dc2626"
COLOR_TEMP_OIL = "#9f1239"
COLOR_PREFIX_ZONE = "rgba(16, 185, 129, 0.10)"
COLOR_ANOMALY_ZONE = "rgba(239, 68, 68, 0.12)"
COLOR_ANOMALY_LINE = "#b91c1c"
COLOR_STOP_ZONE = "rgba(100, 116, 139, 0.22)"
COLOR_STOP_TEXT = "#475569"
COLOR_FREQ_JUMP_LINE = "#7c3aed"
COLOR_PEAK_STOP_ZONE = "rgba(100, 116, 139, 0.28)"
COLOR_PEAK_STOP_TEXT = "#475569"
COLOR_PEAK_OTHER_ZONE = "rgba(249, 115, 22, 0.28)"
COLOR_PEAK_OTHER_TEXT = "#c2410c"
COLOR_CLEANED = "#059669"
COLOR_PRITOK_PREV_ZONE = "rgba(139, 92, 246, 0.16)"
COLOR_PRITOK_PREV_TEXT = "#6d28d9"

STOP_FREQUENCY_THRESHOLD_HZ = 1.0
STOP_MIN_DURATION_MINUTES = 30.0
FREQ_JUMP_THRESHOLD_HZ = 0.5
FREQ_JUMP_BUFFER_AFTER_STOP_HOURS = 12.0
MAX_FREQ_JUMP_MARKERS = 12
PEAK_DEVIATION_THRESHOLD_PCT = 15.0
PEAK_MAX_DURATION_HOURS = 72.0
PEAK_STOP_CHECK_BUFFER_HOURS = 1.0
PEAK_STOP_MIN_SAMPLES = 2
INFLUENCE_FREQ_RECOVERY_RATIO = 0.95
# Согласовано с alma_service.stop_influence (фикс 04.06.2026): 1.05 (5%) обрывал зону на
# спаде пика и оставлял видимый хвост; 1.005 тянет серую полосу до возврата давления к base.
INFLUENCE_PRESSURE_RECOVERY_RATIO = 1.005
INFLUENCE_BASE_WINDOW_HOURS = 12.0
INFLUENCE_MAX_TAIL_HOURS = 36.0
# Минимальный рост давления (бамп) от остановки, чтобы помечать зону: отсекает мелкие
# остановки без заметного пика (эксперт 04.06.2026 — «два пика», а не каждый микро-стоп).
# Порог ниже минимума одобренных зон (602 = +10%), поэтому одобренные не затрагиваются.
MIN_ZONE_EXCESS_PCT = 9.0
INFLUENCE_FULL_RESOLUTION_PAD_HOURS = 2.0

CANONICAL_EXAMPLES = {
    "negermet": (44, "Скачок давления +120% за часы при неизменной частоте (скв. 524)"),
    "pritok": (26, "Нисходящий тренд давления -26% при неизменной частоте (скв. 691)"),
    "salt": (40, "Рост давления при трендовом росте частоты, падение токов и загрузки (скв. 3244Г)"),
    "norm_freq": (18, "Штатная реакция давления на подъём частоты +3 Гц (скв. 1995)"),
}


# ---------------------------------------------------------------------------
# Разбор markdown
# ---------------------------------------------------------------------------

@dataclass
class MdSection:
    level: int
    title: str
    body: str


def split_md_sections(text: str) -> list[MdSection]:
    sections: list[MdSection] = []
    matches = list(re.finditer(r"^(#{2,3})\s+(.+)$", text, flags=re.MULTILINE))
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append(
            MdSection(level=len(match.group(1)), title=match.group(2).strip(), body=text[start:end].strip())
        )
    return sections


def find_section(sections: list[MdSection], title_prefix: str) -> MdSection | None:
    for section in sections:
        if section.title.startswith(title_prefix):
            return section
    return None


def section_with_subsections(sections: list[MdSection], title_prefix: str) -> str:
    collected: list[str] = []
    capture = False
    base_level = 0
    for section in sections:
        if section.title.startswith(title_prefix):
            capture = True
            base_level = section.level
            collected.append(section.body)
            continue
        if capture:
            if section.level <= base_level:
                break
            collected.append("#" * section.level + " " + section.title + "\n\n" + section.body)
    return "\n\n".join(collected)


def parse_md_table(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return []
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def md_inline_to_html(text: str) -> str:
    out = escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"~~(.+?)~~", r"<del>\1</del>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


def md_table_to_html(rows: list[dict[str, str]], css_class: str = "md-table") -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    head_html = "".join(f"<th>{md_inline_to_html(h)}</th>" for h in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{md_inline_to_html(row[h])}</td>" for h in headers) + "</tr>"
        for row in rows
    )
    return f"<div class='table-scroll'><table class='{css_class}'><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table></div>"


def md_block_to_html(text: str, heading_offset: int = 1) -> str:
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        heading = re.match(r"^(#{2,4})\s+(.+)$", stripped)
        if heading:
            level = min(len(heading.group(1)) + heading_offset, 6)
            blocks.append(f"<h{level}>{md_inline_to_html(heading.group(2))}</h{level}>")
            i += 1
            continue
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            blocks.append(md_table_to_html(parse_md_table("\n".join(table_lines))))
            continue
        if re.match(r"^[-*]\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                items.append(re.sub(r"^[-*]\s+", "", lines[i].strip()))
                i += 1
            blocks.append("<ul>" + "".join(f"<li>{md_inline_to_html(item)}</li>" for item in items) + "</ul>")
            continue
        if re.match(r"^\d+[.)]\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^\d+[.)]\s+", lines[i].strip()):
                items.append(re.sub(r"^\d+[.)]\s+", "", lines[i].strip()))
                i += 1
            blocks.append("<ol>" + "".join(f"<li>{md_inline_to_html(item)}</li>" for item in items) + "</ol>")
            continue
        paragraph = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{2,4}\s|\||[-*]\s|\d+[.)]\s)", lines[i].strip()):
            paragraph.append(lines[i].strip())
            i += 1
        blocks.append(f"<p>{md_inline_to_html(' '.join(paragraph))}</p>")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------

@dataclass
class WellRow:
    summary_row: int
    well_label: str
    class_key: str
    class_label: str
    source_kind: str
    expert_comment: str
    usage: str
    physics: dict[str, str] = field(default_factory=dict)
    prefix: dict[str, str] = field(default_factory=dict)
    case_id: str | None = None
    split: str = "norm"
    anomaly_start: pd.Timestamp | None = None
    anomaly_end: pd.Timestamp | None = None
    series: pd.DataFrame | None = None
    extra_intervals: list[dict[str, Any]] = field(default_factory=list)


def normalize_id(value: object) -> str:
    return str(value).strip().lower()


def normalize_column(name: str) -> str:
    return name.strip().lower().replace("ё", "е")


def find_column(df: pd.DataFrame, *needles: str) -> str | None:
    for column in df.columns:
        normalized = normalize_column(str(column))
        if all(normalize_column(needle) in normalized for needle in needles):
            return str(column)
    return None


def class_key_for_type(type_text: str) -> str:
    normalized = type_text.strip().lower()
    for needle, key in CLASS_BY_MD_TYPE.items():
        if normalized.startswith(needle) or needle in normalized:
            return key
    return "norm"


def load_md_rows(md_text: str) -> dict[int, WellRow]:
    sections = split_md_sections(md_text)
    wells_section = find_section(sections, "Скважины из сводной")
    physics_section = find_section(sections, "Проверка физики по каждой строке сводной")
    prefix_section = find_section(sections, "Префиксы 39 аномальных скважин")
    if wells_section is None or physics_section is None:
        raise RuntimeError("В md не найдены разделы «Скважины из сводной» / «Проверка физики»")

    rows: dict[int, WellRow] = {}
    for record in parse_md_table(wells_section.body):
        summary_row = int(record["Строка сводной таблицы"])
        class_label = record["Тип"]
        rows[summary_row] = WellRow(
            summary_row=summary_row,
            well_label=record["Скважина / идентификатор случая"],
            class_key=class_key_for_type(class_label),
            class_label=class_label,
            source_kind=record["Источник данных"],
            expert_comment=record["Физический комментарий / признаки"],
            usage=record["Использование сейчас"],
        )

    for record in parse_md_table(physics_section.body):
        summary_row = int(record["Строка сводной"])
        if summary_row in rows:
            rows[summary_row].physics = record

    if prefix_section is not None:
        for record in parse_md_table(prefix_section.body):
            summary_row = int(record["Строка сводной"])
            if summary_row in rows:
                rows[summary_row].prefix = record

    return rows


def load_anomaly_cases(rows: dict[int, WellRow]) -> None:
    for source in ANOMALY_SOURCES:
        intervals_path = DB_DIR / f"{source}_intervals.parquet"
        dataset_path = DB_DIR / f"{source}_anomaly_database_5min.parquet"
        if not intervals_path.exists() or not dataset_path.exists():
            print(f"[внимание] нет данных для {source}: {intervals_path}")
            continue
        intervals = pd.read_parquet(intervals_path)
        dataset = pd.read_parquet(dataset_path)
        dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
        for _, interval in intervals.iterrows():
            summary_row = int(interval["summary_row"]) if "summary_row" in interval else None
            if summary_row is None or summary_row not in rows:
                continue
            row = rows[summary_row]
            row.case_id = str(interval["well_id"])
            row.split = str(interval["split"])
            row.anomaly_start = pd.Timestamp(interval["start_date"])
            row.anomaly_end = pd.Timestamp(interval["end_date"])
            series = dataset[dataset["well_id"] == row.case_id].sort_values("timestamp").reset_index(drop=True)
            row.series = series if not series.empty else None


def attach_extra_intervals(rows: dict[int, WellRow]) -> None:
    """Подтянуть extra_intervals из alma_summary_overrides.json — дополнительные
    интервалы аномалий другого класса на той же скважине (например приток в
    префиксе соляной скважины 3245(2)), зафиксированные, но не встроенные в
    конвейер. Рисуются отдельной зоной «предыдущая аномалия»."""
    path = PROJECT_ROOT / "configs" / "alma_summary_overrides.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not payload.get("enabled", True):
        return
    for item in payload.get("extra_intervals", []):
        summary_row = item.get("summary_row")
        if summary_row in rows:
            rows[summary_row].extra_intervals.append(dict(item))


def load_norm_cases(rows: dict[int, WellRow]) -> None:
    dataset_path = DB_DIR / "norm_work_database_5min.parquet"
    if not dataset_path.exists():
        print(f"[внимание] нет данных нормы: {dataset_path}")
        return
    dataset = pd.read_parquet(dataset_path)
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
    by_norm_id = {normalize_id(well_id): well_id for well_id in dataset["well_id"].unique()}
    for row in rows.values():
        if row.class_key not in {"norm", "norm_freq"}:
            continue
        case_id = by_norm_id.get(normalize_id(row.well_label))
        if case_id is None:
            continue
        row.case_id = str(case_id)
        row.split = "norm"
        series = dataset[dataset["well_id"] == case_id].sort_values("timestamp").reset_index(drop=True)
        row.series = series if not series.empty else None


def load_excluded_1123l(rows: dict[int, WellRow]) -> None:
    target = next(
        (row for row in rows.values() if normalize_id(row.well_label) == "1123л" and row.class_key == "negermet"),
        None,
    )
    if target is None:
        return
    cache_path = RAW_CACHE_DIR / "negermet" / "1123л_Кустовое_ННКТ.parquet"
    if not cache_path.exists():
        print(f"[внимание] нет кэша для 1123л: {cache_path}")
        return
    raw = pd.read_parquet(cache_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    wide = (
        raw.pivot_table(index="timestamp", columns="param_name", values="value", aggfunc="mean")
        .sort_index()
        .resample("5min")
        .median()
    )
    wide = wide.dropna(how="all").reset_index()
    wide.columns.name = None
    target.case_id = "1123л"
    target.split = "excluded"
    target.series = wide

    if SUMMARY_XLSX_PATH.exists():
        summary = pd.read_excel(SUMMARY_XLSX_PATH, header=2)
        for _, xlsx_row in summary.iterrows():
            if normalize_id(xlsx_row.get("Скважина")) != "1123л":
                continue
            if "негермет" not in normalize_id(xlsx_row.get("Тип аномалии")):
                continue
            target.anomaly_start = pd.to_datetime(xlsx_row.get("Дата начала аномалии"), dayfirst=True, errors="coerce")
            target.anomaly_end = pd.to_datetime(xlsx_row.get("Дата конца аномалии"), dayfirst=True, errors="coerce")
            break


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

def adaptive_resample(
    series: pd.DataFrame,
    anomaly_start: pd.Timestamp | None,
    prefix_target: int = 1100,
    anomaly_target: int = 900,
    full_resolution_periods: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = (),
) -> pd.DataFrame:
    indexed = series.set_index("timestamp").sort_index()
    numeric = indexed.select_dtypes(include=[np.number])
    if numeric.empty:
        return series

    def resample_part(part: pd.DataFrame, target: int) -> pd.DataFrame:
        if part.empty or len(part) <= target:
            return part
        duration_min = (part.index[-1] - part.index[0]).total_seconds() / 60.0
        step = max(5, int(np.ceil(duration_min / max(target, 1))))
        return part.resample(f"{step}min").median()

    if anomaly_start is not None and indexed.index[0] < anomaly_start < indexed.index[-1]:
        prefix_part = resample_part(numeric[numeric.index < anomaly_start], prefix_target)
        anomaly_part = resample_part(numeric[numeric.index >= anomaly_start], anomaly_target)
        combined = pd.concat([prefix_part, anomaly_part])
    else:
        combined = resample_part(numeric, prefix_target + anomaly_target)

    # вокруг указанных периодов (остановки и их влияние) сохраняем полное 5-минутное
    # разрешение — иначе резкие переходы частоты размазываются прореживанием и
    # визуальные границы не совпадают с фактическими
    if full_resolution_periods:
        pad = pd.Timedelta(hours=INFLUENCE_FULL_RESOLUTION_PAD_HOURS)
        parts = [combined]
        for period_start, period_end in full_resolution_periods:
            window = numeric.loc[period_start - pad: period_end + pad]
            if not window.empty:
                parts.append(window)
        combined = pd.concat(parts).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    return combined.reset_index()


def detect_stop_periods(series: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    freq_col = find_column(series, "выходная частота")
    if freq_col is None or "timestamp" not in series.columns:
        return []
    frame = series[["timestamp", freq_col]].dropna().sort_values("timestamp")
    if frame.empty:
        return []
    stopped = frame[freq_col] < STOP_FREQUENCY_THRESHOLD_HZ
    groups = (stopped != stopped.shift()).cumsum()
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, group in frame[stopped].groupby(groups[stopped]):
        start = group["timestamp"].iloc[0]
        end = group["timestamp"].iloc[-1]
        if (end - start).total_seconds() / 60.0 >= STOP_MIN_DURATION_MINUTES:
            periods.append((start, end))
    return periods


def detect_pressure_peaks(series: pd.DataFrame) -> list[dict[str, Any]]:
    pressure_col = find_column(series, "давление на приеме")
    freq_col = find_column(series, "выходная частота")
    if pressure_col is None or "timestamp" not in series.columns:
        return []
    indexed = series.set_index("timestamp").sort_index()
    p = indexed[pressure_col].dropna()
    if len(p) < 50:
        return []
    frequency = indexed[freq_col].dropna() if freq_col else pd.Series(dtype=float)
    baseline = p.rolling("24h", center=True, min_periods=12).median()
    deviation_pct = (p - baseline) / baseline * 100
    in_peak = deviation_pct > PEAK_DEVIATION_THRESHOLD_PCT
    if not in_peak.any():
        return []
    groups = (in_peak != in_peak.shift()).cumsum()
    buffer = pd.Timedelta(hours=PEAK_STOP_CHECK_BUFFER_HOURS)
    peaks: list[dict[str, Any]] = []
    for _, group in p[in_peak].groupby(groups[in_peak]):
        start, end = group.index[0], group.index[-1]
        duration_hours = (end - start).total_seconds() / 3600.0
        if duration_hours > PEAK_MAX_DURATION_HOURS:
            continue
        base_value = float(baseline.loc[start:end].median())
        excess_pct = (float(group.max()) / base_value - 1.0) * 100.0 if base_value > 0 else 0.0
        # классификация по частоте: была ли остановка насоса в окне пика (с буфером)
        freq_window = frequency.loc[start - buffer: end + buffer] if len(frequency) else pd.Series(dtype=float)
        stopped_samples = int((freq_window < STOP_FREQUENCY_THRESHOLD_HZ).sum()) if len(freq_window) else 0
        from_stop = stopped_samples >= PEAK_STOP_MIN_SAMPLES
        peaks.append(
            {
                "start": start,
                "end": end,
                "excess_pct": excess_pct,
                "from_stop": from_stop,
                "stopped_samples": stopped_samples,
            }
        )
    return peaks


def detect_stop_influence_zones(series: pd.DataFrame) -> list[dict[str, Any]]:
    pressure_col = find_column(series, "давление на приеме")
    freq_col = find_column(series, "выходная частота")
    if pressure_col is None or freq_col is None or "timestamp" not in series.columns:
        return []
    frame = series.set_index("timestamp")[[freq_col, pressure_col]].dropna().sort_index()
    if len(frame) < 50:
        return []
    freq = frame[freq_col]
    pressure = frame[pressure_col]
    stopped = freq < STOP_FREQUENCY_THRESHOLD_HZ
    if not stopped.any():
        return []
    base_window = pd.Timedelta(hours=INFLUENCE_BASE_WINDOW_HOURS)
    max_tail = pd.Timedelta(hours=INFLUENCE_MAX_TAIL_HOURS)
    groups = (stopped != stopped.shift()).cumsum()
    zones: list[dict[str, Any]] = []
    for _, core in freq[stopped].groupby(groups[stopped]):
        if len(core) < PEAK_STOP_MIN_SAMPLES:
            continue
        core_start, core_end = core.index[0], core.index[-1]
        before_freq = freq.loc[core_start - base_window: core_start]
        before_freq = before_freq[before_freq >= STOP_FREQUENCY_THRESHOLD_HZ]
        before_pressure = pressure.loc[core_start - base_window: core_start]
        if before_freq.empty or before_pressure.empty:
            continue
        working_freq = float(before_freq.median())
        base_pressure = float(before_pressure.median())
        # начало зоны: последняя точка с рабочей частотой перед остановкой
        normal_before = freq.loc[:core_start]
        normal_before = normal_before[normal_before >= working_freq * INFLUENCE_FREQ_RECOVERY_RATIO]
        zone_start = normal_before.index[-1] if len(normal_before) else core_start
        # конец зоны: частота восстановилась И давление вернулось к базе
        after = frame.loc[core_end:]
        recovered_mask = (after[freq_col] >= working_freq * INFLUENCE_FREQ_RECOVERY_RATIO) & (
            after[pressure_col] <= base_pressure * INFLUENCE_PRESSURE_RECOVERY_RATIO
        )
        recovered_times = after.index[recovered_mask]
        limit = core_end + max_tail
        if len(recovered_times) and recovered_times[0] <= limit:
            zone_end = recovered_times[0]
            pressure_recovered = True
        else:
            zone_end = min(frame.index[-1], limit)
            pressure_recovered = False
        zone_pressure = pressure.loc[zone_start:zone_end]
        peak_pressure = float(zone_pressure.max()) if len(zone_pressure) else base_pressure
        excess_pct = (peak_pressure / base_pressure - 1.0) * 100.0 if base_pressure > 0 else 0.0
        zones.append(
            {
                "start": zone_start,
                "end": zone_end,
                "core_start": core_start,
                "core_end": core_end,
                "excess_pct": excess_pct,
                "pressure_recovered": pressure_recovered,
            }
        )
    # слияние перекрывающихся зон (близкие остановки = одно событие)
    merged: list[dict[str, Any]] = []
    for zone in sorted(zones, key=lambda z: z["start"]):
        if merged and zone["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], zone["end"])
            merged[-1]["core_end"] = max(merged[-1]["core_end"], zone["core_end"])
            merged[-1]["excess_pct"] = max(merged[-1]["excess_pct"], zone["excess_pct"])
            merged[-1]["pressure_recovered"] = merged[-1]["pressure_recovered"] and zone["pressure_recovered"]
        else:
            merged.append(dict(zone))
    return merged


def detect_frequency_jumps(
    series: pd.DataFrame,
    stop_periods: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, float]]:
    freq_col = find_column(series, "выходная частота")
    if freq_col is None or "timestamp" not in series.columns:
        return []
    indexed = series.set_index("timestamp")[freq_col].dropna().sort_index()
    if indexed.empty:
        return []
    medians = indexed.resample("6h").median().dropna()
    deltas = medians.diff()
    jumps: list[tuple[pd.Timestamp, float]] = []
    buffer = pd.Timedelta(hours=FREQ_JUMP_BUFFER_AFTER_STOP_HOURS)
    for ts, delta in deltas.items():
        if not np.isfinite(delta) or abs(float(delta)) < FREQ_JUMP_THRESHOLD_HZ:
            continue
        inside_stop = any(
            stop_start - buffer <= ts <= stop_end + buffer for stop_start, stop_end in stop_periods
        )
        if inside_stop:
            continue
        jumps.append((pd.Timestamp(ts), float(delta)))
    if len(jumps) > MAX_FREQ_JUMP_MARKERS:
        jumps = sorted(jumps, key=lambda item: abs(item[1]), reverse=True)[:MAX_FREQ_JUMP_MARKERS]
        jumps = sorted(jumps, key=lambda item: item[0])
    return jumps


def trace_values(series: pd.DataFrame, column: str | None, digits: int) -> tuple[list[str], list[float | None]] | None:
    if column is None or column not in series.columns:
        return None
    values = series[column]
    if values.notna().sum() == 0:
        return None
    x = series["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    y = [None if pd.isna(v) else round(float(v), digits) for v in values]
    return x, y


def _prefix_zones(zones: list[dict[str, Any]], anomaly_start: pd.Timestamp | None) -> list[dict[str, Any]]:
    # Оставляем только зоны, чьё ядро остановки в нормальной префиксной зоне (до аномалии).
    if anomaly_start is None:
        return list(zones)
    return [z for z in zones if z["core_start"] < anomaly_start]


def _prefix_peaks(peaks: list[dict[str, Any]], anomaly_start: pd.Timestamp | None) -> list[dict[str, Any]]:
    if anomaly_start is None:
        return list(peaks)
    return [p for p in peaks if p["start"] < anomaly_start]


def _clip_end(ts: pd.Timestamp, anomaly_start: pd.Timestamp | None) -> pd.Timestamp:
    # Полоса не должна заходить в аномальную зону: обрезаем правый край по началу аномалии.
    return min(ts, anomaly_start) if anomaly_start is not None else ts


def build_well_figure(row: WellRow) -> dict[str, Any] | None:
    if row.series is None or row.series.empty:
        return None
    # Правило пиков и зон влияния остановок (указания эксперта 02-03.06.2026) применяется
    # только к классу приток: на негермете и солях резкое изменение давления — сама аномалия
    # Зоны влияния остановок (полный охват: падение частоты → восстановление + возврат
    # давления к базе) считаем для ВСЕХ классов — у salt/негермета раньше рисовалось узкое
    # ядро f<1 (бралась только середина пика). Помечаем ТОЛЬКО в нормальной префиксной зоне:
    # внутри размеченной аномалии резкое изменение давления это сама аномалия (эксперт 04.06.2026).
    influence_zones_all = detect_stop_influence_zones(row.series)
    stop_periods = [(zone["core_start"], zone["core_end"]) for zone in influence_zones_all]
    influence_zones = [
        z for z in _prefix_zones(influence_zones_all, row.anomaly_start)
        if z["excess_pct"] >= MIN_ZONE_EXCESS_PCT
    ]
    if row.class_key == "pritok":
        pressure_peaks = detect_pressure_peaks(row.series)
        orange_peaks = _prefix_peaks(
            [peak for peak in pressure_peaks if not peak["from_stop"]], row.anomaly_start
        )
    else:
        pressure_peaks = []
        orange_peaks = []
    freq_jumps = detect_frequency_jumps(row.series, stop_periods)
    full_resolution = tuple((zone["start"], zone["end"]) for zone in influence_zones)
    series = adaptive_resample(row.series, row.anomaly_start, full_resolution_periods=full_resolution)
    if "timestamp" not in series.columns or series.empty:
        return None

    pressure_col = find_column(series, "давление на приеме")
    frequency_col = find_column(series, "выходная частота")
    current_col = find_column(series, "ток на фазе а")
    load_col = find_column(series, "коэффициент загрузки")
    temp_intake_col = find_column(series, "температура на приеме")
    temp_oil_col = find_column(series, "температура масла")

    pressure = trace_values(series, pressure_col, 2)
    if pressure is None:
        return None
    frequency = trace_values(series, frequency_col, 1)
    current = trace_values(series, current_col, 2)
    load = trace_values(series, load_col, 1)
    temp_intake = trace_values(series, temp_intake_col, 1)
    temp_oil = trace_values(series, temp_oil_col, 1)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.46, 0.18, 0.36],
        vertical_spacing=0.05,
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    fig.add_trace(
        go.Scatter(
            x=pressure[0], y=pressure[1], mode="lines", name="Давление на приёме",
            line={"color": COLOR_PRESSURE, "width": 1.7},
            hovertemplate="%{x|%d.%m.%Y %H:%M}<br>Давление: %{y:.2f} кгс/см²<extra></extra>",
        ),
        row=1, col=1,
    )
    # Каналы «после очистки остановок» (как видит банк/энкодер притока): зоны убраны, линия
    # соединена через вырез. Вырезается весь многомерный ряд синхронно — показываем стык на
    # всех параметрах. Только для притока — банк чистит остановки только там.
    cleaned_inside = None
    if row.class_key == "pritok" and influence_zones:
        series_ts = pd.to_datetime(series["timestamp"]).to_numpy()
        cleaned_inside = np.zeros(len(series_ts), dtype=bool)
        for zone in influence_zones:
            cleaned_inside |= (series_ts >= np.datetime64(zone["start"])) & (series_ts <= np.datetime64(zone["end"]))

    def add_cleaned(row_idx, col, digits, unit, name, *, secondary=False):
        if cleaned_inside is None or col is None or col not in series.columns:
            return
        raw_v = series[col].to_numpy(dtype=float)
        y = [
            None if (cleaned_inside[i] or not np.isfinite(raw_v[i])) else round(float(raw_v[i]), digits)
            for i in range(len(raw_v))
        ]
        fig.add_trace(
            go.Scatter(
                x=pressure[0], y=y, mode="lines", name=name, showlegend=(row_idx == 1),
                line={"color": COLOR_CLEANED, "width": 1.3, "dash": "dash"}, connectgaps=True,
                hovertemplate=f"%{{x|%d.%m.%Y %H:%M}}<br>{name}: %{{y:.{digits}f}} {unit}<extra></extra>",
            ),
            row=row_idx, col=1, secondary_y=secondary,
        )

    add_cleaned(1, pressure_col, 2, "кгс/см²", "Давление после очистки остановок")
    if frequency is not None:
        fig.add_trace(
            go.Scatter(
                x=frequency[0], y=frequency[1], mode="lines", name="Выходная частота",
                line={"color": COLOR_FREQUENCY, "width": 1.4},
                hovertemplate="%{x|%d.%m.%Y %H:%M}<br>Частота: %{y:.1f} Гц<extra></extra>",
            ),
            row=2, col=1,
        )
    if current is not None:
        fig.add_trace(
            go.Scatter(
                x=current[0], y=current[1], mode="lines", name="Ток фазы А",
                line={"color": COLOR_CURRENT, "width": 1.3},
                hovertemplate="%{x|%d.%m.%Y %H:%M}<br>Ток фазы А: %{y:.2f} А<extra></extra>",
            ),
            row=3, col=1, secondary_y=False,
        )
    if load is not None:
        fig.add_trace(
            go.Scatter(
                x=load[0], y=load[1], mode="lines", name="Загрузка ПЭД",
                line={"color": COLOR_LOAD, "width": 1.3},
                hovertemplate="%{x|%d.%m.%Y %H:%M}<br>Загрузка ПЭД: %{y:.1f} %<extra></extra>",
            ),
            row=3, col=1, secondary_y=False,
        )
    if temp_intake is not None:
        fig.add_trace(
            go.Scatter(
                x=temp_intake[0], y=temp_intake[1], mode="lines", name="Температура на приёме",
                line={"color": COLOR_TEMP_INTAKE, "width": 1.3, "dash": "dot"},
                hovertemplate="%{x|%d.%m.%Y %H:%M}<br>Темп. на приёме: %{y:.1f} °C<extra></extra>",
            ),
            row=3, col=1, secondary_y=True,
        )
    if temp_oil is not None:
        fig.add_trace(
            go.Scatter(
                x=temp_oil[0], y=temp_oil[1], mode="lines", name="Температура масла",
                line={"color": COLOR_TEMP_OIL, "width": 1.3, "dash": "dot"},
                hovertemplate="%{x|%d.%m.%Y %H:%M}<br>Темп. масла: %{y:.1f} °C<extra></extra>",
            ),
            row=3, col=1, secondary_y=True,
        )

    # Стык «после очистки» для остальных параметров (тот же вырез по времени, синхронно)
    add_cleaned(2, frequency_col, 1, "Гц", "Частота после очистки")
    add_cleaned(3, current_col, 2, "А", "Ток после очистки", secondary=False)
    add_cleaned(3, load_col, 1, "%", "Загрузка после очистки", secondary=False)
    add_cleaned(3, temp_intake_col, 1, "°C", "Темп. на приёме после очистки", secondary=True)
    add_cleaned(3, temp_oil_col, 1, "°C", "Темп. масла после очистки", secondary=True)

    shapes: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    first_ts = series["timestamp"].iloc[0]
    last_ts = series["timestamp"].iloc[-1]
    if row.anomaly_start is not None and first_ts < row.anomaly_start:
        shapes.append(
            {
                "type": "rect", "xref": "x", "yref": "paper", "layer": "below",
                "x0": first_ts.strftime("%Y-%m-%d %H:%M"), "x1": row.anomaly_start.strftime("%Y-%m-%d %H:%M"),
                "y0": 0, "y1": 1, "fillcolor": COLOR_PREFIX_ZONE, "line": {"width": 0},
            }
        )
        anomaly_zone_end = min(row.anomaly_end, last_ts) if row.anomaly_end is not None else last_ts
        shapes.append(
            {
                "type": "rect", "xref": "x", "yref": "paper", "layer": "below",
                "x0": row.anomaly_start.strftime("%Y-%m-%d %H:%M"), "x1": anomaly_zone_end.strftime("%Y-%m-%d %H:%M"),
                "y0": 0, "y1": 1, "fillcolor": COLOR_ANOMALY_ZONE, "line": {"width": 0},
            }
        )
        shapes.append(
            {
                "type": "line", "xref": "x", "yref": "paper",
                "x0": row.anomaly_start.strftime("%Y-%m-%d %H:%M"), "x1": row.anomaly_start.strftime("%Y-%m-%d %H:%M"),
                "y0": 0, "y1": 1, "line": {"color": COLOR_ANOMALY_LINE, "width": 1.6, "dash": "dash"},
            }
        )
        annotations.append(
            {
                "x": row.anomaly_start.strftime("%Y-%m-%d %H:%M"), "y": 1.04, "xref": "x", "yref": "paper",
                "text": "Начало аномалии (разметка эксперта)", "showarrow": False,
                "font": {"size": 11, "color": COLOR_ANOMALY_LINE}, "xanchor": "left",
            }
        )

    # Фиолетовая зона «предыдущая аномалия» — extra_intervals из alma_summary_overrides.json
    # (например приток в префиксе соляной скважины 3245(2): 17.09→04.10 до начала соли).
    for extra in row.extra_intervals:
        try:
            ex_start = pd.Timestamp(extra["start"])
            ex_end = pd.Timestamp(extra["end"])
        except (KeyError, ValueError):
            continue
        ex_x0 = max(ex_start, first_ts)
        ex_x1 = min(ex_end, last_ts)
        if ex_x1 <= ex_x0:
            continue
        ex_label = str(extra.get("anomaly_type", "аномалия"))
        shapes.append(
            {
                "type": "rect", "xref": "x", "yref": "paper", "layer": "below",
                "x0": ex_x0.strftime("%Y-%m-%d %H:%M"), "x1": ex_x1.strftime("%Y-%m-%d %H:%M"),
                "y0": 0, "y1": 1, "fillcolor": COLOR_PRITOK_PREV_ZONE,
                "line": {"color": COLOR_PRITOK_PREV_TEXT, "width": 1, "dash": "dot"},
            }
        )
        shapes.append(
            {
                "type": "line", "xref": "x", "yref": "paper",
                "x0": ex_x0.strftime("%Y-%m-%d %H:%M"), "x1": ex_x0.strftime("%Y-%m-%d %H:%M"),
                "y0": 0, "y1": 1, "line": {"color": COLOR_PRITOK_PREV_TEXT, "width": 1.4, "dash": "dash"},
            }
        )
        annotations.append(
            {
                "x": ex_x0.strftime("%Y-%m-%d %H:%M"), "y": 0.96, "xref": "x", "yref": "paper",
                "text": f"{ex_label} (предыдущая аномалия) {ex_start.strftime('%d.%m')}–{ex_end.strftime('%d.%m')}",
                "showarrow": False, "font": {"size": 10, "color": COLOR_PRITOK_PREV_TEXT}, "xanchor": "left",
            }
        )

    # Серые зоны влияния остановок (все классы): от падения частоты до восстановления частоты
    # И возврата давления к базе — полный охват повышения, пика и спада, только в префиксе.
    for zone in influence_zones:
        label = (
            f"Остановка {zone['core_start'].strftime('%d.%m %H:%M')}–{zone['core_end'].strftime('%H:%M')}, "
            f"давление (+{zone['excess_pct']:.0f}%) вернулось к базе к {zone['end'].strftime('%H:%M')}"
        )
        if not zone["pressure_recovered"]:
            label = (
                f"Остановка {zone['core_start'].strftime('%d.%m %H:%M')}–{zone['core_end'].strftime('%H:%M')}, "
                f"давление (+{zone['excess_pct']:.0f}%) к базе не вернулось"
            )
        zone_x1 = _clip_end(zone["end"], row.anomaly_start)
        shapes.append(
            {
                "type": "rect", "xref": "x", "yref": "paper", "layer": "below",
                "x0": zone["start"].strftime("%Y-%m-%d %H:%M"), "x1": zone_x1.strftime("%Y-%m-%d %H:%M"),
                "y0": 0, "y1": 1, "fillcolor": COLOR_PEAK_STOP_ZONE,
                "line": {"color": COLOR_PEAK_STOP_TEXT, "width": 1, "dash": "dot"},
            }
        )
        if len(influence_zones) <= 6:
            annotations.append(
                {
                    "x": zone["end"].strftime("%Y-%m-%d %H:%M"), "y": 0.5, "xref": "x", "yref": "paper",
                    "text": label, "showarrow": False, "textangle": -90,
                    "font": {"size": 10, "color": COLOR_PEAK_STOP_TEXT}, "xanchor": "left", "yanchor": "middle",
                }
            )

    # Оранжевые зоны пиков НЕ от остановок (приток): причина неизвестна, нужна проверка эксперта
    for peak in orange_peaks:
        period_label = f"{peak['start'].strftime('%d.%m %H:%M')}–{peak['end'].strftime('%H:%M')}"
        shapes.append(
            {
                "type": "rect", "xref": "x", "yref": "paper", "layer": "below",
                "x0": peak["start"].strftime("%Y-%m-%d %H:%M"), "x1": peak["end"].strftime("%Y-%m-%d %H:%M"),
                "y0": 0, "y1": 1, "fillcolor": COLOR_PEAK_OTHER_ZONE,
                "line": {"color": COLOR_PEAK_OTHER_TEXT, "width": 1, "dash": "dot"},
            }
        )
        if len(orange_peaks) <= 6:
            annotations.append(
                {
                    "x": peak["end"].strftime("%Y-%m-%d %H:%M"), "y": 0.5, "xref": "x", "yref": "paper",
                    "text": f"Пик +{peak['excess_pct']:.0f}% ({period_label}) — НЕ остановка, причина неизвестна",
                    "showarrow": False, "textangle": -90,
                    "font": {"size": 10, "color": COLOR_PEAK_OTHER_TEXT}, "xanchor": "left", "yanchor": "middle",
                }
            )

    # Фиолетовые линии скачков частоты: реакция давления в этот момент — штатная, не аномалия
    for jump_ts, jump_delta in freq_jumps:
        shapes.append(
            {
                "type": "line", "xref": "x", "yref": "paper",
                "x0": jump_ts.strftime("%Y-%m-%d %H:%M"), "x1": jump_ts.strftime("%Y-%m-%d %H:%M"),
                "y0": 0, "y1": 1,
                "line": {"color": COLOR_FREQ_JUMP_LINE, "width": 1.0, "dash": "dot"},
            }
        )
        if len(freq_jumps) <= 8:
            annotations.append(
                {
                    "x": jump_ts.strftime("%Y-%m-%d %H:%M"), "y": 0.02, "xref": "x", "yref": "paper",
                    "text": f"частота {jump_delta:+.1f} Гц", "showarrow": False, "textangle": -90,
                    "font": {"size": 9, "color": COLOR_FREQ_JUMP_LINE}, "xanchor": "left", "yanchor": "bottom",
                }
            )

    axis_style = {
        "showgrid": True, "gridcolor": "#eef2f7", "linecolor": "#cbd5e1",
        "ticks": "outside", "tickcolor": "#cbd5e1", "zeroline": False,
    }
    fig.update_layout(
        height=560,
        margin={"l": 64, "r": 56, "t": 26, "b": 40},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
        font={"family": "'Segoe UI', 'PT Sans', Arial, sans-serif", "size": 12, "color": "#1e293b"},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.14, "x": 0, "font": {"size": 11}},
        shapes=shapes,
        annotations=annotations,
    )
    tickformat = {"tickformat": "%d.%m.%y"}
    fig.update_xaxes(**axis_style, **tickformat)
    fig.update_yaxes(**axis_style)
    fig.update_yaxes(title_text="кгс/см²", title_font={"size": 11, "color": "#64748b"}, row=1, col=1)
    fig.update_yaxes(title_text="Гц", title_font={"size": 11, "color": "#64748b"}, row=2, col=1)
    fig.update_yaxes(title_text="А / %", title_font={"size": 11, "color": "#64748b"}, row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="°C", title_font={"size": 11, "color": "#64748b"}, row=3, col=1, secondary_y=True)

    return {
        "json": fig.to_json(),
        "stops": stop_periods if row.class_key != "pritok" else [],
        "influence_zones": influence_zones,
        "jumps": freq_jumps,
        "peaks": orange_peaks,
    }


def parse_drift_pct(text: str) -> float | None:
    match = re.search(r"([+-]?\d+(?:\.\d+)?)%", text)
    if match is None:
        return None
    return float(match.group(1))


def prefix_verdict_kind(verdict: str) -> str:
    normalized = verdict.strip().lower()
    if normalized.startswith("чистый"):
        return "чистый"
    if "дрейф" in normalized or "размах" in normalized:
        return "дрейф/размах"
    return "короткий"


def build_prefix_drift_figure(rows: dict[int, WellRow]) -> str | None:
    items: list[tuple[str, float, str]] = []
    for row in sorted(rows.values(), key=lambda r: r.summary_row):
        if not row.prefix:
            continue
        drift = parse_drift_pct(row.prefix.get("Дрейф давления", ""))
        if drift is None:
            continue
        verdict = row.prefix.get("Вердикт", "")
        items.append((f"{row.well_label} (стр. {row.summary_row})", drift, verdict))
    if not items:
        return None

    labels = [item[0] for item in items]
    drifts = [item[1] for item in items]
    colors = []
    for _, drift, verdict in items:
        kind = prefix_verdict_kind(verdict)
        if kind == "чистый":
            colors.append("#10b981")
        elif abs(drift) >= 30:
            colors.append("#dc2626")
        else:
            colors.append("#f59e0b")

    fig = go.Figure(
        go.Bar(
            x=drifts, y=labels, orientation="h",
            marker={"color": colors},
            hovertemplate="%{y}<br>Дрейф давления: %{x:.1f}%<extra></extra>",
        )
    )
    fig.add_shape(type="line", x0=5, x1=5, y0=-0.5, y1=len(labels) - 0.5, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    fig.add_shape(type="line", x0=-5, x1=-5, y0=-0.5, y1=len(labels) - 0.5, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    fig.add_annotation(x=5, y=len(labels) - 0.5, text="порог нормы ±5%", showarrow=False, yanchor="bottom", font={"size": 10, "color": "#64748b"})
    fig.update_layout(
        height=max(420, 22 * len(labels) + 120),
        margin={"l": 170, "r": 30, "t": 30, "b": 40},
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font={"family": "'Segoe UI', 'PT Sans', Arial, sans-serif", "size": 11, "color": "#1e293b"},
        xaxis={"title": "Дрейф давления за префикс, %", "showgrid": True, "gridcolor": "#eef2f7", "zeroline": True, "zerolinecolor": "#94a3b8"},
        yaxis={"showgrid": False, "autorange": "reversed"},
        showlegend=False,
    )
    return fig.to_json()


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

REPORT_CSS = """
<style>
:root {
  --фон: #f1f5f9; --карточка: #ffffff; --текст: #0f172a; --вторичный: #64748b;
  --линия: #e2e8f0; --акцент: #1d4ed8; --зелёный: #059669; --красный: #dc2626; --янтарный: #d97706;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 16px; }
body {
  margin: 0; background: var(--фон); color: var(--текст);
  font-family: 'Segoe UI', 'PT Sans', Arial, sans-serif; font-size: 15px; line-height: 1.55;
}
code { background: #eef2ff; border-radius: 4px; padding: 1px 5px; font-size: 0.9em; }
a { color: var(--акцент); text-decoration: none; }
a:hover { text-decoration: underline; }

.каркас { display: flex; min-height: 100vh; }
.боковая {
  width: 270px; flex-shrink: 0; position: sticky; top: 0; height: 100vh; overflow-y: auto;
  background: #0f172a; color: #cbd5e1; padding: 22px 0 40px 0;
}
.боковая .лого { padding: 0 20px 14px 20px; font-size: 15px; font-weight: 700; color: #ffffff; border-bottom: 1px solid #1e293b; }
.боковая .лого span { display: block; font-size: 11.5px; font-weight: 400; color: #94a3b8; margin-top: 4px; }
.боковая nav { margin-top: 12px; }
.боковая nav a {
  display: block; padding: 6px 20px; color: #cbd5e1; font-size: 13px; border-left: 3px solid transparent;
}
.боковая nav a.вложенный { padding-left: 36px; font-size: 12.5px; color: #94a3b8; }
.боковая nav a:hover { background: #1e293b; text-decoration: none; }
.боковая nav a.активный { border-left-color: #60a5fa; background: #1e293b; color: #ffffff; }

.содержимое { flex-grow: 1; padding: 28px 36px 80px 36px; max-width: 1280px; }

.шапка { margin-bottom: 28px; }
.шапка h1 { font-size: 26px; margin: 0 0 8px 0; }
.шапка .подзаголовок { color: var(--вторичный); font-size: 14px; }

section.раздел { margin-bottom: 44px; }
section.раздел > h2 {
  font-size: 21px; margin: 0 0 16px 0; padding-left: 12px; border-left: 4px solid var(--акцент);
}
section.раздел > h3 { font-size: 17px; margin: 22px 0 10px 0; }

.панель { background: var(--карточка); border: 1px solid var(--линия); border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; }
.панель h3 { margin-top: 0; }
.панель h4 { margin: 18px 0 8px 0; font-size: 15px; }

.счётчики { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 16px; }
.счётчик {
  background: var(--карточка); border: 1px solid var(--линия); border-radius: 12px;
  padding: 16px 22px; min-width: 150px; flex-grow: 1;
}
.счётчик .число { font-size: 30px; font-weight: 700; line-height: 1.1; }
.счётчик .метка { font-size: 13px; color: var(--вторичный); margin-top: 4px; }

.бейдж {
  display: inline-block; padding: 2px 11px; border-radius: 999px; font-size: 12px; font-weight: 600;
  color: #ffffff; vertical-align: middle; white-space: nowrap;
}

.легенда-зон { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 10px; }
.легенда-зон .элемент { display: flex; align-items: center; gap: 8px; font-size: 13.5px; }
.легенда-зон .образец { width: 34px; height: 16px; border-radius: 4px; display: inline-block; }

.table-scroll { overflow-x: auto; }
table.md-table, table.навигатор {
  border-collapse: collapse; width: 100%; font-size: 13.5px; background: var(--карточка);
}
table.md-table th, table.навигатор th {
  background: #f8fafc; text-align: left; padding: 9px 12px; border: 1px solid var(--линия);
  font-size: 12.5px; position: sticky; top: 0;
}
table.md-table td, table.навигатор td { padding: 8px 12px; border: 1px solid var(--линия); vertical-align: top; }
table.md-table tbody tr:nth-child(even) { background: #fafbfd; }
table.навигатор tbody tr { cursor: pointer; }
table.навигатор tbody tr:hover { background: #eff6ff; }

.фильтры { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; align-items: center; }
.фильтры .кнопка-класс {
  border: 1px solid var(--линия); background: var(--карточка); border-radius: 999px;
  padding: 5px 14px; font-size: 13px; cursor: pointer;
}
.фильтры .кнопка-класс.активный { background: #1d4ed8; color: #ffffff; border-color: #1d4ed8; }
.фильтры select, .фильтры input {
  border: 1px solid var(--линия); border-radius: 8px; padding: 6px 10px; font-size: 13px; background: var(--карточка);
}

.карточка-скважины {
  background: var(--карточка); border: 1px solid var(--линия); border-radius: 14px;
  padding: 20px 24px; margin-bottom: 26px;
}
.карточка-скважины .заголовок { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 6px; }
.карточка-скважины .заголовок h4 { margin: 0; font-size: 17px; }
.карточка-скважины .подпись { color: var(--вторичный); font-size: 13px; margin-bottom: 12px; }
.график { width: 100%; min-height: 560px; }
.график .заглушка {
  display: flex; align-items: center; justify-content: center; height: 560px;
  color: var(--вторичный); font-size: 14px; background: #f8fafc; border-radius: 10px;
}
.нет-данных {
  padding: 36px; text-align: center; color: var(--вторичный); background: #f8fafc; border-radius: 10px; font-size: 14px;
}
.пояснение-зон {
  margin-top: 8px; padding: 10px 14px; background: #f8fafc; border: 1px solid var(--линия);
  border-radius: 10px; font-size: 13px; color: #334155; line-height: 1.5;
}
.правило-чтения ol { padding-left: 22px; }
.правило-чтения li { margin-bottom: 8px; }

.детали { margin-top: 14px; }
.детали details { border: 1px solid var(--линия); border-radius: 10px; margin-bottom: 8px; background: #fbfcfe; }
.детали summary { padding: 10px 16px; cursor: pointer; font-weight: 600; font-size: 13.5px; }
.детали .тело { padding: 4px 16px 14px 16px; font-size: 13.5px; }
.детали .тело table { font-size: 13px; }

.физика-сетка { display: grid; grid-template-columns: 170px 1fr; gap: 6px 14px; font-size: 13.5px; }
.физика-сетка .имя { color: var(--вторичный); }

.вывод-блок {
  margin-top: 12px; padding: 12px 16px; border-radius: 10px; background: #eff6ff;
  border-left: 4px solid var(--акцент); font-size: 13.5px;
}

footer { color: var(--вторичный); font-size: 12.5px; margin-top: 60px; border-top: 1px solid var(--линия); padding-top: 16px; }

@media (max-width: 1100px) {
  .каркас { flex-direction: column; }
  .боковая { width: 100%; height: auto; position: static; }
  .содержимое { padding: 20px 16px 60px 16px; }
}
@media print {
  .боковая { display: none; }
  .карточка-скважины { break-inside: avoid; }
}
</style>
"""

REPORT_JS = """
<script>
(function () {
  var отрисованные = {};
  function отрисовать(контейнер) {
    var данные = document.getElementById(контейнер.dataset.figId);
    if (!данные) { return; }
    var спецификация = JSON.parse(данные.textContent);
    Plotly.newPlot(контейнер, спецификация.data, спецификация.layout, {
      responsive: true,
      displayModeBar: false,
      doubleClick: 'reset',
      scrollZoom: false
    });
    отрисованные[контейнер.id] = true;
  }
  function убрать(контейнер) {
    Plotly.purge(контейнер);
    контейнер.innerHTML = "<div class='заглушка'>График появится при прокрутке…</div>";
    отрисованные[контейнер.id] = false;
  }
  var наблюдатель = new IntersectionObserver(function (записи) {
    записи.forEach(function (запись) {
      var контейнер = запись.target;
      if (запись.isIntersecting && !отрисованные[контейнер.id]) {
        контейнер.innerHTML = "";
        отрисовать(контейнер);
      } else if (!запись.isIntersecting && отрисованные[контейнер.id]) {
        убрать(контейнер);
      }
    });
  }, { rootMargin: '1200px 0px 1200px 0px' });
  document.querySelectorAll('.график[data-fig-id]').forEach(function (контейнер) {
    наблюдатель.observe(контейнер);
  });

  // Фильтры навигационной таблицы
  var активныйКласс = 'все';
  function применитьФильтры() {
    var разбиение = document.getElementById('фильтр-разбиение').value;
    var вердикт = document.getElementById('фильтр-вердикт').value;
    var поиск = document.getElementById('фильтр-поиск').value.trim().toLowerCase();
    document.querySelectorAll('#таблица-навигатор tbody tr').forEach(function (строка) {
      var показать = (активныйКласс === 'все' || строка.dataset.class === активныйКласс)
        && (разбиение === 'все' || строка.dataset.split === разбиение)
        && (вердикт === 'все' || строка.dataset.verdict === вердикт)
        && (поиск === '' || строка.dataset.well.indexOf(поиск) !== -1);
      строка.style.display = показать ? '' : 'none';
    });
  }
  document.querySelectorAll('.кнопка-класс').forEach(function (кнопка) {
    кнопка.addEventListener('click', function () {
      document.querySelectorAll('.кнопка-класс').forEach(function (другая) { другая.classList.remove('активный'); });
      кнопка.classList.add('активный');
      активныйКласс = кнопка.dataset.value;
      применитьФильтры();
    });
  });
  ['фильтр-разбиение', 'фильтр-вердикт'].forEach(function (идентификатор) {
    var элемент = document.getElementById(идентификатор);
    if (элемент) { элемент.addEventListener('change', применитьФильтры); }
  });
  var полеПоиска = document.getElementById('фильтр-поиск');
  if (полеПоиска) { полеПоиска.addEventListener('input', применитьФильтры); }
  document.querySelectorAll('#таблица-навигатор tbody tr').forEach(function (строка) {
    строка.addEventListener('click', function () {
      var цель = document.getElementById(строка.dataset.target);
      if (цель) { цель.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
    });
  });

  // Подсветка активного раздела в оглавлении
  var ссылки = document.querySelectorAll('.боковая nav a[href^="#"]');
  var карта = {};
  ссылки.forEach(function (ссылка) { карта[ссылка.getAttribute('href').slice(1)] = ссылка; });
  var наблюдательРазделов = new IntersectionObserver(function (записи) {
    записи.forEach(function (запись) {
      var ссылка = карта[запись.target.id];
      if (ссылка && запись.isIntersecting) {
        ссылки.forEach(function (другая) { другая.classList.remove('активный'); });
        ссылка.classList.add('активный');
      }
    });
  }, { rootMargin: '-20% 0px -70% 0px' });
  document.querySelectorAll('section.раздел[id]').forEach(function (раздел) {
    наблюдательРазделов.observe(раздел);
  });
})();
</script>
"""


def class_badge(class_key: str, label: str | None = None) -> str:
    color = CLASS_COLORS.get(class_key, "#475569")
    return f"<span class='бейдж' style='background:{color}'>{escape(label or CLASS_LABELS.get(class_key, class_key))}</span>"


def split_badge(split: str) -> str:
    color = SPLIT_COLORS.get(split, "#475569")
    return f"<span class='бейдж' style='background:{color}'>{escape(SPLIT_LABELS.get(split, split))}</span>"


def verdict_badge(verdict: str) -> str:
    kind = prefix_verdict_kind(verdict)
    if kind == "чистый":
        color = "#10b981"
    elif kind == "короткий":
        color = "#64748b"
    else:
        color = "#f59e0b"
    return f"<span class='бейдж' style='background:{color}'>Префикс: {escape(verdict)}</span>"


def render_overview(rows: dict[int, WellRow], md_sections: list[MdSection]) -> str:
    counts: dict[str, int] = {}
    for row in rows.values():
        counts[row.class_key] = counts.get(row.class_key, 0) + 1
    n_train = sum(1 for row in rows.values() if row.split == "train")
    n_test = sum(1 for row in rows.values() if row.split == "test")
    n_excluded = sum(1 for row in rows.values() if row.split == "excluded")

    counters_html = "".join(
        f"<div class='счётчик'><div class='число' style='color:{CLASS_COLORS[key]}'>{counts.get(key, 0)}</div>"
        f"<div class='метка'>{escape(CLASS_LABELS[key])}</div></div>"
        for key in ("negermet", "pritok", "salt", "norm", "norm_freq")
    )
    counters_html += (
        f"<div class='счётчик'><div class='число'>{n_train}</div><div class='метка'>Обучающих случаев</div></div>"
        f"<div class='счётчик'><div class='число' style='color:#7c3aed'>{n_test}</div><div class='метка'>Отложенных проверочных</div></div>"
        f"<div class='счётчик'><div class='число' style='color:#9f1239'>{n_excluded}</div><div class='метка'>Исключено из набора</div></div>"
    )

    sources_section = find_section(md_sections, "Источники и границы")
    sources_html = md_block_to_html(sources_section.body) if sources_section else ""

    return (
        "<section class='раздел' id='обзор'>"
        "<h2>Обзор набора</h2>"
        f"<div class='счётчики'>{counters_html}</div>"
        f"<div class='панель'>{sources_html}</div>"
        "</section>"
    )


def render_legend() -> str:
    return (
        "<section class='раздел' id='как-читать'>"
        "<h2>Как читать отчёт</h2>"
        "<div class='панель'>"
        "<p>Каждой строке сводной таблицы соответствует карточка скважины с интерактивным графиком: "
        "верхний ярус — давление на приёме насоса (главный сигнал для всех классов), средний — выходная частота "
        "(контекст режима), нижний — токи, загрузка ПЭД и температуры (подтверждающие признаки). "
        "Графики можно приближать выделением области; двойной щелчок возвращает исходный масштаб. "
        "Щелчок по названию канала в легенде скрывает или показывает его.</p>"
        "<div class='легенда-зон'>"
        f"<div class='элемент'><span class='образец' style='background:{COLOR_PREFIX_ZONE};border:1px solid #10b981'></span> Нормальный префикс (до начала аномалии)</div>"
        f"<div class='элемент'><span class='образец' style='background:{COLOR_ANOMALY_ZONE};border:1px solid #ef4444'></span> Размеченная аномалия</div>"
        f"<div class='элемент'><span class='образец' style='border-top:2px dashed {COLOR_ANOMALY_LINE};height:0;margin-top:8px'></span> Начало аномалии по разметке эксперта</div>"
        f"<div class='элемент'><span class='образец' style='background:{COLOR_STOP_ZONE};border:1px dotted {COLOR_STOP_TEXT}'></span> Остановка насоса (частота ниже 1 Гц)</div>"
        f"<div class='элемент'><span class='образец' style='border-top:2px dotted {COLOR_FREQ_JUMP_LINE};height:0;margin-top:8px'></span> Скачок выходной частоты</div>"
        f"<div class='элемент'><span class='образец' style='background:{COLOR_PEAK_STOP_ZONE};border:1px dotted {COLOR_PEAK_STOP_TEXT}'></span> Остановка насоса и её влияние на давление (приток): от падения частоты до возврата давления к базе</div>"
        f"<div class='элемент'><span class='образец' style='background:{COLOR_PEAK_OTHER_ZONE};border:1px dotted {COLOR_PEAK_OTHER_TEXT}'></span> Пик давления НЕ от остановки (приток, причина неизвестна)</div>"
        f"<div class='элемент'><span class='образец' style='background:{COLOR_PRITOK_PREV_ZONE};border:1px dotted {COLOR_PRITOK_PREV_TEXT}'></span> Предыдущая аномалия другого класса в префиксе (например приток до начала соли — скв. 3245(2))</div>"
        "</div>"
        "<div class='правило-чтения'>"
        "<p><strong>Главное правило чтения графиков:</strong> рост давления — признак аномалии только тогда, когда насос работает, "
        "а частота не менялась. Два штатных случая, которые выглядят как аномалия, но ею не являются:</p>"
        "<ol>"
        "<li><strong>Пик давления внутри серой зоны (остановка насоса).</strong> Насос остановили — столб жидкости давит на датчик — "
        "давление растёт само по себе (гидростатическое восстановление). После запуска насоса давление возвращается. "
        "Пример: скв. 5021 — два пика +78% и +49% это именно остановки, а не приток.</li>"
        "<li><strong>Изменение давления сразу после фиолетовой линии (скачок частоты).</strong> Подняли частоту — насос откачивает "
        "быстрее — давление падает; снизили частоту — давление растёт. Это реакция на смену режима. "
        "Сила реакции у скважин разная: у эталона 1996л ~0.3% на 1 Гц, у 5021 — ~2% на 1 Гц.</li>"
        "</ol>"
        "<p style='margin-bottom:0'>Под каждым графиком — проверка физики из методологического документа: фактические изменения "
        "давления, частоты, токов и температур между нормальным окном и аномалией, и вывод о согласии с классом.</p>"
        "</div>"
        "</div>"
        "</section>"
    )


def render_class_physics(md_sections: list[MdSection], rows: dict[int, WellRow]) -> str:
    section = find_section(md_sections, "Физика классов")
    table_html = md_table_to_html(parse_md_table(section.body)) if section else ""

    examples_html = ""
    for class_key, (summary_row, description) in CANONICAL_EXAMPLES.items():
        if summary_row not in rows:
            continue
        row = rows[summary_row]
        examples_html += (
            "<div class='панель' style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>"
            f"{class_badge(class_key)}"
            f"<span style='flex-grow:1'>{escape(description)}</span>"
            f"<a href='#карточка-{summary_row}'>Смотреть график → строка {summary_row}, скв. {escape(row.well_label)}</a>"
            "</div>"
        )

    framework_section = find_section(md_sections, "Общая физическая рамка")
    features_section = find_section(md_sections, "Набор из 7 признаков модели")
    framework_html = md_block_to_html(framework_section.body) if framework_section else ""
    features_html = md_block_to_html(features_section.body) if features_section else ""

    return (
        "<section class='раздел' id='физика-классов'>"
        "<h2>Физика классов аномалий</h2>"
        f"{table_html}"
        "<h3>Канонические примеры на реальных рядах</h3>"
        f"{examples_html}"
        "</section>"
        "<section class='раздел' id='порядок-анализа'>"
        "<h2>Порядок экспертного анализа и признаки модели</h2>"
        f"<div class='панель'><h3>Порядок экспертного анализа сигналов</h3>{framework_html}</div>"
        f"<div class='панель'><h3>Набор из 7 признаков модели</h3>{features_html}</div>"
        "</section>"
    )


def navigator_row(row: WellRow) -> str:
    pressure_text = row.physics.get("Давление и частота", "")
    verdict = row.prefix.get("Вердикт", "") if row.prefix else ""
    verdict_kind = prefix_verdict_kind(verdict) if verdict else ""
    has_chart = row.series is not None
    target = f"карточка-{row.summary_row}"
    return (
        f"<tr data-class='{row.class_key}' data-split='{row.split}' "
        f"data-verdict='{escape(verdict_kind or 'нет')}' data-well='{escape(normalize_id(row.well_label))}' "
        f"data-target='{target}'>"
        f"<td>{row.summary_row}</td>"
        f"<td><strong>{escape(row.well_label)}</strong></td>"
        f"<td>{class_badge(row.class_key, row.class_label)}</td>"
        f"<td>{split_badge(row.split)}</td>"
        f"<td>{md_inline_to_html(pressure_text)}</td>"
        f"<td>{escape(verdict) if verdict else '—'}</td>"
        f"<td>{'есть' if has_chart else 'нет данных'}</td>"
        "</tr>"
    )


def render_navigator(rows: dict[int, WellRow]) -> str:
    body_html = "".join(navigator_row(row) for row in sorted(rows.values(), key=lambda r: r.summary_row))
    class_buttons = "<button class='кнопка-класс активный' data-value='все'>Все классы</button>" + "".join(
        f"<button class='кнопка-класс' data-value='{key}'>{escape(CLASS_LABELS[key])}</button>"
        for key in ("negermet", "pritok", "salt", "norm", "norm_freq")
    )
    return (
        "<section class='раздел' id='карта-скважин'>"
        "<h2>Карта скважин сводной</h2>"
        "<div class='панель'>"
        f"<div class='фильтры'>{class_buttons}"
        "<select id='фильтр-разбиение'>"
        "<option value='все'>Все разбиения</option>"
        "<option value='train'>Обучающие</option>"
        "<option value='test'>Отложенные проверочные</option>"
        "<option value='norm'>Источники нормы</option>"
        "<option value='excluded'>Исключённые</option>"
        "</select>"
        "<select id='фильтр-вердикт'>"
        "<option value='все'>Все вердикты префикса</option>"
        "<option value='чистый'>Чистый</option>"
        "<option value='дрейф/размах'>Дрейф или размах</option>"
        "<option value='короткий'>Короткий</option>"
        "</select>"
        "<input id='фильтр-поиск' type='text' placeholder='Поиск по номеру скважины…'>"
        "</div>"
        "<div class='table-scroll' style='max-height:560px;overflow-y:auto'>"
        "<table class='навигатор' id='таблица-навигатор'>"
        "<thead><tr><th>Строка</th><th>Скважина</th><th>Класс</th><th>Разбиение</th>"
        "<th>Давление и частота (норма → аномалия)</th><th>Вердикт префикса</th><th>График</th></tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table></div>"
        "<p style='color:var(--вторичный);font-size:13px;margin-bottom:0'>Щелчок по строке таблицы открывает карточку скважины с графиком.</p>"
        "</div>"
        "</section>"
    )


def render_physics_details(row: WellRow) -> str:
    if not row.physics:
        return ""
    fields = [
        ("Давление и частота", row.physics.get("Давление и частота", "")),
        ("Токи, загрузка, температуры", row.physics.get("Токи, загрузка, температуры", "")),
        ("Вибрации и дисбалансы", row.physics.get("Вибрации и дисбалансы", "")),
    ]
    grid_html = "".join(
        f"<div class='имя'>{escape(name)}</div><div>{md_inline_to_html(value)}</div>"
        for name, value in fields if value
    )
    conclusion = row.physics.get("Физический вывод и сверка с комментарием", "")
    return (
        "<details open><summary>Проверка физики по фактическому ряду</summary>"
        f"<div class='тело'><div class='физика-сетка'>{grid_html}</div>"
        f"<div class='вывод-блок'>{md_inline_to_html(conclusion)}</div></div>"
        "</details>"
    )


def render_prefix_details(row: WellRow) -> str:
    if not row.prefix:
        return ""
    fields = [
        ("Разбиение", row.prefix.get("Разбиение", "")),
        ("Длительность префикса", row.prefix.get("Длительность префикса", "")),
        ("Размах давления", row.prefix.get("Размах давления", "")),
        ("Дрейф давления", row.prefix.get("Дрейф давления", "")),
        ("Остановки", row.prefix.get("Остановки", "")),
        ("Скачок частоты", row.prefix.get("Скачок частоты", "")),
        ("Вердикт", row.prefix.get("Вердикт", "")),
    ]
    grid_html = "".join(
        f"<div class='имя'>{escape(name)}</div><div>{md_inline_to_html(value)}</div>"
        for name, value in fields if value
    )
    return (
        "<details><summary>Качество нормального префикса</summary>"
        f"<div class='тело'><div class='физика-сетка'>{grid_html}</div></div>"
        "</details>"
    )


def render_expert_comment_details(row: WellRow) -> str:
    comment = row.expert_comment.strip()
    usage = row.usage.strip()
    if not comment and not usage:
        return ""
    body = ""
    if comment:
        body += f"<p>{md_inline_to_html(comment)}</p>"
    if usage:
        body += f"<p style='color:var(--вторичный)'>{md_inline_to_html(usage)}</p>"
    return (
        "<details><summary>Комментарий из сводной и использование</summary>"
        f"<div class='тело'>{body}</div>"
        "</details>"
    )


def render_well_card(row: WellRow, figure: dict[str, Any] | None) -> str:
    badges = class_badge(row.class_key, row.class_label) + " " + split_badge(row.split)
    verdict = row.prefix.get("Вердикт", "") if row.prefix else ""
    if verdict:
        badges += " " + verdict_badge(verdict)

    stops = figure["stops"] if figure else []
    jumps = figure["jumps"] if figure else []
    influence_zones = figure.get("influence_zones", []) if figure else []
    other_peaks = figure.get("peaks", []) if figure else []
    if stops:
        badges += f" <span class='бейдж' style='background:{COLOR_STOP_TEXT}'>Остановок насоса: {len(stops)}</span>"
    if influence_zones:
        badges += f" <span class='бейдж' style='background:{COLOR_PEAK_STOP_TEXT}'>Остановок насоса: {len(influence_zones)}</span>"
    if jumps:
        badges += f" <span class='бейдж' style='background:{COLOR_FREQ_JUMP_LINE}'>Скачков частоты: {len(jumps)}</span>"
    if other_peaks:
        badges += f" <span class='бейдж' style='background:{COLOR_PEAK_OTHER_TEXT}'>Пиков неясной природы: {len(other_peaks)}</span>"

    period_text = ""
    if row.series is not None and not row.series.empty:
        start = row.series["timestamp"].iloc[0]
        end = row.series["timestamp"].iloc[-1]
        period_text = f"Период данных: {start.strftime('%d.%m.%Y')} — {end.strftime('%d.%m.%Y')}"
        if row.anomaly_start is not None:
            period_text += f" · Начало аномалии: {row.anomaly_start.strftime('%d.%m.%Y %H:%M')}"

    if figure is not None:
        fig_id = f"данные-графика-{row.summary_row}"
        chart_html = (
            f"<div class='график' id='график-{row.summary_row}' data-fig-id='{fig_id}'>"
            "<div class='заглушка'>График появится при прокрутке…</div></div>"
            f"<script type='application/json' id='{fig_id}'>{figure['json']}</script>"
        )
        zone_notes = []
        if stops:
            stop_list = ", ".join(s.strftime("%d.%m %H:%M") for s, _ in stops[:6])
            suffix = "…" if len(stops) > 6 else ""
            zone_notes.append(
                f"<strong style='color:{COLOR_STOP_TEXT}'>Серые зоны</strong> — остановки насоса ({stop_list}{suffix}): "
                "рост давления внутри них — гидростатическое восстановление, не аномалия."
            )
        if jumps:
            zone_notes.append(
                f"<strong style='color:{COLOR_FREQ_JUMP_LINE}'>Фиолетовые пунктирные линии</strong> — скачки выходной частоты: "
                "изменение давления сразу после такой линии (и в противоположную сторону) — штатная реакция на смену режима, не аномалия."
            )
        if influence_zones:
            zone_list = ", ".join(
                f"{zone['core_start'].strftime('%d.%m %H:%M')}–{zone['core_end'].strftime('%H:%M')} "
                f"(влияние до {zone['end'].strftime('%H:%M')}, +{zone['excess_pct']:.0f}%)"
                for zone in influence_zones[:6]
            )
            suffix = "…" if len(influence_zones) > 6 else ""
            zone_notes.append(
                f"<strong style='color:{COLOR_PEAK_STOP_TEXT}'>Серые зоны</strong> — остановки насоса и их влияние на давление ({zone_list}{suffix}): "
                "зона начинается с падения выходной частоты и заканчивается, когда частота восстановилась И давление вернулось к базе (допуск 5%). "
                "Весь горб давления внутри зоны — гидростатика и переходный процесс, исключается из статистики."
            )
        if other_peaks:
            peak_list = ", ".join(
                f"{peak['start'].strftime('%d.%m %H:%M')} (+{peak['excess_pct']:.0f}%)" for peak in other_peaks[:6]
            )
            suffix = "…" if len(other_peaks) > 6 else ""
            zone_notes.append(
                f"<strong style='color:{COLOR_PEAK_OTHER_TEXT}'>Оранжевые зоны</strong> — пики давления НЕ от остановки ({peak_list}{suffix}): "
                "насос в окне пика работал, причина пика неизвестна (КРС / промывка / замер?). "
                "Исключаются из статистики; природу нужно уточнить у эксперта."
            )
        if zone_notes:
            chart_html += "<div class='пояснение-зон'>" + "<br>".join(zone_notes) + "</div>"
    else:
        chart_html = (
            "<div class='нет-данных'>Ряд этой скважины отсутствует в текущем 5-минутном наборе. "
            "Проверка физики выполнена по исходному кэшу выгрузки.</div>"
        )

    details_html = render_physics_details(row) + render_expert_comment_details(row) + render_prefix_details(row)

    return (
        f"<div class='карточка-скважины' id='карточка-{row.summary_row}'>"
        f"<div class='заголовок'><h4>Строка {row.summary_row} · Скважина {escape(row.well_label)}</h4>{badges}</div>"
        f"<div class='подпись'>{escape(period_text)}</div>"
        f"{chart_html}"
        f"<div class='детали'>{details_html}</div>"
        "</div>"
    )


def render_well_groups(rows: dict[int, WellRow], figures: dict[int, dict[str, Any] | None]) -> str:
    groups = [
        ("карточки-негерметичность", "Негерметичность НКТ", ("negermet",)),
        ("карточки-приток", "Приток", ("pritok",)),
        ("карточки-соли", "Солеотложение", ("salt",)),
        ("карточки-норма", "Нормальная работа и изменение частоты", ("norm", "norm_freq")),
    ]
    html = ""
    for section_id, title, class_keys in groups:
        group_rows = [row for row in sorted(rows.values(), key=lambda r: r.summary_row) if row.class_key in class_keys]
        if not group_rows:
            continue
        cards = "".join(render_well_card(row, figures.get(row.summary_row)) for row in group_rows)
        html += (
            f"<section class='раздел' id='{section_id}'>"
            f"<h2>Карточки скважин — {escape(title)} ({len(group_rows)})</h2>"
            f"{cards}"
            "</section>"
        )
    return html


def render_prefix_section(md_sections: list[MdSection], drift_figure_json: str | None) -> str:
    reference_section = find_section(md_sections, "Эталонное распределение нормы")
    prefix_table_section = find_section(md_sections, "Префиксы 39 аномальных скважин")
    conclusions_section = find_section(md_sections, "Выводы по префиксам")
    frequency_section = find_section(md_sections, "Частотная проверка дрейфов")
    intro_section = find_section(md_sections, "Качество нормальных префиксов аномальных скважин")

    chart_html = ""
    if drift_figure_json is not None:
        chart_html = (
            "<h3>Дрейф давления по префиксам (наглядно)</h3>"
            "<div class='панель'>"
            "<div class='график' id='график-дрейфы' data-fig-id='данные-графика-дрейфы' style='min-height:420px'>"
            "<div class='заглушка'>График появится при прокрутке…</div></div>"
            f"<script type='application/json' id='данные-графика-дрейфы'>{drift_figure_json}</script>"
            "</div>"
        )

    parts = [
        "<section class='раздел' id='префиксы'>",
        "<h2>Качество нормальных префиксов аномальных скважин</h2>",
    ]
    if intro_section:
        parts.append(f"<div class='панель'>{md_block_to_html(intro_section.body)}</div>")
    if reference_section:
        parts.append(f"<h3>Эталонное распределение нормы (23 полностью нормальные скважины)</h3>{md_block_to_html(reference_section.body)}")
    parts.append(chart_html)
    if prefix_table_section:
        parts.append(f"<h3>Префиксы 39 аномальных скважин</h3>{md_block_to_html(prefix_table_section.body)}")
    if conclusions_section:
        parts.append(f"<h3>Выводы по префиксам</h3><div class='панель'>{md_block_to_html(conclusions_section.body)}</div>")
    parts.append("</section>")

    frequency_html = ""
    if frequency_section:
        frequency_html = (
            "<section class='раздел' id='частотная-проверка'>"
            "<h2>Частотная проверка дрейфов</h2>"
            f"{md_block_to_html(frequency_section.body)}"
            "</section>"
        )
    return "".join(parts) + frequency_html


def render_md_passthrough(section_id: str, title: str, md_content: str) -> str:
    if not md_content.strip():
        return ""
    return (
        f"<section class='раздел' id='{section_id}'>"
        f"<h2>{escape(title)}</h2>"
        f"<div class='панель'>{md_block_to_html(md_content)}</div>"
        "</section>"
    )


def render_sidebar() -> str:
    links = [
        ("обзор", "Обзор набора", False),
        ("как-читать", "Как читать отчёт", False),
        ("физика-классов", "Физика классов", False),
        ("порядок-анализа", "Порядок анализа и признаки", False),
        ("карта-скважин", "Карта скважин", False),
        ("карточки-негерметичность", "Негерметичность НКТ", True),
        ("карточки-приток", "Приток", True),
        ("карточки-соли", "Солеотложение", True),
        ("карточки-норма", "Нормальная работа", True),
        ("префиксы", "Качество префиксов", False),
        ("частотная-проверка", "Частотная проверка", False),
        ("ограничения", "Ограничения набора", False),
        ("тезисы", "Экспертные тезисы", False),
        ("меры", "Реализованные меры", False),
        ("выводы", "Выводы для модели", False),
        ("задачи-и-вопросы", "Задачи и вопросы", False),
        ("справка", "Справка и терминология", False),
    ]
    links_html = "".join(
        f"<a href='#{anchor}' class='{'вложенный' if nested else ''}'>{escape(title)}</a>"
        for anchor, title, nested in links
    )
    return (
        "<aside class='боковая'>"
        "<div class='лого'>Физика аномалий УЭЦН<span>Визуальный отчёт по сводной и экспертным комментариям</span></div>"
        f"<nav>{links_html}</nav>"
        "</aside>"
    )


def render_report(md_text: str, rows: dict[int, WellRow]) -> str:
    md_sections = split_md_sections(md_text)

    figures: dict[int, dict[str, Any] | None] = {}
    for summary_row, row in rows.items():
        figures[summary_row] = build_well_figure(row)
    drift_figure_json = build_prefix_drift_figure(rows)

    n_with_charts = sum(1 for value in figures.values() if value is not None)
    generated_at = datetime.now().strftime("%d.%m.%Y %H:%M")

    header_html = (
        "<div class='шапка'>"
        "<h1>Физика аномалий по скважинам из сводной — визуальный отчёт</h1>"
        f"<div class='подзаголовок'>Строк сводной: {len(rows)} · Графиков: {n_with_charts} · "
        f"Сформировано: {generated_at} · Источник: методологический документ проверки физики и фактические ряды телеметрии</div>"
        "</div>"
    )

    theses_section = find_section(md_sections, "Экспертные тезисы из переписки")
    limitations_content = section_with_subsections(md_sections, "Известные ограничения набора")
    measures_content = section_with_subsections(md_sections, "Реализованные меры")
    conclusions_section = find_section(md_sections, "Выводы для текущего")
    tasks_section = find_section(md_sections, "Отложенные инженерные задачи")
    questions_section = find_section(md_sections, "Что ещё нужно уточнить")
    terminology_section = find_section(md_sections, "Терминология")
    global_section = find_section(md_sections, "Зачем глобальный подход")

    tasks_and_questions = ""
    if tasks_section or questions_section:
        body = ""
        if tasks_section:
            body += f"<h3>Отложенные инженерные задачи (без эксперта)</h3>{md_block_to_html(tasks_section.body)}"
        if questions_section:
            body += f"<h3>Что ещё нужно уточнить у эксперта</h3>{md_block_to_html(questions_section.body)}"
        tasks_and_questions = (
            "<section class='раздел' id='задачи-и-вопросы'>"
            "<h2>Отложенные задачи и вопросы эксперту</h2>"
            f"<div class='панель'>{body}</div>"
            "</section>"
        )

    reference_block = ""
    if terminology_section or global_section:
        body = ""
        if terminology_section:
            body += f"<h3>Терминология</h3>{md_block_to_html(terminology_section.body)}"
        if global_section:
            body += f"<h3>Зачем глобальный подход соответствует физике</h3>{md_block_to_html(global_section.body)}"
        reference_block = (
            "<section class='раздел' id='справка'>"
            "<h2>Справка и терминология</h2>"
            f"<div class='панель'>{body}</div>"
            "</section>"
        )

    content_html = (
        header_html
        + render_overview(rows, md_sections)
        + render_legend()
        + render_class_physics(md_sections, rows)
        + render_navigator(rows)
        + render_well_groups(rows, figures)
        + render_prefix_section(md_sections, drift_figure_json)
        + render_md_passthrough("ограничения", "Известные ограничения набора и нерешённые вопросы", limitations_content)
        + render_md_passthrough("тезисы", "Экспертные тезисы из переписки", theses_section.body if theses_section else "")
        + render_md_passthrough("меры", "Реализованные меры", measures_content)
        + render_md_passthrough("выводы", "Выводы для текущего глобального детектора", conclusions_section.body if conclusions_section else "")
        + tasks_and_questions
        + reference_block
        + "<footer>Отчёт сформирован автоматически из методологического документа "
        "«Физика аномалий по скважинам из сводной» и фактических рядов телеметрии. "
        "Все графики работают без подключения к сети.</footer>"
    )

    plotly_js = get_plotlyjs()
    return (
        "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Физика аномалий по скважинам из сводной — визуальный отчёт</title>"
        + REPORT_CSS
        + f"<script>{plotly_js}</script>"
        + "</head><body>"
        + "<div class='каркас'>"
        + render_sidebar()
        + f"<main class='содержимое'>{content_html}</main>"
        + "</div>"
        + REPORT_JS
        + "</body></html>"
    )


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Визуальный отчёт по физике аномалий из сводной")
    parser.add_argument("--md", type=Path, default=MD_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    md_text = args.md.read_text(encoding="utf-8")

    rows = load_md_rows(md_text)
    print(f"Строк сводной в md: {len(rows)}")

    load_anomaly_cases(rows)
    load_norm_cases(rows)
    attach_extra_intervals(rows)
    load_excluded_1123l(rows)

    n_series = sum(1 for row in rows.values() if row.series is not None)
    missing = [f"{row.summary_row} ({row.well_label})" for row in rows.values() if row.series is None]
    print(f"Рядов загружено: {n_series} из {len(rows)}")
    if missing:
        print(f"Без рядов: {', '.join(sorted(missing))}")

    html = render_report(md_text, rows)
    output_path = ensure_parent(args.output)
    output_path.write_text(html, encoding="utf-8")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Отчёт: {output_path} ({size_mb:.1f} МБ)")


if __name__ == "__main__":
    main()
