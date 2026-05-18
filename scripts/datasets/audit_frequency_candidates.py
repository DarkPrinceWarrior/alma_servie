from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.paths import DB_DIR  # noqa: E402
from alma_service.tabular_io import read_table  # noqa: E402


RAW_CACHE_DIRS = {
    "negermet": DB_DIR / "raw_cache" / "negermet",
    "pritok": DB_DIR / "raw_cache" / "pritok",
    "salt": DB_DIR / "raw_cache" / "salt",
    "norm_work": DB_DIR / "raw_cache" / "norm_work",
}
CANDIDATE_FREQS = ("2min", "5min", "10min", "15min")


@dataclass(frozen=True)
class DatasetRawSummary:
    dataset: str
    files: int
    parameters: int
    parameter_series: int
    raw_points: int
    start: str | None
    end: str | None
    median_delta_seconds: float | None
    p10_delta_seconds: float | None
    p25_delta_seconds: float | None
    p75_delta_seconds: float | None
    p90_delta_seconds: float | None
    p95_delta_seconds: float | None
    p99_delta_seconds: float | None
    share_delta_lt_2min: float | None
    share_delta_lt_5min: float | None
    share_delta_lt_10min: float | None
    share_delta_lt_15min: float | None


@dataclass(frozen=True)
class FrequencyCandidateSummary:
    dataset: str
    frequency: str
    grid_points_total: int
    median_grid_points_per_well: float
    max_grid_points_per_well: int
    grid_to_raw_ratio: float
    fast_delta_share_below_frequency: float | None


@dataclass(frozen=True)
class ParameterDeltaSummary:
    dataset: str
    parameter: str
    series_count: int
    raw_points: int
    median_delta_seconds: float | None
    p90_delta_seconds: float | None
    share_delta_lt_5min: float | None
    share_delta_lt_10min: float | None


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _freq_seconds(freq: str) -> float:
    return float(pd.Timedelta(freq).total_seconds())


def _safe_quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _share_lt(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float((arr < threshold).mean())


def _read_cache(path: Path) -> pd.DataFrame:
    df = read_table(path, dtypes={"param_name": str}, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values(["param_name", "timestamp"]).reset_index(drop=True)


def _series_deltas_seconds(series_timestamps: pd.Series) -> list[float]:
    ts = pd.to_datetime(series_timestamps, errors="coerce").dropna().drop_duplicates().sort_values()
    if len(ts) < 2:
        return []
    deltas = ts.diff().dropna().dt.total_seconds()
    return [float(value) for value in deltas if value > 0]


def _dataset_files(dataset: str) -> list[Path]:
    root = RAW_CACHE_DIRS[dataset]
    if not root.exists():
        return []
    return sorted(root.glob("*.parquet"))


def _summarize_dataset(dataset: str, files: list[Path]) -> tuple[DatasetRawSummary, list[FrequencyCandidateSummary], list[ParameterDeltaSummary]]:
    all_deltas: list[float] = []
    raw_points = 0
    parameters: set[str] = set()
    parameter_series = 0
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    grid_points_by_freq: dict[str, list[int]] = {freq: [] for freq in CANDIDATE_FREQS}
    parameter_deltas: dict[str, list[float]] = {}
    parameter_points: dict[str, int] = {}
    parameter_series_count: dict[str, int] = {}

    for path in files:
        df = _read_cache(path)
        if df.empty:
            continue
        starts.append(df["timestamp"].min())
        ends.append(df["timestamp"].max())
        raw_points += int(df["timestamp"].notna().sum())
        union_start = df["timestamp"].min()
        union_end = df["timestamp"].max()
        for freq in CANDIDATE_FREQS:
            grid = pd.date_range(union_start.ceil(freq), union_end.floor(freq), freq=freq)
            grid_points_by_freq[freq].append(int(len(grid)))

        for parameter, part in df.groupby("param_name", sort=True):
            parameter_name = str(parameter)
            parameters.add(parameter_name)
            parameter_series += 1
            points = int(part["timestamp"].notna().sum())
            parameter_points[parameter_name] = parameter_points.get(parameter_name, 0) + points
            parameter_series_count[parameter_name] = parameter_series_count.get(parameter_name, 0) + 1
            deltas = _series_deltas_seconds(part["timestamp"])
            parameter_deltas.setdefault(parameter_name, []).extend(deltas)
            all_deltas.extend(deltas)

    dataset_summary = DatasetRawSummary(
        dataset=dataset,
        files=len(files),
        parameters=len(parameters),
        parameter_series=parameter_series,
        raw_points=raw_points,
        start=str(min(starts)) if starts else None,
        end=str(max(ends)) if ends else None,
        median_delta_seconds=_safe_quantile(all_deltas, 0.50),
        p10_delta_seconds=_safe_quantile(all_deltas, 0.10),
        p25_delta_seconds=_safe_quantile(all_deltas, 0.25),
        p75_delta_seconds=_safe_quantile(all_deltas, 0.75),
        p90_delta_seconds=_safe_quantile(all_deltas, 0.90),
        p95_delta_seconds=_safe_quantile(all_deltas, 0.95),
        p99_delta_seconds=_safe_quantile(all_deltas, 0.99),
        share_delta_lt_2min=_share_lt(all_deltas, _freq_seconds("2min")),
        share_delta_lt_5min=_share_lt(all_deltas, _freq_seconds("5min")),
        share_delta_lt_10min=_share_lt(all_deltas, _freq_seconds("10min")),
        share_delta_lt_15min=_share_lt(all_deltas, _freq_seconds("15min")),
    )

    frequency_summaries: list[FrequencyCandidateSummary] = []
    for freq, grid_points in grid_points_by_freq.items():
        total = int(sum(grid_points))
        frequency_summaries.append(
            FrequencyCandidateSummary(
                dataset=dataset,
                frequency=freq,
                grid_points_total=total,
                median_grid_points_per_well=float(np.median(grid_points)) if grid_points else 0.0,
                max_grid_points_per_well=int(max(grid_points)) if grid_points else 0,
                grid_to_raw_ratio=float(total / raw_points) if raw_points else 0.0,
                fast_delta_share_below_frequency=_share_lt(all_deltas, _freq_seconds(freq)),
            )
        )

    parameter_summaries: list[ParameterDeltaSummary] = []
    for parameter in sorted(parameter_deltas):
        deltas = parameter_deltas[parameter]
        parameter_summaries.append(
            ParameterDeltaSummary(
                dataset=dataset,
                parameter=parameter,
                series_count=parameter_series_count.get(parameter, 0),
                raw_points=parameter_points.get(parameter, 0),
                median_delta_seconds=_safe_quantile(deltas, 0.50),
                p90_delta_seconds=_safe_quantile(deltas, 0.90),
                share_delta_lt_5min=_share_lt(deltas, _freq_seconds("5min")),
                share_delta_lt_10min=_share_lt(deltas, _freq_seconds("10min")),
            )
        )

    return dataset_summary, frequency_summaries, parameter_summaries


def _recommendation(freq_summaries: list[FrequencyCandidateSummary], raw_summaries: list[DatasetRawSummary]) -> dict[str, Any]:
    by_freq: dict[str, list[FrequencyCandidateSummary]] = {}
    for item in freq_summaries:
        by_freq.setdefault(item.frequency, []).append(item)

    combined: list[dict[str, Any]] = []
    for freq in CANDIDATE_FREQS:
        items = by_freq.get(freq, [])
        combined.append(
            {
                "frequency": freq,
                "grid_points_total": int(sum(item.grid_points_total for item in items)),
                "max_dataset_grid_points": int(max((item.grid_points_total for item in items), default=0)),
                "mean_fast_delta_share_below_frequency": float(
                    np.mean([item.fast_delta_share_below_frequency for item in items if item.fast_delta_share_below_frequency is not None])
                )
                if items
                else None,
            }
        )

    return {
        "primary_recommendation": "5min",
        "control_frequency": "10min",
        "reason": (
            "5min is the first benchmark candidate: it is less destructive for fast negermet events than 10/15min, "
            "while avoiding the large 2min grid for salt/pritok/norm_work. 10min should be kept as a cheaper control."
        ),
        "combined": combined,
        "datasets": [asdict(item) for item in raw_summaries],
    }


def _format_seconds(value: float | None) -> str:
    if value is None:
        return ""
    if value >= 3600:
        return f"{value / 3600:.2f}h"
    if value >= 60:
        return f"{value / 60:.2f}m"
    return f"{value:.1f}s"


def _format_share(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.1f}%"


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    raw = report["raw_summaries"]
    freq = report["frequency_summaries"]
    params = report["parameter_summaries"]
    recommendation = report["recommendation"]

    lines: list[str] = []
    lines.append("# Аудит кандидатов общей частоты")
    lines.append("")
    lines.append("Файл сгенерирован скриптом `scripts/datasets/audit_frequency_candidates.py`.")
    lines.append("Анализ использует сырые cache-файлы `db/raw_cache/*/*.parquet`, то есть timestamp каждого параметра до регулярной сетки.")
    lines.append("")
    lines.append("## Сырые интервалы измерений")
    lines.append("")
    lines.append("| Датасет | Файлов | Параметр-рядов | Raw points | p10 | p25 | median | p75 | p90 | p95 | p99 | <5min | <10min | <15min |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in raw:
        lines.append(
            "| {dataset} | {files} | {series} | {points} | {p10} | {p25} | {p50} | {p75} | {p90} | {p95} | {p99} | {lt5} | {lt10} | {lt15} |".format(
                dataset=item["dataset"],
                files=item["files"],
                series=item["parameter_series"],
                points=item["raw_points"],
                p10=_format_seconds(item["p10_delta_seconds"]),
                p25=_format_seconds(item["p25_delta_seconds"]),
                p50=_format_seconds(item["median_delta_seconds"]),
                p75=_format_seconds(item["p75_delta_seconds"]),
                p90=_format_seconds(item["p90_delta_seconds"]),
                p95=_format_seconds(item["p95_delta_seconds"]),
                p99=_format_seconds(item["p99_delta_seconds"]),
                lt5=_format_share(item["share_delta_lt_5min"]),
                lt10=_format_share(item["share_delta_lt_10min"]),
                lt15=_format_share(item["share_delta_lt_15min"]),
            )
        )

    lines.append("")
    lines.append("## Размер регулярной сетки")
    lines.append("")
    lines.append("| Датасет | Частота | Grid points total | Median per well | Max per well | Grid/raw ratio | Доля raw-интервалов быстрее частоты |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for item in freq:
        lines.append(
            "| {dataset} | {frequency} | {total} | {median:.0f} | {max_points} | {ratio:.3f} | {fast} |".format(
                dataset=item["dataset"],
                frequency=item["frequency"],
                total=item["grid_points_total"],
                median=item["median_grid_points_per_well"],
                max_points=item["max_grid_points_per_well"],
                ratio=item["grid_to_raw_ratio"],
                fast=_format_share(item["fast_delta_share_below_frequency"]),
            )
        )

    lines.append("")
    lines.append("## Рекомендация")
    lines.append("")
    lines.append(f"Первый кандидат для общего benchmark: `{recommendation['primary_recommendation']}`.")
    lines.append(f"Контрольная частота: `{recommendation['control_frequency']}`.")
    lines.append("")
    lines.append(
        "Логика: `2min` лучше сохраняет быстрые события, но сильно раздувает сетку для соли/притока/нормальной работы. "
        "`10min` дешевле и совпадает с текущим притоком/norm_work, но грубее для негермета. "
        "`15min` слишком грубая как единая частота. Поэтому первый честный компромисс для общей модели - `5min`, "
        "а `10min` нужно оставить как дешевый baseline-контроль."
    )

    lines.append("")
    lines.append("## Сводно по всем датасетам")
    lines.append("")
    lines.append("| Частота | Grid points total | Max dataset grid points | Mean share raw-интервалов быстрее частоты |")
    lines.append("|---|---:|---:|---:|")
    for item in recommendation["combined"]:
        lines.append(
            "| {frequency} | {total} | {max_dataset} | {share} |".format(
                frequency=item["frequency"],
                total=item["grid_points_total"],
                max_dataset=item["max_dataset_grid_points"],
                share=_format_share(item["mean_fast_delta_share_below_frequency"]),
            )
        )

    lines.append("")
    lines.append("## Параметры с наиболее частыми обновлениями")
    lines.append("")
    for dataset in sorted({item["dataset"] for item in params}):
        subset = [item for item in params if item["dataset"] == dataset and item["share_delta_lt_5min"] is not None]
        subset = sorted(subset, key=lambda item: item["share_delta_lt_5min"], reverse=True)[:8]
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| Параметр | Series | Raw points | Median delta | P90 delta | <5min | <10min |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for item in subset:
            lines.append(
                "| {parameter} | {series} | {points} | {median} | {p90} | {lt5} | {lt10} |".format(
                    parameter=item["parameter"],
                    series=item["series_count"],
                    points=item["raw_points"],
                    median=_format_seconds(item["median_delta_seconds"]),
                    p90=_format_seconds(item["p90_delta_seconds"]),
                    lt5=_format_share(item["share_delta_lt_5min"]),
                    lt10=_format_share(item["share_delta_lt_10min"]),
                )
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_report() -> dict[str, Any]:
    raw_summaries: list[DatasetRawSummary] = []
    frequency_summaries: list[FrequencyCandidateSummary] = []
    parameter_summaries: list[ParameterDeltaSummary] = []

    for dataset in RAW_CACHE_DIRS:
        files = _dataset_files(dataset)
        raw, freq, params = _summarize_dataset(dataset, files)
        raw_summaries.append(raw)
        frequency_summaries.extend(freq)
        parameter_summaries.extend(params)

    return {
        "raw_summaries": [asdict(item) for item in raw_summaries],
        "frequency_summaries": [asdict(item) for item in frequency_summaries],
        "parameter_summaries": [asdict(item) for item in parameter_summaries],
        "recommendation": _recommendation(frequency_summaries, raw_summaries),
    }


def main() -> None:
    output_json = Path("artifacts/analysis/frequency_candidates_audit.json")
    output_md = Path("artifacts/analysis/frequency_candidates_audit.md")
    report = build_report()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_markdown(report, output_md)
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")
    print(f"Recommended first benchmark frequency: {report['recommendation']['primary_recommendation']}")


if __name__ == "__main__":
    main()
