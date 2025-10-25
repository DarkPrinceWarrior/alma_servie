"""
Compute anomaly metrics by comparing pre-window averages with anomaly-period averages.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG_PATH, load_config

logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    well: str
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    raw_row: Dict


def parse_datetime_range(value: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    pattern = r"(\d{2})\.(\d{2})\.(\d{2})\s+(\d{2}):(\d{2})\s*-\s*(\d{2})\.(\d{2})\.(\d{2})\s+(\d{2}):(\d{2})"
    match = re.search(pattern, str(value))
    if not match:
        return None, None
    day1, month1, year1, hour1, minute1, day2, month2, year2, hour2, minute2 = match.groups()
    start = pd.Timestamp(f"20{year1}-{month1}-{day1} {hour1}:{minute1}:00")
    end = pd.Timestamp(f"20{year2}-{month2}-{day2} {hour2}:{minute2}:00")
    return start, end


def load_anomaly_table(config: Dict, override_path: Optional[Path] = None) -> pd.DataFrame:
    workbook_path = override_path or Path(config["anomalies"]["source_workbook"])
    if not workbook_path.exists():
        raise FileNotFoundError(
            f"Anomaly workbook not found at {workbook_path}. "
            "Provide the file or use --source to override the path."
        )
    sheet = config["anomalies"]["sheet_name"]
    return pd.read_excel(workbook_path, sheet_name=sheet)


def extract_events(anomalies_df: pd.DataFrame) -> List[AnomalyEvent]:
    events: List[AnomalyEvent] = []
    for _, row in anomalies_df.iterrows():
        start, end = parse_datetime_range(row.get("Дата и время"))
        if start is None or end is None:
            continue
        events.append(
            AnomalyEvent(
                well=str(row.get("Скважина")),
                name=str(row.get("Название предупреждения")),
                start=start,
                end=end,
                raw_row=row.to_dict(),
            )
        )
    return events


def compute_percent_change(before: float, during: float) -> Optional[float]:
    if pd.isna(before) or pd.isna(during):
        return None
    if before == 0:
        if during == 0:
            return 0.0
        return 100.0
    return round(((during - before) / before) * 100.0, 2)


def compute_metrics_for_event(
    merged: pd.DataFrame,
    event: AnomalyEvent,
    config: Dict,
) -> Dict[str, Optional[float]]:
    anomalies_conf = config["anomalies"]
    pre_window_days = anomalies_conf.get("pre_window_days", 3)
    agzu_metrics = anomalies_conf.get("agzu_metrics", [])
    su_metrics = anomalies_conf.get("su_metrics", [])
    agzu_prefix = anomalies_conf.get("metrics_agzu_prefix", "АГЗУ")
    su_prefix = anomalies_conf.get("metrics_su_prefix", "СУ")

    df_well = merged[merged[config["su"]["well_column"]].astype(str) == event.well]
    if df_well.empty:
        return {}

    timestamp_col = "timestamp"
    df_well = df_well.sort_values(timestamp_col)

    pre_start = event.start - pd.Timedelta(days=pre_window_days)
    window_before = df_well[(df_well[timestamp_col] >= pre_start) & (df_well[timestamp_col] < event.start)]
    window_anomaly = df_well[(df_well[timestamp_col] >= event.start) & (df_well[timestamp_col] <= event.end)]

    metrics: Dict[str, Optional[float]] = {}

    def _window_mean(frame: pd.DataFrame, column: str) -> float:
        return pd.to_numeric(frame[column], errors="coerce").mean()

    for metric in su_metrics:
        if metric not in df_well.columns:
            continue
        before_mean = _window_mean(window_before, metric)
        during_mean = _window_mean(window_anomaly, metric)
        metrics[f"{su_prefix}_{metric}_изм%"] = compute_percent_change(before_mean, during_mean)

    for metric in agzu_metrics:
        if metric not in df_well.columns:
            continue
        before_mean = _window_mean(window_before, metric)
        during_mean = _window_mean(window_anomaly, metric)
        metrics[f"{agzu_prefix}_{metric}_изм%"] = compute_percent_change(before_mean, during_mean)

    metrics[f"{agzu_prefix}_точек_до"] = int(len(window_before))
    metrics[f"{agzu_prefix}_точек_во_время"] = int(len(window_anomaly))
    return metrics


def evaluate_conditions(metrics: Dict[str, Optional[float]], conditions: List[Dict]) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for condition in conditions:
        name = condition.get("name", "condition")
        rules = condition.get("rules", [])
        condition_met = True
        for rule in rules:
            metric_name = rule.get("metric")
            if metric_name not in metrics or metrics[metric_name] is None:
                condition_met = False
                break
            value = metrics[metric_name]
            operator = rule.get("operator", ">=")
            threshold = rule.get("threshold", 0)
            if operator == ">=":
                if not (value >= threshold):
                    condition_met = False
                    break
            elif operator == "<=":
                if not (value <= threshold):
                    condition_met = False
                    break
            elif operator == ">":
                if not (value > threshold):
                    condition_met = False
                    break
            elif operator == "<":
                if not (value < threshold):
                    condition_met = False
                    break
            else:
                condition_met = False
                break
        results[name] = condition_met
    return results


ConditionDict = Dict[str, Any]


@dataclass
class RuleDefinition:
    name: str
    label: str
    description: str = ""
    min_duration_hours: float = 1.0
    cooldown_hours: float = 0.0
    focus_metrics: List[str] = field(default_factory=list)
    all: List[ConditionDict] = field(default_factory=list)
    any: List[ConditionDict] = field(default_factory=list)
    none: List[ConditionDict] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "RuleDefinition":
        return cls(
            name=cfg["name"],
            label=cfg.get("label", cfg["name"]),
            description=cfg.get("description", ""),
            min_duration_hours=cfg.get("min_duration_hours", 1.0),
            cooldown_hours=cfg.get("cooldown_hours", 0.0),
            focus_metrics=cfg.get("focus_metrics", []),
            all=cfg.get("all", []),
            any=cfg.get("any", []),
            none=cfg.get("none", []),
        )


@dataclass
class DetectionResult:
    well: str
    rule_name: str
    rule_label: str
    rule_description: str
    start: pd.Timestamp
    end: pd.Timestamp
    duration_hours: float
    samples: int
    score: float
    peak_time: Optional[pd.Timestamp]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "well_number": self.well,
            "rule_name": self.rule_name,
            "rule_label": self.rule_label,
            "rule_description": self.rule_description,
            "start": self.start,
            "end": self.end,
            "duration_hours": self.duration_hours,
            "sample_count": self.samples,
            "score": self.score,
            "peak_time": self.peak_time,
        }
        record.update(self.details)
        return record


class RuleBasedDetector:
    def __init__(self, detection_conf: Dict[str, Any], rules_conf: List[Dict[str, Any]], well_column: str) -> None:
        if detection_conf is None:
            raise ValueError(
                "Отсутствует секция 'anomalies.detection' в конфигурации. "
                "Добавьте её в config/pipeline.yaml."
            )
        self.metrics: List[str] = detection_conf.get("metrics") or [
            "Выходная частота",
            "Давление на приеме насоса",
            "Ток фазы A",
            "Объемный дебит жидкости, м3/сут",
        ]
        self.window_hours: int = int(detection_conf.get("window_hours", 6))
        self.baseline_shift_hours: int = int(detection_conf.get("baseline_shift_hours", self.window_hours))
        self.min_valid_fraction: float = float(detection_conf.get("min_valid_fraction", 0.6))
        self.min_valid_fraction = max(0.0, min(1.0, self.min_valid_fraction))
        self.default_min_baseline_abs: float = float(detection_conf.get("default_min_baseline_abs", 0.0))
        self.min_baseline_abs: Dict[str, float] = {
            key: float(value) for key, value in detection_conf.get("min_baseline_abs", {}).items()
        }
        self.rules: List[RuleDefinition] = [RuleDefinition.from_config(rule_conf) for rule_conf in rules_conf]
        self.well_column = well_column
        self.default_sampling_hours: float = 1.0
        self._sampling_by_well: Dict[str, float] = {}
        if not self.rules:
            logger.warning("Аномальные правила не заданы (anomalies.rules). Детектор вернёт пустой результат.")

    def run(self, merged: pd.DataFrame) -> pd.DataFrame:
        if merged.empty:
            return pd.DataFrame()

        if "timestamp" not in merged.columns:
            raise ValueError("Merged dataset must contain 'timestamp' column.")

        merged = merged.copy()
        merged["timestamp"] = pd.to_datetime(merged["timestamp"])
        merged[self.well_column] = merged[self.well_column].astype(str)

        records: List[Dict[str, Any]] = []
        for well, df_well in merged.groupby(self.well_column):
            df_well = df_well.sort_values("timestamp")
            features = self._compute_features(well, df_well)
            if features is None or features.empty:
                continue
            for rule in self.rules:
                detections = self._evaluate_rule_for_well(well, df_well, features, rule)
                records.extend(result.to_record() for result in detections)

        if not records:
            return pd.DataFrame(columns=[
                "well_number",
                "rule_name",
                "rule_label",
                "rule_description",
                "start",
                "end",
                "duration_hours",
                "sample_count",
                "score",
                "peak_time",
            ])

        detections_df = pd.DataFrame(records)
        detections_df.sort_values(["start", "well_number", "rule_name"], inplace=True)
        detections_df.reset_index(drop=True, inplace=True)
        return detections_df

    def _compute_features(self, well: str, df_well: pd.DataFrame) -> Optional[pd.DataFrame]:
        df = df_well.set_index("timestamp")
        if df.empty:
            return None

        sampling_hours = self._infer_sampling_hours(df.index)
        self._sampling_by_well[well] = sampling_hours

        window = max(1, int(round(self.window_hours / sampling_hours)))
        baseline_shift = max(1, int(round(self.baseline_shift_hours / sampling_hours)))
        min_periods = max(1, int(math.ceil(window * self.min_valid_fraction)))

        features = pd.DataFrame(index=df.index)

        for metric in self.metrics:
            if metric not in df.columns:
                logger.debug("Metric '%s' отсутствует в данных, пропускаю её.", metric)
                continue
            series = pd.to_numeric(df[metric], errors="coerce")
            if series.isna().all():
                continue

            current_mean = series.rolling(window=window, min_periods=min_periods).mean()
            baseline_mean = series.shift(baseline_shift).rolling(window=window, min_periods=min_periods).mean()
            delta = current_mean - baseline_mean

            min_abs = self.min_baseline_abs.get(metric, self.default_min_baseline_abs)
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = baseline_mean.where(baseline_mean.abs() >= min_abs)
                pct_change = (delta / denom) * 100.0

            valid_count = series.rolling(window=window, min_periods=1).count()
            baseline_count = series.shift(baseline_shift).rolling(window=window, min_periods=1).count()
            valid_fraction = valid_count / window
            baseline_fraction = baseline_count / window
            current_std = series.rolling(window=window, min_periods=min_periods).std()
            current_min = series.rolling(window=window, min_periods=min_periods).min()
            current_max = series.rolling(window=window, min_periods=min_periods).max()
            current_range = current_max - current_min
            with np.errstate(divide="ignore", invalid="ignore"):
                current_cv = current_std / current_mean.abs()

            features[f"{metric}__pct_change"] = pct_change
            features[f"{metric}__delta"] = delta
            features[f"{metric}__current_mean"] = current_mean
            features[f"{metric}__baseline_mean"] = baseline_mean
            features[f"{metric}__valid_fraction"] = valid_fraction
            features[f"{metric}__baseline_fraction"] = baseline_fraction
            features[f"{metric}__missing_fraction"] = 1.0 - valid_fraction
            features[f"{metric}__current_std"] = current_std
            features[f"{metric}__current_min"] = current_min
            features[f"{metric}__current_max"] = current_max
            features[f"{metric}__current_range"] = current_range
            features[f"{metric}__current_cv"] = current_cv

        return features

    def _evaluate_rule_for_well(
        self,
        well: str,
        df_well: pd.DataFrame,
        features: pd.DataFrame,
        rule: RuleDefinition,
    ) -> List[DetectionResult]:
        if features.empty:
            return []

        mask = self._evaluate_conditions(features, rule)
        if mask is None or not mask.any():
            return []

        sampling_hours = self._sampling_by_well.get(well, self.default_sampling_hours)
        min_samples = max(1, int(math.ceil(rule.min_duration_hours / sampling_hours)))
        cooldown_samples = max(0, int(math.ceil(rule.cooldown_hours / sampling_hours)))
        segments = self._mask_to_segments(mask, min_samples, cooldown_samples)
        results: List[DetectionResult] = []

        for start, end in segments:
            segment = features.loc[start:end]
            if segment.empty:
                continue
            result = self._summarise_segment(well, rule, segment, sampling_hours)
            results.append(result)
        return results

    def _evaluate_conditions(self, features: pd.DataFrame, rule: RuleDefinition) -> Optional[pd.Series]:
        if features.empty:
            return None
        mask = pd.Series(True, index=features.index, dtype=bool)

        for cond in rule.all:
            cond_mask = self._condition_mask(features, cond)
            mask &= cond_mask

        if rule.any:
            any_mask = pd.Series(False, index=features.index, dtype=bool)
            for cond in rule.any:
                any_mask |= self._condition_mask(features, cond)
            mask &= any_mask

        for cond in rule.none:
            mask &= ~self._condition_mask(features, cond)

        return mask

    def _condition_mask(self, features: pd.DataFrame, cond: ConditionDict) -> pd.Series:
        feature_name = cond.get("feature")
        if feature_name not in features:
            return pd.Series(False, index=features.index, dtype=bool)
        series = features[feature_name]
        if cond.get("abs", False):
            series = series.abs()

        operator = cond.get("operator", ">=")
        value = cond.get("value")
        if operator == ">=":
            mask = series >= value
        elif operator == ">":
            mask = series > value
        elif operator == "<=":
            mask = series <= value
        elif operator == "<":
            mask = series < value
        elif operator == "between":
            lower = cond.get("min", -np.inf)
            upper = cond.get("max", np.inf)
            mask = series.between(lower, upper)
        elif operator == "outside":
            lower = cond.get("min", -np.inf)
            upper = cond.get("max", np.inf)
            mask = (series < lower) | (series > upper)
        elif operator == "==":
            mask = series == value
        elif operator == "!=":
            mask = series != value
        elif operator == "isna":
            mask = series.isna()
        elif operator == "notna":
            mask = series.notna()
        else:
            raise ValueError(f"Неизвестный оператор условия: {operator}")

        if cond.get("fill", False):
            fill_value = bool(cond["fill"])
            mask = mask.fillna(fill_value)
        else:
            mask = mask.fillna(False)
        return mask

    def _mask_to_segments(
        self,
        mask: pd.Series,
        min_samples: int,
        cooldown_samples: int,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        if mask.empty:
            return []

        values = mask.fillna(False).to_numpy(dtype=bool)
        index = mask.index.to_list()
        segments: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

        n = len(values)
        i = 0
        while i < n:
            if values[i]:
                start_idx = i
                while i < n and values[i]:
                    i += 1
                end_idx = i - 1
                length = end_idx - start_idx + 1
                if length >= min_samples:
                    segments.append((index[start_idx], index[end_idx]))
                    i += cooldown_samples
                    continue
            i += 1
        return segments

    def _summarise_segment(
        self,
        well: str,
        rule: RuleDefinition,
        segment: pd.DataFrame,
        sampling_hours: float,
    ) -> DetectionResult:
        start = segment.index[0]
        end = segment.index[-1]
        samples = len(segment)
        duration_hours = float(samples * sampling_hours)

        details: Dict[str, Any] = {}
        score_values: List[float] = []
        peak_time: Optional[pd.Timestamp] = None

        for metric in rule.focus_metrics:
            prefix = self._metric_prefix(metric)
            pct_col = f"{metric}__pct_change"
            delta_col = f"{metric}__delta"
            mean_col = f"{metric}__current_mean"
            std_col = f"{metric}__current_std"
            valid_col = f"{metric}__valid_fraction"

            if pct_col in segment:
                pct_series = segment[pct_col]
                if not pct_series.isna().all():
                    details[f"{prefix}__pct_mean"] = self._to_float(pct_series.mean())
                    details[f"{prefix}__pct_max"] = self._to_float(pct_series.max())
                    details[f"{prefix}__pct_min"] = self._to_float(pct_series.min())
                    score_values.append(abs(pct_series).mean())
                    if peak_time is None:
                        try:
                            peak_time = pct_series.abs().idxmax()
                        except ValueError:
                            peak_time = None

            if delta_col in segment:
                delta_series = segment[delta_col]
                if not delta_series.isna().all():
                    details[f"{prefix}__delta_mean"] = self._to_float(delta_series.mean())

            if mean_col in segment:
                mean_series = segment[mean_col]
                if not mean_series.isna().all():
                    details[f"{prefix}__current_mean"] = self._to_float(mean_series.mean())

            if std_col in segment:
                std_series = segment[std_col]
                if not std_series.isna().all():
                    details[f"{prefix}__std_mean"] = self._to_float(std_series.mean())

            if valid_col in segment:
                valid_series = segment[valid_col]
                if not valid_series.isna().all():
                    details[f"{prefix}__valid_fraction_mean"] = self._to_float(valid_series.mean())

        score = self._to_float(np.nanmean(score_values)) if score_values else float("nan")

        if peak_time is None and len(segment.index) > 0:
            peak_time = segment.index[min(len(segment.index) // 2, len(segment.index) - 1)]

        notes = []
        for metric in rule.focus_metrics:
            prefix = self._metric_prefix(metric)
            key = f"{prefix}__pct_mean"
            if key in details and not math.isnan(details[key]):
                notes.append(f"{metric}: Δ% ср.{details[key]:.1f}")
        if notes:
            details["notes"] = "; ".join(notes)

        details.setdefault("focus_metrics", ", ".join(rule.focus_metrics))

        return DetectionResult(
            well=well,
            rule_name=rule.name,
            rule_label=rule.label,
            rule_description=rule.description,
            start=start,
            end=end,
            duration_hours=duration_hours,
            samples=samples,
            score=score,
            peak_time=peak_time,
            details=details,
        )

    def _infer_sampling_hours(self, index: pd.Index) -> float:
        if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
            return self.default_sampling_hours
        diffs = index.to_series().diff().dropna().dt.total_seconds()
        if diffs.empty:
            return self.default_sampling_hours
        median_seconds = float(diffs.median())
        if median_seconds <= 0:
            return self.default_sampling_hours
        return median_seconds / 3600.0

    @staticmethod
    def _metric_prefix(metric: str) -> str:
        prefix = (
            metric.lower()
            .replace(" ", "_")
            .replace(",", "")
            .replace("/", "_")
            .replace("?", "")
        )
        return prefix

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")


def run_anomaly_analysis(config_path: Path, workbook_override: Optional[Path] = None) -> pd.DataFrame:
    config = load_config(config_path)
    processed_path = Path(config["paths"]["processed_dir"]) / "merged_hourly.parquet"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Не найден файл с объединёнными данными: {processed_path}. "
            "Выполните шаги clean/align перед детекцией аномалий."
        )

    merged = pd.read_parquet(processed_path)

    detector = RuleBasedDetector(
        detection_conf=config["anomalies"].get("detection"),
        rules_conf=config["anomalies"].get("rules", []),
        well_column=config["su"]["well_column"],
    )

    if workbook_override is not None:
        logger.info(
            "Получен параметр --source (%s), однако для правил детекции он не используется. "
            "Для анализа эталонных интервалов воспользуйтесь командой `python -m pipeline events`.",
            workbook_override,
        )

    detections = detector.run(merged)
    if not detections.empty:
        detections["rule_label"] = "Аномалия по условиям"
        detections["rule_description"] = ""

    reports_dir = Path(config["paths"]["reports_dir"]) / "anomalies"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "anomaly_analysis.parquet"
    detections.to_parquet(output_path, index=False)

    excel_path = output_path.with_suffix(".xlsx")
    detections.to_excel(excel_path, index=False)

    summary_records: List[Dict[str, Any]] = []
    for record in detections.to_dict(orient="records"):
        converted: Dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                converted[key] = value.isoformat()
            elif isinstance(value, (np.floating, float)):
                converted[key] = None if math.isnan(value) else float(value)
            else:
                converted[key] = value
        summary_records.append(converted)

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary_records, ensure_ascii=False, indent=2), encoding="utf-8")
    return detections


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect rule-based anomalies in merged well datasets.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to pipeline configuration.")
    parser.add_argument("--source", type=Path, help="Override path to anomaly workbook.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else None)
    result = run_anomaly_analysis(args.config, workbook_override=args.source)
    print(f"Detected {len(result)} anomaly segments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
