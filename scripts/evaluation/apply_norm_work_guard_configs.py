from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_DIR = Path("artifacts/results/norm_work_false_alarms")
DEFAULT_DB_DIR = Path("db")
DEFAULT_ANOMALIES = ("negermet", "pritok", "salt")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_float(value: Any, default: float = float("inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _split_metrics(payload: dict[str, Any], variant: str, split: str = "all") -> dict[str, Any]:
    return dict(payload.get(variant, {}).get("splits", {}).get(split, {}))


def _norm_metrics(payload: dict[str, Any], variant: str) -> dict[str, Any]:
    return dict(payload.get(variant, {}).get("norm_summary", {}))


def _acceptance_decision(
    payload: dict[str, Any],
    *,
    p90_delay_ratio_tolerance: float,
) -> tuple[bool, list[str]]:
    saved_main = _split_metrics(payload, "saved")
    selected_main = _split_metrics(payload, "selected")
    saved_norm = _norm_metrics(payload, "saved")
    selected_norm = _norm_metrics(payload, "selected")

    reasons: list[str] = []
    accepted = True

    saved_hits = _safe_int(saved_main.get("hit_count"))
    selected_hits = _safe_int(selected_main.get("hit_count"))
    if selected_hits < saved_hits:
        accepted = False
        reasons.append(f"hit_count worsened: {selected_hits} < {saved_hits}")

    saved_norm_starts = _safe_int(saved_norm.get("total_false_alarm_starts"))
    selected_norm_starts = _safe_int(selected_norm.get("total_false_alarm_starts"))
    if selected_norm_starts > saved_norm_starts:
        accepted = False
        reasons.append(f"norm false starts worsened: {selected_norm_starts} > {saved_norm_starts}")

    saved_norm_far = _safe_float(saved_norm.get("false_alarm_starts_per_day"))
    selected_norm_far = _safe_float(selected_norm.get("false_alarm_starts_per_day"))
    if selected_norm_far > saved_norm_far:
        accepted = False
        reasons.append(f"norm FAR/day worsened: {selected_norm_far:.6f} > {saved_norm_far:.6f}")

    saved_p90 = _safe_float(saved_main.get("p90_delay_ratio"), 0.0)
    selected_p90 = _safe_float(selected_main.get("p90_delay_ratio"), float("inf"))
    allowed_p90 = saved_p90 * p90_delay_ratio_tolerance
    if selected_p90 > allowed_p90:
        accepted = False
        reasons.append(
            f"p90_delay_ratio exceeded tolerance: {selected_p90:.6f} > {allowed_p90:.6f}"
        )

    if accepted:
        reasons.append("accepted: hit-rate preserved, norm false alarms not worse, delay within tolerance")
    return accepted, reasons


def _summarize_one(
    anomaly_key: str,
    payload: dict[str, Any],
    *,
    p90_delay_ratio_tolerance: float,
) -> dict[str, Any]:
    accepted, reasons = _acceptance_decision(
        payload,
        p90_delay_ratio_tolerance=p90_delay_ratio_tolerance,
    )
    saved_main = _split_metrics(payload, "saved")
    selected_main = _split_metrics(payload, "selected")
    saved_norm = _norm_metrics(payload, "saved")
    selected_norm = _norm_metrics(payload, "selected")
    return {
        "anomaly": anomaly_key,
        "accepted": accepted,
        "reasons": reasons,
        "saved_config": payload.get("saved_config", {}),
        "selected_config": payload.get("selected_config", {}),
        "saved": {
            "hit_count": _safe_int(saved_main.get("hit_count")),
            "interval_count": _safe_int(saved_main.get("interval_count")),
            "p90_delay_ratio": _safe_float(saved_main.get("p90_delay_ratio")),
            "false_alarms_per_day": _safe_float(saved_main.get("false_alarms_per_day")),
            "norm_false_starts": _safe_int(saved_norm.get("total_false_alarm_starts")),
            "norm_far_per_day": _safe_float(saved_norm.get("false_alarm_starts_per_day")),
        },
        "selected": {
            "hit_count": _safe_int(selected_main.get("hit_count")),
            "interval_count": _safe_int(selected_main.get("interval_count")),
            "p90_delay_ratio": _safe_float(selected_main.get("p90_delay_ratio")),
            "false_alarms_per_day": _safe_float(selected_main.get("false_alarms_per_day")),
            "norm_false_starts": _safe_int(selected_norm.get("total_false_alarm_starts")),
            "norm_far_per_day": _safe_float(selected_norm.get("false_alarm_starts_per_day")),
        },
    }


def apply_configs(
    *,
    results_dir: Path,
    db_dir: Path,
    anomalies: tuple[str, ...],
    detector: str,
    p90_delay_ratio_tolerance: float,
    apply: bool,
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = results_dir / f"config_backup_{timestamp}"

    for anomaly_key in anomalies:
        result_path = results_dir / f"norm_guard_tuning_{anomaly_key}_{detector}.json"
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        payload = _load_json(result_path)
        decision = _summarize_one(
            anomaly_key,
            payload,
            p90_delay_ratio_tolerance=p90_delay_ratio_tolerance,
        )
        decisions.append(decision)

        if not apply or not decision["accepted"]:
            continue

        config_path = db_dir / f"{anomaly_key}_{detector}_config.json"
        if config_path.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            (backup_dir / config_path.name).write_text(
                config_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        _write_json(
            config_path,
            {
                "detector": detector,
                "config": decision["selected_config"],
                "norm_work_guard": {
                    "source": str(result_path),
                    "accepted_at": timestamp,
                    "p90_delay_ratio_tolerance": p90_delay_ratio_tolerance,
                    "saved": decision["saved"],
                    "selected": decision["selected"],
                },
            },
        )

    report = {
        "detector": detector,
        "applied": apply,
        "p90_delay_ratio_tolerance": p90_delay_ratio_tolerance,
        "backup_dir": str(backup_dir) if backup_dir.exists() else None,
        "decisions": decisions,
    }
    report_path = results_dir / "norm_guard_acceptance_summary.json"
    _write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Accept/reject norm_work guarded tuning configs and optionally apply accepted configs.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--detector", default="paano_shared")
    parser.add_argument("--anomalies", nargs="+", default=list(DEFAULT_ANOMALIES))
    parser.add_argument("--p90-delay-ratio-tolerance", type=float, default=1.15)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = apply_configs(
        results_dir=args.results_dir,
        db_dir=args.db_dir,
        anomalies=tuple(args.anomalies),
        detector=args.detector,
        p90_delay_ratio_tolerance=args.p90_delay_ratio_tolerance,
        apply=bool(args.apply),
    )
    for decision in report["decisions"]:
        status = "ACCEPT" if decision["accepted"] else "REJECT"
        print(
            f"{status} {decision['anomaly']}: "
            f"norm starts {decision['saved']['norm_false_starts']} -> "
            f"{decision['selected']['norm_false_starts']}, "
            f"p90 ratio {decision['saved']['p90_delay_ratio']:.4f} -> "
            f"{decision['selected']['p90_delay_ratio']:.4f}"
        )
        for reason in decision["reasons"]:
            print(f"  - {reason}")
    print(f"Report saved: {args.results_dir / 'norm_guard_acceptance_summary.json'}")


if __name__ == "__main__":
    main()
