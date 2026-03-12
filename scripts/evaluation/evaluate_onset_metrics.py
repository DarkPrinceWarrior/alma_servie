from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.anomaly_specs import get_detection_spec
from alma_service.benchmark_metrics import (
    load_intervals,
    load_predicted_starts,
    load_scores,
    summarize_splits,
    write_interval_frame,
)
from alma_service.detection_artifacts import DEFAULT_DETECTOR, predicted_starts_path, scores_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate delay-aware onset detection metrics.")
    parser.add_argument("--anomaly", choices=["negermet", "pritok", "salt"], default=None)
    parser.add_argument("--detector", default=DEFAULT_DETECTOR, help="Detector key when --anomaly is used.")
    parser.add_argument("--intervals", default=None, help="Path to *_intervals.parquet")
    parser.add_argument("--predicted-starts", default=None, help="Path to *_predicted_starts.parquet")
    parser.add_argument("--scores", default=None, help="Path to *_scores.parquet")
    parser.add_argument("--prestart-hours", type=float, default=2.0, help="Early-detection tolerance for interval hit.")
    parser.add_argument("--name", default="run", help="Label for outputs")
    parser.add_argument("--output-prefix", default=None, help="If set, writes JSON summary and per-interval CSV")
    args = parser.parse_args()

    if args.anomaly:
        spec = get_detection_spec(args.anomaly)
        intervals_path = args.intervals or str(spec.dataset.intervals_path)
        predicted_path = args.predicted_starts or str(predicted_starts_path(spec, args.detector))
        scores_path_value = args.scores or str(scores_path(spec, args.detector))
    else:
        if not args.intervals or not args.predicted_starts:
            parser.error("Either --anomaly or both --intervals and --predicted-starts are required.")
        intervals_path = args.intervals
        predicted_path = args.predicted_starts
        scores_path_value = args.scores

    intervals = load_intervals(intervals_path)
    predictions = load_predicted_starts(predicted_path)
    scores = load_scores(scores_path_value)
    split_summaries, split_frames = summarize_splits(
        intervals=intervals,
        predictions=predictions,
        scores=scores,
        prestart_hours=args.prestart_hours,
    )

    summary = {
        "name": args.name,
        "anomaly": args.anomaly,
        "detector": args.detector if args.anomaly else None,
        "prestart_hours": args.prestart_hours,
        "splits": split_summaries,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_prefix:
        prefix = Path(args.output_prefix)
        summary_path = prefix.with_suffix(".json")
        interval_path = prefix.with_name(prefix.name + "_intervals.parquet")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        interval_frame = split_frames.get("all", split_frames.get("train", split_frames.get("test", None)))
        if interval_frame is not None:
            write_interval_frame(interval_path, interval_frame)
        print(f"Saved: {summary_path}")
        if interval_frame is not None:
            print(f"Saved: {interval_path}")


if __name__ == "__main__":
    main()
