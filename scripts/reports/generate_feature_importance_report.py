"""
Generate feature importance report for any anomaly type and any detector.

Usage:
    python scripts/reports/generate_feature_importance_report.py --anomaly salt
    python scripts/reports/generate_feature_importance_report.py --anomaly negermet --detector pca_spe
    python scripts/reports/generate_feature_importance_report.py --anomaly pritok --detector fused --output /tmp/report.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alma_service.feature_importance import generate_feature_importance_report
from alma_service.detection_artifacts import DETECTOR_KEYS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Отчёт влияния признаков на детекцию аномалий (permutation importance + PCA-аналитика)"
    )
    parser.add_argument(
        "--anomaly",
        required=True,
        choices=["negermet", "pritok", "salt"],
        help="Тип аномалии",
    )
    parser.add_argument(
        "--detector",
        choices=sorted(DETECTOR_KEYS),
        default=None,
        help="Детектор (по умолчанию — выбранный из benchmark_summary)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь к выходному HTML-файлу",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Путь к исходному parquet-файлу данных",
    )
    args = parser.parse_args()

    generate_feature_importance_report(
        anomaly_key=args.anomaly,
        detector=args.detector,
        output_path=args.output,
        source_path=args.source,
        verbose=True,
    )


if __name__ == "__main__":
    main()
