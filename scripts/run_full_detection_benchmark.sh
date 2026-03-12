#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
RETUNE_FLAG="${RETUNE_FLAG:---retune}"
LOG_DIR="${LOG_DIR:-artifacts/logs}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/full_detection_benchmark_${RUN_TS}.log"

mkdir -p "$LOG_DIR"

ANOMALIES=(negermet pritok salt)
DETECTORS=(paano_feat pca_spe lof iforest fused tranad_global)

run_cmd() {
  echo
  echo "[$(date '+%F %T')] $*"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

echo "Log file: $LOG_FILE"
echo "Python: $PYTHON_BIN"
echo "Retune flag: $RETUNE_FLAG"
echo | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Python: $PYTHON_BIN" | tee -a "$LOG_FILE"
echo "Retune flag: $RETUNE_FLAG" | tee -a "$LOG_FILE"

for anomaly in "${ANOMALIES[@]}"; do
  for detector in "${DETECTORS[@]}"; do
    run_cmd "$PYTHON_BIN" "scripts/detection/detect_${anomaly}.py" --detector "$detector" "$RETUNE_FLAG"
  done
done

run_cmd "$PYTHON_BIN" scripts/reports/generate_negermet_paano_report.py --detector fused
run_cmd "$PYTHON_BIN" scripts/reports/generate_pritok_paano_report.py --detector fused
run_cmd "$PYTHON_BIN" scripts/reports/generate_salt_paano_report.py --detector fused

run_cmd "$PYTHON_BIN" scripts/evaluation/evaluate_onset_metrics.py --anomaly negermet --detector fused --name negermet_fused
run_cmd "$PYTHON_BIN" scripts/evaluation/evaluate_onset_metrics.py --anomaly pritok --detector fused --name pritok_fused
run_cmd "$PYTHON_BIN" scripts/evaluation/evaluate_onset_metrics.py --anomaly salt --detector fused --name salt_fused

echo
echo "Done. Log saved to $LOG_FILE"
