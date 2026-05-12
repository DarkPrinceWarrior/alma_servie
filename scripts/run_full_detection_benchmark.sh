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
DETECTOR="${DETECTOR:-paano_shared}"

run_cmd() {
  echo
  echo "[$(date '+%F %T')] $*"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

selected_detector() {
  local anomaly="$1"
  local summary="artifacts/results/${anomaly}_benchmark_summary.json"
  if [[ ! -f "$summary" ]]; then
    echo "$DETECTOR"
    return
  fi
  "$PYTHON_BIN" - <<'PY' "$summary"
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload.get("selected_default_detector", "paano_shared"))
PY
}

echo "Log file: $LOG_FILE"
echo "Python: $PYTHON_BIN"
echo "Detector: $DETECTOR"
echo "Retune flag: $RETUNE_FLAG"
echo | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Python: $PYTHON_BIN" | tee -a "$LOG_FILE"
echo "Detector: $DETECTOR" | tee -a "$LOG_FILE"
echo "Retune flag: $RETUNE_FLAG" | tee -a "$LOG_FILE"

for anomaly in "${ANOMALIES[@]}"; do
  run_cmd "$PYTHON_BIN" "scripts/detection/detect_${anomaly}.py" --detector "$DETECTOR" "$RETUNE_FLAG"

  detector="$(selected_detector "$anomaly")"
  run_cmd "$PYTHON_BIN" "scripts/reports/generate_${anomaly}_paano_report.py" --detector "$detector"
  run_cmd "$PYTHON_BIN" scripts/evaluation/evaluate_onset_metrics.py --anomaly "$anomaly" --detector "$detector" --name "${anomaly}_${detector}"
done

echo
echo "Done. Log saved to $LOG_FILE"
