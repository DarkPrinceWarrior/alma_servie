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
NEGERMET_DETECTORS=(${NEGERMET_DETECTORS:-pca_spe lof fused})
PRITOK_DETECTORS=(${PRITOK_DETECTORS:-pca_spe fused})
SALT_DETECTORS=(${SALT_DETECTORS:-pca_spe fused})

run_cmd() {
  echo
  echo "[$(date '+%F %T')] $*"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

selected_detector() {
  local anomaly="$1"
  local summary="artifacts/results/${anomaly}_benchmark_summary.json"
  if [[ ! -f "$summary" ]]; then
    echo "pca_spe"
    return
  fi
  "$PYTHON_BIN" - <<'PY' "$summary"
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload.get("selected_default_detector", "pca_spe"))
PY
}

detectors_for_anomaly() {
  local anomaly="$1"
  case "$anomaly" in
    negermet) printf '%s\n' "${NEGERMET_DETECTORS[@]}" ;;
    pritok) printf '%s\n' "${PRITOK_DETECTORS[@]}" ;;
    salt) printf '%s\n' "${SALT_DETECTORS[@]}" ;;
    *) return 1 ;;
  esac
}

echo "Log file: $LOG_FILE"
echo "Python: $PYTHON_BIN"
echo "Retune flag: $RETUNE_FLAG"
echo | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Python: $PYTHON_BIN" | tee -a "$LOG_FILE"
echo "Retune flag: $RETUNE_FLAG" | tee -a "$LOG_FILE"

for anomaly in "${ANOMALIES[@]}"; do
  while IFS= read -r detector; do
    run_cmd "$PYTHON_BIN" "scripts/detection/detect_${anomaly}.py" --detector "$detector" "$RETUNE_FLAG"
  done < <(detectors_for_anomaly "$anomaly")

  detector="$(selected_detector "$anomaly")"
  run_cmd "$PYTHON_BIN" "scripts/reports/generate_${anomaly}_paano_report.py" --detector "$detector"
  run_cmd "$PYTHON_BIN" scripts/evaluation/evaluate_onset_metrics.py --anomaly "$anomaly" --detector "$detector" --name "${anomaly}_${detector}"
done

echo
echo "Done. Log saved to $LOG_FILE"
