#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
LOG_DIR="${LOG_DIR:-artifacts/logs}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/full_dataset_build_${RUN_TS}.log"

NEGERMET_FREQ="${NEGERMET_FREQ:-15s}"
PRITOK_FREQ="${PRITOK_FREQ:-2min}"
SALT_FREQ="${SALT_FREQ:-2min}"

mkdir -p "$LOG_DIR"

run_cmd() {
  echo
  echo "[$(date '+%F %T')] $*"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

echo "Log file: $LOG_FILE"
echo "Python: $PYTHON_BIN"
echo "Negermet freq: $NEGERMET_FREQ"
echo "Pritok freq: $PRITOK_FREQ"
echo "Salt freq: $SALT_FREQ"
echo | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Python: $PYTHON_BIN" | tee -a "$LOG_FILE"
echo "Negermet freq: $NEGERMET_FREQ" | tee -a "$LOG_FILE"
echo "Pritok freq: $PRITOK_FREQ" | tee -a "$LOG_FILE"
echo "Salt freq: $SALT_FREQ" | tee -a "$LOG_FILE"

run_cmd "$PYTHON_BIN" scripts/datasets/build_negermet_dataset.py --freq "$NEGERMET_FREQ"
run_cmd "$PYTHON_BIN" scripts/datasets/build_pritok_dataset.py --freq "$PRITOK_FREQ"
run_cmd "$PYTHON_BIN" scripts/datasets/build_salt_dataset.py --freq "$SALT_FREQ"

echo
echo "Done. Log saved to $LOG_FILE"
