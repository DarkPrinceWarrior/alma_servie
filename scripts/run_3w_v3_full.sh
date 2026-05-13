#!/usr/bin/env bash
# v3 pipeline: keep all instances (no drop on canonical channel mismatch),
# balanced source-type split, NaN-safe scoring, full re-run across 5 GPUs.
set -uo pipefail
cd /root/projects/alma_servie

DONE_MARKER="runs/3w_v3.done"
LOG="runs/3w_v3.log"
rm -f "${DONE_MARKER}"

{
  echo "===== v3 step 1: rebuild ====="
  uv run python scripts/datasets/build_3w_dataset.py --config configs/3w_paano.json
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "REBUILD_FAILED rc=${RC}"
    echo "FAILED rc=${RC}" > "${DONE_MARKER}"
    exit ${RC}
  fi

  echo "===== v3 step 2: clear old encoders/scores/metrics ====="
  rm -f artifacts/3w/checkpoints/3w_class_*.pt
  rm -f artifacts/3w/scores/class_*_scores.parquet artifacts/3w/scores/class_*_predicted_starts.parquet
  rm -f artifacts/3w/metrics/class_*_selected.json artifacts/3w/metrics/class_*_metrics.json
  rm -f runs/3w_v3_worker_*.log runs/3w_v3_worker_*.done

  echo "===== v3 step 3: launch 5 v3 workers ====="
  declare -A WORKER_CLASSES
  WORKER_CLASSES[1]="1 6"
  WORKER_CLASSES[2]="2 7"
  WORKER_CLASSES[3]="3 8"
  WORKER_CLASSES[4]="4 9"
  WORKER_CLASSES[5]="5"
  for GPU in 1 2 3 4 5; do
    CLASSES="${WORKER_CLASSES[$GPU]}"
    SESSION="3w_v3_w${GPU}"
    WLOG="runs/3w_v3_worker_${GPU}.log"
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    tmux new -d -s "${SESSION}" "bash scripts/_3w_v3_worker.sh ${GPU} ${CLASSES} 2>&1 | tee ${WLOG}"
    echo "  started ${SESSION} GPU=${GPU} classes=${CLASSES}"
  done

  echo "===== v3 step 4: wait for workers ====="
  while true; do
    cnt=$(ls runs/3w_v3_worker_*.done 2>/dev/null | wc -l)
    if [ "$cnt" -ge 5 ]; then break; fi
    sleep 30
  done
  echo "all 5 workers done"

  echo "===== v3 step 5: aggregate ====="
  uv run python scripts/evaluation/aggregate_3w_phase05.py --out artifacts/results/3w_benchmark_summary_v3.json

  echo "===== v3 step 6: diagnostic snapshot ====="
  uv run python scripts/evaluation/diagnose_3w_train_test_gap.py --out docs/3w_phase05_diagnostic_v3.md

  echo "V3_FINISHED" > "${DONE_MARKER}"
  echo "===== DONE ====="
} 2>&1 | tee -a "${LOG}"
