#!/usr/bin/env bash
# Phase 2 orchestration: rebuild dataset with v2 logic (per-instance intersection
# + source-type balanced split), then rerun all 9 classes across 5 GPUs and
# aggregate. Writes runs/3w_full_v2.done when complete.
set -uo pipefail
cd /root/projects/alma_servie

DONE_MARKER="runs/3w_full_v2.done"
LOG="runs/3w_full_v2.log"
rm -f "${DONE_MARKER}"

echo "===== v2 step 1: rebuild dataset =====" | tee -a "${LOG}"
uv run python scripts/datasets/build_3w_dataset.py --config configs/3w_paano.json 2>&1 | tee -a "${LOG}"
RC=$?
if [ $RC -ne 0 ]; then
  echo "REBUILD_FAILED rc=${RC}" | tee -a "${LOG}"
  echo "FAILED rc=${RC}" > "${DONE_MARKER}"
  exit ${RC}
fi

echo "===== v2 step 2: clear old encoders/scores =====" | tee -a "${LOG}"
rm -f artifacts/3w/checkpoints/3w_class_*.pt
rm -f artifacts/3w/scores/class_*_scores.parquet artifacts/3w/scores/class_*_predicted_starts.parquet
rm -f artifacts/3w/metrics/class_*_selected.json artifacts/3w/metrics/class_*_metrics.json
rm -f runs/3w_v2_worker_*.log runs/3w_v2_worker_*.done

echo "===== v2 step 3: launch 5 workers =====" | tee -a "${LOG}"
declare -A WORKER_CLASSES=(
  [1]="1 6"
  [2]="2 7"
  [3]="3 8"
  [4]="4 9"
  [5]="5"
)
for GPU in 1 2 3 4 5; do
  CLASSES="${WORKER_CLASSES[$GPU]}"
  SESSION="3w_v2_w${GPU}"
  WLOG="runs/3w_v2_worker_${GPU}.log"
  WDONE="runs/3w_v2_worker_${GPU}.done"
  rm -f "${WDONE}" "${WLOG}"
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  tmux new -d -s "${SESSION}" "GPU_OUT=${GPU} bash scripts/_3w_v2_worker.sh ${GPU} ${CLASSES} 2>&1 | tee ${WLOG}"
  echo "  started ${SESSION} GPU=${GPU} classes=${CLASSES}" | tee -a "${LOG}"
done

echo "===== v2 step 4: wait for workers =====" | tee -a "${LOG}"
while true; do
  cnt=$(ls runs/3w_v2_worker_*.done 2>/dev/null | wc -l)
  if [ "$cnt" -ge 5 ]; then break; fi
  sleep 30
done
echo "all 5 workers done" | tee -a "${LOG}"

echo "===== v2 step 5: aggregate =====" | tee -a "${LOG}"
uv run python scripts/evaluation/aggregate_3w_phase05.py --out artifacts/results/3w_benchmark_summary_v2.json 2>&1 | tee -a "${LOG}"

echo "===== v2 step 6: diagnostic snapshot =====" | tee -a "${LOG}"
uv run python scripts/evaluation/diagnose_3w_train_test_gap.py --out docs/3w_phase05_diagnostic_v2.md 2>&1 | tee -a "${LOG}"

echo "PHASE2_FINISHED" > "${DONE_MARKER}"
echo "===== DONE =====" | tee -a "${LOG}"
