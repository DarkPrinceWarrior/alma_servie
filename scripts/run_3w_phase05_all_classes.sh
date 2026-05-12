#!/usr/bin/env bash
# Launch 3W PaAno Phase 0.5 across all event classes (1..9) in 5 parallel tmux workers,
# one worker per free GPU. Each worker chains: detect_3w -> evaluate (tune+eval) -> report
# for its assigned class subset, then writes runs/3w_worker_<GPU>.done as the completion marker.
set -euo pipefail

cd /root/projects/alma_servie

declare -A WORKER_CLASSES=(
  [1]="1 6"
  [2]="2 7"
  [3]="3 8"
  [4]="4 9"
  [5]="5"
)

mkdir -p runs

for GPU in 1 2 3 4 5; do
  CLASSES="${WORKER_CLASSES[$GPU]}"
  SESSION="3w_w${GPU}"
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  LOG="runs/3w_worker_${GPU}.log"
  DONE="runs/3w_worker_${GPU}.done"
  rm -f "${DONE}"
  CMD="cd /root/projects/alma_servie && \
    export CUDA_VISIBLE_DEVICES=${GPU} && \
    set -e && \
    for C in ${CLASSES}; do \
      echo \"=== worker${GPU} class \$C: detect ===\" && \
      uv run python scripts/detection/detect_3w.py --event-class \$C 2>&1 && \
      echo \"=== worker${GPU} class \$C: tune+eval ===\" && \
      uv run python scripts/evaluation/evaluate_3w_onset.py --event-class \$C --mode both 2>&1 && \
      echo \"=== worker${GPU} class \$C: report ===\" && \
      uv run python scripts/reports/generate_3w_report.py --event-class \$C 2>&1 ; \
    done ; \
    echo WORKER_${GPU}_DONE > ${DONE}"
  tmux new -d -s "${SESSION}" "${CMD} 2>&1 | tee ${LOG}"
  echo "[launcher] started ${SESSION} GPU=${GPU} classes=${CLASSES} -> ${LOG}"
done

echo
echo "All workers launched. Watch progress:"
echo "  tmux ls"
echo "  tail -f runs/3w_worker_*.log"
echo
echo "Completion markers: runs/3w_worker_<GPU>.done"
