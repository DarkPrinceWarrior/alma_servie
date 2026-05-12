#!/usr/bin/env bash
# Worker script: detect + tune + evaluate + report for a sequence of 3W event classes.
# Usage: _3w_worker.sh GPU_ID CLASS_1 [CLASS_2 ...]
# Writes runs/3w_worker_${GPU}.done after all classes processed (success or partial).
set -u
GPU="$1"
shift
CLASSES="$@"

cd /root/projects/alma_servie
export CUDA_VISIBLE_DEVICES="${GPU}"
DONE_MARKER="runs/3w_worker_${GPU}.done"
rm -f "${DONE_MARKER}"

OVERALL_RC=0

for C in $CLASSES; do
  echo "===== worker ${GPU} class ${C}: detect ====="
  uv run python scripts/detection/detect_3w.py --event-class "${C}" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "WORKER_${GPU}_CLASS_${C}_DETECT_FAILED rc=${RC}"
    OVERALL_RC=$RC
    continue
  fi

  echo "===== worker ${GPU} class ${C}: tune+evaluate ====="
  uv run python scripts/evaluation/evaluate_3w_onset.py --event-class "${C}" --mode both 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "WORKER_${GPU}_CLASS_${C}_EVAL_FAILED rc=${RC}"
    OVERALL_RC=$RC
    continue
  fi

  echo "===== worker ${GPU} class ${C}: report ====="
  uv run python scripts/reports/generate_3w_report.py --event-class "${C}" --max-instances 40 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "WORKER_${GPU}_CLASS_${C}_REPORT_FAILED rc=${RC}"
    OVERALL_RC=$RC
    continue
  fi

  echo "===== worker ${GPU} class ${C}: DONE ====="
done

echo "WORKER_${GPU}_FINISHED overall_rc=${OVERALL_RC}" > "${DONE_MARKER}"
exit ${OVERALL_RC}
