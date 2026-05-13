#!/usr/bin/env bash
set -u
GPU="$1"
shift
CLASSES="$@"
cd /root/projects/alma_servie
export CUDA_VISIBLE_DEVICES="${GPU}"
DONE_MARKER="runs/3w_v3_worker_${GPU}.done"
rm -f "${DONE_MARKER}"
OVERALL_RC=0
for C in $CLASSES; do
  echo "===== v3 worker ${GPU} class ${C}: detect (force) ====="
  uv run python scripts/detection/detect_3w.py --event-class "${C}" --force-retrain 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "V3_W${GPU}_CLASS_${C}_DETECT_FAILED rc=${RC}"
    OVERALL_RC=$RC
    continue
  fi
  echo "===== v3 worker ${GPU} class ${C}: tune+evaluate ====="
  uv run python scripts/evaluation/evaluate_3w_onset.py --event-class "${C}" --mode both 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "V3_W${GPU}_CLASS_${C}_EVAL_FAILED rc=${RC}"
    OVERALL_RC=$RC
    continue
  fi
  echo "===== v3 worker ${GPU} class ${C}: report ====="
  uv run python scripts/reports/generate_3w_report.py --event-class "${C}" --max-instances 40 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "V3_W${GPU}_CLASS_${C}_REPORT_FAILED rc=${RC}"
    OVERALL_RC=$RC
    continue
  fi
  echo "===== v3 worker ${GPU} class ${C}: DONE ====="
done
echo "V3_WORKER_${GPU}_FINISHED overall_rc=${OVERALL_RC}" > "${DONE_MARKER}"
exit ${OVERALL_RC}
