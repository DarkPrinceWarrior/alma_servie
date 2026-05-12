#!/usr/bin/env bash
# Rerun classes 3 and 4 which failed initially due to absent class==0 (normal) rows
# in train instances. Fix in detect_3w.py uses pre-undesirable points as fallback
# reference. Run after the main workers have freed up their GPUs.
set -u
cd /root/projects/alma_servie

CLASSES="${1:-3 4}"
GPU="${2:-1}"
export CUDA_VISIBLE_DEVICES="${GPU}"
LOG="runs/3w_rerun_failed.log"
DONE="runs/3w_rerun_failed.done"
rm -f "${DONE}"

OVERALL_RC=0
for C in $CLASSES; do
  echo "===== rerun class ${C}: detect (forced) ====="
  uv run python scripts/detection/detect_3w.py --event-class "${C}" --force-retrain 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "RERUN_CLASS_${C}_DETECT_FAILED rc=${RC}"
    OVERALL_RC=$RC
    continue
  fi
  echo "===== rerun class ${C}: tune+evaluate ====="
  uv run python scripts/evaluation/evaluate_3w_onset.py --event-class "${C}" --mode both 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "RERUN_CLASS_${C}_EVAL_FAILED rc=${RC}"
    OVERALL_RC=$RC
    continue
  fi
  echo "===== rerun class ${C}: report ====="
  uv run python scripts/reports/generate_3w_report.py --event-class "${C}" --max-instances 40 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "RERUN_CLASS_${C}_REPORT_FAILED rc=${RC}"
    OVERALL_RC=$RC
    continue
  fi
  echo "===== rerun class ${C}: DONE ====="
done
echo "RERUN_FINISHED overall_rc=${OVERALL_RC}" > "${DONE}"
exit ${OVERALL_RC}
