#!/usr/bin/env bash
# Phase 3 (P10): transfer the strongest 3W encoder (class 9 by default) into
# each ALMA anomaly key, fine-tune, then run the production ALMA detection
# pipeline with the transferred weights and compare against the frozen
# baseline (baseline-pre-3w-2026-05-12).
#
# IMPORTANT: production ALMA encoders are preserved at
# models/baseline_2026-05-12_<anomaly>_paano_shared_encoder.pt and at the
# git tag baseline-pre-3w-2026-05-12. This script writes transferred
# encoders into models/<anomaly>_paano_shared_encoder.pt (the live path),
# but the rollback is the cp from baseline_2026-05-12_* documented in
# docs/baseline_state_2026-05-12.md.
set -uo pipefail
cd /root/projects/alma_servie

SOURCE_CLASS="${1:-9}"
GPU="${2:-1}"
ITERS="${3:-200}"

DONE_MARKER="runs/3w_transfer.done"
LOG="runs/3w_transfer.log"
rm -f "${DONE_MARKER}"
mkdir -p runs

echo "===== transfer step 0: snapshot ALMA encoders =====" | tee -a "${LOG}"
for A in negermet pritok salt; do
  if [ -f "models/${A}_paano_shared_encoder.pt" ]; then
    cp -p "models/${A}_paano_shared_encoder.pt" "models/pre_transfer_${A}_paano_shared_encoder.pt"
    echo "  snapshotted ${A}" | tee -a "${LOG}"
  fi
done

echo "===== transfer step 1: 3W class ${SOURCE_CLASS} -> each ALMA anomaly =====" | tee -a "${LOG}"
export CUDA_VISIBLE_DEVICES="${GPU}"
for A in negermet pritok salt; do
  echo "----- transfer 3W class ${SOURCE_CLASS} -> ${A} -----" | tee -a "${LOG}"
  uv run python scripts/evaluation/transfer_3w_to_alma.py \
    --source-class "${SOURCE_CLASS}" \
    --target-anomaly "${A}" \
    --fine-tune-iters "${ITERS}" \
    --verbose 2>&1 | tee -a "${LOG}"
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "TRANSFER_${A}_FAILED rc=${RC}" | tee -a "${LOG}"
  fi
done

echo "===== transfer step 2: run detection with transferred encoders =====" | tee -a "${LOG}"
for A in negermet pritok salt; do
  echo "----- detect ${A} with transferred encoder -----" | tee -a "${LOG}"
  uv run python "scripts/detection/detect_${A}.py" --detector paano_shared 2>&1 | tee -a "${LOG}"
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "DETECT_${A}_FAILED rc=${RC}" | tee -a "${LOG}"
  fi
done

echo "===== transfer step 3: snapshot transferred metrics =====" | tee -a "${LOG}"
mkdir -p artifacts/results/transfers/3w_class_${SOURCE_CLASS}
for A in negermet pritok salt; do
  for f in "artifacts/results/${A}_paano_shared_results.summary.json" "artifacts/results/${A}_benchmark_summary.json"; do
    if [ -f "$f" ]; then
      cp -p "$f" "artifacts/results/transfers/3w_class_${SOURCE_CLASS}/$(basename $f)"
    fi
  done
done

echo "===== transfer step 4: restore production encoders =====" | tee -a "${LOG}"
for A in negermet pritok salt; do
  if [ -f "models/baseline_2026-05-12_${A}_paano_shared_encoder.pt" ]; then
    cp -p "models/baseline_2026-05-12_${A}_paano_shared_encoder.pt" "models/${A}_paano_shared_encoder.pt"
    echo "  restored ${A} from frozen baseline" | tee -a "${LOG}"
  fi
done

echo "===== transfer step 5: re-run baseline detection for clean comparison =====" | tee -a "${LOG}"
mkdir -p artifacts/results/transfers/baseline_post_restore
for A in negermet pritok salt; do
  uv run python "scripts/detection/detect_${A}.py" --detector paano_shared 2>&1 | tee -a "${LOG}"
  for f in "artifacts/results/${A}_paano_shared_results.summary.json" "artifacts/results/${A}_benchmark_summary.json"; do
    if [ -f "$f" ]; then
      cp -p "$f" "artifacts/results/transfers/baseline_post_restore/$(basename $f)"
    fi
  done
done

echo "TRANSFER_FINISHED source_class=${SOURCE_CLASS}" > "${DONE_MARKER}"
echo "===== DONE =====" | tee -a "${LOG}"
