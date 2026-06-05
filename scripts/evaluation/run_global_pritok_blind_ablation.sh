#!/usr/bin/env bash
# A/B/C-сравнение физики ПРИТОКА на 5 слепых тестовых скважинах через paano_global.
#
# Грузит сохранённый глобальный энкодер + population memory bank (штатный global
# blind-контракт, одинаков во всех вариантах). Различается только обработка
# притока: V1 чистый нейро-скор; V2 + ветка давления до порога (sweep веса);
# V3 + гейт decision layer. Скважины: 42-713, 42-723, 45-790, 46-806, 48-812.

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset ALMA_PAANO_INPUT_PADDING ALMA_GLOBAL_MEMORY_BANK_MODE

ROOT="${1:-artifacts/test_wells_phys_ablation}"
XLSX="${2:-data/raw/test_wells}"

run() {  # name  VAR=val...
  local name="$1"; shift
  local out="$ROOT/$name"
  mkdir -p "$out"
  echo "================ blind ${name} :: $* ================"
  env "$@" uv run python scripts/detection/detect_test_wells_batch.py \
    --detector paano_global --anomalies pritok \
    --use-population-memory-bank \
    --xlsx-dir "$XLSX" --output-root "$out" 2>&1 | tee "$out/run.log"
  echo "---- done blind ${name} (exit ${PIPESTATUS[0]}) ----"
}

run V1_baseline             ALMA_GLOBAL_PRESSURE_BRANCH=0 ALMA_GLOBAL_DOMAIN_GATE=0
run V2_pressure_w0025       ALMA_GLOBAL_PRESSURE_BRANCH=1 ALMA_GLOBAL_PRESSURE_WEIGHT=0.0025 ALMA_GLOBAL_DOMAIN_GATE=0
run V2_pressure_w0050       ALMA_GLOBAL_PRESSURE_BRANCH=1 ALMA_GLOBAL_PRESSURE_WEIGHT=0.005  ALMA_GLOBAL_DOMAIN_GATE=0
run V2_pressure_w0100       ALMA_GLOBAL_PRESSURE_BRANCH=1 ALMA_GLOBAL_PRESSURE_WEIGHT=0.01   ALMA_GLOBAL_DOMAIN_GATE=0
run V3_pressure_gate_w0025  ALMA_GLOBAL_PRESSURE_BRANCH=1 ALMA_GLOBAL_PRESSURE_WEIGHT=0.0025 ALMA_GLOBAL_DOMAIN_GATE=1

echo "ALL BLIND DONE :: $ROOT"
