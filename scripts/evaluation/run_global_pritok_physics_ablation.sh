#!/usr/bin/env bash
# A/B/C-сравнение физики ПРИТОКА на ОДНОМ глобальном энкодере.
#
# Энкодер обучается на полном пуле всех классов (настоящий global, --no-retune,
# детерминированный seed), поэтому во всех вариантах он идентичен; различается
# только обработка притока. negermet/salt обязаны совпасть бит-в-бит во всех
# прогонах — это встроенная проверка чистоты сравнения.
#
#   V1 = global как есть (чистый нейро-скор нормы);
#   V2 = + ветка давления ДО порога (как у shared), sweep веса 0.0025/0.005/0.01;
#   V3 = V2(0.0025) + гейт decision layer ПОСЛЕ порога (reject реально снимает старт).
#
# Критерий выбора (зафиксирован с экспертом): max hit_rate при FAR/сутки <= V1,
# тай-брейк — меньшая задержка onset. Считаем по held-out test split притока.

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset ALMA_PAANO_INPUT_PADDING ALMA_GLOBAL_MEMORY_BANK_MODE

ROOT="${1:-artifacts/results/global_phys_ablation_$(date +%Y%m%d_%H%M%S)}"
COMMON="--patch-short 192 --patch-long 384 --common-source-freq 5min --include-norm-work --global-iters 200 --no-retune"

run() {  # name  VAR=val...
  local name="$1"; shift
  local out="$ROOT/$name"
  mkdir -p "$out"
  echo "================ ${name} :: $* ================"
  env "$@" uv run python scripts/evaluation/benchmark_global_normality_detector.py \
    --output-dir "$out" $COMMON 2>&1 | tee "$out/run.log"
  echo "---- done ${name} (exit ${PIPESTATUS[0]}) ----"
}

run V1_baseline             ALMA_GLOBAL_PRESSURE_BRANCH=0 ALMA_GLOBAL_DOMAIN_GATE=0
run V2_pressure_w0025       ALMA_GLOBAL_PRESSURE_BRANCH=1 ALMA_GLOBAL_PRESSURE_WEIGHT=0.0025 ALMA_GLOBAL_DOMAIN_GATE=0
run V2_pressure_w0050       ALMA_GLOBAL_PRESSURE_BRANCH=1 ALMA_GLOBAL_PRESSURE_WEIGHT=0.005  ALMA_GLOBAL_DOMAIN_GATE=0
run V2_pressure_w0100       ALMA_GLOBAL_PRESSURE_BRANCH=1 ALMA_GLOBAL_PRESSURE_WEIGHT=0.01   ALMA_GLOBAL_DOMAIN_GATE=0
run V3_pressure_gate_w0025  ALMA_GLOBAL_PRESSURE_BRANCH=1 ALMA_GLOBAL_PRESSURE_WEIGHT=0.0025 ALMA_GLOBAL_DOMAIN_GATE=1

echo "ALL DONE :: $ROOT"
