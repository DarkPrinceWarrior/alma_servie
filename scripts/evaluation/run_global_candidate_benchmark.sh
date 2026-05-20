#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export ALMA_RETUNE_MODE="${ALMA_RETUNE_MODE:-fast}"
export ALMA_OPTUNA_N_JOBS="${ALMA_OPTUNA_N_JOBS:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Exercise code defaults for the candidate contract instead of hiding them
# behind shell overrides.
unset ALMA_PAANO_INPUT_PADDING
unset ALMA_GLOBAL_MEMORY_BANK_MODE

output_dir="${1:-artifacts/results/global_normality_detector/5min_candidate_$(date +%Y%m%d_%H%M%S)}"

uv run python scripts/evaluation/benchmark_global_normality_detector.py \
  --output-dir "$output_dir" \
  --patch-short 192 \
  --patch-long 384 \
  --common-source-freq 5min \
  --include-norm-work \
  --global-iters 200
