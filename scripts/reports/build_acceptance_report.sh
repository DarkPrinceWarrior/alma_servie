#!/usr/bin/env bash
# Сборка итогового отчёта по тестированию (.docx, Приложение к договору).
# Детекция для графиков детерминирована — запускается ТОЛЬКО если данных ещё нет
# (artifacts/_report_src сохраняется между прогонами). Правка текста отчёта → повторный
# запуск рендерит .docx без повторной детекции (~30 сек).
#   bash scripts/reports/build_acceptance_report.sh            # переиспользует данные
#   FORCE_DETECT=1 bash scripts/reports/build_acceptance_report.sh   # принудительная детекция
set +e
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SRC=artifacts/_report_src
WELLS=(42-713/pritok 42-723/pritok 45-790/pritok 46-806/pritok 48-812/pritok 524/negermet)

need=0
for w in "${WELLS[@]}"; do [ -f "$SRC/$w/source.parquet" ] || need=1; done
[ "${FORCE_DETECT:-0}" = "1" ] && need=1

if [ "$need" = "1" ]; then
  echo "================ ДЕТЕКЦИЯ для графиков (per-class авто) ================"
  mkdir -p data/raw/_rep_neg && cp -p data/67_days/29-524.xlsx data/raw/_rep_neg/524.xlsx
  uv run python scripts/detection/detect_test_wells_batch.py \
    --xlsx-dir data/raw/test_wells --output-root "$SRC" \
    --detector paano_global --anomalies pritok --use-population-memory-bank 2>&1 | tail -3
  uv run python scripts/detection/detect_test_wells_batch.py \
    --xlsx-dir data/raw/_rep_neg --output-root "$SRC" \
    --detector paano_global --anomalies negermet --use-population-memory-bank 2>&1 | tail -3
  rm -rf data/raw/_rep_neg
else
  echo "================ ДЕТЕКЦИЯ ПРОПУЩЕНА (данные есть в $SRC) ================"
fi

echo "================ СБОРКА .docx ================"
uv run --with python-docx --with matplotlib python scripts/reports/build_acceptance_report_docx.py 2>&1 | tail -4
echo "================ DONE ================"
