# ALMA cleanup inventory: `paano_shared` + `paano_global`

Дата: 2026-05-21.

Цель чистки: оставить в репозитории только актуальный рабочий контур вокруг
двух PaAno-детекторов:

- `paano_shared` — production/default;
- `paano_global` — production-candidate.

В cleanup не входят `db/`, `artifacts/`, `models/`, исходные Excel/Parquet
данные и backend/frontend. Они не являются кодовым мусором текущей задачи.

## Таблица решений

| Область | Файлы | Решение | Причина |
|---|---|---|---|
| Core library | `alma_service/generic_detection.py`, `generic_detectors.py`, `global_normality.py`, `shared_encoder.py`, `onset_detection.py`, `benchmark_metrics.py`, `prediction_postprocess.py`, `domain_decision_layer.py`, `domain_rule_diagnostics.py`, `feature_schema.py`, `engineered_features.py`, `dataset_*`, `paths.py`, `tabular_io.py` | Оставить | Общий runtime `paano_shared` / `paano_global` |
| Physical/domain layers | `pressure_trend.py`, `salt_trend.py`, `negermet_signature.py`, `anomaly_physics.py`, `operational_events.py`, `telemetry_status.py`, `zone_labels.py` | Оставить | Встроены в текущий score/domain decision layer |
| Dataset builders | `scripts/datasets/build_negermet_dataset.py`, `build_pritok_dataset.py`, `build_salt_dataset.py`, `build_norm_work_dataset.py` | Оставить | Нужны для текущих ALMA datasets |
| Current detection CLIs | `scripts/detection/detect_negermet.py`, `detect_pritok.py`, `detect_salt.py`, `build_shared_encoders.py`, `detect_uploaded_well.py` | Оставить | Рабочие entry points текущих детекторов |
| Current reports | `scripts/reports/generate_negermet_paano_report.py`, `generate_pritok_paano_report.py`, `generate_salt_paano_report.py`, `generate_feature_importance_report.py` | Оставить | Рабочие HTML/FI отчеты |
| Current evaluation | `evaluate_onset_metrics.py`, `benchmark_global_normality_detector.py`, `run_global_candidate_benchmark.sh`, `tune_saved_onset.py`, `tune_with_norm_work_guard.py`, `apply_norm_work_guard_configs.py`, `screen_norm_work_false_alarms.py`, `optuna_sweep_alma.py` | Оставить | Текущие метрики, global candidate benchmark, norm-work/domain guard support |
| Current configs | `configs/alma_global_feature_schema.json`, `configs/alma_global_normality_5min.json` | Оставить | Fixed schema и 5min global candidate |
| Current docs | `README.md`, `AGENTS.md`, `CLAUDE.md`, `docs/ALMA_система_детекции_аномалий.md`, `docs/ALMA_подготовка_данных_от_сырых_к_обучению.md`, `docs/pritok_pressure_trend_method.md` | Оставить/обновить | Документируют текущий контур |
| PaAno dependency | `paano/` | Оставить | Внешняя neural-библиотека, не трогать в cleanup |
| 3W research | `configs/3w_paano.json`, `scripts/*3w*`, `scripts/detection/*3w*`, `scripts/evaluation/*3w*`, `scripts/reports/generate_3w_report.py`, `docs/3w/*`, `docs/archive/*3w*` | Удалить | Старый research/transfer контур, не часть `paano_shared/global` runtime |
| External model experiments | `scripts/evaluation/train_dacad.py`, `_s4_swap_test.py`, old external adapters/benchmarks if present | Удалить | Сравнение сторонних моделей завершено, не production |
| Transfer/pretrain experiments | `benchmark_global_pretrain_finetune.py`, `compare_transfer_variants.py`, `transfer_3w_to_alma.py`, `alma_service/anomaly_injection.py` | Удалить | Не используется текущим `paano_shared/global`; synthetic injection был только для transfer experiments |
| Salym-only package pipeline | `alma_service/salym_raw_pipeline.py`, `scripts/salym/*`, `scripts/detection/screen_salym_unlabeled.py`, `scripts/reports/generate_salym_expert_package.py`, `scripts/build_salym_human_pdf.sh`, `docs/salym/*`, `docs/assets/salym_ws1333_gantt.png` | Удалить | Salym/test35 вне текущего scope и не участвует в двух PaAno-детекторах |
| One-off preprocessing audits | `scripts/datasets/audit_frequency_candidates.py`, `audit_preprocessing_defaults_and_channels.py` | Удалить | Выводы перенесены в текущий документ подготовки данных |
| Old notes | `docs/ALMA_class_specific_vs_global_pretrain.md`, `docs/ALMA_notes_from_pro55_experiments.md`, `docs/notes_from_pro5.5.md`, old archive baseline/roadmap notes | Удалить | Содержимое устарело и сведено в текущие документы |

## Не удалять в этой чистке

| Область | Почему |
|---|---|
| `db/`, `artifacts/`, `models/` | Runtime/generated outputs, gitignored |
| `data/raw/*`, `data/processed/*`, `data/reference/*` | Исходные и справочные данные, не кодовый мусор |
| `app/back`, `app/front` | Отдельный backend/frontend scope |
| `paano/` | Внешний dependency; менять только по отдельной задаче |

## Проверки после удаления

Минимальный набор:

```bash
uv run python -m py_compile alma_service/*.py scripts/datasets/*.py scripts/detection/*.py scripts/evaluation/*.py scripts/reports/*.py
uv run python scripts/detection/detect_negermet.py --detector paano_shared --help
uv run python scripts/detection/detect_negermet.py --detector paano_global --help
uv run python scripts/reports/generate_negermet_paano_report.py --help
bash -n scripts/run_full_dataset_build.sh scripts/run_full_detection_benchmark.sh scripts/evaluation/run_global_candidate_benchmark.sh
```
