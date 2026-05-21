# ALMA cleanup inventory: `paano_shared` + `paano_global`

Дата: 2026-05-21.

Серверная рабочая копия: `/root/projects/alma_servie`.
Ветка: `app/webapp`.

Цель документа: держать один источник правды по чистке репозитория и серверной
рабочей копии после перехода к двум актуальным PaAno-контурам:

- `paano_shared` - production/default;
- `paano_global` - production-candidate.

## Короткий вывод

Git-кодовая база после cleanup приведена к текущему контуру:

- core/runtime вокруг `paano_shared` и `paano_global`;
- текущие dataset builders;
- текущие detection/report/evaluation scripts;
- текущие domain/physics/explainability layers;
- изолированный Salym raw pipeline:
  `alma_service/salym/raw_pipeline.py` и `scripts/salym/build_salym_raw.py`.

Старые research-ветки и одноразовые эксперименты удалены из Git. Серверная
рабочая копия дополнительно очищена от старых локальных leftovers: 3W,
runtime logs, `.claude/`, пустой `data/processed/` и архива test35.

## Таблица решений

| Область | Файлы | Решение | Причина |
|---|---|---|---|
| Core library | `alma_service/generic_detection.py`, `generic_detectors.py`, `global_normality.py`, `shared_encoder.py`, `onset_detection.py`, `benchmark_metrics.py`, `prediction_postprocess.py`, `domain_decision_layer.py`, `domain_rule_diagnostics.py`, `feature_schema.py`, `engineered_features.py`, `dataset_*`, `paths.py`, `tabular_io.py` | Оставить | Общий runtime `paano_shared` / `paano_global` |
| Physical/domain layers | `pressure_trend.py`, `salt_trend.py`, `negermet_signature.py`, `anomaly_physics.py`, `operational_events.py`, `telemetry_status.py`, `zone_labels.py` | Оставить | Встроены в текущий score/domain decision layer |
| Dataset builders | `scripts/datasets/build_negermet_dataset.py`, `build_pritok_dataset.py`, `build_salt_dataset.py`, `build_norm_work_dataset.py` | Оставить | Нужны для текущих ALMA datasets |
| Current detection CLIs | `scripts/detection/detect_negermet.py`, `detect_pritok.py`, `detect_salt.py`, `build_shared_encoders.py`, `detect_uploaded_well.py` | Оставить | Рабочие entry points текущих детекторов |
| Current reports | `scripts/reports/generate_negermet_paano_report.py`, `generate_pritok_paano_report.py`, `generate_salt_paano_report.py`, `generate_feature_importance_report.py` | Оставить | Рабочие HTML/FI отчёты |
| Current evaluation | `evaluate_onset_metrics.py`, `benchmark_global_normality_detector.py`, `run_global_candidate_benchmark.sh`, `tune_saved_onset.py`, `tune_with_norm_work_guard.py`, `apply_norm_work_guard_configs.py`, `screen_norm_work_false_alarms.py`, `optuna_sweep_alma.py` | Оставить | Текущие метрики, global candidate benchmark, norm-work/domain guard support |
| Current configs | `configs/alma_global_feature_schema.json`, `configs/alma_global_normality_5min.json` | Оставить | Fixed schema и 5min global candidate |
| Current docs | `README.md`, `AGENTS.md`, `CLAUDE.md`, `docs/ALMA_система_детекции_аномалий.md`, `docs/ALMA_подготовка_данных_от_сырых_к_обучению.md`, `docs/pritok_pressure_trend_method.md` | Оставить/обновлять | Документируют текущий контур |
| Raw ALMA inputs | `data/raw/negermet`, `data/raw/pritok`, `data/raw/salt`, `data/raw/norm_work` | Оставить в Git | Нужны для воспроизводимой сборки датасетов |
| `uv` environment lock | `uv.lock` | Оставить в Git | Серверный workflow официально использует `uv` |
| PaAno dependency | `paano/` | Оставить, но разбирать отдельно | Внешняя neural-библиотека; не чистить обычной логикой cleanup |
| Salym raw pipeline | `alma_service/salym/raw_pipeline.py`, `scripts/salym/build_salym_raw.py` | Оставить изолированно | Нужен для отдельной Salym-задачи, но не участвует в основном runtime |
| 3W research | `configs/3w_paano.json`, `scripts/*3w*`, `scripts/detection/*3w*`, `scripts/evaluation/*3w*`, `scripts/reports/generate_3w_report.py`, `docs/3w/*`, `docs/archive/*3w*` | Удалено | Старый research/transfer контур, не часть `paano_shared/global` runtime |
| External model experiments | `scripts/evaluation/train_dacad.py`, `_s4_swap_test.py`, external adapters/benchmarks | Удалено | Сравнение сторонних моделей завершено, не production |
| Transfer/pretrain experiments | `benchmark_global_pretrain_finetune.py`, `compare_transfer_variants.py`, `transfer_3w_to_alma.py`, `alma_service/anomaly_injection.py` | Удалено | Не используется текущим `paano_shared/global` |
| Salym-only reports/packages | `scripts/detection/screen_salym_unlabeled.py`, `scripts/reports/generate_salym_expert_package.py`, `scripts/build_salym_human_pdf.sh`, `docs/salym/*`, `docs/assets/salym_ws1333_gantt.png` | Удалено | Старые экспертные пакеты и human docs не относятся к двум основным детекторам |
| One-off preprocessing audits | `scripts/datasets/audit_frequency_candidates.py`, `audit_preprocessing_defaults_and_channels.py` | Удалено | Выводы перенесены в текущий документ подготовки данных |
| Old notes | `docs/ALMA_class_specific_vs_global_pretrain.md`, `docs/ALMA_notes_from_pro55_experiments.md`, `docs/notes_from_pro5.5.md`, old archive baseline/roadmap notes | Удалено | Содержимое устарело и сведено в текущие документы |

## Подтверждение по коду

Tracked-файлов из удаляемых групп больше нет:

- старый 3W research / transfer контур;
- external model experiments;
- transfer/pretrain experiments;
- Salym expert/report package;
- old notes;
- one-off preprocessing audits.

После пересборки серверного CodeGraph старые удалённые символы не находятся:

- `transfer_3w_to_alma`;
- `train_dacad`;
- `generate_salym_expert_package`;
- `screen_salym_unlabeled`;
- `anomaly_injection`;
- `benchmark_global_pretrain_finetune`;
- `compare_transfer_variants`;
- `build_3w_dataset`;
- `mtgflow`;
- `timesnet`;
- `mtad`.

## Проверки, которые прошли

```bash
uv run python -m py_compile $(git ls-files "alma_service/*.py" "scripts/**/*.py")
bash -n scripts/run_full_dataset_build.sh scripts/run_full_detection_benchmark.sh scripts/evaluation/run_global_candidate_benchmark.sh
uv run python scripts/detection/detect_negermet.py --help
uv run python scripts/detection/detect_pritok.py --help
uv run python scripts/detection/detect_salt.py --help
uv run python scripts/detection/detect_uploaded_well.py --help
uv run python scripts/datasets/build_norm_work_dataset.py --freqs 10min
uv run python scripts/datasets/build_pritok_dataset.py --freq 10min
```

`pytest` в текущем серверном `uv`-окружении не установлен:

```text
error: Failed to spawn: `pytest`
Caused by: No such file or directory
```

Это проблема окружения, а не подтверждённая ошибка кода.

## Что уже очищено локально на сервере

| Путь | Что было | Решение |
|---|---|---|
| `.claude/` | локальные настройки Claude Code | удалено как tool-state |
| `runs/` | логи последних benchmark/global прогонов | удалено как runtime logs |
| `data/raw/3w/` | raw Petrobras 3W dataset | удалено, потому что 3W больше не часть текущего runtime |
| `data/processed/3w/` | processed 3W parquet/features | удалено вместе с raw 3W |
| `data/processed/` | пустая папка после удаления 3W processed | удалено |
| `data/raw/test_unlabeled_35.rar` | исходный архив test35 | удалено, распакованная папка оставлена |

## Что не чистить автоматически

| Путь | Почему |
|---|---|
| `db/`, `artifacts/`, `models/` | Runtime/generated outputs, gitignored |
| `salym/`, `salym_prepared/` | Большие Salym-данные, отдельная задача |
| `data/raw/test_unlabeled_35/` | Оставлено локально на сервере по решению пользователя |
| `data/reference/test35_эксперт_27_скважин_v2.xlsx` | Экспертные комментарии по test35, оставлено локально |
| `data/reference/Сводная информация.backup.xlsx` | Локальный backup сводной, можно удалить только отдельным решением |
| `.codegraph/`, `.serena/`, `.venv/` | Tool/runtime state |
| `paano/` | External/submodule-style dependency, требует отдельного разбора |

## Что осталось посмотреть

1. `paano/` dirty state.

   Нужно отдельно проверить, какие локальные изменения лежат внутри external
   dependency, и решить:

   - это наши нужные патчи к PaAno;
   - это временный мусор;
   - это надо оформить как отдельный commit/submodule update;
   - или синхронизировать с upstream PaAno.

2. Локальные test35/reference данные.

   Сейчас они осознанно оставлены вне Git. Если test35 вернётся в работу,
   нужно отдельно решить структуру хранения и не смешивать его с текущими
   benchmark-датасетами.

3. `pytest` в серверном `uv` окружении.

   Если хотим запускать test suite на сервере, нужно добавить/установить
   `pytest` в dev dependencies. Пока smoke-проверки выполняются через
   `py_compile`, CLI `--help` и целевые builders.

4. Полный benchmark после обновления raw inputs.

   Новые `pritok` и `norm_work` уже включены и builders проходят. Следующий
   runtime-шаг - запускать целевые detection/benchmark pipelines только после
   отдельного решения, потому что это уже вычислительная задача, а не cleanup.
