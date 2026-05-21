# ALMA: аудит локального состояния серверной рабочей копии

Дата: 2026-05-21.

Серверная рабочая копия: `/root/projects/alma_servie`.
Ветка: `app/webapp`.

Цель документа: отделить актуальный код репозитория от локальных данных,
логов, tool-state и временных артефактов, которые остались на сервере вне Git
после cleanup.

## Короткий вывод

Git-кодовая база после cleanup приведена к текущему контуру:

- `paano_shared` - production/default;
- `paano_global` - production-candidate;
- текущие dataset builders, detection scripts, reports, evaluation scripts;
- изолированный Salym raw pipeline:
  `alma_service/salym/raw_pipeline.py` и `scripts/salym/build_salym_raw.py`.

После ручной чистки 2026-05-21 серверная рабочая папка стала заметно чище:
старые 3W leftovers, `runs/`, `.claude/` и архив `test_unlabeled_35.rar`
удалены. Вне Git остаются только данные, по которым принято отдельное решение,
и несколько технических хвостов.

## Что уже подтверждено по коду

Tracked-файлов из удаляемых групп `docs/ALMA_cleanup_inventory_2026-05-21.md`
на сервере не осталось:

- старый 3W research / transfer контур;
- external model experiments;
- transfer/pretrain experiments;
- Salym expert/report package;
- old notes;
- one-off preprocessing audits.

После пересборки серверного CodeGraph старые удалённые символы также не
находятся:

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
```

`pytest` в текущем серверном `uv`-окружении не установлен:

```text
error: Failed to spawn: `pytest`
Caused by: No such file or directory
```

Это проблема окружения, а не подтверждённая ошибка кода.

## Untracked: то, что видно в git status

### Уже удалено 2026-05-21

| Путь | Что было | Решение |
|---|---|---|
| `.claude/` | локальные настройки Claude Code | удалено как tool-state |
| `runs/` | логи последних benchmark/global прогонов | удалено как runtime logs |
| `data/raw/3w/` | raw Petrobras 3W dataset | удалено, потому что 3W больше не часть текущего runtime |
| `data/processed/3w/` | processed 3W parquet/features | удалено вместе с raw 3W |
| `data/raw/test_unlabeled_35.rar` | исходный архив test35 | удалено, распакованная папка оставлена |
| `data/processed/` | пустая папка после удаления 3W processed | удалено |

### test35 leftovers

| Путь | Размер | Что это | Решение |
|---|---:|---|---|
| `data/raw/test_unlabeled_35/` | 27 Excel-файлов | оставить локально на сервере |
| `data/reference/test35_эксперт_27_скважин_v2.xlsx` | 125 KB | экспертная разметка/комментарии по test35 | не удалять без явного решения, если комментарии ещё нужны |

test35 по текущему правилу не участвует в benchmark текущих детекторов.

### norm_work

| Путь | Что это | Решение |
|---|---|---|
| `data/raw/norm_work/` | 20 Excel-файлов нормальной работы | включить в Git как исходные данные текущего `norm_work` pipeline |

Эти файлы потенциально полезны для:

- проверки false alarms;
- norm-work guard;
- будущего обучения/калибровки global-normality подхода.

### Новые raw-файлы pritok

В `data/raw/pritok/` есть untracked Excel-файлы:

- `122_Кус_Приток.xlsx`;
- `1442л_ЮЯ_Приток.xlsx`;
- `1449_ЮЯ_Приток.xlsx`;
- `1508_ЮЯ_Приток.xlsx`;
- `1995_ЮЯ_Изменение_частоты_Нормальная.xlsx`;
- `1996л_ЮЯ_Изменение_частоты_Нормальная.xlsx`;
- `305г_ВИК_Изменение_частоты_Нормальная.xlsx`;
- `4203у_ЮЯ_Приток.xlsx`;
- `5021_ЮЯ_Приток.xlsx`.

Решение: включить в Git. `PRITOK_WELL_FILES` уже ссылается на эти файлы, а
значит без них сборка `pritok` из чистого clone будет неполной.

### `uv.lock`

| Путь | Размер | Что это | Решение |
|---|---:|---|---|
| `uv.lock` | 4 KB | lock-файл `uv` | включить в Git, потому что серверный workflow официально использует `uv` |

Так как серверный workflow теперь использует `uv`, lock-файл может быть
полезен, но его нельзя просто удалить/игнорировать без решения по dependency
policy.

## Ignored, но большие локальные папки

Эти папки уже игнорируются Git, но физически занимают место на сервере:

| Путь | Размер | Что это | Решение |
|---|---:|---|---|
| `salym/` | 67 GB | raw Salym data | не удалять без явного решения |
| `salym_prepared/` | 32 GB | prepared Salym data | не удалять без явного решения |
| `artifacts/` | 1.6 GB | runtime reports/results | не трогать без задачи на cleanup артефактов |
| `db/` | 322 MB | parquet/config/scores runtime data | не трогать без задачи на cleanup runtime data |
| `models/` | 67 MB | веса моделей | не трогать |
| `.codegraph/` | 11 MB | локальный индекс CodeGraph | оставить, это tool-state |
| `.serena/` | tool-state | локальное состояние Serena | оставить или чистить только отдельно |
| `.venv/` | Python env | server uv environment | не трогать |

## Рекомендованная классификация

### Уже очищено

- `.claude/`;
- `runs/`;
- `data/raw/3w/`;
- `data/processed/3w/`;
- `data/processed/`;
- `data/raw/test_unlabeled_35.rar`.

### Не чистить автоматически

- `data/reference/test35_эксперт_27_скважин_v2.xlsx`;
- `salym/`;
- `salym_prepared/`;
- `db/`;
- `artifacts/`;
- `models/`;
- `data/raw/test_unlabeled_35/`.

## Что сделать следующим шагом

1. Закоммитить `uv.lock`, этот audit-документ, `data/raw/norm_work/` и новые
   `data/raw/pritok/*.xlsx`.

2. Оставить `data/raw/test_unlabeled_35/` и
   `data/reference/test35_эксперт_27_скважин_v2.xlsx` локально на сервере.

3. Отдельно решить судьбу dirty `paano`, потому что это external/submodule-style
   dependency и его нельзя чистить обычной логикой cleanup.
