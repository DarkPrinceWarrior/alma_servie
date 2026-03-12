# Система обнаружения аномалий в нефтяных скважинах

Проект собирает датасеты по Excel-выгрузкам, запускает единый blind detection pipeline по трём типам аномалий и генерирует HTML-отчёты.

## Структура

- `alma_service/` — общие модули проекта, включая `onset_detection.py` и централизованные пути.
- `alma_service/dataset_config.py` — единая конфигурация исходных скважин, test-split и списка параметров модели.
- `scripts/datasets/` — сборка CSV-датасетов и интервалов из исходных Excel-файлов.
- `scripts/detection/` — новый unified detector stack (`paano_feat`, `pca_spe`, `lof`, `iforest`, `fused`, `tranad_global`) и legacy PaAno-обёртки.
- `scripts/reports/` — HTML-отчёты по результатам blind-детекции и legacy feature-importance отчёты.
- `scripts/evaluation/` — метрики качества детекции стартов аномалий.
- `data/raw/` — исходные Excel-файлы по типам аномалий.
- `data/reference/` — общие справочные Excel-файлы.
  - `Параметры для модели.xlsx` — эталонный список признаков, который используют dataset builders.
- `db/` — подготовленные датасеты, интервалы, скоры и конфиги детекторов.
- `models/` — сохранённые веса моделей PaAno.
- `artifacts/results/` — итоговые таблицы детекции.
- `artifacts/reports/` — HTML-отчёты.
- `paano/` — исходный код библиотеки PaAno.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Базовый workflow

### 1. Сборка датасетов

```bash
python scripts/datasets/build_negermet_dataset.py --freq 15s
python scripts/datasets/build_pritok_dataset.py --freq 2min
python scripts/datasets/build_salt_dataset.py --freq 2min
```

Результат:

- `db/*_anomaly_database_*.csv`
- `db/*_intervals.csv`

Примечания:

- builders собирают полный набор параметров, который реально есть в Excel по конкретной скважине;
- каналы, которых нет у конкретной скважины, остаются `NaN` в общем CSV, но позже не подаются в blind PaAno для этой скважины;
- в `db/*_intervals.csv` пишется колонка `split`, где train/test-скважины задаются через `alma_service/dataset_config.py`.

### 2. Запуск детекции

Новый основной CLI:

```bash
python scripts/detection/detect_negermet.py --detector fused
python scripts/detection/detect_pritok.py --detector fused
python scripts/detection/detect_salt.py --detector fused
```

Доступные детекторы:

- `paano_feat` — engineered-features версия локального PaAno
- `pca_spe` — PCA + Hotelling T²/SPE
- `lof` — Local Outlier Factor
- `iforest` — Isolation Forest
- `fused` — weighted fusion локальных детекторов
- `tranad_global` — глобальный transformer benchmark по train-скважинам

Legacy baseline сохранён отдельно:

```bash
python scripts/detection/detect_negermet_paano.py
python scripts/detection/detect_pritok_paano.py
python scripts/detection/detect_salt_paano.py
```

Результат:

- `artifacts/results/*_<detector>_results.csv`
- `artifacts/results/*_<detector>_results.summary.json`
- `artifacts/results/*_benchmark_summary.json`
- `db/*_<detector>_scores.csv`
- `db/*_<detector>_predicted_starts.csv`
- `db/*_<detector>_config.json`
- `db/*_<detector>_tuning.json`
- `models/*_tranad_global.pt` для глобального benchmark-моделя

Примечания:

- у всех трёх аномалий теперь один и тот же staged blind pipeline: causal preprocessing, instability mask, engineered features, unified onset layer;
- конфигурация onset-детектора тюнится только по `train`-скважинам и затем применяется к `train` и `test`;
- `salt` больше не использует `dev`/LOIO режим и не вырезает аномальные интервалы из train-mask по ground truth;
- дефолтный production-кандидат выбирается через `artifacts/results/*_benchmark_summary.json`.

### 3. Генерация HTML-отчётов

```bash
python scripts/reports/generate_negermet_paano_report.py
python scripts/reports/generate_pritok_paano_report.py
python scripts/reports/generate_salt_paano_report.py
```

Результат:

- `artifacts/reports/*_<detector>_report.html`

По умолчанию report читает detector из `*_benchmark_summary.json`. Можно переопределить:

```bash
python scripts/reports/generate_negermet_paano_report.py --detector pca_spe
```

### 4. Оценка качества стартов

```bash
python scripts/evaluation/evaluate_onset_metrics.py \
  --anomaly salt \
  --detector fused \
  --name salt_fused
```

Метрики теперь единые для всех детекторов:

- `hit_count` / `hit_rate`
- `p90_delay_ratio`
- `p90_abs_delay_hours`
- `false_alarms_per_day`
- `avg_starts_per_interval`

## Типы аномалий

- `Негерметичность` — резкие изменения давления.
- `Приток` — устойчивые постепенные тренды.
- `Солеотложение` — тот же unified pipeline, но с дополнительными soft-sensor derived features.

## Примечания

- Скрипты больше не зависят от запуска строго из корня: пути резолвятся относительно репозитория.
- Новые отчёты и итоговые CSV по умолчанию больше не складываются в корень проекта.
- HTML-отчёты читают те же CSV/JSON, которые пишет детектор, поэтому summary в HTML, `results.csv` и `summary.json` синхронизированы.
- `db/*_anomaly_database_*.csv` остаётся raw-источником; engineered features считаются поверх него внутри detection pipeline.
