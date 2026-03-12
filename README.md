# Система обнаружения аномалий в нефтяных скважинах

Проект собирает датасеты по Excel-выгрузкам, запускает единый blind PaAno-пайплайн по трём типам аномалий и генерирует HTML-отчёты.

## Структура

- `alma_service/` — общие модули проекта, включая `onset_detection.py` и централизованные пути.
- `alma_service/dataset_config.py` — единая конфигурация исходных скважин, test-split и списка параметров модели.
- `scripts/datasets/` — сборка CSV-датасетов и интервалов из исходных Excel-файлов.
- `scripts/detection/` — единый blind PaAno-детектор для `negermet`, `pritok` и `salt`.
- `scripts/reports/` — HTML-отчёты по результатам blind-детекции и feature-importance отчёты.
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

```bash
python scripts/detection/detect_negermet_paano.py
python scripts/detection/detect_pritok_paano.py
python scripts/detection/detect_salt_paano.py
```

Результат:

- `artifacts/results/*_paano_results.csv`
- `artifacts/results/*_paano_results.summary.json`
- `db/*_paano_scores.csv`
- `db/*_paano_predicted_starts.csv`
- `db/*_paano_config.json`
- `db/*_paano_tuning.json`

Примечания:

- у всех трёх аномалий теперь один и тот же blind PaAno-пайплайн;
- конфигурация onset-детектора тюнится только по `train`-скважинам и затем применяется к `train` и `test`;
- `salt` больше не использует `dev`/LOIO режим и не вырезает аномальные интервалы из train-mask по ground truth.

### 3. Генерация HTML-отчётов

```bash
python scripts/reports/generate_negermet_paano_report.py
python scripts/reports/generate_pritok_paano_report.py
python scripts/reports/generate_salt_paano_report.py
```

Результат:

- `artifacts/reports/*_paano_report.html`

### 4. Оценка качества стартов

```bash
python scripts/evaluation/evaluate_onset_metrics.py \
  --intervals db/salt_intervals.csv \
  --predicted-starts db/salt_paano_predicted_starts.csv \
  --scores db/salt_paano_scores.csv \
  --name salt
```

## Типы аномалий

- `Негерметичность` — резкие изменения давления.
- `Приток` — устойчивые постепенные тренды.
- `Солеотложение` — blind multivariate детекция тем же unified PaAno-пайплайном, что и другие аномалии.

## Примечания

- Скрипты больше не зависят от запуска строго из корня: пути резолвятся относительно репозитория.
- Новые отчёты и итоговые CSV по умолчанию больше не складываются в корень проекта.
- HTML-отчёты читают те же CSV/JSON, которые пишет детектор, поэтому summary в HTML, `results.csv` и `summary.json` синхронизированы.
