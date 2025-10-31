# Repository Guidelines

## Project Structure & Module Organization
- `src/pipeline` теперь содержит только рабочие этапы `anomalies` и `events`, которые читают `alma/Общая_таблица.xlsx`. Запуск: `python -m src.pipeline anomalies` и `python -m src.pipeline events`.
- `src/pipeline/anomalies/` — пакет из нескольких модулей:
  - `__init__.py` держит публичный API и CLI;
  - `models.py` — dataclass-ы и конфигурации;
  - `settings.py` — загрузка параметров из `config/pipeline.yaml`;
  - `preprocessing.py` — чтение рабочей книги и санитация рядов;
  - `detection.py` — baseline, признаки и логика детекции.
- Соблюдаем разделение ответственности между подмодулями; новые функции помещаем в соответствующий файл, а не в `__init__.py`.
- `config/pipeline.yaml` хранит пути (`alma`, `reports`) и параметры детектора. Любые правки конфигурации фиксируйте рядом с rollout-заметками.
- `alma/` — актуальная рабочая книга: лист `svod` плюс отдельные листы по скважинам.
- `reports/` — итоговые артефакты (`anomaly_analysis.*`, `events_features.*`, `reference_wells_report.html`).
- `scripts/` — вспомогательные CLI (`generate_reference_report.py` и пр.).

## Build, Test, and Development Commands
- Создать среду: `python -m venv venv && source venv/bin/activate`.
- Установить зависимости: `pip install -r requirements.txt`.
- Пересчитать детектор: `python -m src.pipeline anomalies --config config/pipeline.yaml`.
- Обновить справочную статистику: `python -m src.pipeline events --config config/pipeline.yaml`.
- Сформировать интерактивный отчёт по всем скважинам (по умолчанию): `python scripts/generate_reference_report.py --config config/pipeline.yaml`. Ограничить набор можно опцией `--wells 5271г,1123л`.

## Coding Style & Naming Conventions
- Follow PEP 8 (4 spaces), придерживайтесь существующих type hints и коротких docstring’ов.
- Новые модули называем по ответственности (`anomalies_report.py`, `events_export.py` и т.п.), избегаем «misc/utils».
- Используем snake_case и именование колонок, совпадающее с рабочей книгой (`Intake_Pressure`, `timestamp_Current`).
- Предпочитаем чистые функции, принимающие пути/конфигурацию, без жёстко прописанных директорий.

## Testing Guidelines
- Add automated checks under `tests/` with `pytest -q`; isolate filesystem effects using temporary directories and fixture configs.
- Cover resampling, thresholding, and anomaly rule branches with parametrised cases and synthetic parquet slices.
- Assert on returned summary dictionaries (row counts, metrics) to catch regressions without relying on large data dumps.

## Commit & Pull Request Guidelines
- Match the existing imperative commits (e.g., `Add anomaly rule for pressure spikes`) and scope one logical change per commit.
- In PRs, summarise affected modules, note configuration toggles, and link related issue IDs or task documents.
- Attach before/after snippets or report paths from `reports/` when altering detection logic or QC outputs.
- Explicitly call out new dependencies, data migrations, or workbook schema changes in a checklist bullet.

## Configuration & Data Handling
- Не добавляем исходные Excel в git; путь до них задаём через `config/pipeline.yaml`.
- Любые обновления `alma/Общая_таблица.xlsx` или схемы фиксируем в комментариях к задаче/PR вместе с датой выгрузки.
