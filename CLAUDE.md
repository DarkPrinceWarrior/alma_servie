# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tooling

Code navigation uses three MCP servers, each with one job — do not duplicate them:

- **fff** — locate files and literal text (strings, comments, log messages).
  Use fff instead of shell `find`/`grep`/`rg`. One bare identifier per query;
  after two greps, read the code.
- **codegraph** — structural questions over a tree-sitter symbol graph.
  `codegraph_context "<task>"` is the primary tool (entry points + related
  symbols + code in one call). Also `codegraph_search` (symbol by name —
  prefer over `fff grep`), `codegraph_callers`/`codegraph_callees`,
  `codegraph_impact` (blast radius before a refactor), `codegraph_node`,
  `codegraph_explore` (deeper architecture/module exploration after
  `codegraph_search` or `codegraph_context` surfaces concrete symbol/file
  names; in CodeGraph 0.8+ source sections include line numbers for direct
  `file:line` citations).
  Trust its results — full AST parse; do not re-verify with grep.
- **serena** — LSP-precise symbol navigation and the only tool that *edits*
  at symbol level (`find_symbol`, `get_symbols_overview`,
  `find_referencing_symbols`, `replace_symbol_body`, `insert_*`,
  `rename_symbol`, `safe_delete_symbol`). Prefer over reading whole files.

Cycle: locate (fff / `codegraph_search`) → understand (`codegraph_context`,
then one precise `codegraph_explore` for deep architecture questions) → assess
risk (`codegraph_impact`) → read and edit (serena) → verify.

**codegraph index sync** — the MCP server auto-syncs (~2 s debounce), but keep
it fresh explicitly: run `codegraph status` at the start of a session and
`codegraph sync` if it reports pending changes, or after any bulk change the
watcher may miss (`git pull`, branch switch, mass file generation). If a
codegraph answer contradicts the file, the index is stale — `codegraph sync`
and re-query. Do not query the index in the same turn as an edit (~500 ms lag).
On slow or WSL `/mnt/*` filesystems, use `codegraph serve --mcp --no-watch`
and rely on explicit `codegraph sync` or CodeGraph-installed git hooks.

Other MCP: **context7** for version-sensitive library docs (Next.js, React,
FastAPI, PyTorch — prefer over web search); **tavily** for general web search;
**playwright** for browser smoke-checks after UI changes.

Verified local tool versions on 2026-05-26: `fff-mcp 0.8.4`, `Serena 1.5.3`,
and `codegraph 0.9.5`.

## Memory (Honcho)

Use Honcho as the memory layer for this repository. Before answering questions
about project preferences, working rules, prior decisions, or remembered
context, consult Honcho in addition to this file and local repository docs.

Current Honcho MCP tools expose peer cards, conclusions, chat over peer
representations, and dream scheduling: `get_peer_card`, `set_peer_card`,
`list_conclusions`, `create_conclusions`, `chat`, `schedule_dream`. Use those
current names rather than older `search`/`create_conclusion` notes.

Separate confirmed facts from inference. Treat files and command outputs as
confirmed; treat Honcho memory and architectural guesses as inference unless
verified locally.

## Project Purpose

**alma_servie** — oil-well anomaly detection system for three anomaly types:
- `negermet` (`Негерметичность`) — sharp pressure drops
- `pritok` (`Приток`) — gradual pressure trends
- `salt` (`Солеотложение`) — salt deposition, with additional soft-sensor derived features

End-to-end pipeline: Excel exports → Parquet datasets → blind detection → interactive HTML reports.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Key Commands

### Build datasets
```bash
python scripts/datasets/build_negermet_dataset.py --freq 5min
python scripts/datasets/build_pritok_dataset.py --freq 10min
python scripts/datasets/build_pritok_dataset.py --freq 5min
python scripts/datasets/build_salt_dataset.py --freq 10min
python scripts/datasets/build_salt_dataset.py --freq 5min
python scripts/datasets/build_norm_work_dataset.py --freqs 5min,10min
# Or all at once:
bash scripts/run_full_dataset_build.sh
```
Current retained grids in `db/` after cleanup are: `negermet=5min`,
`pritok=10min+5min`, `salt=10min+5min`, `norm_work=10min+5min`. Some builder
defaults are legacy; pass explicit `--freq` / `--freqs` for reproducible runs.

### Run detection
```bash
# Current production/default detector
python scripts/detection/detect_negermet.py --detector paano_shared
python scripts/detection/detect_pritok.py --detector paano_shared
python scripts/detection/detect_salt.py --detector paano_shared

# Production-candidate global detector
python scripts/detection/detect_negermet.py --detector paano_global
python scripts/detection/detect_pritok.py --detector paano_global
python scripts/detection/detect_salt.py --detector paano_global

# Full current benchmarks:
bash scripts/run_full_detection_benchmark.sh
bash scripts/evaluation/run_global_candidate_benchmark.sh
```

### Generate reports
```bash
python scripts/reports/generate_negermet_paano_report.py [--detector paano_shared]
python scripts/reports/generate_pritok_paano_report.py [--detector paano_shared]
python scripts/reports/generate_salt_paano_report.py [--detector paano_shared]
```
By default, report reads detector from `artifacts/results/*_benchmark_summary.json`.

### Evaluate onset quality
```bash
python scripts/evaluation/evaluate_onset_metrics.py \
  --anomaly salt --detector paano_shared --name salt_paano_shared
```

No linting, formatting, or test runner is configured. After changes, run the affected script directly to check for import/runtime errors.

## Server workflow (a100)

**Рабочая модель:** сервер — основная рабочая копия и место вычислений для этого репозитория. Все исследовательские расчёты, сборка датасетов/отчётов, detector runs, benchmark-прогоны, обучения, tuning jobs и прочие project workloads запускаются на сервере. Локальная/ноутбучная копия — только синхронизированное зеркало для MCP-навигации, просмотра и handoff; не тратить compute ноутбука на проектные прогоны.

### SSH хосты (уже в `~/.ssh/config`, глобально)

- **`ssh a100`** — LAN (`192.168.101.12`), только из офиса/VPN.
- **`ssh a100-remote`** — удалённый доступ через jump host (из дома, любая сеть): `ProxyJump root@37.9.4.106:12921` → target `10.10.40.201`. Тот же физический сервер.

### Hardware

6× A100-SXM4-40GB (240 ГБ суммарно), Debian 13, CUDA 13.0. Проектная директория на серваке: `/root/projects/alma_servie/` внутри каталога `/root/projects` в нижнем регистре. Каталога `/root/Projects` на сервере нет.

Окружение на сервере управляется через `uv`, не через обычный pip-style `venv`:

- `.python-version`: `3.13`;
- виртуальное окружение: `.venv/`, создано через `uv venv --python 3.13 .venv`;
- команда установки:
  `uv pip install -e . -r requirements.txt torch==2.11.0`;
- проверенная связка: Python `3.13.5`, PyTorch `2.11.0+cu130`,
  `torch.version.cuda == "13.0"`, A100 видна при `CUDA_VISIBLE_DEVICES=1`.

Не ставить `paano/requirements.txt` как есть: там закреплен старый `torch==2.7.1`, который может откатить рабочую CUDA 13.0 связку.

### Правила

1. **Source of truth теперь сервер**: актуальная рабочая копия находится в `/root/projects/alma_servie`. Вся текущая работа, анализ, правки, прогоны и отчёты выполняются на сервере.
2. **Локальная/ноутбучная копия больше не источник работы**. Она только подтягивает изменения с сервера, когда это нужно.
3. MCP-инструменты (`fff`, `codegraph`, `serena`) видят локальный checkout, а не удалённую файловую систему сервера напрямую. Перед MCP-анализом кода локалка должна быть синхронизирована с сервером. После серверных правок, генерации файлов, git-операций или bulk transfer сначала подтянуть состояние сервера на локалку, затем выполнить `codegraph sync`, если дальше используется CodeGraph.
4. На сервере запускать команды через `uv run python ...`. Для GPU-only PaAno запусков явно задавать `CUDA_VISIBLE_DEVICES=1` или другой не-нулевой GPU.
5. **Артефакты** (`models/`, `artifacts/`, `db/`) **остаются на серваке.** В git они и так в `.gitignore`. Финальные веса/отчёты — `scp` обратно на лэптоп.
6. **Long-running** (тренировка PaAno, full benchmark) запускать через `tmux new -d -s <name>` чтобы SSH-разрыв не убивал процесс.
7. **GPU0 на серваке занят чужим процессом (~8.4 ГБ)** — использовать `CUDA_VISIBLE_DEVICES=1..5`.
8. Serena CLI установлена на сервере через `uv tool install -p 3.13 serena-agent@latest --prerelease=allow`; проверенная версия — `Serena 1.5.1`. `.serena/` перенесена на сервер для этого проекта. Это локальное состояние инструмента, не исходный код репозитория.
9. На сервер перенесены runtime-данные `db/`, `artifacts/`, `models/`, `salym/`, `salym_prepared/`. После чистки 2026-05-21: `artifacts` ≈ 110 MB, `db` ≈ 122 MB, `models` ≈ 15 MB. Исторические Salym-данные остаются локально на сервере: `salym` = `71894737195` bytes, `salym_prepared` = `34105867675` bytes.

### Команды

```bash
# PaAno training в tmux на GPU 1
ssh a100 'tmux new -d -s paano \
    "cd /root/projects/alma_servie && \
     CUDA_VISIBLE_DEVICES=1 uv run python paano/train.py 2>&1 | tee runs/paano.log"'
ssh a100 'tmux capture-pane -t paano -p | tail -30'   # прогресс
ssh a100 'tmux ls'                                     # активные сессии

# full benchmark detection (длинный прогон)
ssh a100 'tmux new -d -s benchmark \
    "cd /root/projects/alma_servie && \
     CUDA_VISIBLE_DEVICES=1 uv run bash scripts/run_full_detection_benchmark.sh 2>&1 | tee runs/benchmark.log"'

# одиночный detector run на серваке
ssh a100 'cd /root/projects/alma_servie && \
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/detection/detect_salt.py --detector paano_shared'

# пересоздание серверного uv-окружения
ssh a100 'cd /root/projects/alma_servie && \
    printf "3.13\n" > .python-version && \
    uv venv --python 3.13 .venv && \
    uv pip install -e . -r requirements.txt torch==2.11.0'

# проверка CUDA/PyTorch
ssh a100 'cd /root/projects/alma_servie && \
    CUDA_VISIBLE_DEVICES=1 uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"'

# перенос файлов (rsync на серваке нет — только scp)
scp -p data/raw/<new>.xlsx a100:/root/projects/alma_servie/data/raw/
scp a100:/root/projects/alma_servie/models/<new>.pt models/
scp -r a100:/root/projects/alma_servie/artifacts/reports/ artifacts/
scp -rp .serena a100:/root/projects/alma_servie/
scp -rp salym salym_prepared a100:/root/projects/alma_servie/
```

### Известные грабли

- IPv6 routing сломан → `/etc/gai.conf` уже содержит `precedence ::ffff:0:0/96 100` для приоритета IPv4. Не откатывать.
- `rsync` отсутствует — использовать `scp -p`.
- На серваке отдельный SSH-ключ для GitLab (`/root/.ssh/id_ed25519`, Title `a100-server` в GitLab) — push с сервера работает напрямую.
- DDP (если PaAno будет тренироваться на нескольких GPU): NCCL на VM135 требует `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`.
- Основной репозиторий хранит `paano` как gitlink без `.gitmodules`. На сервере `paano/` восстановлен вручную из `https://github.com/jinnnju/PaAno.git` на commit `0e93e93a857af216642d1685f2ce2d2589b35e2e`, затем поверх перенесены локальные изменения **четырёх** файлов: `paano/main.py`, `paano/train.py`, `paano/utils/evaluation.py`, `paano/utils/utils.py` (GPU-kmeans коресет банка, seed=42; guard'ы скоринга). Поэтому на сервере ожидаемый `git status` показывает `m paano`. Сверка с апстримом 2026-06-10: после `0e93e93` в апстриме менялись только `main.py` (раннер TSB-AD), README и картинки — ядро метода (`model.py`, `train.py`, `utils/`) актуально, обновление не требуется.

## Веб-приложение (app/) — деплой и данные

Публичный веб-сервис детекции аномалий: **https://alma-anomaly.ds-mind-lab.ru/**.
Фронт `app/front` (Next.js/React/TS), бэк `app/back` (FastAPI/SQLAlchemy/Postgres).

**Где крутится:** прод развёрнут **на самом a100** через `docker compose -f
app/back/compose.yaml` (проект `alma_servie`; сервисы `postgres`/`backend`/`frontend`;
контейнеры `alma_servie_postgres`/`alma_servie_api`/`alma_servie_front`; порты 8000 API,
3000 фронт). Есть авторизация (Postgres + RBAC); учётку запрашивать у владельца — **в
репозиторий не коммитить**.

**Деплой РУЧНОЙ — `git push` сам по себе ничего не выкатывает** (контейнеры крутят
запечённый образ). После правок `app/` на a100:
```bash
cd /root/projects/alma_servie/app/back
docker compose build backend frontend     # только изменённые сервисы
docker compose up -d backend frontend
```
Сборка может падать на TLS-таймауте к `auth.docker.io` (Docker Hub + IPv6-грабли) —
тогда `DOCKER_BUILDKIT=0 docker compose build ...` (классический билдер берёт
кэшированный базовый образ без обращения к реестру).

**Данные — живой mount, НЕ через git:** прод читает `/data` = `db/` + `artifacts/` из
репозитория на a100. Правки `db/*.parquet` на a100 меняют живой сайт **сразу, без
пересборки**. `db/` в `.gitignore` — push никогда не переносит данные.

**Модель данных приложения** (всё data-driven из `db/`):
- списки скважин — `db/{anomaly}_intervals.parquet`, поле `split` = `test`/`train`;
- метрики/график — `db/{anomaly}_{detector}_scores.parquet` + `_predicted_starts.parquet`
  + телеметрия `db/{anomaly}_anomaly_database_{freq}.parquet` + результаты
  `artifacts/results/{anomaly}_{detector}_results.parquet`;
- доступность отчёта — наличие `artifacts/reports/{anomaly}/{anomaly}_{detector}_report.html`.

**Текущая конфигурация (06.2026):** активный детектор — `paano_global`; классы — только
`negermet` + `pritok` (`salt` убран из `app/`, в research-коде остаётся); частота
телеметрии 5min; главная = «Тест» (8 слепых, `split=test`) + «Обучение» (28 размеченных,
`split=train`). Слепые тест-скважины сведены в `db/` скриптом
`runs/consolidate_app_data.py` (a100), бэкап — `db/_backup_app_20260618/`.

**Слепые (неразмеченные) тест-скважины:** у них нет фактической разметки — приложение
НЕ рисует им зону аномалии и факт. начало/окончание, только одну «предполагаемую дату»
= «Время обнаружения». В `db/` им задан один выбранный онсет (из docx-отчёта) в
`predicted_starts` и вырожденный интервал с `source_kind=test_wells` (нужен лишь для
списка); `load_well_series` фильтрует `source_kind=test_wells`, чтобы не отдавать его как
зону; чарт берёт «Время обнаружения» из `predicted_starts`, когда нет размеченного
результата. Скрипт — `runs/fix_blind_app_data.py`. Размеченные скв. (напр. негермет 524)
сохраняют полный вид (зона + факт. начало/конец + задержка).

**Загрузка скважины** (`/upload`) запускает воркером
`scripts/detection/detect_uploaded_well.py --detector paano_global
--anomalies negermet,pritok --use-population-memory-bank` — тот же боевой пайплайн, что
последний прогон тест-скважин (per-class пара энкодер+банк). Команда строится в
`app/back/.../uploads/views.py`; воркер берёт её из БД (рестарт воркера для смены команды
не нужен, нужен rebuild backend). **История загрузок** — таблица Postgres
`detection_runs` (`anomaly="multi"`); чистится из UI кнопкой «Очистить всё» (bulk-delete).
Не входит в git/данные.

## Architecture

### Data flow
```
data/raw/*.xlsx
  → scripts/datasets/build_*_dataset.py
  → db/*_anomaly_database_*.parquet + db/*_intervals.parquet
  → scripts/detection/detect_*.py
  → db/*_<detector>_scores.parquet + artifacts/results/*
  → scripts/reports/generate_*_report.py
  → artifacts/reports/*_<detector>_report.html
```

### Library: `alma_service/`
The shared library that all scripts import from. Key modules:
- `paths.py` — all canonical paths; `PROJECT_ROOT` resolved from file location so scripts work from any cwd
- `dataset_config.py` — well lists, train/test split, parameter renames (single source of truth for data config)
- `generic_detection.py` — `run_detection()` unified entry point; onset config, tune grids, runtime config per anomaly type
- `onset_detection.py` — `detect_causal_onsets()`, numba-JIT compiled CUSUM + EMA; calibrated via `CausalThresholds`

### Current detectors

The active detector keys are:

- `paano_shared` — current production/default detector.
- `paano_global` — production-candidate global normality detector.

`SharedPaAnoDetector` scores each well with a frozen/shared PaAno encoder and a local memory bank.
For `paano_shared`, anomaly-specific physics is fused inside the same detector:
- `pritok`: pressure trend branch
- `salt`: grouped multichannel deposition/residual branch
- `negermet`: pressure-step/load-response signature branch

`paano_global` uses a common 5min schema, shared/global encoder, `edge_hold`
input contract, local memory bank, coverage-aware statuses, and domain decision
layer. It is benchmarked but not default.

### Staged blind pipeline (all three anomaly types)
1. Causal preprocessing
2. Instability mask
3. Engineered features
4. Unified onset layer (Optuna/TPE-tuned on train wells only, then applied to both splits)

`salt` uses the same pipeline but adds soft-sensor derived features.

### `paano/` submodule
The PaAno neural library (ICLR 2026 paper). The repository uses it through `SharedPaAnoDetector` and `alma_service/shared_encoder.py`. Entry points: `paano/main.py`, `paano/train.py`, `paano/model.py`.

### Removed experiment families

The repository was cleaned to keep only the active `paano_shared` and
`paano_global` pipelines. Old 3W transfer, external-model comparisons,
Salym-only package generators, and one-off ablation scripts are intentionally
removed from source control. Do not reintroduce them unless there is a new
explicit research task.

## Conventions

- `from __future__ import annotations` at the top of every module
- Type hints throughout; union types via `X | Y` (Python 3.10+)
- `@dataclass` for data-holding structs (e.g. `DetectorScoreOutput`, `CausalThresholds`)
- No docstrings — names are self-documenting
- All heavy I/O in Parquet; JSON only for summary/config/tuning outputs
- `torch.compile` enabled only for local `PatchEncoder` with a safe fallback

## Output locations
- Raw datasets: `db/*_anomaly_database_*.parquet`
- Intervals + splits: `db/*_intervals.parquet`
- Per-point scores: `db/*_<detector>_scores.parquet`
- Predicted starts: `db/*_<detector>_predicted_starts.parquet`
- Detector config/tuning: `db/*_<detector>_config.json`, `db/*_<detector>_tuning.json`
- Result tables: `artifacts/results/*_<detector>_results.parquet` + `.summary.json`
- Benchmark summary: `artifacts/results/*_benchmark_summary.json`
- HTML reports: `artifacts/reports/*_<detector>_report.html`
- Model weights: `models/`

`db/`, `artifacts/`, and `models/` are gitignored — do not commit their contents.
