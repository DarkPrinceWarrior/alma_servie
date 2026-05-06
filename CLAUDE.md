# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tooling

For any file search or grep in the current git indexed directory use fff tools.

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
python scripts/datasets/build_negermet_dataset.py --freq 15s
python scripts/datasets/build_pritok_dataset.py --freq 2min
python scripts/datasets/build_salt_dataset.py --freq 2min
# Or all at once:
bash scripts/run_full_dataset_build.sh
```

### Run detection
```bash
# Unified stack (preferred): detectors = pca_spe | paano_feat
python scripts/detection/detect_negermet.py --detector pca_spe
python scripts/detection/detect_pritok.py --detector pca_spe
python scripts/detection/detect_salt.py --detector pca_spe

# Legacy PaAno baseline:
python scripts/detection/detect_negermet_paano.py
python scripts/detection/detect_pritok_paano.py
python scripts/detection/detect_salt_paano.py

# Full benchmark:
bash scripts/run_full_detection_benchmark.sh
```

### Generate reports
```bash
python scripts/reports/generate_negermet_paano_report.py [--detector pca_spe]
python scripts/reports/generate_pritok_paano_report.py [--detector pca_spe]
python scripts/reports/generate_salt_paano_report.py [--detector pca_spe]
```
By default, report reads detector from `artifacts/results/*_benchmark_summary.json`.

### Evaluate onset quality
```bash
python scripts/evaluation/evaluate_onset_metrics.py \
  --anomaly salt --detector pca_spe --name salt_pca_spe
```

No linting, formatting, or test runner is configured. After changes, run the affected script directly to check for import/runtime errors.

## Server workflow (a100)

**Когда переключаться на сервер:** все тренировки PaAno от `epochs ≥ 50`, длинные benchmark-прогоны (`run_full_detection_benchmark.sh` по всем скважинам), Optuna/TPE-тюнинг с большим числом trials. Smoke-тесты, разработка и отладка кода — локально.

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
3. На сервере запускать команды через `uv run python ...`. Для GPU-only PaAno запусков явно задавать `CUDA_VISIBLE_DEVICES=1` или другой не-нулевой GPU.
4. **Артефакты** (`models/`, `artifacts/`, `db/`) **остаются на серваке.** В git они и так в `.gitignore`. Финальные веса/отчёты — `scp` обратно на лэптоп.
5. **Long-running** (тренировка PaAno, full benchmark) запускать через `tmux new -d -s <name>` чтобы SSH-разрыв не убивал процесс.
6. **GPU0 на серваке занят чужим процессом (~8.4 ГБ)** — использовать `CUDA_VISIBLE_DEVICES=1..5`.
7. `.serena/` перенесена на сервер для этого проекта. Это локальное состояние инструмента, не исходный код репозитория.
8. На сервер перенесены runtime-данные `db/`, `artifacts/`, `models/`, `salym/`, `salym_prepared/`. Размеры, сверенные 2026-05-06: `salym` = `71894737195` bytes, `salym_prepared` = `34105867675` bytes.

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
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/detection/detect_salt.py --detector pca_spe'

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
- Основной репозиторий хранит `paano` как gitlink без `.gitmodules`. На сервере `paano/` восстановлен вручную из `https://github.com/jinnnju/PaAno.git` на commit `0e93e93a857af216642d1685f2ce2d2589b35e2e`, затем поверх перенесены локальные изменения `paano/main.py` и `paano/train.py`. Поэтому на сервере ожидаемый `git status` показывает `m paano`.

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
- `paano_pipeline.py` — `run_detection()` legacy PaAno entry point
- `onset_detection.py` — `detect_causal_onsets()`, numba-JIT compiled CUSUM + EMA; calibrated via `CausalThresholds`

### Detector interface (`BaseDetector`)
All detectors implement:
- `fit_reference(X_ref, mask_ref)` — train on reference (train-split) data
- `score_stream(X_all, mask_all)` → anomaly scores

Concrete implementations: `PaAnoFeatureDetector`, `PCASPEDetector`, `SharedPaAnoDetector`.

### Staged blind pipeline (all three anomaly types)
1. Causal preprocessing
2. Instability mask
3. Engineered features
4. Unified onset layer (Optuna/TPE-tuned on train wells only, then applied to both splits)

`salt` uses the same pipeline but adds soft-sensor derived features.

### `paano/` submodule
The PaAno neural library (ICLR 2026 paper). Used via `PaAnoFeatureDetector` and the legacy scripts. Entry points: `paano/main.py`, `paano/train.py`, `paano/model.py`.

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
