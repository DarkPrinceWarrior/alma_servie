# Repository instructions

## Memory and context

Use Honcho as the memory layer for this repository. Before answering questions
about project preferences, working rules, prior decisions, or remembered
context, consult Honcho in addition to this file and local repository docs.
Current Honcho MCP tools expose peer cards, conclusions, chat over peer
representations, and dream scheduling (`get_peer_card`, `set_peer_card`,
`list_conclusions`, `create_conclusions`, `chat`, `schedule_dream`). Use those
current names rather than older `search`/`create_conclusion` notes.

Separate confirmed facts from inference. Treat facts from files and command
outputs as confirmed; treat remembered context and architectural guesses as
inference unless verified locally.

## Tooling rules

Code navigation uses three MCP servers with a strict division of labour:
`fff` to locate, `codegraph` to understand structure, `serena` to read a
symbol precisely and edit it. Do not duplicate them — each owns one job.

Verified local tool versions on 2026-05-26: `fff-mcp 0.8.4`, `Serena 1.5.3`,
and `codegraph 0.9.5`.

### fff — locate files and literal text

For any file search or grep in the current git repository, use fff first.
Do not use shell `find`, `grep`, or `rg` when fff can express the query.

Use fff for:

- finding files by name or pattern;
- searching literal text — strings, comments, log messages;
- discovering entry points, routers, services, scripts, and module structure;
- narrowing the area to inspect before using codegraph or Serena.

Search one bare identifier per query; after two grep calls, read the code
instead of grepping variations.

### codegraph — structural questions over the symbol graph

`codegraph` is a tree-sitter knowledge graph (SQLite) of every symbol, edge,
and file. Use it for structural questions, not literal text:

- `codegraph_context "<task>"` — PRIMARY: entry points + related symbols +
  code in one call. Start here for any feature, bug, or unfamiliar area.
- `codegraph_search` — find a symbol by name (kind + signature + location);
  prefer this over `fff grep` when looking up a symbol by name.
- `codegraph_callers` / `codegraph_callees` — who calls / what is called.
- `codegraph_impact <symbol>` — blast radius before a refactor.
- `codegraph_node` — a symbol's source / signature / docstring.
- `codegraph_explore` — deeper architecture/module exploration. Use it after
  `codegraph_search` or `codegraph_context` has surfaced concrete symbol or
  file names; prefer one precise explore call over a grep/read loop. In
  CodeGraph 0.8+, explore source sections include line numbers for direct
  `file:line` citations.
- `codegraph_files` / `codegraph_status` — directory layout / index health.

Trust codegraph results — they come from a full AST parse; do not re-verify
with grep. Do not query the index in the same turn as a file edit — the
watcher debounces ~500 ms behind writes.

### serena — symbolic navigation and symbol-level edits

After fff/codegraph identify the relevant area, use Serena for LSP-precise
navigation and symbol-level edits — Serena is the only one of the three that
edits code:

- `get_symbols_overview`;
- `find_symbol`;
- `find_referencing_symbols` — LSP-accurate references; final check before
  `rename_symbol` (use `codegraph_impact` for the quick blast-radius estimate);
- `replace_symbol_body`;
- `insert_before_symbol`;
- `insert_after_symbol`;
- `rename_symbol`;
- `safe_delete_symbol`.

Prefer Serena tools over reading or rewriting full source files when symbolic
tools are sufficient.

### codegraph index sync

The MCP server watches the project and auto-syncs the graph (~2 s debounce).
Still, keep the index fresh explicitly:

- at the start of a work session, run `codegraph status` — if it reports
  pending changes, run `codegraph sync`;
- after any bulk external change the watcher may have missed — `git pull`,
  branch switch, mass file generation, returning from the server — run
  `codegraph sync` before relying on codegraph answers;
- if a codegraph result contradicts what you see in a file, the index is
  stale: `codegraph sync` and re-query.
- if a workspace is on a slow or WSL `/mnt/*` filesystem and watcher startup is
  a problem, run MCP with `codegraph serve --mcp --no-watch` and rely on
  explicit `codegraph sync` or CodeGraph-installed git hooks.

Standard cycle: locate (`fff` / `codegraph_search`) → understand
(`codegraph_context`, then `codegraph_explore` for deep architecture questions)
→ assess risk (`codegraph_impact`) → read and edit (`serena`) → verify (run
the affected script; `playwright` smoke for UI).

Use Context7 before relying on memory for version-sensitive framework/library
behavior, especially FastAPI, Starlette, Pydantic, SQLAlchemy, Alembic, HTTPX,
pytest, or other actively changing libraries.

Use Tavily MCP tools for fresh external information: releases, changelogs,
documentation gaps, incidents, comparisons, or web research.

Use the `fastapi` skill when working on FastAPI routes, dependencies,
request/response schemas, Pydantic models, startup/lifespan logic, auth/RBAC,
or API design.

## Project purpose

`alma_servie` is an oil-well anomaly detection system for three anomaly types:

- `negermet` (`Негерметичность`) - sharp pressure drops;
- `pritok` (`Приток`) - gradual pressure trends;
- `salt` (`Солеотложение`) - salt deposition with additional soft-sensor
  derived features.

The end-to-end research pipeline is:

```text
Excel exports -> Parquet datasets -> blind detection -> interactive HTML reports
```

## Repository areas

Root research code:

- `alma_service/` - shared research library used by scripts;
- `scripts/datasets/` - dataset builders for `negermet`, `pritok`, `salt`, and `norm_work`;
- `scripts/detection/` - anomaly detection entry points (`detect_negermet.py`, `detect_pritok.py`, `detect_salt.py`);
- `scripts/reports/` - HTML report generators;
- `scripts/evaluation/` - onset metrics and current global-candidate benchmark;
- `paano/` - PaAno neural library/submodule;
- `configs/alma_global_feature_schema.json`, `configs/alma_global_normality_5min.json` - current global pipeline configs;
- `db/`, `artifacts/`, `models/` - generated outputs, gitignored.

Active detector keys:

- `paano_shared` - current production/default detector.
- `paano_global` - production-candidate global normality detector.

Old 3W transfer, external-model comparisons, Salym-only package generators, and
one-off ablation scripts were removed. Do not reintroduce them unless the user
explicitly starts a new research task for that family.

Backend code:

- `app/back/` - FastAPI backend;
- `app/back/AGENTS.md` contains more specific backend instructions and takes
  precedence inside that subtree.

When working under `app/back/`, follow both this file and the nested
`app/back/AGENTS.md`; if they conflict, prefer the more specific nested
instruction.

## Key commands

Research setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build datasets:

```bash
python scripts/datasets/build_negermet_dataset.py --freq 5min
python scripts/datasets/build_pritok_dataset.py --freq 10min
python scripts/datasets/build_pritok_dataset.py --freq 5min
python scripts/datasets/build_salt_dataset.py --freq 10min
python scripts/datasets/build_salt_dataset.py --freq 5min
python scripts/datasets/build_norm_work_dataset.py --freqs 5min,10min
bash scripts/run_full_dataset_build.sh
```

Current retained grids in `db/` after cleanup are: `negermet=5min`,
`pritok=10min+5min`, `salt=10min+5min`, `norm_work=10min+5min`. Some builder
defaults are legacy; pass explicit `--freq` / `--freqs` for reproducible runs.

Run detection:

```bash
python scripts/detection/detect_negermet.py --detector paano_shared
python scripts/detection/detect_pritok.py --detector paano_shared
python scripts/detection/detect_salt.py --detector paano_shared
bash scripts/run_full_detection_benchmark.sh

python scripts/detection/detect_negermet.py --detector paano_global
python scripts/detection/detect_pritok.py --detector paano_global
python scripts/detection/detect_salt.py --detector paano_global
bash scripts/evaluation/run_global_candidate_benchmark.sh
```

Generate reports:

```bash
python scripts/reports/generate_negermet_paano_report.py --detector paano_shared
python scripts/reports/generate_pritok_paano_report.py --detector paano_shared
python scripts/reports/generate_salt_paano_report.py --detector paano_shared
```

Evaluate onset quality:

```bash
python scripts/evaluation/evaluate_onset_metrics.py \
  --anomaly salt --detector paano_shared --name salt_paano_shared
```

For root research changes, no global lint/format/test runner is configured in
`CLAUDE.md`; run the affected script directly to check imports/runtime behavior.

For backend changes under `app/back/`, use the validation commands documented
in `app/back/AGENTS.md`.

## Server workflow (a100)

The server is the working copy and compute host for this repository. Run all
research calculations, dataset/report generation, detector runs, benchmarks,
training jobs, tuning jobs, and other project workloads on the server. The
laptop/local checkout is a synchronized consumer copy used for MCP navigation,
inspection, and handoff only; do not spend laptop compute on project runs.

SSH hosts (already in global `~/.ssh/config`):

- `ssh a100` - LAN (`192.168.101.12`), office/VPN only.
- `ssh a100-remote` - remote access via jump host (any network):
  `ProxyJump root@37.9.4.106:12921` -> target `10.10.40.201`. Same physical
  server as `a100`.

Hardware: 6x A100-SXM4-40GB, Debian 13, CUDA 13.0. Project path on server:
`/root/projects/alma_servie/` inside the lowercase `/root/projects` directory.
There is no `/root/Projects` directory on the server.

Server Python environment is managed with `uv`, not plain `pip`:

- `.python-version`: `3.13`;
- virtual environment: `.venv/`, created by `uv venv --python 3.13 .venv`;
- install command:
  `uv pip install -e . -r requirements.txt torch==2.11.0`;
- verified stack: Python `3.13.5`, PyTorch `2.11.0+cu130`,
  `torch.version.cuda == "13.0"`, A100 visible with
  `CUDA_VISIBLE_DEVICES=1`.

Do not install `paano/requirements.txt` as-is on the server: it pins the old
`torch==2.7.1` and would replace the CUDA 13.0 PyTorch stack.

Rules:

1. Source of truth is the server copy at `/root/projects/alma_servie`. Current
   active work happens on the server. The laptop/local copy is only a consumer
   that pulls/syncs changes from the server when needed.
2. MCP tools (`fff`, `codegraph`, `serena`) see the local checkout, not the
   remote server filesystem directly. Before using MCP for code navigation or
   analysis, make sure the local checkout mirrors the server state. After any
   server-side edits, generated files, git operations, or bulk transfers, sync
   the server state back to local first, then run `codegraph sync` if relying
   on CodeGraph.
3. Use `uv run python ...` for server commands. For GPU-only PaAno runs, set
   `CUDA_VISIBLE_DEVICES=1` or another non-zero GPU explicitly.
4. Artifacts (`models/`, `artifacts/`, `db/`) stay on the server. They are
   already gitignored. Pull final weights/reports back via `scp` when needed.
5. Long-running jobs go through `tmux new -d -s <name>` so an SSH disconnect
   does not kill the process.
6. GPU0 on the server is taken by another process (~8.4 GB). Use
   `CUDA_VISIBLE_DEVICES=1..5`.
7. Serena CLI is installed on the server with
   `uv tool install -p 3.13 serena-agent@latest --prerelease=allow`; verified
   version is `Serena 1.5.1`. `.serena/` was copied to the server for this
   project. Treat it as local tool state, not as repository source.
8. Current transferred runtime data on the server includes `db/`, `artifacts/`,
   `models/`, `salym/`, and `salym_prepared/`. After cleanup on 2026-05-21:
   `artifacts` is ~110 MB, `db` is ~122 MB, and `models` is ~15 MB. Historical
   raw Salym transfers remain server-local: `salym` = `71894737195` bytes,
   `salym_prepared` = `34105867675` bytes.

Commands:

```bash
# PaAno training in tmux on GPU 1
ssh a100 'tmux new -d -s paano \
    "cd /root/projects/alma_servie && \
     CUDA_VISIBLE_DEVICES=1 uv run python paano/train.py 2>&1 | tee runs/paano.log"'
ssh a100 'tmux capture-pane -t paano -p | tail -30'   # progress
ssh a100 'tmux ls'                                     # active sessions

# Full benchmark detection
ssh a100 'tmux new -d -s benchmark \
    "cd /root/projects/alma_servie && \
     CUDA_VISIBLE_DEVICES=1 uv run bash scripts/run_full_detection_benchmark.sh 2>&1 | tee runs/benchmark.log"'

# Single detector run
ssh a100 'cd /root/projects/alma_servie && \
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/detection/detect_salt.py --detector paano_shared'

# Recreate server uv environment
ssh a100 'cd /root/projects/alma_servie && \
    printf "3.13\n" > .python-version && \
    uv venv --python 3.13 .venv && \
    uv pip install -e . -r requirements.txt torch==2.11.0'

# Verify PyTorch CUDA stack
ssh a100 'cd /root/projects/alma_servie && \
    CUDA_VISIBLE_DEVICES=1 uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"'

# File transfer (no rsync on server - use scp)
scp -p data/raw/<new>.xlsx a100:/root/projects/alma_servie/data/raw/
scp a100:/root/projects/alma_servie/models/<new>.pt models/
scp -r a100:/root/projects/alma_servie/artifacts/reports/ artifacts/
scp -rp .serena a100:/root/projects/alma_servie/
scp -rp salym salym_prepared a100:/root/projects/alma_servie/
```

Known gotchas:

- IPv6 routing is broken on the server. `/etc/gai.conf` already contains
  `precedence ::ffff:0:0/96 100` to prefer IPv4. Do not revert.
- `rsync` is not installed - use `scp -p`.
- The server has its own SSH key for GitLab (`/root/.ssh/id_ed25519`, title
  `a100-server` in GitLab) - pushing from the server works directly.
- For multi-GPU DDP (if PaAno is scaled out): NCCL on VM135 requires
  `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`.
- The main repo records `paano` as a gitlink without `.gitmodules`. On the
  server, `paano/` was restored manually from `https://github.com/jinnnju/PaAno.git`
  at commit `0e93e93a857af216642d1685f2ce2d2589b35e2e`, then local changes to
  `paano/main.py` and `paano/train.py` were copied over. Therefore server
  `git status` is expected to show `m paano`.

## Web application (app/) — deployment & data

Public anomaly-detection web service: **https://alma-anomaly.ds-mind-lab.ru/**.
Frontend `app/front` (Next.js/React/TS), backend `app/back` (FastAPI/SQLAlchemy/Postgres).

**Where it runs:** production is deployed **on the a100 itself** via
`docker compose -f app/back/compose.yaml` (project `alma_servie`; services
`postgres`/`backend`/`frontend`; containers `alma_servie_postgres`/`alma_servie_api`/
`alma_servie_front`; ports 8000 API, 3000 front). Auth is enabled (Postgres + RBAC);
ask the owner for a login — **never commit credentials**.

**Deploy is MANUAL — a `git push` alone does NOT redeploy** (containers run a baked
image). After changing `app/` code, on the a100:

```bash
cd /root/projects/alma_servie/app/back
docker compose build backend frontend     # only changed services
docker compose up -d backend frontend
```

Builds may fail with a TLS timeout to `auth.docker.io` (Docker Hub + IPv6 grief) — then
use `DOCKER_BUILDKIT=0 docker compose build ...` (the classic builder reuses the cached
base image without contacting the registry).

**Data is a live mount, NOT via git:** prod reads `/data` = `db/` + `artifacts/` from
the a100 repo. Editing `db/*.parquet` on the a100 changes the live site **immediately,
without a rebuild**. `db/` is gitignored — a push never carries data.

**App data model** (fully data-driven from `db/`):
- well lists — `db/{anomaly}_intervals.parquet`, `split` column = `test`/`train`;
- per-well metrics/chart — `db/{anomaly}_{detector}_scores.parquet` +
  `_predicted_starts.parquet` + telemetry `db/{anomaly}_anomaly_database_{freq}.parquet`
  + results `artifacts/results/{anomaly}_{detector}_results.parquet`;
- report availability — existence of
  `artifacts/reports/{anomaly}/{anomaly}_{detector}_report.html`.

**Current config (2026-06):** active detector `paano_global`; anomaly classes
`negermet` + `pritok` only (`salt` removed from `app/`, kept in research code);
telemetry freq 5min; home page = "Тест" (8 blind wells, `split=test`) + "Обучение"
(28 labeled, `split=train`). Blind test wells are consolidated into `db/` by
`runs/consolidate_app_data.py` (a100); backup at `db/_backup_app_20260618/`.

**Upload history** — Postgres table `detection_runs` (`anomaly="multi"`); cleared from
the UI ("Очистить всё", bulk-delete). Not part of git/data.

## Coding conventions

- Use `from __future__ import annotations` at the top of Python modules.
- Use type hints throughout; prefer `X | Y` union syntax.
- Use `@dataclass` for data-holding structs.
- Keep names self-documenting; do not add docstrings by default.
- Use Parquet for heavy I/O; JSON only for summaries, config, and tuning
  outputs.
- Keep `torch.compile` guarded with a safe fallback where applicable.

## Scope discipline

Before changing code, briefly state what was found, what will change, and why.

Prefer minimal, localized edits over broad refactors.

Do not rename files, move modules, change public interfaces, install
dependencies, edit secrets, or touch generated outputs unless explicitly
required.

Do not commit contents of `db/`, `artifacts/`, or `models/`.

Do not modify `paano/` casually; treat it as an external/submodule-style
dependency unless the task explicitly targets it.

Preserve the separation between research code and backend integration. The
backend should not import heavy research modules directly; backend integration
rules are detailed in `app/back/AGENTS.md`.
