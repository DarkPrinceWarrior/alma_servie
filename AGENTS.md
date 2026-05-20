# Repository instructions

## Memory and context

Use Honcho as the memory layer for this repository. Before answering questions
about project preferences, working rules, prior decisions, or remembered
context, consult Honcho in addition to this file and local repository docs.

Separate confirmed facts from inference. Treat facts from files and command
outputs as confirmed; treat remembered context and architectural guesses as
inference unless verified locally.

## Tooling rules

Code navigation uses three MCP servers with a strict division of labour:
`fff` to locate, `codegraph` to understand structure, `serena` to read a
symbol precisely and edit it. Do not duplicate them — each owns one job.

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
- `codegraph_explore` — survey an unfamiliar module (token-heavy; onboarding
  only, not for narrow questions).
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

Standard cycle: locate (`fff` / `codegraph_search`) → understand
(`codegraph_context`) → assess risk (`codegraph_impact`) → read and edit
(`serena`) → verify (run the affected script; `playwright` smoke for UI).

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
- `scripts/datasets/` - dataset builders (`build_*_dataset.py`, `build_3w_dataset.py`);
- `scripts/detection/` - anomaly detection entry points (`detect_*.py`, `detect_3w.py`, `physical_branches_3w.py`);
- `scripts/reports/` - HTML report generators;
- `scripts/evaluation/` - evaluation utilities, 3W sweeps, 3W -> ALMA transfer;
- `paano/` - PaAno neural library/submodule;
- `configs/3w_paano.json` - Petrobras 3W pipeline config;
- `db/`, `artifacts/`, `models/` - generated outputs, gitignored.

The Petrobras 3W Dataset 2.0.0 is used as an oil-domain pretrain/benchmark for
ALMA (not a replacement for customer data). The transfer hook
`alma_service.shared_encoder.load_or_train_shared_encoder()` warm-starts the
production `paano_shared` encoder from a 3W per-class encoder. The single public
detector key stays `paano_shared`. Full report and roadmap: `docs/3w_pipeline.md`.

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
python scripts/datasets/build_negermet_dataset.py --freq 15s
python scripts/datasets/build_pritok_dataset.py --freq 10min
python scripts/datasets/build_salt_dataset.py --freq 2min
bash scripts/run_full_dataset_build.sh
python scripts/datasets/build_3w_dataset.py --config configs/3w_paano.json
```

`build_pritok_dataset.py` defaults to `--freq 10min`; it must match the
production detect grid `db/pritok_anomaly_database_10min.parquet`. A freq
mismatch silently yields `n_channels=0` and undetected wells.

Run detection:

```bash
python scripts/detection/detect_negermet.py --detector paano_shared
python scripts/detection/detect_pritok.py --detector paano_shared
python scripts/detection/detect_salt.py --detector paano_shared
bash scripts/run_full_detection_benchmark.sh
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

When to switch to the server: PaAno training with `epochs >= 50`, long
benchmark runs (`run_full_detection_benchmark.sh` over all wells), Optuna/TPE
tuning with many trials. Keep smoke tests, code edits, and debugging on the
laptop.

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
2. Use `uv run python ...` for server commands. For GPU-only PaAno runs, set
   `CUDA_VISIBLE_DEVICES=1` or another non-zero GPU explicitly.
3. Artifacts (`models/`, `artifacts/`, `db/`) stay on the server. They are
   already gitignored. Pull final weights/reports back via `scp` when needed.
4. Long-running jobs go through `tmux new -d -s <name>` so an SSH disconnect
   does not kill the process.
5. GPU0 on the server is taken by another process (~8.4 GB). Use
   `CUDA_VISIBLE_DEVICES=1..5`.
6. `.serena/` was copied to the server for this project. Treat it as local
   tool state, not as repository source.
7. Current transferred runtime data on the server includes `db/`, `artifacts/`,
   `models/`, `salym/`, and `salym_prepared/`. Sizes verified on 2026-05-06:
   `salym` = `71894737195` bytes, `salym_prepared` = `34105867675` bytes.

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
