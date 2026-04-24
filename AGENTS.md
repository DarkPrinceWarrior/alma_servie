# Repository instructions

## Memory and context

Use Honcho as the memory layer for this repository. Before answering questions
about project preferences, working rules, prior decisions, or remembered
context, consult Honcho in addition to this file and local repository docs.

Separate confirmed facts from inference. Treat facts from files and command
outputs as confirmed; treat remembered context and architectural guesses as
inference unless verified locally.

## Tooling rules

For any file search or grep in the current git repository, use fff first.
Do not use shell `find`, `grep`, or `rg` when fff can express the query.

Use fff for:

- finding files by name or pattern;
- searching code across the repository;
- discovering entry points, routers, services, scripts, and module structure;
- narrowing the area to inspect before using Serena.

After fff identifies the relevant area, use Serena for symbolic code navigation
and symbol-level edits:

- `get_symbols_overview`;
- `find_symbol`;
- `find_referencing_symbols`;
- `replace_symbol_body`;
- `insert_before_symbol`;
- `insert_after_symbol`;
- `rename_symbol`;
- `safe_delete_symbol`.

Prefer Serena tools over reading or rewriting full source files when symbolic
tools are sufficient.

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
- `scripts/datasets/` - dataset builders;
- `scripts/detection/` - anomaly detection entry points;
- `scripts/reports/` - HTML report generators;
- `scripts/evaluation/` - evaluation utilities;
- `paano/` - PaAno neural library/submodule;
- `db/`, `artifacts/`, `models/` - generated outputs, gitignored.

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
python scripts/datasets/build_pritok_dataset.py --freq 2min
python scripts/datasets/build_salt_dataset.py --freq 2min
bash scripts/run_full_dataset_build.sh
```

Run detection:

```bash
python scripts/detection/detect_negermet.py --detector pca_spe
python scripts/detection/detect_pritok.py --detector pca_spe
python scripts/detection/detect_salt.py --detector pca_spe
bash scripts/run_full_detection_benchmark.sh
```

Generate reports:

```bash
python scripts/reports/generate_negermet_paano_report.py --detector pca_spe
python scripts/reports/generate_pritok_paano_report.py --detector pca_spe
python scripts/reports/generate_salt_paano_report.py --detector pca_spe
```

Evaluate onset quality:

```bash
python scripts/evaluation/evaluate_onset_metrics.py \
  --anomaly salt --detector pca_spe --name salt_pca_spe
```

For root research changes, no global lint/format/test runner is configured in
`CLAUDE.md`; run the affected script directly to check imports/runtime behavior.

For backend changes under `app/back/`, use the validation commands documented
in `app/back/AGENTS.md`.

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
