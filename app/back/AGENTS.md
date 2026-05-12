# Repository instructions

## Project

FastAPI backend for the alma_servie anomaly detection system. Живёт в
единой продуктовой ветке `app/webapp` рядом с Next.js фронтом
(`app/front/`). Research-код остаётся в корне репо и ветках research.

Архитектура **feature-first** (МТО-стиль): каждая доменная сущность —
отдельная папка в `src/back/api/<entity>/` с `views.py`, `crud.py`,
`schemas.py`.

## Architecture defaults

- **feature-first**: ручки (`views.py`) + persistence (`crud.py` — async
  функции, НЕ классы) + pydantic (`schemas.py`) внутри папки сущности;
- async SQLAlchemy 2.0 (`Mapped/mapped_column`, `AsyncSession`);
- thin routers — бизнес-логика в `crud.py` или в `service.py` (если нужно
  выделить multi-step процесс);
- модели в `src/back/models/<entity>.py`;
- JWT access + refresh (httpOnly cookie) в `src/back/api/auth/`, RBAC в
  `src/back/rbac/` (`RoleCode`, `PERMISSIONS`, `require_permission`);
- Pydantic v2 style, `Annotated` для DI, общие `DbDep` / `UserDep` /
  `AdminUserDep` в `src/back/api/deps.py`.

ORM-модели храним отдельно от API-схем.

## Integration with research code

Корневой `alma_service/` подключён к бэку как **editable local package**
(`alma-service` в `[tool.uv.sources]`). Research-зависимости (torch,
numba, optuna и т.п.) в бэк **не тащим** — `pyproject.toml` корня
объявляет пустой `dependencies = []`, все тяжёлые пакеты по-прежнему
ставятся research-венвом из корневого `requirements.txt`.

Правила импорта:

- **Лёгкие модули** (`alma_service.paths`, `alma_service.dataset_config`)
  импортируем напрямую в бэке через `from alma_service.paths import PROJECT_ROOT`.
- **Тяжёлые модули** (`alma_service.generic_detection`,
  `alma_service.onset_detection` и
  вызывающие их `scripts/detection/*.py`, `scripts/datasets/*.py`,
  `scripts/reports/*.py`) — **не импортируем** в api-контейнер.
  Они запускаются через `asyncio.create_subprocess_exec` в отдельном
  worker-контейнере, который видит `alma_servie/venv/` с полным
  research-стеком.
- `paano/` — submodule, не часть пакета `alma_service`. Доступна только
  воркеру, никогда — api.

Коммуникация api ↔ worker — через таблицу `detection_runs` в Postgres
(статус задач) и общий filesystem volume с `db/`, `artifacts/`, `models/`.

## Tools

Use **fff** first for file/code search (`find_files`, `grep`, `multi_grep`).

Use **Serena** for symbol-level navigation and edits (`find_symbol`,
`find_referencing_symbols`, `replace_symbol_body`).

Use **Context7** when FastAPI / Starlette / Pydantic / SQLAlchemy /
Alembic / HTTPX / pytest behaviour зависит от версии.

Use **Tavily** для внешнего веб-поиска (релизы, changelogs, docs gaps).

## Validation

После значимых изменений:

```bash
uv sync
uv run ruff check
uv run ruff format
uv run pytest
uv run alembic upgrade head
```

## Scope discipline

- Новые/изменённые колонки/таблицы — **только через alembic-миграцию**.
- Не расширять scope задачи без необходимости (в том числе не
  рефакторить соседние модули «заодно»).
- Не трогать `compose.yaml`, секреты, CI без явного запроса.
- Исследовательский код в корне репо (`scripts/`, `alma_service/`,
  `paano/`) из бэка не трогаем — интегрируем через service-обёртки.
