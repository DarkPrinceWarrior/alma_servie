# Repository instructions

## Project

FastAPI backend for the alma_servie anomaly detection system.
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

Основная часть проекта `alma_servie` (в корне репо, вне `app/back/`)
остаётся исследовательским кодом: `scripts/`, `alma_service/`, `paano/`,
`db/`, `artifacts/`. Из бэка её **не импортируем напрямую** в ручки —
обёртываем в сервисы/задачи (`src/back/api/<entity>/service.py`),
которые вызывают нужные скрипты как подпроцесс или переиспользуют
функции по чётко очерченному API.

Пока что `app/back/` живёт своим pyproject/venv — не смешиваем с
исследовательским `requirements.txt` в корне.

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
