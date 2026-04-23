# alma_servie back

FastAPI backend for the alma_servie anomaly detection system.

Architecture — feature-first (МТО-стиль): каждая сущность в своей папке
`src/back/api/<entity>/` с `views.py`, `crud.py`, `schemas.py`. Модели
общие в `src/back/models/<entity>.py`. Async SQLAlchemy 2.0 + asyncpg.

```
src/back/
  api/
    deps.py            # DbDep (UserDep/AdminUserDep — по мере появления auth)
    health/
  core/
    config.py
  db/
    base.py            # DeclarativeBase
    database.py        # async engine, AsyncSessionLocal, get_db
  models/              # Mapped/mapped_column
  rbac/
    roles.py           # RoleCode: USER | ADMIN + implications + BASE_ROLES
    permissions.py     # "wells:read", "detection:run", …
  main.py              # FastAPI + lifespan
```

Код детекции/отчётов из корня `alma_servie` (модуль `alma_service/`,
скрипты `scripts/detection/`, `scripts/reports/`) подключается
постепенно — отдельными фичами под `src/back/api/<entity>/`.

## Local development

1. Скопируй `.env.example` в `.env` и заполни `SECRET_KEY` (при
   будущем включении auth — `ADMIN_EMAIL`, `ADMIN_PASSWORD`).
2. Подними Postgres:

   ```bash
   docker compose up -d postgres
   ```

3. Накати миграции (после того, как появятся первые модели и `alembic revision`):

   ```bash
   uv run alembic upgrade head
   ```

4. Запусти API:

   ```bash
   uv run uvicorn back.main:app --reload --host 0.0.0.0 --port 8000
   ```

API доступен на `http://localhost:8000`, Swagger — `http://localhost:8000/docs`.

## Docker Compose

Полный стек:

```bash
docker compose up --build
```

Старт включает `alma_servie_postgres` (`localhost:5432`) и
`alma_servie_backend` (`localhost:8000`). Контейнер backend дожидается
healthcheck Postgres, применяет `alembic upgrade head` и запускает uvicorn.

Остановить:

```bash
docker compose down           # контейнеры
docker compose down -v        # + удалить том Postgres
```

## API

Все ручки под префиксом `/api`. OpenAPI: `GET /openapi.json`, Swagger `/docs`.

### Health

- `GET /api/health` — общий статус.
- `GET /api/health/db` — проверка соединения с Postgres.

Остальные ручки (auth, users, wells, detection, reports) добавляются
отдельными фичами.

## Checks

```bash
uv sync
uv run ruff check
uv run ruff format
uv run pytest
uv run alembic upgrade head
```

## Конфигурация

Все настройки читаются из `.env` (см. `.env.example`). Ключевые:

| Переменная | Назначение |
|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://…` |
| `SECRET_KEY` | HMAC-ключ для JWT |
| `ACCESS_TOKEN_EXPIRE_MINUTES` / `REFRESH_TOKEN_EXPIRE_DAYS` | TTL токенов |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | Seed-админ при старте |
| `ADMIN_RESET_PASSWORD=true` | Сбросить пароль seed-админа при старте |
| `REFRESH_COOKIE_SECURE` | `true` в проде за HTTPS |
