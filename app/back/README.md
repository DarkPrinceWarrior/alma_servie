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

Лёгкий стек (api + postgres, по дефолту):

```bash
docker compose up --build -d
```

Старт включает `alma_servie_postgres` (`localhost:5432`) и
`alma_servie_api` (`localhost:8000`). Контейнер api дожидается
healthcheck Postgres, применяет `alembic upgrade head` и запускает uvicorn.
Детекция в этом режиме работает в mock-режиме (`DETECTION_MOCK=true`).

Полный стек с worker-контейнером (тяжёлый, ~3 ГБ образ с torch/numba):

```bash
# 1) в .env переключить api на реального worker'a
echo "DETECTION_MOCK=false" >> .env

# 2) поднять с профилем heavy
docker compose --profile heavy up --build -d
```

worker (`alma_servie_worker`) крутит polling-loop по таблице
`detection_runs` (`SELECT ... FOR UPDATE SKIP LOCKED`), забирает
`pending` задачи и запускает реальные `scripts/detection/detect_*.py`
через subprocess в `/workspace`.

Остановить:

```bash
docker compose down           # контейнеры
docker compose down -v        # + удалить том Postgres
```

## API

Все ручки под префиксом `/api`. OpenAPI: `GET /openapi.json`, Swagger `/docs`.

### Auth (JWT)

- `POST /api/auth/login` — форма `username`/`password` (OAuth2). Возвращает
  `access_token` + выставляет `refresh_token` в httpOnly cookie на
  `/api/auth/refresh`.
- `POST /api/auth/refresh` — ротация refresh-токена с reuse-detection:
  если переданный RT уже отозван, отзываются **все** сессии пользователя.
- `POST /api/auth/logout` — blacklist access JWT по `jti`, отзыв RT.

### Users & Roles

- `GET /api/users/me` — текущий пользователь (роль `USER` или выше).
- `GET /api/users` — admin-only, `?role=&search=&limit=&offset=`.
- `POST /api/users` — admin-only создание.
- `GET /api/users/{uuid}` — admin-only.
- `PATCH /api/users/{uuid}/password` — admin-only смена пароля
  (отзывает все refresh-токены пользователя).

### Health

- `GET /api/health` — общий статус.
- `GET /api/health/db` — проверка соединения с Postgres.

### Wells

Read-only API поверх `{DATA_ROOT}/db/{anomaly}_intervals.parquet`.
Каталог скважин и split'ов берётся из самих parquet (single source of
truth), без импорта research-конфигов.

- `GET /api/wells?anomaly=negermet|pritok|salt` — список скважин с
  сводкой (split, n_intervals, data_start, data_end).
- `GET /api/wells/{well_id}?anomaly=...` — карточка скважины + её
  интервалы.
- `GET /api/wells/{well_id}/intervals?anomaly=...` — только интервалы.

Ответы кэшируются по `(path, mtime_ns)` parquet-файла через
`@lru_cache` — любое перезаписывание файла worker'ом автоматически
инвалидирует кэш.

### Detections

Асинхронный запуск прогонов детекции с трекингом статуса через таблицу
`detection_runs` в Postgres.

- `POST /api/detections` `{anomaly, detector}` → 201 с `DetectionRunRead`.
  `anomaly ∈ {negermet, pritok, salt}`, `detector = paano_shared`.
  Single-flight: если уже есть активный
  прогон `(anomaly, detector)` в статусе `pending|running` — 409 с `run_id`.
- `GET /api/detections?anomaly=&detector=&status=&limit=&offset=` — листинг.
- `GET /api/detections/{run_id}` — карточка прогона со `stdout_tail`,
  `summary_json`, `exit_code`.

Статусы: `pending → running → (succeeded | failed | cancelled)`.

Режим выполнения задаётся через `DETECTION_MOCK`:

- `true` (дефолт api-контейнера в легковесном стеке) — при POST api
  сам запускает `asyncio.sleep` + синтетический `summary_json`.
  Реальный subprocess не стартует — api-образу это и не нужно, у
  него нет research-стека.
- `false` — api **только вставляет** `pending` в `detection_runs` и
  оставляет выполнение worker-контейнеру (`alma_servie_worker`),
  который крутит polling-loop со `SELECT ... FOR UPDATE SKIP LOCKED`.
  При запуске `docker compose --profile heavy up` — переключай api
  в этот режим (иначе и api, и worker будут пытаться обработать
  одну задачу).

Shortcut: `GET /api/detections/{run_id}/report` → 307 redirect на
`/api/reports/{anomaly}/{detector}/html`.

### Reports

Отчёты и точечные scores прогона детектора.

- `GET /api/reports/{anomaly}/{detector}/html` →
  `artifacts/reports/{anomaly}/{anomaly}_{detector}_report.html` как
  `FileResponse` с `Content-Security-Policy: frame-ancestors *` для
  встраивания в `<iframe>` фронта.
- `GET /api/reports/{anomaly}/{detector}/scores?well_id=&from=&to=&limit=2000`
  → `ScoreSeries{well_id, anomaly, detector, n_points, n_downsampled,
  points: [{t, score, split}]}`. Downsampling — равномерная decimation
  по stride, `limit` ограничен 20000.
- `GET /api/reports/{anomaly}/{detector}/starts?well_id=&split=` →
  список `PredictedStart{well_id, detected_time, split}`, отсортированных
  по `detected_time`.

Остальные ручки (auth, users) добавляются отдельными фичами — см.
`docs/backend_roadmap.md`.

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
