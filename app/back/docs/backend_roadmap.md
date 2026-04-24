# alma_servie backend — roadmap

> Продуктовый трек `app/backend` ветки. Research (`ai_paano_detection`) живёт своей жизнью.
> Документ — единственный источник правды о последовательности работ до MVP.
> Обновляется после каждого merged-этапа: отмечаем DoD, фиксируем hash коммита.

---

## 0. Цели и нецели

### Что строим

FastAPI-бэкенд (+ позже Next.js-фронт, + worker для тяжёлых расчётов), который:

1. Позволяет инженеру открыть список скважин по аномалии (`negermet | pritok | salt`), увидеть их таймлайн и интервалы.
2. Даёт запустить детекцию по выбранной скважине/датасету и следить за статусом прогона.
3. Показывает HTML-отчёт и точечные scores детектора, полученные текущим прогоном.
4. Позже — логин/роли, разметка интервалов пользователем, экспорт.

### Чего сознательно НЕ делаем в MVP

- Переписывание пайплайна детекции, paano, scripts/datasets — **всё это остаётся в корне репо**, не трогаем.
- Свой ML-код в бэке. Весь compute — в research-пакете `alma_service` + `scripts/`.
- Сложная очередь задач (celery/rq/arq). MVP — таблица `detection_runs` + loop в worker.
- Multi-tenant, биллинг, квоты.
- Автоматическая переобучающая петля. Запуск детекции — по кнопке или по CLI.

---

## 1. Архитектурные инварианты

Эти правила не меняются без явного решения в этом документе.

1. **Два Python-окружения, два образа**:
   - `api` (python 3.13 + uv + FastAPI + polars + asyncpg, ~200 МБ). Не тянет torch/numba.
   - `worker` (python 3.12 + research venv + torch + numba + paano, ~3 ГБ). Запускает `scripts/*.py`.
2. **Никаких тяжёлых импортов в api**. Бэк импортирует только лёгкие модули `alma_service.paths`, `alma_service.dataset_config`. Детекция — всегда через subprocess или через job-таблицу.
3. **Данные — через volumes, не в образе**. `/data/db`, `/data/artifacts`, `/data/models`, `/data/raw` монтируются извне.
4. **Пути через env**: `Settings.data_root`, `Settings.research_root`. Никаких `PROJECT_ROOT / "db"` в ручках.
5. **Связка api ↔ worker через Postgres**. `detection_runs(id, anomaly, detector, status, ...)`. worker делает `SELECT ... WHERE status='pending' FOR UPDATE SKIP LOCKED`.
6. **Feature-first** в `src/back/api/<entity>/{views,crud,schemas,service}.py`, как в `master_nsi/back`.
7. **Миграции только через alembic**. Руками схему не трогаем.
8. **Research code read-only из api**. api никогда не пишет в `db/`, `artifacts/` — только читает готовый вывод worker'а.

---

## 2. Checkpoint — что уже есть

| # | Коммит | Что |
|---|---|---|
| 1 | `23801b0` | Scaffold: FastAPI + lifespan, health ручки, async SQLAlchemy skeleton, RBAC-каркас (roles, permissions), alembic-пустой, Dockerfile + compose, httpx smoke-тест |
| 2 | `3eba2e2` | compose project name `alma_servie`, контейнер `alma_servie_api`, volume `alma_servie_pgdata` |
| 3 | `3067174` | Корневой `pyproject.toml` (пустые deps), `alma-service` как editable dep в `app/back`, правила импорта в AGENTS.md |

**DoD пройдено**: `docker compose up -d` поднимает postgres + api, `GET /api/health`, `/api/health/db` отвечают 200, `GET /docs` открывается.

---

## 3. Roadmap

### Этап A — `wells` (read-only API над parquet)

**Цель**: фронт/CLI видит скважины и их интервалы без запуска детекции.

**Что делаем**:

1. Добавить в `core/config.py`: `data_root: Path = Path("/data")` (локально дефолт = `../../`).
2. Новая фича `src/back/api/wells/`:
   - `schemas.py`:
     ```python
     class WellSummary(BaseModel):
         well_id: str
         anomaly: Literal["negermet", "pritok", "salt"]
         split: Literal["train", "test"]
         n_intervals: int
         time_range: tuple[datetime, datetime] | None
     ```
   - `crud.py`: `list_wells(anomaly)`, `get_well(well_id, anomaly)`, `get_well_intervals(well_id, anomaly)`. Чистые функции над polars.
   - `views.py`:
     - `GET /api/wells?anomaly=negermet`
     - `GET /api/wells/{well_id}?anomaly=negermet`
     - `GET /api/wells/{well_id}/intervals?anomaly=negermet`
3. Кэш результатов с инвалидацией по mtime parquet-файла.
4. `polars>=0.20` в зависимостях бэка.
5. Использовать `alma_service.dataset_config.NEGERMET_WELLS` (и аналоги) через прямой импорт — проверить, что не тянет тяжёлые deps.
6. Фикстуры в `tests/fixtures/db/` с мини-parquet (3-5 строк).

**Пример ответа**:
```json
GET /api/wells?anomaly=negermet
[
  {"well_id": "WS-1333", "anomaly": "negermet", "split": "train", "n_intervals": 3, "time_range": ["2015-03-06T00:00:00", "2016-12-14T00:00:00"]},
  {"well_id": "WS-5638", "anomaly": "negermet", "split": "test", "n_intervals": 1, "time_range": ["2016-04-02T00:00:00", "2016-07-18T00:00:00"]}
]
```

**DoD**:
- [ ] три ручки возвращают данные на реальных parquet из `db/`;
- [ ] 4 pytest-теста (list, get, intervals, 404);
- [ ] ruff + pytest зелёные;
- [ ] README.md бэка обновлён (раздел Wells);
- [ ] коммит `feat(app/back): wells read API over parquet`.

**Оценка**: 1 коммит, ~300 строк.

---

### Этап B — `detections` (запуск прогонов с фоновой задачей)

**Цель**: кнопка «Прогнать детектор» работает end-to-end без отдельного worker'а (для быстрой итерации). worker появится в этапе D.

**Что делаем**:

1. Миграция `0001_detection_runs.py`:
   ```sql
   CREATE TABLE detection_runs (
     id UUID PRIMARY KEY,
     anomaly TEXT NOT NULL,
     detector TEXT NOT NULL,
     status TEXT NOT NULL,  -- pending | running | succeeded | failed
     started_at TIMESTAMPTZ,
     finished_at TIMESTAMPTZ,
     command TEXT NOT NULL,
     exit_code INT,
     stdout_tail TEXT,
     summary_json JSONB,
     error_message TEXT,
     created_at TIMESTAMPTZ DEFAULT NOW(),
     updated_at TIMESTAMPTZ DEFAULT NOW()
   );
   CREATE INDEX idx_detection_runs_status ON detection_runs (status);
   CREATE INDEX idx_detection_runs_anomaly_detector ON detection_runs (anomaly, detector, created_at DESC);
   ```
2. Модель `src/back/models/detection_run.py` + `src/back/models/__init__.py` регистрирует её.
3. Сервис `src/back/api/detections/service.py`:
   ```python
   async def launch(db, anomaly, detector) -> DetectionRun:
       # single-flight: если есть running с тем же (anomaly, detector) — возвращаем его
       run = DetectionRun(
           anomaly=anomaly,
           detector=detector,
           status="pending",
           command=f"python scripts/detection/detect_{anomaly}.py --detector {detector}",
       )
       db.add(run); await db.commit()
       asyncio.create_task(_execute(run.id))
       return run
   
   async def _execute(run_id):
       # subprocess в venv корня, ring-buffer stdout, по exit code обновить статус
   ```
4. Ручки `src/back/api/detections/views.py`:
   - `POST /api/detections` `{anomaly, detector}` → возвращает `DetectionRun`.
   - `GET /api/detections?anomaly=&detector=&status=&limit=&offset=`.
   - `GET /api/detections/{id}`.
   - (опционально, в отдельном коммите) `GET /api/detections/{id}/stream` — SSE со stdout.
5. Защита от дубликатов: если `(anomaly, detector)` уже `pending|running` — 409 с `existing_run_id`.
6. Тесты: mock-subprocess возвращает заранее известный exit code и синтетический `summary.json`.

**DoD**:
- [ ] миграция `uv run alembic upgrade head` применяется;
- [ ] `POST /api/detections {"anomaly":"negermet","detector":"pca_spe"}` запускает реальную детекцию на test-скважине (или на mock);
- [ ] `GET /api/detections/{id}` показывает переход `pending → running → succeeded`;
- [ ] single-flight 409 покрыт тестом;
- [ ] коммит `feat(app/back): detection runs with background subprocess + status API`.

**Оценка**: 1-2 коммита, ~500 строк + миграция.

---

### Этап C — `reports` (HTML-отчёт + точечные scores)

**Цель**: после прогона фронт может встроить HTML и нарисовать линию scores поверх телеметрии.

**Что делаем**:

1. Ручки `src/back/api/reports/`:
   - `GET /api/reports/{anomaly}/{detector}/html` — отдаёт `artifacts/reports/{anomaly}_{detector}_report.html` как `StreamingResponse` с правильным `Content-Type: text/html; charset=utf-8`.
   - `GET /api/reports/{anomaly}/{detector}/scores?well_id=&frm=&to=` — читает `db/{anomaly}_{detector}_scores.parquet`, фильтрует по скважине и диапазону, возвращает `[{t, score, mask}]` (downsampled до ~2000 точек, если больше).
   - `GET /api/reports/{anomaly}/{detector}/starts` — `db/{anomaly}_{detector}_predicted_starts.parquet`.
2. Shortcut: `GET /api/detections/{id}/report` → 307 redirect на `reports/{anomaly}/{detector}/html`.
3. Content-Security-Policy для встраивания HTML в `<iframe>` фронта.
4. Тесты на миниатюрных fixture-parquet.

**DoD**:
- [ ] `curl /api/reports/negermet/pca_spe/html > report.html` — открывается в браузере;
- [ ] scores endpoint возвращает корректный JSON с downsampling;
- [ ] коммит `feat(app/back): reports and scores API`.

**Оценка**: 1 коммит, ~300 строк.

---

### Этап D — worker-контейнер (правильное разделение compute)

**Цель**: вынести исполнение детекции в отдельный сервис, чтобы api был тонким.

**Что делаем**:

1. Новая директория `app/back/worker/`:
   - `Dockerfile.worker` (`python:3.12` + copy корневого `requirements.txt` + copy `alma_service/`, `paano/`, `scripts/`).
   - `worker/loop.py` — цикл `SELECT ... FOR UPDATE SKIP LOCKED`, `asyncio.create_subprocess_exec`, обновление `detection_runs`.
   - либо: worker живёт в отдельной папке `/worker/` в корне репо, но с общим `compose.yaml` в `app/back/`.
2. Обновить `compose.yaml`:
   ```yaml
   services:
     worker:
       build: { context: ../.., dockerfile: app/back/worker/Dockerfile.worker }
       depends_on: [postgres]
       volumes:
         - ../../db:/data/db
         - ../../artifacts:/data/artifacts
         - ../../models:/data/models
   ```
3. В api убираем `asyncio.create_task(_execute(...))` (осталось от этапа B); теперь `POST /api/detections` только вставляет строку со статусом `pending`, а worker её заберёт.
4. `docker compose up` поднимает 3 сервиса: postgres, api, worker. Health у обоих.
5. Тесты интеграционные — через `docker compose up -d worker` в CI (если будет CI). Локально — можно руками.

**DoD**:
- [ ] `docker compose up -d` поднимает три здоровых контейнера;
- [ ] `POST /api/detections` + ожидание 30 сек → `GET /api/detections/{id}` показывает `succeeded`;
- [ ] worker видит `/data/db`, `/data/artifacts`, `/data/models`;
- [ ] api больше не умеет сам запускать subprocess — тесты этого не позволяют;
- [ ] коммит `feat(worker): standalone compute container, api becomes pure web layer`.

**Оценка**: 2 коммита, ~400 строк.

---

### Этап E — `auth` (JWT, users, roles)

**Цель**: фронт может логиниться, ручки `detections` защищены permissions.

**Что делаем**:

1. Модели: `User`, `Role`, `UserRole` (association), `RefreshToken`, `TokenBlacklist`. Структура — как в `master_nsi/back/src/back/models/`.
2. Миграция `0002_auth.py`.
3. Seed ролей (`USER`, `ADMIN`) и админа из env в `lifespan`.
4. `src/back/api/auth/` — `views.py` (login, refresh, logout), `crud.py`, `schemas.py`.
5. `src/back/api/users/` — `GET /api/users/me`, admin-ручки.
6. Расширить `src/back/api/deps.py`: `UserDep`, `AdminUserDep`, `get_current_user` на основе JWT.
7. Навесить `require_permission("detection:run")` на `POST /api/detections`.
8. Тесты: login flow + 401/403 на защищённых ручках.

**DoD**:
- [ ] `POST /api/auth/login` возвращает access + выставляет refresh cookie;
- [ ] `GET /api/users/me` с access работает;
- [ ] `POST /api/detections` без токена → 401, с user-ролью → 200;
- [ ] коммит `feat(app/back): JWT auth + users + RBAC wiring`.

**Оценка**: 2-3 коммита, ~1000 строк + миграция.

---

### Этап F — frontend MVP

**Цель**: человек-инженер может зайти на `/`, увидеть список скважин, прогнать детектор, посмотреть отчёт.

**Что делаем** (в `app/front/`, Next.js 16 + shadcn, копия стека `master_nsi/front`):

1. `app/front/` scaffold — кладём как есть.
2. API-клиент на основе openapi.json бэка (openapi-typescript-codegen).
3. Страницы:
   - `/` — tabs по трём классам аномалий, таблица скважин.
   - `/wells/[id]` — карточка скважины: таймлайн интервалов, embedded `<iframe>` HTML-отчёта.
   - `/detections` — список прогонов со статусами.
   - `/detections/new` — форма запуска.
   - `/login`, `/logout`.
4. Подключение auth — cookie-based refresh + access в header.

**DoD**:
- [ ] локально `pnpm dev` поднимает фронт, логин работает, список скважин видно, запуск детекции работает, отчёт показывается;
- [ ] `pnpm tsc --noEmit`, `pnpm test:run`, `pnpm build` зелёные;
- [ ] коммит(ы) `feat(app/front): wells list, well detail, detection runs, auth`.

**Оценка**: 3-4 коммита, позже — когда дизайнер закончит.

---

### Этап G — деплой

**Цель**: система поднимается на сервере одной командой.

**Что делаем**:

1. `docker-compose.prod.yaml` в корне репо:
   - services: postgres, api, worker, front, (позже) nginx как reverse proxy.
   - volumes: `/srv/alma_servie_data/{db,artifacts,models,raw}` bind-mount.
   - env: read-only из `/etc/alma_servie/env` или через docker secrets.
2. `.env.prod.example` — без секретов, только шаблон.
3. `docs/deploy.md` — step-by-step:
   ```bash
   ssh server
   cd /opt/alma_servie
   git fetch && git checkout app/backend
   git submodule update --init paano
   docker compose -f docker-compose.prod.yaml build
   docker compose -f docker-compose.prod.yaml up -d
   docker compose -f docker-compose.prod.yaml exec api uv run alembic upgrade head
   ```
4. Healthcheck: Prometheus scrape `/api/health` + Grafana позже.
5. Бэкапы: `pg_dump` raspberry-simple + ротация по дате. `/srv/alma_servie_data/` — rsync на второй диск.

**DoD**:
- [ ] из чистого VPS/сервера по `docs/deploy.md` за 30 мин поднимается работающая система;
- [ ] SSL через Caddy или nginx+certbot, домен подключён;
- [ ] коммит `feat(deploy): production compose, docs, secrets template`.

**Оценка**: 1-2 коммита + ручной setup сервера.

---

## 4. Сводная таблица этапов

| Этап | Ветка | Коммитов | Строк | Время |
|---|---|---|---|---|
| A. wells | `app/backend` | 1 | ~300 | 0.5 дня |
| B. detections (inline) | `app/backend` | 1-2 | ~500 + migration | 1 день |
| C. reports | `app/backend` | 1 | ~300 | 0.5 дня |
| D. worker-контейнер | `app/backend` | 2 | ~400 | 1 день |
| E. auth | `app/backend` | 2-3 | ~1000 + migration | 1.5 дня |
| F. frontend | `app/backend` | 3-4 | ~2000 | 2-3 дня (после дизайна) |
| G. deploy | `app/backend` | 1-2 | ~200 | 0.5-1 день |

**Итого до production-MVP**: ~11-16 коммитов в `app/backend`, ~5-8 рабочих дней чистой разработки (без учёта дизайна и серверного setup).

---

## 5. Зависимости между этапами

```
A wells  ─┐
          ├─► C reports ─┐
B detect ─┘              ├─► F frontend ──► G deploy
          └─► D worker ──┘
                          ┌─► E auth (может идти параллельно A-D)
```

- Этапы A, B, C, D — последовательны (B → D, C опирается на существование артефактов, которые делает B+D).
- E auth — независим от A-D, можно вставить между любыми. Но **фронт (F) требует E**.
- G deploy — после F (или минимум после D + A/B/C без F, если хотим «API-only продакшен»).

---

## 6. Риски и открытые вопросы

### R1. Размер `paano` в worker-образе

`paano/` тянет torch (~800 МБ) + numba. Если worker крутится в CI на runner без GPU, torch-cpu хватит. Но если на сервере CUDA — нужна отдельная torch-версия. **Решение**: держать два варианта Dockerfile.worker: `.cpu` и `.cuda`, выбирать через build-arg.

### R2. Размер данных в volume

`db/` может вырасти до 50-100 ГБ (когда добавим Salym). Bind-mount в docker требует, чтобы диск хватало. **Решение**: отделить `db/` в отдельный физический том, cleanup policy для старых scores (`*_scores.parquet` можно пересчитать).

### R3. Миграции и back-compat

`detection_runs` со временем обрастёт полями. **Правило**: только `ADD COLUMN NULL` или `ALTER TYPE` с дефолтом. `DROP COLUMN` — только после 2 релизов depreciation.

### R4. Кто владеет `/data/db`?

api читает, worker пишет. Если worker падает в середине записи — получаем corrupted parquet. **Решение**: worker пишет во временный файл и атомарно `rename`. Уже сейчас так в `scripts/*.py`? — **TODO проверить**.

### R5. CORS для фронта

Если api и front на разных доменах (например, `api.alma.local` и `alma.local`) — нужен CORS middleware. **Решение**: `fastapi.middleware.cors.CORSMiddleware` с точным allow-list из env.

### R6. Где хранить raw Excel

Сейчас `data/raw/` в репо (в gitignore). На сервере — отдельный volume. **Вопрос**: откуда заливаются новые Excel? Ручной rsync, API-загрузка, Airflow — не решено.

---

## 7. Глоссарий

- **api** — FastAPI-контейнер (`alma_servie_api`), web-слой.
- **worker** — compute-контейнер, крутит subprocess пайплайна.
- **research** — всё, что в корне репо: `alma_service/`, `scripts/`, `paano/`, `data/`, experiment-ветки.
- **Anomaly classes**: `negermet` (негерметичность), `pritok` (приток), `salt` (солеотложение).
- **Detectors**: `pca_spe`, `paano_feat`, `paano_shared`, `ensemble`.
- **DoD** — Definition of Done, чек-лист готовности этапа.

---

## 8. Лог выполнения

| Дата | Этап | Коммит | Заметка |
|---|---|---|---|
| 2026-04-23 | scaffold | `23801b0` | FastAPI + health + compose |
| 2026-04-23 | compose fix | `3eba2e2` | переименование контейнера + project name |
| 2026-04-23 | pkg wiring | `3067174` | корневой pyproject + alma-service editable |
| 2026-04-23 | push | — | ветка `app/backend` на origin |
| 2026-04-24 | **A. wells** | `9b35428` | ручки list/get/intervals над parquet, polars, Dockerfile build context=../.., volume ../..:/data:ro; 5/7 тестов |
| 2026-04-24 | **B. detections** | `122c7d3` | таблица detection_runs + миграция 0001, api/detections (POST/GET list/GET by id) + single-flight 409, fake executor (DETECTION_MOCK), +3 теста; docker smoke ok (pending→running→succeeded за 2 сек) |
| 2026-04-24 | **C. reports** | `024f4d4` | api/reports (html/scores/starts), GET /api/detections/{id}/report shortcut (307), path-резолвер, downsample stride, CSP frame-ancestors; +8 тестов; docker smoke: 9.8 МБ HTML, 1365→N scores, 18 предсказанных стартов |
| | B. detections | | |
| | C. reports | | |
| | D. worker | | |
| | E. auth | | |
| | F. frontend | | |
| | G. deploy | | |
