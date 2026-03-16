# Shared PaAno Encoder — Полный отчёт о реализации

**Дата:** 15–16 марта 2026  
**Участники:** Safae + AI-ассистент  
**Проект:** alma_service — детекция аномалий на скважинах

---

## 1. Исходная задача

Пользователь предложил архитектурную идею: **вместо обучения отдельного PaAno encoder'а на каждой скважине**, использовать **один shared encoder на семейство аномалий** (negermet, pritok, salt). Каждый shared encoder обучается на объединённом пуле «нормальных» данных со всех train-скважин данного типа аномалии. После этого для каждой конкретной скважины строится **local memory bank** через замороженный shared encoder.

### Ключевые принципы архитектуры:

1. **Число моделей = числу типов аномалий** (3 модели, не N скважин)
2. **Shared representation** — encoder учит общую «норму» семейства
3. **Local decision layer** — per-well memory bank + per-well threshold
4. **4-зонная разметка данных** — чистые нормальные данные отделяются от переходных фаз

---

## 2. План реализации

Был составлен и утверждён план из 3 компонентов:

### Компонент 1 — 4-Zone Data Curation
Разделение временного ряда каждой скважины на 4 зоны:
- **Zone 0: CLEAN_NORMAL** — чистая норма, используется для обучения
- **Zone 1: PRE_ANOMALY_BUFFER** — предвестники аномалии, исключаются
- **Zone 2: ANOMALY** — сама аномалия
- **Zone 3: POST_ANOMALY_RECOVERY** — восстановление после аномалии, исключается

### Компонент 2 — Shared Encoder
- Сбор clean_normal пула со всех train-скважин
- Определение общих каналов через intersection
- Обучение dual-scale encoder (short + long patch)
- Scoring через shared model + local memory bank

### Компонент 3 — Интеграция в пайплайн
- Регистрация нового детектора `paano_shared`
- Модификация `generic_detection.py` для shared training phase
- Fallback логика при недостаточном пуле

---

## 3. Реализация

### 3.1 Новый модуль: `alma_service/zone_labels.py`

**Цель:** 4-зонная разметка данных.

**Что содержит:**
- `Zone` enum — `CLEAN_NORMAL`, `PRE_ANOMALY_BUFFER`, `ANOMALY`, `POST_ANOMALY_RECOVERY`
- `ZONE_BUFFER_CONFIG` — конфигурация буферов per-family:
  - negermet: 30 мин pre / 60 мин post
  - pritok: 60 мин pre / 180 мин post
  - salt: 60 мин pre / 180 мин post
- `label_zones()` — назначает зону каждому таймстемпу на основе интервалов аномалий
- `make_clean_normal_mask()` — маска True только для CLEAN_NORMAL точек
- `make_training_exclusion_mask()` — маска для исключения буферов и аномалий
- `make_onset_allowed_from_zones()` — маска для onset detection, подавляющая буферы

**Особенность:** минимальная ширина буфера = `patch_size` точек (чтобы ни один "грязный" патч не попал в обучение).

---

### 3.2 Новый модуль: `alma_service/shared_encoder.py`

**Цель:** Обучение shared encoder и scoring.

**Что содержит:**
- `SharedEncoderState` — dataclass с обученными моделями, статистиками нормализации, списком shared каналов
- `collect_shared_train_pool()` — собирает clean_normal данные из всех train-скважин:
  - Определяет intersection каналов (только каналы, общие для всех скважин)
  - Проецирует feature matrix каждой скважины на общий набор каналов
  - Объединяет в один pool
  - Возвращает pool, shared_channels, well_ids
- `train_shared_encoder()` — обучает два encoder'а (short patch + long patch):
  - Нормализация pool → z-score
  - Создание патчей
  - Обучение PatchEncoder через `run_training()`
  - Возвращает `SharedEncoderState`
- `_score_well_single_scale()` — scoring одной скважины через shared model:
  - Проецирует данные скважины на shared каналы
  - Нормализует через сохранённые mean/std
  - Строит local memory bank из reference-точек
  - Вычисляет anomaly score
- `select_shared_columns()` — utility для проекции feature matrix на shared каналы

---

### 3.3 Новый класс: `SharedPaAnoDetector` (в `generic_detectors.py`)

**Цель:** Реализация `BaseDetector` интерфейса для shared encoder.

**Что делает:**
- Принимает готовый `SharedEncoderState` (обученный shared encoder)
- `fit_reference()` — строит local memory bank из reference-точек скважины
- `score_stream()` — scoring через shared encoder + local bank
- Поддерживает dual-scale fusion (short + long) с весами

---

### 3.4 Модификация: `alma_service/engineered_features.py`

**Что изменено:**
- Функция `prepare_engineered_well()` теперь принимает опциональный параметр `anomaly_intervals`
- Если `anomaly_intervals` передан:
  - Вызывается `label_zones()` для разметки на 4 зоны
  - `reference_mask` пересекается с `clean_normal_mask` (reference ∩ clean_normal)
  - `onset_allowed_mask` обновляется через `make_onset_allowed_from_zones()`
  - В `detail` добавляется флаг `zone_aware: True`
- Backward compatibility: если `anomaly_intervals=None`, поведение не меняется

---

### 3.5 Модификация: `alma_service/detection_artifacts.py`

**Что изменено:**
- `DETECTOR_KEYS` — добавлен `"paano_shared"`
- `LOCAL_DETECTOR_KEYS` — добавлен `"paano_shared"`

---

### 3.6 Модификация: `alma_service/generic_detection.py`

**Главная интеграция.** Изменения:

1. **Импорт** `SharedPaAnoDetector`
2. **`LOCAL_DEFAULT_PRIORITY`** — добавлен `"paano_shared": 5` (выше чем paano_feat=3 и pca_spe=4)
3. **`_default_onset_config()`** — `fusion_weight_short=0.60` теперь для `paano_feat` и `paano_shared`
4. **`_prepare_all_wells()`**:
   - Новый параметр `zone_aware: bool = False`
   - Если `zone_aware=True`, передаёт `anomaly_intervals` в `prepare_engineered_well()`
   - **Truncation:** данные каждой скважины обрезаются по `end_date` первого аномального интервала (без пост-аномального нормального сегмента)
5. **`_build_local_runs()`**:
   - Новый параметр `shared_state`
   - Для `paano_shared`: проецирует features на shared каналы, создаёт `SharedPaAnoDetector`
   - Для остальных: прежнее поведение
6. **`run_detection()`**:
   - Фильтрация intervals до первого на скважину (`groupby("well_id").first()`)
   - Shared encoder training phase перед per-well scoring
   - Fallback: если shared pool мал → переключается на `paano_feat`
7. **`run_single_well()`**:
   - Для `paano_shared` → автоматический fallback на `paano_feat` (single-well mode не имеет pool)

---

### 3.7 Модификация: `alma_service/detection_report.py`

**Что изменено:**
- `DETECTOR_LABELS` — добавлен `"paano_shared": "PaAno Shared Encoder"` для красивого отображения в HTML-отчёте

---

### 3.8 Модификация: `alma_service/feature_importance.py`

**Цель:** Поддержка `paano_shared` в анализе важности каналов.

**Что изменено:**
- Импорт `SharedPaAnoDetector`
- `DETECTOR_LABELS` — добавлен `"paano_shared"`
- `_build_detector()` — принимает `shared_state`, создаёт `SharedPaAnoDetector` для `paano_shared`
- `compute_channel_importance()` — принимает `shared_state`, при пертурбации каналов:
  - Shared encoder заморожен (не переобучается)
  - Перестраивается только local memory bank
  - Проекция на shared каналы перед scoring
- `generate_feature_importance_report()` — обучает shared encoder перед анализом всех скважин

---

### 3.9 Модификация: `alma_service/dataset_config.py`

**Исправления данных:**
- `TEST_WELLS["pritok"]` — убрана `"129л"` (перенесена в train)
- `SALT_WELL_FILES` — убрана `"3244г"` (полностью исключена из БД)

---

### 3.10 Тесты

#### `tests/test_zone_labels.py` — 8 тестов:
1. Пустые интервалы → всё CLEAN_NORMAL
2. Один интервал → правильная 4-зонная разметка
3. Буферы ≥ patch_size точек
4. Два интервала → корректная обработка
5. Negermet config → 30/60 мин буферы
6. Pritok config → 60/180 мин буферы
7. Clean normal mask
8. Training exclusion mask
9. Onset allowed mask подавляет буферы и reference

#### `tests/test_shared_encoder.py` — 5 тестов:
1. Intersection каналов при 2 скважинах (ch_y, ch_z общие)
2. Пустой intersection → ValueError
3. Только train split попадает в pool
4. Reference mask фильтрация (только 50 из 200 точек)
5. Column projection (select_shared_columns)

---

## 4. Правки данных

### Parquet файлы:
```bash
# Притока: 129л → train
python -c "import pandas as pd; df=pd.read_parquet('db/pritok_intervals.parquet'); df.loc[df['well_id']=='129л','split']='train'; df.to_parquet('db/pritok_intervals.parquet', index=False)"

# Соль: удалить 3244г
python -c "import pandas as pd; df=pd.read_parquet('db/salt_intervals.parquet'); df=df[df['well_id']!='3244г']; df.to_parquet('db/salt_intervals.parquet', index=False)"
```

---

## 5. Результаты бенчмарков

### Все тесты: 25/25 PASSED ✅

### negermet (`paano_shared`):
| Метрика | Значение |
|---------|----------|
| Hit rate | **5/5 (100%)** |
| Pool size | 9053 точек, 234 канала, 4 train-скважины |
| Training time | ~7 сек (3с short + 4с long) |
| p90 delay ratio | 0.134 |
| FAR/day | 0.400 |
| starts/interval | 2.50 |
| **Default detector** | ✅ **paano_shared** (побил paano_feat и pca_spe) |

### pritok (`paano_shared`):
| Метрика | Значение |
|---------|----------|
| Pool size | 3014 точек, 234 канала, 2 train-скважины |
| Training time | ~13.5 сек |
| **Default detector** | pca_spe (shared не побил baseline) |

### salt (`paano_shared`):
| Метрика | Значение |
|---------|----------|
| Pool size | 17313 точек, 186 каналов, 7 train-скважин |
| Training time | ~7 сек |
| **Default detector** | pca_spe (shared не побил baseline) |

**Итог:** `paano_shared` стал лучшим для negermet. Для pritok/salt — baseline (`pca_spe`) пока лучше, возможно из-за малого train pool (pritok — 2 скважины) и сильного сужения каналов на intersection (salt — 186 из 198–396).

---

## 6. Итоговый список скважин

### NEGERMET
| Split | Скважина | Интервал | Начало → Конец |
|-------|----------|----------|----------------|
| train | 1123л | 1 | 2025-06-21 08:32 → 2025-06-21 23:42 |
| train | 172г | 1 | 2025-05-03 11:30 → 2025-05-07 02:00 |
| train | 524 | 1 | 2025-06-09 19:38 → 2025-06-10 15:30 |
| train | 5271г | 1 | 2025-08-15 00:37 → 2025-08-15 02:46 |
| test | 3509г | 1 | 2025-08-10 09:40 → 2025-08-10 21:25 |

### PRITOK
| Split | Скважина | Интервал | Начало → Конец |
|-------|----------|----------|----------------|
| train | 129л | 1 | 2026-01-10 02:00 → 2026-02-01 02:00 |
| train | 3261 | 1 | 2025-10-23 00:00 → 2025-11-05 02:00 |
| train | 495 | 1 | 2025-10-08 00:00 → 2025-10-17 02:00 |
| test | 902 | 1 | 2025-10-22 00:00 → 2025-11-15 02:00 |

### SALT
| Split | Скважина | Интервал | Начало → Конец |
|-------|----------|----------|----------------|
| train | 149г | 1 | 2025-06-20 15:12 → 2025-06-28 02:00 |
| train | 3245 | 1 | 2025-04-14 16:44 → 2025-05-12 02:00 |
| train | 3245(2) | 1 | 2025-10-04 02:00 → 2025-11-07 02:00 |
| train | 3269 | 1 | 2025-10-15 02:00 → 2025-10-27 11:45 |
| train | 4039 | 1 | 2025-05-29 11:00 → 2025-06-07 15:10 |
| train | 408 | 1 | 2025-06-02 04:40 → 2025-06-26 11:20 |
| test | 2991г | 1 | 2025-12-11 04:00 → 2025-12-24 02:00 |

**Исключены:** 3244г (соль) — полностью удалена из БД.  
**Перемещены:** 129л (приток) — из test в train.  
**Обрезка:** Данные обрезаются по `end_date` первого аномального интервала (408 interval 2 фактически не используется).

---

## 7. Команды для воспроизведения

### Детекция:
```bash
python -m alma_service.generic_detection negermet --detector paano_shared --retune
python -m alma_service.generic_detection pritok --detector paano_shared --retune
python -m alma_service.generic_detection salt --detector paano_shared --retune
```

### Отчёты:
```bash
python scripts/reports/generate_negermet_paano_report.py --detector paano_shared
python scripts/reports/generate_pritok_paano_report.py --detector paano_shared
python scripts/reports/generate_salt_paano_report.py --detector paano_shared
```

### Feature importance:
```bash
python scripts/reports/generate_feature_importance_report.py --anomaly negermet --detector paano_shared
python scripts/reports/generate_feature_importance_report.py --anomaly pritok --detector paano_shared
python scripts/reports/generate_feature_importance_report.py --anomaly salt --detector paano_shared
```

### Тесты:
```bash
python -m pytest tests/ -v
```

---

## 8. Архитектурная схема

```
┌─────────────────────────────────────────────────────────────────┐
│                    ANOMALY FAMILY (e.g. negermet)                │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Well 1123│  │ Well 172г│  │ Well 524 │  │ Well 5271│        │
│  │  (train) │  │  (train) │  │  (train) │  │  (train) │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │              │              │              │              │
│       ▼              ▼              ▼              ▼              │
│  ┌─────────────────────────────────────────────────────┐        │
│  │            4-Zone Labeling (zone_labels.py)          │        │
│  │  clean_normal | pre_buffer | anomaly | post_recovery │        │
│  └────────────────────────┬────────────────────────────┘        │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────┐        │
│  │        Shared Train Pool (clean_normal only)         │        │
│  │    intersection of channels across all train wells    │        │
│  │         9053 points × 234 channels (negermet)         │        │
│  └────────────────────────┬────────────────────────────┘        │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────┐        │
│  │          Shared Encoder Training                     │        │
│  │    PatchEncoder (short=32) + PatchEncoder (long=64)  │        │
│  │              → SharedEncoderState                     │        │
│  └────────────────────────┬────────────────────────────┘        │
│                           │                                      │
│           ┌───────────────┼───────────────┐                      │
│           ▼               ▼               ▼                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │Local Bank #1 │ │Local Bank #2 │ │Local Bank #N │            │
│  │  (well 1123) │ │  (well 172г) │ │  (well 3509) │            │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘            │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌──────────────────────────────────────────────────┐           │
│  │              Anomaly Scores per Well              │           │
│  │     → Onset Detection → Predicted Starts          │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Файлы (полный список изменений)

### Новые файлы:
| Файл | Описание |
|------|----------|
| `alma_service/zone_labels.py` | 4-зонная разметка с per-family буферами |
| `alma_service/shared_encoder.py` | Сбор пула, обучение shared encoder, scoring |
| `tests/test_zone_labels.py` | 8 unit-тестов для зонной разметки |
| `tests/test_shared_encoder.py` | 5 unit-тестов для shared encoder |

### Модифицированные файлы:
| Файл | Описание изменений |
|------|-------------------|
| `alma_service/engineered_features.py` | +anomaly_intervals, zone-aware masks |
| `alma_service/generic_detectors.py` | +SharedPaAnoDetector класс |
| `alma_service/generic_detection.py` | Shared training phase, truncation, fallbacks |
| `alma_service/detection_artifacts.py` | +paano_shared key |
| `alma_service/detection_report.py` | +detector label |
| `alma_service/feature_importance.py` | +paano_shared support |
| `alma_service/dataset_config.py` | 129л→train, убрана 3244г |
| `db/pritok_intervals.parquet` | 129л split=train |
| `db/salt_intervals.parquet` | Удалена 3244г |
