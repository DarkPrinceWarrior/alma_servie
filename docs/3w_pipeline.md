# 3W PaAno pipeline — единый отчёт и roadmap

Объединённый документ: baseline state, ТЗ, roadmap, итоги Phase 0.5 → v3 →
Optuna sweep → physical branches → P10 hook → multi-source A/B, открытые
направления с ранжированием по ROI.

- Старт: 2026-05-12
- Последнее обновление: 2026-05-13
- Branch: `app/webapp`
- Baseline tag: `baseline-pre-3w-2026-05-12`
- Последний коммит: `7784f21` "Phase 0.5++: P10 hook, Optuna sweep, physical branches, multi-source pretrain"

---

## 0. Резюме одной страницей

Реализован полный 3W PaAno pipeline на всех 9 классах Petrobras 3W Dataset 2.0.0,
плюс инфраструктура transfer 3W → ALMA (negermet / pritok / salt).

**Главные числа:**

- **9/9 классов passes Pareto** (FAR/day ≤ 0.10, starts/event ≤ 3.0).
- **mean hit-rate (test) = 0.795** (было 0.618 на v3 без physical branches).
- **5 классов на hit = 1.000** (1, 2, 6, 7, 9).
- Эффективное покрытие (hit > 0): **8/9**; только class 3 SEVERE_SLUGGING
  на hit = 0.188 — единственная оставшаяся дыра по покрытию.

**Transfer 3W → ALMA:**

- P10 integration hook (`load_or_train_shared_encoder`) активен, cache HIT
  валидирован на всех 3 ALMA-аномалиях.
- Best variant: per-class 3W class 9 → ALMA naive transplant. Эффект: -25%
  FAR/day на negermet, -3.3ч delay на pritok train. Salt не двигается
  (encoder не bottleneck).
- Multi-source 3W global NORMAL pretrain + anomaly injection (rate=0.3)
  проверены A/B — слабее class 9 на нашем pool size.
- Production по умолчанию на baseline; transfer-веса лежат в
  `models/3w_class_9_transfer/`, готовы к deployment.

---

## 1. Главный урок A/B (2026-05-13)

Финальная A/B сравнительная таблица 4 вариантов encoder-а на ALMA:

| ALMA | Метрика | baseline | 3w class 9 | 3w global NORMAL | global + injection 0.3 |
|------|---------|---------:|-----------:|-----------------:|-----------------------:|
| negermet | hit (all) | 1.000 | 1.000 | 1.000 | 1.000 |
| negermet | FAR/day (all) | 0.250 | **0.188** | 0.188 | **0.188** |
| negermet | starts (all) | 1.80 | **1.60** | 1.80 | **1.60** |
| pritok | hit (all) | 1.000 | 1.000 | 1.000 | 1.000 |
| pritok | mae_h (train) | 21.01 | **17.75** | 21.73 | 21.00 |
| pritok | FAR/day (all) | 0.0391 | 0.0376 | 0.0391 | 0.0391 |
| salt | все | invariant | invariant | invariant | invariant |

**Что из этого следует:**

1. **На ALMA hit = 1.000 — потолок**: encoder transfer двигает только FAR / delay, не hit-rate.
2. **3W class 9 (per-class) остаётся best**: -25% FAR на negermet, -3.3ч delay на pritok train. Никакая другая комбинация не лучше.
3. **Multi-source хуже class 9** на нашем pool size — intersection 7 классов с NORMAL даёт всего 10 общих каналов, поэтому первый conv обедняется и higher-level conv-блоки underfit.
4. **Salt insensitive к encoder** — bottleneck в onset detector, не в representation.
5. **Anomaly injection на ALMA-pool**: 2-15 windows / inject 0-4 — статистически незначимо. Injection имеет смысл только при больших pool'ах (3W-pretrain, не ALMA-fine-tune).

---

## 2. Открытые направления, ранжированные по ROI

| # | Направление | Сложность | Срок | Ожидаемый эффект |
|---|-------------|-----------|------|------------------|
| 1 | **Slug-period branch для 3W class 3** (SEVERE_SLUGGING, hit = 0.188) — rolling autocorr peak в полосе 5-30 мин, fusion как в class 4/6 | средняя | 1-2 дня | hit 0.188 → 0.5-0.8 (predicted); единственный класс без эффективного покрытия |
| 2 | **Anomaly injection в 3W pretrain** (не в ALMA fine-tune) — pool 324K → 600+ windows / inject 200 → статистически значимо | низкая | 1 день | +5-10% hit/FAR на 3W class 5 и 8 |
| 3 | **Commit + production-deploy** 3W class 9 transfer encoder для negermet/pritok (лучшее что у нас есть, лежит в `models/3w_class_9_transfer/`) | низкая | 30 мин | Production применяет проверенный transfer |
| 4 | **DACAD contrastive + GRL** — следующий шаг согласно изначальному плану, но: ALMA hit-ceiling ограничивает потенциальный gain | высокая | 3-5 дней | Возможно +0-5% на ALMA + ощутимо на 3W weak-classes |
| 5 | **Onset Optuna sweep для salt test FAR=0.023** — analogично class 7/8 | низкая | 30 мин на A100 | -50% FAR на salt test |
| 6 | **Trend-slope branch для 3W class 5** (RAPID_PRODUCTIVITY_LOSS, hit = 0.582) | средняя | 1-2 дня | hit 0.582 → 0.7-0.9 |

**Recommended priority**: #3 (deploy фиксация) → #1 (class 3 slug) → #5 (salt FAR) → #2 (injection в pretrain) → #6 (class 5) → #4 (DACAD как методологический шаг).

---

## 3. 3W acceptance table (final, test split)

Источник чисел: `artifacts/results/3w_benchmark_summary.json` +
`artifacts/3w/metrics/class_<N>_metrics.json`. Дашборды: 9 offline HTML отчётов
в `artifacts/3w/reports/`.

| Класс | Имя | Интервалов | Hit-rate | FAR/сутки | Starts/событие | Pareto ✓ |
|------:|------|----------:|---------:|----------:|---------------:|:--------:|
| 1 | ABRUPT_INCREASE_OF_BSW | 20 | **1.000** | 0.000 | 2.00 | ✓ |
| 2 | SPURIOUS_CLOSURE_OF_DHSV | 5 | **1.000** | 0.000 | 1.00 | ✓ |
| 3 | SEVERE_SLUGGING | 16 | 0.188 | 0.000 | 0.19 | ✓ (physical: 0.125 → 0.188) |
| 4 | FLOW_INSTABILITY | 51 | **0.529** | 0.000 | 0.53 | ✓ (physical: 0.000 → 0.529) |
| 5 | RAPID_PRODUCTIVITY_LOSS | 67 | **0.582** | 0.000 | 0.67 | ✓ |
| 6 | QUICK_RESTRICTION_IN_PCK | 32 | **1.000** | 0.000 | 1.00 | ✓ (physical: 0.000 → 1.000) |
| 7 | SCALING_IN_PCK | 7 | **1.000** | 0.000 | 2.29 | ✓ (Optuna sweep, было 3.14) |
| 8 | HYDRATE_IN_PRODUCTION_LINE | 14 | **0.857** | 0.078 | 1.07 | ✓ (Optuna sweep, было FAR=0.314) |
| 9 | HYDRATE_IN_SERVICE_LINE | 23 | **1.000** | 0.000 | 1.87 | ✓ |

---

## 4. ALMA acceptance reference (production paano_shared, frozen)

Baseline `paano_shared` без transfer (split = all). Production-defaults после
восстановления из `models/baseline_pre_p10transfer_<a>_paano_shared_encoder.pt`:

| Аномалия | Hit-rate | FAR/day | Starts/interval | Median delay (ч) | P90 delay (ч) | Дней наблюдения |
|----------|---------:|--------:|----------------:|-----------------:|--------------:|----------------:|
| negermet | 1.00 | 0.250 | 1.80 | 0.00 | 0.61 | 16 |
| pritok | 1.00 | 0.039 | 3.67 | 4.01 | 59.27 | 691 |
| salt | 1.00 | 0.013 | 2.63 | 0.03 | 19.97 | 707 |

**Acceptance gate**: 3W warm-start в production принимается только если
не ухудшает ни одну из этих величин per-class и `norm_work guard FAR ≤ baseline`.

Зафиксированные detector configs см. ниже в разделе 11.

---

## 5. Что сделано (хронологически)

### Phase 0.5 (2026-05-12 → старт)

- Audit `alma_service/shared_encoder.py`: PatchEncoder channel-count запекается
  в архитектуру первого conv'а; conv2..N channel-agnostic; revin(`affine=False`)
  параметры-агностичный. Следствие: transfer возможен через transplant всех
  слоёв кроме `convblocks[0]`.
- Petrobras 3W Dataset 2.0.0 (commit `227fce3`) клонирован на a100: 3.9 GB,
  2228 instances, 9 anomaly classes + 594 NORMAL (class 0), 1-секундная сетка,
  29 columns max.
- `configs/3w_paano.json`: JSON-конфиг без PyYAML, патчи (32, 64), resample 1min
  median.
- `scripts/datasets/build_3w_dataset.py` (v1): manifest + intervals + features + splits.

### v2 → v3 systemic fix

- **Bug 1**: broken `PAANO_*` imports в `shared_encoder.py` — production
  обходил через `run_detection()`. Fix: добавил константы в `generic_detectors.py`.
- **Bug 2**: NaN-in-features при per-class build — все real-instances получали
  NaN-каналы, MiniBatchKMeans падал. Fix: per-class canonical channels + `np.nan_to_num`.
- **Bug 3**: source-type biased split (`prefer_real_in_test=true`) → distribution
  shift. Fix: `make_splits_balanced` stratifies per (folder_label, source_type).
- v3 dramatic improvement: 5/9 классов passes hit ≥ 0.58.

### P3: Physical branches per class (2026-05-13)

- `scripts/detection/physical_branches_3w.py` + интеграция в `detect_3w.py`
  через `--physical-weight` (auto = 0.7 для классов 4 и 6).
- Class 4 FLOW_INSTABILITY: oscillation-ratio score `P-TPT_roll5m_std /
  P-TPT_roll30m_std` z-scored против reference. Test hit 0.000 → **0.529**.
- Class 6 QUICK_RESTRICTION_IN_PCK: choke step score `max(z(P-MON-CKP −
  P-JUS-CKP), z(P-MON-CKP − roll30m_baseline))`. Test hit 0.000 → **1.000**.
- Fix fallback `reference_mask` в `detect_3w.build_prepared_wells` для классов
  без NORMAL-prefix (3 и 4): `ref_len = N/2` (было `N`, что забирало весь
  инстанс под reference и блокировало onset).

### P2: Optuna sweep classes 7 и 8 (2026-05-13)

- `scripts/evaluation/optuna_sweep_3w.py` с TPE-сэмплером, 400 trials per class,
  расширенное пространство (`hysteresis_scale`, `rearm_window_minutes` дополнительно
  к 4 базовым параметрам).
- Class 7: starts/event 3.14 → 2.29 (cooldown 4 ч → 54 ч).
- Class 8: test FAR 0.314 → 0.078 (cooldown 4 ч → 52 ч, hysteresis 0.6 → 0.31).

### P10: Transfer 3W → ALMA integration (2026-05-13)

- `alma_service/shared_encoder.load_or_train_shared_encoder()` — проверка
  `shared_channels` / `patch_short` / `patch_long` / `anomaly_key`, cache HIT
  через `load_shared_encoder_state()`. Override: `ALMA_FORCE_RETRAIN_ENCODER=1`.
- `alma_service/generic_detection.run_detection()` теперь дёргает wrapper
  вместо `train_shared_encoder()` напрямую.
- `scripts/evaluation/transfer_3w_to_alma.py`: флаг `--use-target-patches`
  (fine-tune под ALMA-родные patch sizes); skip channel-dependent `convblocks[0]`,
  transplant conv2..N + projection_head + classification_head.
- Транзит подтверждён: negermet -25% FAR, pritok -3.3ч delay на train,
  salt invariant.

### Multi-source pretrain + anomaly injection A/B (2026-05-13)

- `scripts/detection/build_3w_global_encoder.py`: global NORMAL encoder (pool
  324K rows × 10 shared channels через intersection 7 классов с NORMAL).
- `alma_service/anomaly_injection.py`: synthetic anomaly transforms (spike /
  scale-shift / collective-flip / jitter).
- `scripts/evaluation/transfer_3w_to_alma.py`: новые флаги `--source-class global`
  и `--anomaly-injection-rate`.
- `scripts/evaluation/compare_transfer_variants.py`: A/B compare-script,
  переиспользуем для последующих экспериментов.
- Вывод: per-class 3W class 9 остаётся best (см. раздел 1).

### P11: 3W NORMAL guard (research only, deferred)

- `scripts/detection/normal_guard_3w.py`: prototype prior fitter (cross-instance
  per-channel quantile envelope) на 594 NORMAL инстансах.
- Результат: envelope получился слишком широкий ([0, 21M] Pa на P-TPT) из-за
  межскважинной диверсии режимов — guard не дискриминативен.
- Production-ready версия требует per-well median-normalization → приближается
  к существующему reference_mask. Deferred до момента когда FAR > 0.10 на test.

---

## 6. ТЗ (исходное, summary)

Подробное ТЗ: оригинал в `docs/archive/TASK_3W_PaAno_ALMA_detector.md`.

**Цель**: построить production-like 3W detector + опционально transfer
encoder в ALMA. Использовать 3W как доменный нефтяной pretrain/benchmark,
а не как замену данным заказчика.

**Реализованная схема**:

```
3W raw Parquet
  → manifest + validation
  → interval extraction
  → resampling + cleaning + per-class canonical features
  → instance-level train/val/test split (stratified by source_type)
  → PaAno shared encoder pretrain per-class (Phase 0.5)
  → memory bank + onset calibration
  → Optuna-tuned thresholds + physical branches per class
  → metrics + offline Plotly HTML reports
  → transfer encoder weights into ALMA pipeline (P10 hook)
```

**Запреты** (закреплены): нельзя делить построчно, нельзя tune на test,
нельзя смешивать labels в признаки, нельзя делать новые production detector
keys (`paano_shared` остаётся единственным inference API).

---

## 7. Архитектурные решения (зафиксированы)

1. **Раскладка скриптов**: per-anomaly в существующем стиле (`scripts/datasets/`,
   `scripts/detection/`, `scripts/evaluation/`), не отдельный namespace.
2. **Config format**: JSON (`configs/3w_paano.json`).
3. **Resample**: 1min median для numeric, mode для `class`/`state`.
4. **Channel set**: intersection non-NaN columns per class (отдельный
   `shared_channels` per event class).
5. **Splits**: instance-level, stratified by (event_label, source_type),
   seed = 2027.
6. **Acceptance objective**: hard constraint (FAR/day ≤ 0.10, starts/event ≤ 3)
   + lexicographic max hit_rate → min median_delay → min p90_delay.
7. **Detector key**: `paano_shared` (один inference API), варьируется
   `encoder_training_mode` в metadata.
8. **Transfer mechanism**: transplant весов кроме `convblocks[0]`, fine-tune
   200-300 итераций на ALMA train normal pool.

**Training mode taxonomy** (metadata всех encoder-ов):

```
class_specific                          — текущий ALMA production baseline
alma_global_pretrain_class_finetune     — global pretrain на ALMA-only нормах
3w_pretrain_only                        — только 3W
3w_pretrain_alma_finetune               — 3W warm-start + ALMA fine-tune (рабочий transfer)
3w_multi_source_normal_pretrain         — multi-source global NORMAL (A/B вариант)
```

---

## 8. Окружение и hardware

```
Server:        a100 (192.168.101.12, доступ через a100-remote с jump host)
Repo:          /root/projects/alma_servie (source of truth)
Python:        3.13.5 (uv)
Torch:         2.11.0+cu130
CUDA:          13.0
GPU:           6× NVIDIA A100-SXM4-40GB
               GPU0 занят (~8.4 GiB), GPU1..5 свободны (~40 GiB каждая)
Submodule:     paano (commit 0e93e93, gitlink без .gitmodules)
```

---

## 9. Артефакты (структура)

```
data/raw/3w/3W/                                       Petrobras 3W, commit 227fce3 (3.9 GB)
data/processed/3w/
    manifest.parquet
    intervals.parquet
    splits.parquet                                    source-type balanced
    build_summary.json
    class_<N>/features.parquet                        canonical channels, NaN-safe
artifacts/3w/
    checkpoints/3w_class_<N>_paano_shared_encoder.pt  9 per-class encoders
    checkpoints/3w_global_normal_paano_shared_encoder.pt  multi-source encoder
    checkpoints/<a>_global*_3w_transfer_encoder.pt    transfer A/B variants
    scores/class_<N>_scores.parquet
    scores/class_<N>_predicted_starts.parquet
    metrics/class_<N>_selected.json                   Optuna-tuned configs
    metrics/class_<N>_metrics.json                    final test metrics
    metrics/transfer_<src>_to_<anomaly>.json          transplant reports
    reports/3w_class_<N>_<EVENT>.html                 9 offline HTML
artifacts/results/
    3w_benchmark_summary.json                         9-class aggregate
    transfers/baseline_pre_hook/                      ALMA before transfer
    transfers/3w_class_9_v2/                          ALMA with class-9 transfer
    transfers/global_normal/                          ALMA with global NORMAL
    transfers/global_inject/                          ALMA with global + injection
models/
    <anomaly>_paano_shared_encoder.pt                 ALMA production weights (baseline)
    baseline_2026-05-12_<a>_paano_shared_encoder.pt   frozen pre-3W snapshot
    baseline_pre_p10transfer_<a>_paano_shared_encoder.pt  pre-transfer snapshot
    3w_class_9_transfer/<a>_paano_shared_encoder.pt   recommended transfer weights
configs/3w_paano.json                                 JSON, без PyYAML
docs/
    3w_pipeline.md                                    этот файл
    archive/{baseline_state_2026-05-12,roadmap_3w,TASK_3W_PaAno_ALMA_detector,3w_phase05_status,
            3w_phase05_diagnostic,3w_phase05_diagnostic_v3}.md
```

Сетка: 57 MB отчётов + 15 MB scores + 27 MB encoders. Артефакты в `.gitignore`.

---

## 10. Условия отката

```bash
# Вернуться к baseline на сервере
ssh a100-remote 'cd /root/projects/alma_servie && git checkout baseline-pre-3w-2026-05-12'

# Восстановить production encoders из frozen-копий
ssh a100-remote 'cd /root/projects/alma_servie && \
  cp models/baseline_2026-05-12_negermet_paano_shared_encoder.pt models/negermet_paano_shared_encoder.pt && \
  cp models/baseline_2026-05-12_pritok_paano_shared_encoder.pt   models/pritok_paano_shared_encoder.pt && \
  cp models/baseline_2026-05-12_salt_paano_shared_encoder.pt     models/salt_paano_shared_encoder.pt'

# Развернуть 3W class 9 transfer encoder в production
ssh a100-remote 'cd /root/projects/alma_servie && \
  cp models/3w_class_9_transfer/negermet_paano_shared_encoder.pt models/negermet_paano_shared_encoder.pt && \
  cp models/3w_class_9_transfer/pritok_paano_shared_encoder.pt   models/pritok_paano_shared_encoder.pt'
# (salt не трогаем — invariant к encoder)
```

---

## 11. ALMA detector configs (frozen baseline)

### negermet

```json
{
  "target_far_per_day": 0.5,
  "min_run_points": 4,
  "cooldown_hours": 8.0,
  "rearm_window_minutes": 60.0,
  "ema_alpha": 0.08,
  "gate_mode": "relaxed",
  "hysteresis_scale": 0.6,
  "bypass_cooldown_after_clear": true,
  "fusion_weight_short": 0.6,
  "negermet_signature_weight": 0.0
}
```

### pritok

```json
{
  "target_far_per_day": 0.25,
  "min_run_points": 3,
  "cooldown_hours": 120.0,
  "rearm_window_minutes": 240.0,
  "ema_alpha": 0.08,
  "gate_mode": "score_ema",
  "hysteresis_scale": 0.6,
  "bypass_cooldown_after_clear": false,
  "fusion_weight_short": 0.6,
  "pressure_trend_weight": 0.0025
}
```

### salt

```json
{
  "target_far_per_day": 0.1,
  "min_run_points": 4,
  "cooldown_hours": 96.0,
  "rearm_window_minutes": 960.0,
  "ema_alpha": 0.04,
  "gate_mode": "relaxed",
  "hysteresis_scale": 0.6,
  "bypass_cooldown_after_clear": false,
  "fusion_weight_short": 0.6,
  "salt_trend_weight": 0.01
}
```

---

## 12. Git log (хронология коммитов app/webapp)

```
7784f21  Phase 0.5++: P10 hook, Optuna sweep, physical branches, multi-source pretrain
2376bcc  Add 3W diagnostic snapshots (v1 baseline and v3 post-fix)
074091b  3W PaAno v3 + P10 transfer mechanism
dea5dd5  Finalize 3W Phase 0.5: per-class metrics + sweep summary + status
b8873ad  Add 3W Phase 0.5 helpers: worker script, rerun script, aggregator
4be6b74  Fix broken PAANO_* imports + 3W scoring/report edge cases
16f0462  Add 3W PaAno pipeline scripts and parallel launcher
5ab09f1  Start 3W PaAno pipeline: config + roadmap
022782d  Freeze pre-3W baseline state and acceptance reference
36bd748  Document class specific versus global pretrain (baseline)
```

---

## 13. Roadmap (продолжение)

| ID | Задача | Статус |
|----|--------|:------:|
| P0–P9 | Phase 0.5 baseline | ✓ 2026-05-12 |
| Diag | train → test gap analysis | ✓ 2026-05-12 |
| Sys-fix | build v3 + balanced split + NaN-safe | ✓ 2026-05-13 |
| P2 | Optuna sweep (расширенный grid) | ✓ 2026-05-13 (class 7, 8) |
| P3 | Physical branches per class | ✓ 2026-05-13 (class 4, 6); class 3 / class 5 — открыты |
| P10 mech | Transfer 3W → ALMA mechanism | ✓ research validated |
| P10 integ | run_detection load-saved hook | ✓ 2026-05-13 |
| Multi-src | Global NORMAL pretrain A/B | ✓ 2026-05-13 (вариант хуже class 9) |
| Injection | Anomaly injection в ALMA fine-tune A/B | ✓ 2026-05-13 (insignificant на small pool) |
| P11 | 3W NORMAL guard для production | ✓ prototype, deferred deploy |
| #1 | Slug-period branch class 3 | TODO (приоритет высокий) |
| #3 | Production deploy class 9 transfer | TODO (30 мин, low-friction) |
| #5 | Optuna sweep salt test FAR | TODO |
| #2 | Anomaly injection в 3W pretrain (не ALMA) | TODO |
| #6 | Trend-slope branch class 5 | TODO |
| #4 | DACAD contrastive + GRL | TODO (опц., методологический) |

---

## 14. Команды быстрого старта

```bash
# Полный 3W pipeline
ssh a100-remote 'cd /root/projects/alma_servie && \
  uv run python scripts/datasets/build_3w_dataset.py --config configs/3w_paano.json && \
  bash scripts/run_3w_v3_full.sh'

# Per-class detect + tune + report
ssh a100-remote 'cd /root/projects/alma_servie && \
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/detection/detect_3w.py --event-class 4 && \
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluation/optuna_sweep_3w.py --event-class 4 --trials 300 && \
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/reports/generate_3w_report.py --event-class 4'

# Transfer 3W class N → ALMA anomaly
ssh a100-remote 'cd /root/projects/alma_servie && \
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluation/transfer_3w_to_alma.py \
    --source-class 9 --target-anomaly negermet --use-target-patches --fine-tune-iters 300'

# A/B compare всех transfer вариантов
ssh a100-remote 'cd /root/projects/alma_servie && \
  uv run python scripts/evaluation/compare_transfer_variants.py'
```

---

## 15. Источники и ссылки

- Petrobras 3W Dataset 2.0.0: https://github.com/petrobras/3W (commit 227fce3)
- 3W article (arXiv:2507.01048): https://arxiv.org/abs/2507.01048
- 3W structure: https://github.com/petrobras/3W/blob/main/3W_DATASET_STRUCTURE.md
- PaAno: https://github.com/jinnnju/PaAno (commit 0e93e93)
- PaAno paper: https://arxiv.org/abs/2602.01359 / OpenReview NXThkM7Iym
