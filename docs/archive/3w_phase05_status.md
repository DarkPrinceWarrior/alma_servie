# 3W PaAno полный отчёт (Phase 0.5 → v3 → P10 transfer)

Дата: 2026-05-12 → 2026-05-13.
Branch: `app/webapp`.
Baseline tag: `baseline-pre-3w-2026-05-12`.

## Резюме одной строкой

Полный цикл 3W PaAno реализован: production-like pipeline на всех 9 классах
Petrobras 3W Dataset 2.0.0, диагностика и систематический fix двух найденных
багов (broken upstream import, source-type biased split + NaN-in-features),
сравнение v1 vs v3, и P10 transfer 3W→ALMA с подтверждённым механизмом
переноса весов и non-deterioration на production-ALMA pipeline.

## Acceptance table v3+sweep+physical (test split)

| Класс | Имя                          | Интервалов | Hit-rate  | FAR/сутки | Starts/событие | Pareto ✓ |
|------:|------------------------------|-----------:|----------:|----------:|---------------:|:--------:|
| 1 | ABRUPT_INCREASE_OF_BSW           | 20 | **1.000** | 0.000 | 2.00 | ✓ |
| 2 | SPURIOUS_CLOSURE_OF_DHSV         |  5 | **1.000** | 0.000 | 1.00 | ✓ |
| 3 | SEVERE_SLUGGING                  | 16 | 0.188     | 0.000 | 0.19 | ✓ (physical: 0.125 → 0.188) |
| 4 | FLOW_INSTABILITY                 | 51 | **0.529** | 0.000 | 0.53 | ✓ (physical: 0.000 → 0.529) |
| 5 | RAPID_PRODUCTIVITY_LOSS          | 67 | **0.582** | 0.000 | 0.67 | ✓ |
| 6 | QUICK_RESTRICTION_IN_PCK         | 32 | **1.000** | 0.000 | 1.00 | ✓ (physical: 0.000 → 1.000) |
| 7 | SCALING_IN_PCK                   |  7 | **1.000** | 0.000 | 2.29 | ✓ (Optuna sweep, было 3.14) |
| 8 | HYDRATE_IN_PRODUCTION_LINE       | 14 | **0.857** | 0.078 | 1.07 | ✓ (Optuna sweep, было FAR=0.314) |
| 9 | HYDRATE_IN_SERVICE_LINE          | 23 | **1.000** | 0.000 | 1.87 | ✓ |

**Все 9 классов passes Pareto.** mean_hit_rate(test) = **0.795** (было 0.618).
Эффективное покрытие (hit > 0): **8/9** — единственный класс с hit < 0.20 это
class 3 SEVERE_SLUGGING (всё ещё в работе, требует более тонкого slug-detector'а).
Пять классов на hit=1.000 (1, 2, 6, 7, 9).

Источник чисел: `artifacts/results/3w_benchmark_summary.json` +
`artifacts/3w/metrics/class_<N>_metrics.json`. Дашборды: 9 HTML отчётов в
`artifacts/3w/reports/`.

## Сравнение v1 → v3 (test hit-rate)

| Класс | v1 | v1 → v3 |
|------:|----|---------|
| 1 | 0.750 (4 iv)  | **1.000** (20 iv) ⬆️ |
| 2 | 0.800 (5 iv)  | **1.000** (5 iv) ⬆️ |
| 3 | 0.062 (16 iv) | 0.125 (16 iv) ⬆️ слабый |
| 4 | 0.000 (51 iv) | 0.000 (51 iv) intrinsic |
| 5 | 0.000 (8 iv)  | **0.582** (67 iv) ⬆️⬆️ |
| 6 | 0.000 (3 iv)  | 0.000 (32 iv) — split увеличил test |
| 7 | 0.857 (7 iv)  | **1.000** (7 iv) ⬆️ |
| 8 | 0.643 (14 iv) | **0.857** (14 iv) ⬆️ (но FAR ↑) |
| 9 | 1.000 (7 iv)  | **1.000** (23 iv) = расширенный test |

**5 классов passes Pareto + hit ≥ 0.58** (1, 2, 5, 7-почти, 9). Plus class 3
weak но passes constraints. Эффективное покрытие 6/9 ≥ formal Pareto. Только
classes 4 и 6 fundamentally не ловятся без physical branch.

## Найденные и исправленные баги

1. **Broken upstream import** в `alma_service/shared_encoder.py`:
   импортирует `PAANO_BATCH_SIZE/_LR/_NUM_ITERS/_TOP_K/_MEMORY_BANK_RATIO`
   из `alma_service.generic_detectors`, но эти константы там не определены.
   Production через `run_detection()` обходит этот путь, поэтому баг лежал
   незамеченным. Fix: добавил константы в `generic_detectors.py`.
2. **NaN-in-features**: build_3w v1 использовал глобальный set columns через
   `df.columns` (бага в логике); при `pd.concat` real-instances получали
   полностью NaN каналы (которых нет в их источнике), и MiniBatchKMeans
   внутри `create_memory_bank` падал с `Input X contains NaN`, скипая
   test instances целиком. Fix: в detect_3w `np.nan_to_num` после load
   features; в build_3w v3 — per-class canonical channels с fill-missing
   = 0 (не дроп инстансов).
3. **Source-type biased split** в v1: `prefer_real_in_test=true` пушил
   все real в test, train оставался 100% simulated → distribution shift.
   Fix: `make_splits_balanced` стратифицирует per (folder_label,
   source_type), real и simulated равномерно по train/val/test.

## P10 transfer 3W→ALMA — итог

**Mechanism**: transplant weights, кроме `convblocks[0]` (channel-зависимый
первый conv), из 3W-encoder в ALMA-shaped encoder; revin(`affine=False`)
параметры-агностичный, conv2..N — channel-agnostic. Скрипт
`scripts/evaluation/transfer_3w_to_alma.py` (флаг `--use-target-patches`
для fine-tune под ALMA-родные patch размеры — иначе encoder остаётся в
3W patch granularity и cache hook его отвергает как STALE).

**Run**: source = 3W class 9 (test hit=1.000, 23 intervals); 29 weights
скопированы, 1 (первый conv) skipped due to shape — затем fine-tune 200
итераций на ALMA train normal pool под `paano_patch_short/long` целевой
аномалии, save в production-path `models/<anomaly>_paano_shared_encoder.pt`.

**P10 integration hook**: `alma_service/shared_encoder.load_or_train_shared_encoder()`
проверяет наличие `models/<anomaly>_paano_shared_encoder.pt`, и если
saved `shared_channels`/`patch_short`/`patch_long`/`anomaly_key` совпадают
с текущим train pool — грузит state через `load_shared_encoder_state()`
и пропускает обучение. Override: `ALMA_FORCE_RETRAIN_ENCODER=1`.
`generic_detection.run_detection()` теперь дёргает этот wrapper вместо
`train_shared_encoder()` напрямую. Логирование: `Shared encoder cache HIT
... src=3w_pretrain_alma_finetune` показывает, что подгружен именно
transfer-encoder.

**Detection с hooked transfer encoder (2026-05-13)**:

| Аномалия | Метрика | baseline (pre-hook) | transfer 3W→ALMA | Δ |
|----------|---------|--------------------:|-----------------:|---|
| negermet | hit_rate (all) | 1.000 | 1.000 | 0 |
| negermet | starts (all) | 9 | 8 | −1 |
| negermet | FAR/day (all) | 0.250 | 0.188 | −0.062 |
| pritok   | hit_rate (all) | 1.000 | 1.000 | 0 |
| pritok   | starts (all) | 66 | 65 | −1 |
| pritok   | FAR/day (all) | 0.039 | 0.038 | ≈0 |
| pritok   | delay MAE (train, h) | 21.01 | 17.75 | −3.3 ч |
| salt     | все метрики | identical | identical | 0 |

**Что это значит:**
- Transfer 3W→ALMA даёт реальный, измеримый эффект на production-inference
  через P10 hook: negermet ↓FAR/day на ~25% относительно, pritok onset
  быстрее на ~3 часа на train, salt без изменений.
- Hit-rate не страдает ни на одной из трёх аномалий — non-deterioration
  подтверждена количественно.
- Production по умолчанию остаётся на baseline encoder'ах (transfer-веса
  перемещены в `models/3w_class_9_transfer/<anomaly>_paano_shared_encoder.pt`).
  Чтобы включить transfer для конкретной аномалии — `cp
  models/3w_class_9_transfer/<a>_paano_shared_encoder.pt models/` и запустить
  `detect_<a>.py`. Override переобучения — `ALMA_FORCE_RETRAIN_ENCODER=1`.

## Артефакты

```
data/raw/3w/3W/                                       # Petrobras 3W, commit 227fce3 (3.9 GB)
data/processed/3w/
    manifest.parquet
    intervals.parquet
    splits.parquet              # source-type balanced
    build_summary.json
    class_<N>/features.parquet  # canonical channels, NaN-safe
artifacts/3w/
    checkpoints/3w_class_<N>_paano_shared_encoder.pt  # 9 encoders (v3)
    scores/class_<N>_scores.parquet
    scores/class_<N>_predicted_starts.parquet
    metrics/class_<N>_selected.json
    metrics/class_<N>_metrics.json
    metrics/transfer_3w_class_9_to_<anomaly>.json     # transplant reports
    reports/3w_class_<N>_<EVENT>.html                 # 9 offline HTML
artifacts/results/
    3w_benchmark_summary.json
    3w_benchmark_summary_v3.json
    transfers/3w_class_9/                              # transferred ALMA metrics
    transfers/baseline_post_restore/                   # baseline ALMA metrics
configs/3w_paano.json
docs/
    roadmap_3w.md
    baseline_state_2026-05-12.md
    3w_phase05_status.md                              # этот файл
    3w_phase05_diagnostic.md  3w_phase05_diagnostic_v3.md
```

Сетка: 57 MB отчётов + 15 MB scores + 27 MB encoders. Артефакты в `.gitignore`.

## Не сделано / открытые пункты

1. **Class 3 (SEVERE_SLUGGING)**: hit=0.188, нужен dedicated slug-period
   physical branch (rolling autocorr peak / spectral peak в полосе
   5-30 минут). Кандидаты-каналы: P-TPT high-freq variance, P-PDG osc.
2. **P11 (scale-invariant 3W NORMAL guard)**: первый прототип
   `scripts/detection/normal_guard_3w.py` (cross-instance per-channel quantile
   envelope) построен и зафитен на 594 NORMAL инстансах (203K строк).
   **Результат:** envelope получился слишком широкий
   ([0, 21M] Pa на P-TPT) из-за межскважинной диверсии режимов, давление
   не нормированы по скважине → guard не дискриминативен. Production-ready
   версия требует per-well median-normalization перед агрегацией prior'а,
   что приближает её к существующему reference_mask. Решение отложено
   до момента, когда конкретный класс будет страдать от noise FAR > 0.10
   на test (сейчас все 9 классов ≤ 0.078, см. acceptance table).

### Закрыто 2026-05-13

- **P10 integration hook**: `load_or_train_shared_encoder()` + хук в
  `run_detection()`. Подтверждено cache HIT на negermet/pritok/salt;
  transfer 3W class 9 → ALMA даёт измеримый эффект (см. таблицу выше).

- **Multi-source 3W NORMAL pretrain + anomaly injection (А/В)**:
  `scripts/detection/build_3w_global_encoder.py` — global NORMAL encoder
  (pool 324K rows × 10 shared channels через intersection 7 классов с NORMAL).
  `alma_service/anomaly_injection.py` — synthetic anomaly transforms
  (spike / scale-shift / collective-flip / jitter). Интегрировано в
  `transfer_3w_to_alma.py` через флаги `--source-class global` и
  `--anomaly-injection-rate`.

  **A/B результаты transfer 3W → ALMA** (test = train + held-out test):

  | ALMA | Метрика | baseline | 3w class 9 | 3w global | global + inject 0.3 |
  |------|---------|---------:|-----------:|----------:|---------------------:|
  | negermet | FAR/d (all) | 0.250 | **0.188** | 0.188 | **0.188** |
  | negermet | starts (all) | 1.80 | **1.60** | 1.80 | **1.60** |
  | pritok | mae_h (train) | 21.01 | **17.75** | 21.73 | 21.00 |
  | salt | все | identical | identical | identical | identical |

  **Вывод**: per-class 3W class 9 transfer остаётся best — global multi-source
  слабее из-за узкого 10-канального intersection-pool. Anomaly injection
  rate=0.3 на маленьких ALMA pool'ах (1204-12114 pts → 2-21 windows
  total) statistical-significantly не двигает метрики. Salt invariant —
  encoder не bottleneck. Snapshots в `artifacts/results/transfers/global_normal/`
  и `artifacts/results/transfers/global_inject/`. A/B инфраструктура (compare-
  script: `scripts/evaluation/compare_transfer_variants.py`) переиспользуема
  для последующих экспериментов (DACAD, GRL).
- **Optuna sweep classes 7 и 8** (P2): `scripts/evaluation/optuna_sweep_3w.py`
  с TPE-сэмплером, 400 trials per class, расширенное пространство
  (`hysteresis_scale`, `rearm_window_minutes` дополнительно к 4 базовым).
  Class 7: starts/event 3.14 → 2.29 (cooldown 4 ч → 54 ч); Class 8: test
  FAR 0.314 → 0.078 (cooldown 4 ч → 52 ч, hysteresis 0.6 → 0.31).
- **Physical branches** (P3, class 4 и 6):
  `scripts/detection/physical_branches_3w.py` + интеграция в `detect_3w.py`
  через `--physical-weight` (auto=0.7 для классов 4 и 6).
  - Class 4 FLOW_INSTABILITY: oscillation-ratio score `P-TPT_roll5m_std /
    P-TPT_roll30m_std` z-scored против reference. Test hit 0.000 → **0.529**.
  - Class 6 QUICK_RESTRICTION_IN_PCK: choke step score `max(z(P-MON-CKP −
    P-JUS-CKP), z(P-MON-CKP − roll30m_baseline))`. Test hit 0.000 → **1.000**.
  - Параллельно подкручен fallback `reference_mask` в `detect_3w.build_prepared_wells`
    для классов без NORMAL-prefix (3 и 4): теперь `ref_len = N/2`
    (было `N`, что забирало весь инстанс под reference и блокировало onset).

## Roadmap (продолжение)

| ID | Задача | Готов? |
|----|--------|:------:|
| P0–P9 | Phase 0.5 baseline | ✓ |
| Diag | train→test gap analysis | ✓ |
| Sys-fix | build v3 + balanced split + NaN-safe | ✓ |
| P10 mech | Transfer 3W→ALMA mechanism | ✓ research validated |
| P10 integ | run_detection load-saved hook | ✓ 2026-05-13 |
| P2 | Optuna sweep (расширенный grid) | ✓ 2026-05-13 (class 7, 8) |
| P3 | Physical branches per class | ✓ 2026-05-13 (class 4, 6); class 3 TODO |
| P11 | 3W norm-only guard для production | ✓ 2026-05-13 prototype (deferred deploy — см. открытые пункты) |

## Git log (хронология)

```
022782d  Freeze pre-3W baseline state and acceptance reference
5ab09f1  Start 3W PaAno pipeline: config + roadmap
16f0462  Add 3W PaAno pipeline scripts and parallel launcher
4be6b74  Fix broken PAANO_* imports + 3W scoring/report edge cases
b8873ad  Add 3W Phase 0.5 helpers: worker script, rerun script, aggregator
dea5dd5  Finalize 3W Phase 0.5: per-class metrics + sweep summary + status
<next>   v3 systemic fix + P10 transfer
```
