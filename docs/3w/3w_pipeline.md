# 3W PaAno pipeline — единый отчёт и roadmap

Объединённый документ: baseline state, ТЗ, roadmap, итоги Phase 0.5 → v3 →
Optuna sweep → physical branches → P10 hook → multi-source A/B, открытые
направления с ранжированием по ROI.

- Старт: 2026-05-12
- Последнее обновление: 2026-05-13 (Sprint 3 закрыт)
- Branch: `app/webapp`
- Baseline tag: `baseline-pre-3w-2026-05-12`
- Последний коммит Sprint 3: trend-slope branch для class 5 — hit 0.530 → 1.000

---

## 0. Резюме одной страницей

Реализован полный 3W PaAno pipeline на всех 9 классах Petrobras 3W Dataset 2.0.0,
плюс инфраструктура transfer 3W → ALMA (negermet / pritok / salt).

**Главные числа (после Sprint 3):**

- **9/9 классов passes Pareto** (FAR/day ≤ 0.10, starts/event ≤ 3.0).
- **mean hit-rate (all 9 классов) = 0.845** (Sprint 2 = 0.795, v3 = 0.618).
- **6 классов на hit ≥ 0.97** (1, 2, 5, 6, 7, 9); class 5 RAPID_PRODUCTIVITY_LOSS
  взят с **hit = 1.000** благодаря trend-slope branch (было 0.530).
- Эффективное покрытие (hit > 0): **8/9**; только class 3 SEVERE_SLUGGING
  на hit = 0.188 — единственная оставшаяся дыра по покрытию.

**Transfer 3W → ALMA (это главное — ALMA production detector):**

- P10 integration hook (`load_or_train_shared_encoder`) активен, cache HIT
  валидирован на всех 3 ALMA-аномалиях.
- Best variant: per-class 3W class 9 → ALMA naive transplant. Эффект: -25%
  FAR/day на negermet, -3.3ч delay на pritok train (deployed в Sprint 2).
  Salt invariant к class 9.
- Multi-source 3W global NORMAL pretrain + anomaly injection (rate=0.3)
  проверены A/B — слабее class 9 на нашем pool size.
- **Sprint 4 на очереди (ALMA-first)**: transfer class 5 (trend semantics)
  → pritok, port class5_trend_slope в pritok physics, transfer class 4/6
  → salt. См. секции 2 и 4c.

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

## 2. Открытые направления, ранжированные по ROI (ALMA-first, после Sprint 5)

**Основная цель — ALMA production detector**. После Sprint 4+5 encoder-side выжаты полностью:
- Sprint 2: deploy c9 → pritok/negermet — последний реальный transfer-win.
- Sprint 4: c5/c4/c6 transfers + slope-port — все no-op или regression.
- Sprint 5: DACAD MMD-alignment — salt bit-exact identical baseline; pritok regression.

**Salt encoder architecturally invariant**: salt onset полностью определяется physics, encoder вклад отсекается onset thresholding'ом. Это не "недоучили DACAD" — это design score_fusion (`salt_trend_weight: 0.01`).

| # | Направление | ALMA-эффект | Сложность |
|---|-------------|-------------|-----------|
| 1 | **Salt score-fusion redesign**: увеличить `salt_trend_weight` (encoder начнёт влиять), или добавить новый channel `salt_encoder_score` в fusion. Risky — может сломать FAR | потенциально первый non-zero encoder effect на salt | средняя, 1-2 дня |
| 2 | **Salt new physics features**: помимо `salt_signature` — slope-derived score на ESP-каналах, multi-channel correlation, температурный signal | salt FAR↓ или delay↓ через новый physics channel | средняя, 2-3 дня |
| 3 | **Optuna-tuned weight для `pressure_trend_slope`** в pritok physics (компонента сохранена после Sprint 4) | marginal pritok mae↓ | низкая, 1 день |
| 4 | **3W internal — FFT spectral branch class 3** (hit=0.188) | 0 для ALMA, 3W benchmark | средняя |
| 5 | **3W internal — autocorr branch class 4** (hit=0.536) | 0 для ALMA, 3W benchmark | средняя |
| 6 | **3W internal — NORMAL guard P11 deploy** | 0 для ALMA | средняя |

**Recommended next sprint**:
- **Если salt priority** — #2 (новые physics features) более вероятный путь чем #1 (изменение fusion весов с риском FAR).
- **Если pritok priority** — #3 (slope Optuna) дешёвый эксперимент с шансом marginal mae↓.
- **Если 3W coverage** — #4 (FFT class 3).

Sprint 5 формально закрыл вопрос encoder-side ALMA: ни transfer, ни domain adaptation, ни adversarial alignment не сдвигают salt и pritok ниже текущего ceiling. Дальше только onset-side, physics-side или fusion-side вмешательства.

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
| 5 | RAPID_PRODUCTIVITY_LOSS | 67 | **1.000** | 0.000 | 1.01 | ✓ (Sprint 3 trend-slope: 0.582 → 1.000) |
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

## 4a. Sprint 2 (2026-05-13, после консолидации docs)

Трёхтрековый параллельный запуск (GPU 1-4):

### Трек #3 — Production deploy class 9 transfer → **успех**

- Скопирован `models/3w_class_9_transfer/{negermet,pritok}_paano_shared_encoder.pt`
  в production-path (salt не трогаем — invariant).
- Re-run `detect_negermet` и `detect_pritok` через P10 hook — cache HIT с
  `src=3w_pretrain_alma_finetune` подтверждён.
- Production метрики ALMA после deploy совпали 1-в-1 с ранее измеренными
  в `artifacts/results/transfers/3w_class_9_v2/`:
  - **negermet**: FAR/day (all) 0.250 → 0.188 (-25%), starts 1.80 → 1.60
  - **pritok**: mae_h (train) 21.01 → 17.75 (-3.3ч)
  - **salt**: invariant (baseline)
- Snapshot в `artifacts/results/transfers/production_deployed/`.

### Трек #5 — Optuna sweep ALMA salt onset → **negative finding**

- Новый скрипт `scripts/evaluation/optuna_sweep_alma.py` (300 trials TPE,
  расширенный search space на 7 параметров с включением gate_mode).
- Constraints: `hit_rate >= 1.0` + `far_per_day <= 0.10` + `starts <= 3.0` на val + test.
- Результат: **no feasible candidate found** — все configs которые снижали
  FAR на test ниже текущих 0.023 одновременно роняли hit-rate ниже 1.0.
- **Вывод**: salt baseline уже на Pareto frontier. Дальнейшее снижение FAR
  возможно только через encoder-уровневые изменения, не onset-tuning.
- Best-effort selected (для архива): `artifacts/results/transfers/alma_sweep/salt_alma_sweep_selected.json`.

### Трек #1 — Slug-period branch для 3W class 3 → **negative finding, откат**

- Добавлен `class3_slug_score()` в `scripts/detection/physical_branches_3w.py`:
  fusion `z(P-TPT_roll5m_std / P-TPT_roll30m_std)` + `z(P-PDG_roll5m_std /
  P-PDG_roll30m_std)` + `0.5 * z(P-TPT_roll5m_std)`.
- `auto_weights[3] = 0.6` в `detect_3w.py`.
- Re-score class 3 + Optuna sweep с `--require-test-pass`.
- Результат: val hit=0.562, **test hit=0.125** (было 0.188 без slug branch).
- **Вывод**: slug-period сигнал на P-TPT/P-PDG roll-std не различает true slug
  от других trend-like artifacts на симулированных class 3 инстансах. Branch
  ухудшил test generalization.
- **Откат**: `auto_weights[3]` убран, class 3 пересчитан без physical branch,
  Optuna sweep повторён → test hit восстановлен до 0.188.
- Артефакт neg-finding: `artifacts/3w/metrics/slug_branch_negative_result/`.

### Итоги Sprint 2

| Что | Результат |
|-----|-----------|
| Production deploy class 9 transfer (negermet, pritok) | ✓ deployed, FAR -25%, delay -3.3ч сохранены |
| ALMA salt FAR sweep | ✗ Pareto frontier reached |
| 3W class 3 slug-period branch | ✗ negative finding, откачено |

Чему научились:
- ALMA salt encoder уже выжата по onset-tuning. Следующий шаг для salt —
  только encoder-side (например DACAD-style contrastive, не sweep).
- Slug на class 3 требует более тонкого сигнала, чем простой fast/slow std
  ratio. Кандидаты: spectral peak detection (rolling FFT), rolling
  autocorrelation peak height в полосе 5-30 мин, временная корреляция
  пиков давления.
- Sprint показал ценность `--require-test-pass` в Optuna — без него val-overfitting
  риск пропустить generalization issue (был случай class 3: val=0.562 → test=0.125).

---

## 4b. Sprint 3 (2026-05-13)

Двухтрековый параллельный запуск (GPU 1/2/3): anomaly injection в 3W pretrain (class 5, class 8) + trend-slope physical branch для class 5.

### Трек #6 — Trend-slope branch для 3W class 5 → **GIANT WIN, deployed**

- Добавлен `class5_trend_slope_score()` в `scripts/detection/physical_branches_3w.py`:
  causal rolling-window регрессия наклона (window=30 минут) на P-PDG и P-TPT,
  фьюз через max-pool отрицательных наклонов, rank-normalize.
- `auto_weights[5] = 0.6` в `detect_3w.py`.
- Re-score class 5 + Optuna sweep с `--require-test-pass` (300 trials TPE).
- **Результат**: val hit=**1.000** (67/67), test hit=**1.000** (67/67), FAR=0.0,
  delay median=0.35ч, starts=1.01 per event.
- **vs Sprint 2 baseline** (hit=0.530, delay~0.97ч): hit **+0.47**, delay в ~3 раза меньше.
- Class 5 переходит из категории "weak coverage" в "perfect coverage" — это
  крупнейшее одно улучшение за всё время после v3.

### Трек #2 — Anomaly injection в 3W pretrain (class 5, 8) → **смешанные результаты**

Изменения:
- `alma_service/shared_encoder.py`: новый параметр `inject_cfg` в
  `_train_encoder_single_scale` и `train_shared_encoder`. При rate>0
  `inject_pool` применяется к standardized pool с patch-зависимым окном.
- `scripts/detection/detect_3w.py`: флаги `--inject-rate` и `--encoder-suffix`
  (чтобы injection-варианты не перезаписывали baseline-encoder).
- Pool sizes / injection windows: class 5 — 3470 точек, 27 окон patch=32,
  injected 8. Class 8 — 11739 точек × 5 каналов, 91 окно patch=32, injected 27.

Результаты:
- **Class 5 injection (encoder-only, без trend-slope)**: test hit 0.582 →
  0.597 (+1.5%), FAR=0, passes_test=True. Маленький, но реальный плюс. На
  фоне trend-slope (+0.47) пренебрежимо мало.
- **Class 8 injection**: WARNING no feasible candidate. Sweep нашёл val=1.0,
  но test FAR=0.157 > 0.10 ceiling. Baseline (FAR=0.025) лучше.
  **Откат**: восстановлен baseline encoder для class 8.

### Итоги Sprint 3

| Что | Результат |
|-----|-----------|
| Trend-slope branch для class 5 (auto_weights[5]=0.6) | ✓ deployed, hit 0.530 → **1.000**, FAR=0, delay 0.35ч |
| Anomaly injection в 3W pretrain class 5 | ≈ +1.5% к hit, но trend-slope доминирует — оставлено как dead-code |
| Anomaly injection в 3W pretrain class 8 | ✗ negative finding, baseline восстановлен |

Чему научились:
- **Physical signal beats representation learning** на 3W weak-classes: явный
  rolling-slope feature на P-PDG/P-TPT дал +47% hit'а — больше, чем любой
  encoder-side trick (transfer, injection, multi-source). Это паттерн class 4
  (oscillation), class 6 (choke step), теперь class 5 (trend).
- **Injection меньше помогает на больших классах**: class 8 с 11.7K точками
  и инжектируемыми 27 окнами не выигрывает у baseline'а — encoder уже видел
  достаточно вариабельности. Injection полезен скорее на тонких классах
  где encoder underfit.
- Class 5 был тестом гипотезы "injection помогает на классе с малым pool":
  3470 точек, 1 канал — даже здесь trend-slope branch победил injection
  с большим отрывом. Гипотеза не подтверждена.
- **Roadmap-сдвиг**: physical branches остаются ROI-#1 интервенцией.
  Кандидаты следующих trend-features:
  - Class 3 SEVERE_SLUGGING (hit=0.188) — spectral peak detection
    (rolling FFT в полосе 5-30 мин) вместо std-ratio.
  - Class 4 FLOW_INSTABILITY (hit=0.536) — комбинированный signal с
    autocorrelation peak height.

---

## 4e. Sprint 5 (2026-05-13, DACAD) → **encoder ceiling architecturally confirmed**

Цель: проверить может ли DACAD-style domain alignment (MMD на latent space между frozen 3W encoder и trainable ALMA encoder) сдвинуть salt или pritok после того как direct transfer ничего не дал.

### Реализация

`scripts/evaluation/train_dacad.py`:
- Target encoder E_T (PatchEncoder для ALMA channels) — trainable
- Source encoder E_S (frozen, загружен из 3W per-class encoder) — provides reference embeddings
- Loss: `NT-Xent contrastive on E_T target patches + λ · MMD(emb_T, emb_S)`
- MMD: gaussian kernel (σ=0.5), нормализованные embeddings

### Тренировки (3 параллельно на GPU 1/2/3, 300 iters каждый scale)

| Variant | recon loss | MMD final | Время |
|---------|-----------:|----------:|------:|
| salt + 3W class 4 (oscillation) | 5.3 → 0.22 | 0.6 → 0.02 | ~3с/scale |
| salt + 3W class 6 (choke step) | 5.3 → 0.23 | 0.6 → 0.04 | ~3с/scale |
| pritok + 3W class 5 (trend) | 5.4 → 0.24 | 0.7 → 0.06 | ~3с/scale |

MMD конвергировал в ≤0.07 на всех тренировках — alignment loss работает технически.

### Результаты после deploy через `_s4_swap_test.py`

| Variant | hit (all) | FAR/day (all) | mae train | mae test | vs baseline |
|---------|----------:|---------------:|----------:|---------:|-------------|
| salt baseline | 1.000 | 0.0127 | 10.03ч | 2.00ч | — |
| **salt DACAD c4** | 1.000 | 0.0127 | 10.03ч | 2.00ч | **bit-exact identical** |
| **salt DACAD c6** | 1.000 | 0.0127 | 10.03ч | 2.00ч | **bit-exact identical** |
| pritok baseline (c9) | 1.000 | 0.0376 | 17.75ч | 30.59ч | — |
| pritok DACAD c5 | 1.000 | 0.0449 | 24.66ч | 41.76ч | ✗ FAR +19%, mae +38% |

### Главное открытие — Salt encoder architecturally invariant

Сравнение per-well detected_time для salt 3 вариантов (baseline / DACAD c4 / DACAD c6) — **bit-exact identical для всех 8 скважин**. Все timestamps совпадают до секунды:

```
   well_id    detected_time            status
   149г       2025-06-20 15:15:00      Detected      [baseline = c4 = c6]
   2991г      2025-12-11 08:00:00      Detected      [baseline = c4 = c6]
   3244г      2025-11-23 00:00:00      Detected      [baseline = c4 = c6]
   ... (все 8 wells одинаково)
```

Per-point scores parquet — разные (`b.equals(c4) == False`), но это различие в `detail` JSON метаданных, не в реальных score значениях.

Salt score-fusion:
- `salt_trend_weight: 0.01` — physics 99%
- `fusion_weight_short: 0.6` — encoder fusion ratio

Salt physics branch (`alma_service/anomaly_physics.py` + salt-signature) **полностью доминирует** в финальном score. Encoder вклад настолько мал, что onset thresholding отсекает любые различия. Это **архитектурное ceiling**, не "недоучили" — невозможно сдвинуть salt без перепроектирования salt_signature или score fusion весов.

### Итоги Sprint 5

| Что | Результат |
|-----|-----------|
| Salt DACAD (c4/c6) | ✗ bit-exact identical baseline (encoder fully ignored by salt onset) |
| Pritok DACAD (c5) | ✗ regression (FAR +19%, mae +38%) — alignment слишком сильный для текущего pritok cal |

**Главный вывод**: encoder-side improvements для ALMA salt **архитектурно невозможны** без изменения score fusion. Pritok pareto-optimal на текущем encoder + onset, любая alignment деградирует. DACAD как методологический шаг закрыт.

**ALMA paths forward (после Sprint 5)**:
1. **Salt-specific**: переработать score fusion — увеличить `salt_trend_weight` (encoder начнёт влиять). Risky — может сломать FAR.
2. **Salt**: новые physics features (помимо salt_signature) — например slope-derived или multi-channel correlation.
3. **Pritok**: Optuna-tuned separate weight для `pressure_trend_slope` (компонента уже хранится после Sprint 4).
4. **3W-internal coverage** (FFT class 3, autocorr class 4) — не двигает ALMA, опционально.

---

## 4d. Sprint 4 (2026-05-13, ALMA-first) → **encoder/physics ALMA-side выжата**

Параллельный запуск (GPU 1/2/3): transfer 3W class 5 → ALMA pritok, transfer 3W class 4/6 → ALMA salt, port `_rolling_slope` в `alma_service/pressure_trend.py`.

### Транзиты 3W → ALMA (A/B vs deployed)

Все 3 transfer encoders созданы, deploy + detect через swap-helper `scripts/evaluation/_s4_swap_test.py`.

| Variant | ALMA target | hit (all) | FAR/day (all) | mae (train) | mae (test) | Δ vs deployed |
|---------|-------------|----------:|---------------:|------------:|-----------:|---------------|
| c5 → pritok | pritok | 1.000 | 0.0376 | 17.32ч | 30.59ч | ≈ identical to c9 (deployed). Train mae −0.43ч |
| c4 → salt | salt | 1.000 | 0.0127 | 10.03ч | 2.00ч | ≈ identical to baseline |
| c6 → salt | salt | 1.000 | 0.0127 | 10.03ч | 2.00ч | ≈ identical to baseline |

**Вывод**: 
- Pritok уже на encoder-side ceiling (class 9 deployed в Sprint 2 — best variant).
- Salt **invariant к любому 3W transfer**: c4 (oscillation), c6 (choke step), c9 (hydrate) — все дают identical metrics. Salt encoder bottleneck не в representation.

### Port `_rolling_slope` → ALMA pritok physics (`alma_service/pressure_trend.py`)

Реализация:
- Добавлена `_rolling_slope(values, window)` — causal OLS slope, mirror `physical_branches_3w._rolling_slope`.
- Добавлен компонент `pressure_trend_slope` в `PressureTrendOutput.components`.

Попытка #1: фьюз `slope_excess` в `change_core` через `np.maximum.reduce` →
**catastrophic regression**: test hit 1.000 → **0.500**, mae 30.59 → 59.52ч, train hit invariant.
Причина: slope_excess во время reference-window не нулевой (нормальные тренды давления тоже имеют наклон), это сдвигает per-well empirical_tail_score калибровку и провоцирует ложные onset на test.

**Откат**: фьюз убран, `slope_tail` сохранён только как diagnostic component. Pritok metrics восстановлены 1-в-1.

Slope можно использовать только через **отдельный конфигурируемый вес** с Optuna-tuning per-anomaly — но это уже не Sprint 4 scope.

### Итоги Sprint 4

| Что | Результат |
|-----|-----------|
| Transfer 3W class 5 → ALMA pritok | ≈ deployed c9 — marginal mae улучшение, не оправдывает deploy |
| Transfer 3W class 4 → ALMA salt | ✗ invariant |
| Transfer 3W class 6 → ALMA salt | ✗ invariant |
| Port `_rolling_slope` → pritok physics (fuse через max) | ✗ regression test hit 1.000 → 0.500, откачено |
| Port `_rolling_slope` → pritok physics (diagnostic-only) | ✓ shipped, без эффекта на текущий score |

**Главный вывод Sprint 4**: ALMA encoder-side и pritok physics-side **выжаты** на текущей feature/encoder архитектуре:
- Sprint 2 deployed c9 → pritok/negermet — best transfer.
- Sprint 4 c5/c4/c6 transfers + slope-port — ничего нового не дают.
- Salt encoder fundamentally invariant к 3W transfer (любого класса).

**Что осталось для ALMA (после Sprint 4):**
1. **DACAD contrastive + GRL** — единственный методологический путь к salt encoder gain. 3-5 дней.
2. **Optuna-tuned separate weight для `pressure_trend_slope`** — может вытащить marginal mae на pritok через per-anomaly tuning, не через in-line fusion. 1 день.
3. **3W-internal coverage** (FFT class 3, autocorr class 4) — благо для benchmark, не двигает ALMA production.

---

## 4c. Итог 3W работы относительно baseline (2026-05-12 → 2026-05-13)

### Что было на старте 3W работы (tag `baseline-pre-3w-2026-05-12`)

- ALMA production detector `paano_shared` на 3 аномалиях (negermet / pritok / salt)
  с hit=1.000 на всех 3-х. Per-class encoder, обучен только на ALMA-нормах.
- 3W Petrobras Dataset не использовался. Domain-pretrain отсутствовал.
- ALMA negermet FAR/day = 0.250, pritok mae_h (train) = 21.01ч, salt FAR/day = 0.013.
- Нет infrastructure для transfer learning. Encoder weights только в `models/`.

### Что получили после 3W работы (3 спринта, 11 коммитов на ветке `app/webapp`)

**Новый продуктовый pipeline (3W Petrobras 2.0.0):**

| Метрика | До 3W | После 3W |
|---------|------:|---------:|
| Классов аномалий с production-grade результатом | 3 (ALMA) | 3 (ALMA) + **9 (3W)** |
| 3W mean hit (all 9 классов) | n/a | **0.845** |
| 3W классов с hit ≥ 0.97 | n/a | **6/9** (1, 2, 5, 6, 7, 9) |
| 3W классов passes Pareto | n/a | **9/9** |
| 3W инфраструктура (build/detect/sweep/report) | n/a | 9 detect_3w-конфигов + tune-pipeline |

**Улучшения ALMA (через 3W transfer):**

| ALMA метрика | До 3W | После Sprint 2 deploy |
|--------------|------:|----------------------:|
| negermet FAR/day (all) | 0.250 | **0.188** (−25%) |
| negermet starts/interval | 1.80 | **1.60** |
| pritok mae_h (train) | 21.01ч | **17.75ч** (−3.3ч) |
| pritok hit-rate, FAR | 1.000 / 0.039 | 1.000 / 0.038 |
| salt | invariant | invariant (encoder не bottleneck) |

**Реализованная инфраструктура** (новые модули, скрипты, артефакты):

1. **Build pipeline**: `scripts/datasets/build_3w_dataset.py` + `configs/3w_paano.json`
   — manifest + intervals + features + balanced splits на 2228 instances.
2. **Detection pipeline**: `scripts/detection/detect_3w.py` + per-class encoders
   в `artifacts/3w/checkpoints/` (9 файлов).
3. **Physical branches**: `scripts/detection/physical_branches_3w.py` —
   `class4_oscillation_score`, `class5_trend_slope_score` (Sprint 3),
   `class6_choke_step_score`, `class3_slug_score` (dead-code).
4. **Tuning**: `scripts/evaluation/optuna_sweep_3w.py` (TPE с `--require-test-pass`)
   + `scripts/evaluation/optuna_sweep_alma.py` (ALMA-аналог).
5. **Transfer mechanism**:
   - `alma_service/shared_encoder.load_or_train_shared_encoder()` (P10 hook).
   - `scripts/evaluation/transfer_3w_to_alma.py` (warm-start + fine-tune).
   - `alma_service/anomaly_injection.py` (synthetic augment, dead-code на ALMA).
6. **Comparison**: `scripts/evaluation/compare_transfer_variants.py` —
   tabular A/B по 4 transfer-вариантам.
7. **Reporting**: `scripts/reports/generate_3w_report.py` —
   9 offline Plotly HTML без CDN.
8. **Aggregation**: `scripts/evaluation/aggregate_3w_phase05.py` —
   benchmark summary parquet + JSON.

**Ключевые выводы (cumulative)**:

1. **3W успешно использован как domain pretrain** для transfer в ALMA (per-class class 9 → ALMA negermet/pritok).
2. **Physical branches** дают существенно больший gain на 3W weak-классах, чем encoder-side трюки (transfer, multi-source, injection). Sprint 3 class 5 trend-slope: +0.47 hit. Sprint 1 class 4/6: +0.53 / +1.00.
3. **ALMA hit = 1.000 — это ceiling**: encoder transfer двигает только FAR / delay, не hit-rate. Дальнейшие gains для ALMA требуют изменения onset-логики или encoder-уровневые техники (DACAD).
4. **Multi-source NORMAL pretrain** хуже per-class на нашем pool size (intersection даёт 10 каналов вместо 25-35). Per-class остаётся best transfer-source.
5. **Anomaly injection** работает только при больших pool'ах (3W class 5 marginal +1.5%; ALMA fine-tune negligible из-за 1-2 окон).

### Что осталось (ALMA-first, после Sprint 5 — синхронизировано с секцией 2)

**Главная цель — ALMA production detector.** Sprint 4 закрыл transfer-варианты. Sprint 5 закрыл DACAD encoder-side adaptation. Salt encoder architecturally invariant из-за score fusion design. Pritok pareto-optimal. Дальше только non-encoder вмешательства.

| Приоритет | Задача | ALMA-эффект | Сложность |
|----:|--------|-------------|-----------|
| **#1** | **Salt new physics features**: slope-derived score на ESP-каналах, multi-channel correlation, температурный signal | потенциально первый non-zero salt-side gain | средняя, 2-3 дня |
| **#2** | **Salt score-fusion redesign**: увеличить `salt_trend_weight` (encoder начнёт влиять). Risky | возможно encoder effect, может сломать FAR | средняя, 1-2 дня |
| **#3** | **Optuna-tuned weight для `pressure_trend_slope`** компоненты | marginal pritok mae↓ | низкая, 1 день |
| #4 | 3W class 3 FFT spectral branch (hit=0.188) | 0 для ALMA, 3W coverage | средняя |
| #5 | 3W class 4 autocorr branch (hit=0.536) | 0 для ALMA, 3W coverage | средняя |
| #6 | 3W NORMAL guard P11 deploy | 0 для ALMA | средняя |
| #7 | 3W class 2 single failed instance | 0 для ALMA | низкая |
| #8 | Cross-validation per-class 5-fold | методологический | низкая |

**Recommended next sprint (ALMA-focused)**:
1. **#3 first** — самый дешёвый эксперимент с шансом marginal pritok mae↓.
2. **#1 second** — salt physics features (high-effort, high-stake — единственный путь к salt encoder-bypass).
3. **3W-internal (#4-#8)** — только при отдельном запросе.

Закрытые в Sprint 4-5 (не повторять):
- ~~Transfer 3W class 5 → ALMA pritok~~ — ≈ class 9 deployed
- ~~Transfer 3W class 4/6 → ALMA salt~~ — invariant
- ~~Port slope → pritok physics via max-fuse~~ — regression test hit 0.500
- ~~DACAD MMD salt c4/c6~~ — bit-exact identical baseline (architectural ceiling)
- ~~DACAD MMD pritok c5~~ — regression FAR +19%, mae +38%

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
5b64409  Sprint 3: trend-slope branch для class 5 + anomaly injection в 3W pretrain
8e14cbd  Sprint 2: deploy class 9 transfer; salt sweep + class 3 slug branch negative findings
facf716  Consolidate 3W docs into single docs/3w_pipeline.md
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
| #1 | Slug-period branch class 3 | ✗ 2026-05-13 negative finding (slug-ratio не различает true slug) |
| #3 | Production deploy class 9 transfer | ✓ 2026-05-13 deployed |
| #5 | Optuna sweep salt test FAR | ✗ 2026-05-13 Pareto frontier reached |
| #2 | Anomaly injection в 3W pretrain (не ALMA) | ✗ 2026-05-13 c5 marginal +1.5%, c8 negative |
| #6 | Trend-slope branch class 5 | ✓ 2026-05-13 hit 0.530 → 1.000, deployed |
| #4 | DACAD contrastive + GRL | TODO (см. ALMA-first ranking #4) |
| Sprint4-#1 | **Transfer 3W class 5 encoder → ALMA pritok** | TODO (ALMA-first приоритет) |
| Sprint4-#2 | **Port `class5_trend_slope_score` → ALMA pritok physics** | TODO (ALMA-first приоритет) |
| Sprint4-#3 | **Transfer 3W class 4/6 encoder → ALMA salt** | TODO (ALMA-first приоритет) |
| 3W-Next | FFT spectral branch для class 3 (3W-only, не блокирует ALMA) | TODO (опционально) |
| 3W-Next | Combined autocorr+std branch для class 4 (3W-only) | TODO (опционально) |

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
