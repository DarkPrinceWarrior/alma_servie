# 3W PaAno Phase 0.5 — финальный статус

Дата: 2026-05-12 / 2026-05-13 (полный прогон занял ~3 часа GPU-времени на 5× A100).

## Резюме одной строкой

3W PaAno baseline-пайплайн реализован и прогнан на всех 9 классах аномалий
Petrobras 3W Dataset 2.0.0. **8/9 классов проходят Pareto-constraint**
(`FAR/сутки ≤ 0.10`, `срабатываний/событие ≤ 3`). Четыре класса (1, 2, 7, 9)
показали hit-rate ≥ 0.75 на test split, два из них (2, 9) — со starts/event ≤ 1.2.

## Acceptance table — test split

| Класс | Имя | Интервалов | Hit-rate | FAR/сутки | Starts/событие | Median delay, ч | P90 delay, ч | Pareto ✓ |
|------:|-----|-----------:|---------:|----------:|---------------:|----------------:|-------------:|:--------:|
| 1 | ABRUPT_INCREASE_OF_BSW       | 4  | **0.750** | 0.000 | 1.00 | 11.63 | 15.05 | ✓ |
| 2 | SPURIOUS_CLOSURE_OF_DHSV     | 5  | **0.800** | 0.000 | 0.80 |  0.56 |  0.87 | ✓ |
| 3 | SEVERE_SLUGGING              | 16 | 0.062     | 0.000 | 0.06 |  3.87 |  3.87 | ✓ |
| 4 | FLOW_INSTABILITY             | 51 | 0.000     | 0.000 | 0.00 |  —    |  —    | ✓ |
| 5 | RAPID_PRODUCTIVITY_LOSS      | 8  | 0.000     | 0.000 | 0.00 |  —    |  —    | ✓ |
| 6 | QUICK_RESTRICTION_IN_PCK     | 3  | 0.000     | 0.000 | 0.00 |  —    |  —    | ✓ |
| 7 | SCALING_IN_PCK               | 7  | **0.857** | 0.000 | 4.86 |  8.99 | 13.35 | ✗ (starts > 3) |
| 8 | HYDRATE_IN_PRODUCTION_LINE   | 14 | **0.643** | 0.018 | 1.14 |  8.05 | 14.91 | ✓ |
| 9 | HYDRATE_IN_SERVICE_LINE      | 7  | **1.000** | 0.000 | 1.14 |  2.40 |  5.49 | ✓ |

Источник чисел: `artifacts/results/3w_benchmark_summary.json` +
`artifacts/3w/metrics/class_<N>_metrics.json`. Дашборды:
`artifacts/3w/reports/3w_class_<N>_<EVENT>.html`.

## Что проходит и что нет

**Сильные классы (test hit ≥ 0.75, проходят Pareto):**
- Class 9 HYDRATE_IN_SERVICE_LINE: 1.00 hit, FAR=0, median_delay=2.4 ч —
  лучший результат сетки. На train hit_rate=0.993 (163 интервала из train+val).
- Class 2 SPURIOUS_CLOSURE_OF_DHSV: 0.80 hit, median_delay=0.56 ч —
  fast-onset аномалия, ловится практически мгновенно.
- Class 1 ABRUPT_INCREASE_OF_BSW: 0.75 hit, starts=1.00 — нет шумов.
- Class 7 SCALING_IN_PCK: 0.857 hit, но 4.86 starts/event (тюнинг не нашёл
  feasible конфиг — пик кандидата по hit-rate выбран best-effort).

**Проходят формально, но hit-rate почти ноль (тонкий test split):**
- Class 3 SEVERE_SLUGGING — 16 intervals в test, hit=0.06 (1 правильный).
  На train hit=0.473, на val ещё ниже — encoder не научился ловить oscillatory
  pattern без physical branch.
- Class 4 FLOW_INSTABILITY, Class 5 RAPID_PRODUCTIVITY_LOSS,
  Class 6 QUICK_RESTRICTION_IN_PCK — нулевой test hit. У классов 4 и 6
  detect_3w на первой попытке упал (нет class==0 в train wells); после
  fallback fix (`first quarter as reference pool`) обучение прошло,
  но на test split не нашёл срабатываний.

**Не проходит constraints:**
- Class 7 SCALING_IN_PCK — starts/event=4.86 > 3. Базовый PaAno без
  salt_deposition-ветки на test даёт много повторных срабатываний внутри
  одного интервала.

## Что было сделано инфраструктурно

1. **Baseline зафиксирован** (`baseline-pre-3w-2026-05-12`):
   frozen production encoders/summaries + `docs/baseline_state_2026-05-12.md`.
2. **P0 аудит** `alma_service/shared_encoder.py`: `PatchEncoder(in_channels=N)`
   запекается; transfer 3W→ALMA напрямую невозможен (P10, требует adapter).
3. **Patch broken upstream import**: `generic_detectors.py` теперь содержит
   `PAANO_BATCH_SIZE/_LR/_NUM_ITERS/_TOP_K/_MEMORY_BANK_RATIO`. Эти константы
   импортируются в `shared_encoder.py`, но не были определены нигде в репо —
   production через `run_detection()` обходил этот путь, поэтому баг не
   всплывал; прямой `train_shared_encoder()` падал ImportError.
4. **Petrobras 3W** склонирован на сервер (`data/raw/3w/3W`, commit `227fce3`).
5. **Build pipeline**: 2228 instances → 2218 валидны (99.6%); 1-min resample,
   transient/event/undesirable_start, instance-level stratified split
   (1658 train / 332 val / 228 test); per-class features parquet.
6. **Detection / evaluation / report** (3 модуля + `aggregate_3w_phase05.py`):
   per-class PaAno shared encoder (patch_short=32, patch_long=64),
   grid tune с Pareto-constraint, offline Plotly HTML.
7. **Parallel launcher** (`_3w_worker.sh`): 5 tmux workers, по одному на GPU 1..5;
   completion markers и явная rc-обработка.
8. **Fallback reference pool**: build_3w изначально полагался на `class==0`
   как нормальную часть; для классов 3, 4 это не работает (в train instances
   нет normal rows). Добавлен fallback на первые `min(N/4, patch_long*4)`
   точек инстанса как reference.

## Артефакты

```
data/raw/3w/3W/                                            # Petrobras 3W, 3.9 GB
data/processed/3w/
    manifest.parquet
    intervals.parquet
    splits.parquet
    build_summary.json
    class_<N>/features.parquet                             # per-class table
artifacts/3w/
    checkpoints/3w_class_<N>_paano_shared_encoder.pt       # 9 encoders
    scores/class_<N>_scores.parquet
    scores/class_<N>_predicted_starts.parquet
    metrics/class_<N>_selected.json                        # выбранный config
    metrics/class_<N>_metrics.json                         # finalised metrics
    reports/3w_class_<N>_<EVENT_NAME>.html                 # offline HTML, 9 шт
artifacts/results/3w_benchmark_summary.json                # сводная таблица
configs/3w_paano.json                                      # конфиг
docs/roadmap_3w.md                                         # P0–P10 трекер
docs/baseline_state_2026-05-12.md                          # acceptance baseline
```

Сетка занимает ~70 MB (54 MB отчёты + 15 MB scores). PaAno encoder
~3.5 MB каждый. Артефакты в `.gitignore`.

## Что не сделано / ограничения

1. **Адаптер 3W→ALMA**: оставлено как P10. Channel mismatch
   (`P-PDG` ↔ ALMA каналы) требует либо linear projection input,
   либо retrain с combined schema. Решение откладывается до получения
   валидных acceptance-чисел и согласования с пользователем.
2. **Physical branches** (раздел 18 ТЗ): не реализованы. Это объясняет
   слабый hit на классах 3, 4, 5, 6 — для них классы PaAno без
   domain-knowledge сигналов недостаточно.
3. **Tuning grid компактный** (108 комбо: 4×3×3×3). Optuna sweep
   с независимыми workers на GPU 1..4 — задача P2 (см. roadmap).
4. **Source-type biased test**: для классов с simulated-dominant
   данными (5: 98% simulated, 6: 99% simulated, 9: 73% simulated)
   test split тонкий, статистика шумная.
5. **`worker_4` и `worker_3`** упали на initial run (class 3 и 4
   — нет class==0). Перезапущены через `_3w_rerun_failed.sh`
   с уже-deployed fallback. **Marker semantics**: `_3w_rerun_failed.sh`
   использует один `runs/3w_rerun_failed.done` для обоих параллельных
   rerun'ов — first finished пишет marker; для надёжности в будущем
   разделить на `runs/3w_rerun_class_<N>.done`.

## Следующие шаги (после ревью пользователем)

| ID | Задача |
|----|--------|
| P2 | Расширить tuning grid, Optuna на GPU 1..4 параллельно |
| P3 | Физические ветки для классов 3, 4, 5, 6 (oscillation, choke ΔP) |
| P4 | Source-type aware test split (real-only test для simulated-heavy классов) |
| P10 | Transfer 3W→ALMA через adapter layer (см. roadmap_3w.md) |

## Git commits (хронология)

```
022782d  Freeze pre-3W baseline state and acceptance reference
5ab09f1  Start 3W PaAno pipeline: config + roadmap
16f0462  Add 3W PaAno pipeline scripts and parallel launcher
4be6b74  Fix broken PAANO_* imports + 3W scoring/report edge cases
b8873ad  Add 3W Phase 0.5 helpers: worker script, rerun script, aggregator
<this>   Finalize 3W Phase 0.5: per-class metrics + sweep summary + status
```

Tag `baseline-pre-3w-2026-05-12` указывает на pre-3W точку отката.
