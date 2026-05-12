# 3W PaAno Phase 0.5 — статус

Дата: 2026-05-12

## Что сделано

1. **Baseline зафиксирован** (`baseline-pre-3w-2026-05-12`):
   - frozen production encoders/summaries
   - acceptance reference в `docs/baseline_state_2026-05-12.md`
2. **P0 audit** `alma_service/shared_encoder.py`:
   - `PatchEncoder` channel-count baked into архитектуру
   - Standalone 3W detector работает без правок alma_service
   - Transfer 3W→ALMA через существующий `fine_tune_shared_encoder` невозможен (разные каналы)
3. **Patch broken upstream import**: `alma_service/generic_detectors.py` теперь содержит
   `PAANO_BATCH_SIZE/_LR/_NUM_ITERS/_TOP_K/_MEMORY_BANK_RATIO`. Эти константы
   импортируются в `shared_encoder.py`, но **не были определены нигде в репо** —
   `train_shared_encoder()` падал ImportError при прямом вызове. Production
   через `run_detection()` обходил этот путь, поэтому баг не всплывал.
4. **Petrobras 3W склонирован** на сервер: `data/raw/3w/3W`, commit `227fce3`, 3.9 GB.
5. **Build pipeline** (`scripts/datasets/build_3w_dataset.py`):
   - 2228 instances → 2218 валидны (99.6%)
   - 1-min resample, label extraction (transient/event/undesirable_start)
   - instance-level stratified split (1658 train / 332 val / 228 test)
   - per-class features parquet с 40-70 channels (intersection non-NaN)
6. **Detection / evaluation / report** (`detect_3w.py`, `evaluate_3w_onset.py`,
   `generate_3w_report.py`):
   - per-class PaAno shared encoder (patch_short=32, patch_long=64)
   - grid tune (target_far × min_run × ema_alpha × cooldown)
     с Pareto constraint (FAR/day ≤ 0.10, starts/event ≤ 3) +
     lexicographic max hit_rate / min median_delay
   - offline Plotly HTML отчёт
7. **Phase 0.5 launch**: 5 параллельных tmux workers, по одному на GPU 1..5.

## Раскладка workers

| GPU | Классы | Instances total |
|-----|--------|---:|
| 1 | 1, 6 | ~349 |
| 2 | 2, 7 | ~84 |
| 3 | 3, 8 | ~201 |
| 4 | 4, 9 | ~550 |
| 5 | 5 | ~450 |

## Метрики (заполняются после завершения workers)

| Класс | Hit-rate | FAR/сутки | Starts/событие | Median delay (ч) | P90 delay (ч) | Проходит acceptance? |
|------:|---------:|----------:|---------------:|-----------------:|--------------:|----------------------|
| 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD | TBD | TBD | TBD |
| 6 | TBD | TBD | TBD | TBD | TBD | TBD |
| 7 (smoke OK) | 0.857 | 0.000 | 4.71 | 8.99 | 13.35 | starts > 3 (best-effort) |
| 8 | TBD | TBD | TBD | TBD | TBD | TBD |
| 9 | TBD | TBD | TBD | TBD | TBD | TBD |

## Артефакты

- `models/`: encoders не сохранены в `models/` (3W держим в `artifacts/3w/checkpoints/`)
- `artifacts/3w/checkpoints/3w_class_<N>_paano_shared_encoder.pt`
- `artifacts/3w/scores/class_<N>_scores.parquet`
- `artifacts/3w/scores/class_<N>_predicted_starts.parquet`
- `artifacts/3w/metrics/class_<N>_selected.json` — выбранный config после tuning
- `artifacts/3w/metrics/class_<N>_metrics.json` — финальные test metrics
- `artifacts/3w/reports/3w_class_<N>_<EVENT_NAME>.html` — offline отчёт
- `data/processed/3w/{manifest, intervals, splits}.parquet`
- `data/processed/3w/class_<N>/features.parquet`

## Известные нюансы

1. **Distribution sources перекошен**: classes 0/4 = real-only, 5/6/9 преимущественно
   simulated. Test может оказаться heavy on simulated для simulated-dominant classes.
2. **Channel set уменьшен после reduce_features**: реальное число каналов encoder'а
   2-7 для некоторых классов (старая ALMA логика отбирает stability-top); 3W
   nature такова что разные instances имеют разный non-NaN profile.
3. **`paano_shared` artifacts production** не перезаписаны — 3W encoder
   сохраняется отдельным path в `artifacts/3w/checkpoints/`.
4. **patch_long=64 fallback**: для instances с очень коротким reference period
   (< 64*8 точек) используется first N points of well_matrix как ref pool.

## Транзит к production ALMA

**P10 (research, не Phase 0.5)**: transfer 3W→ALMA encoder требует adapter layer
из-за channel mismatch (см. `docs/roadmap_3w.md`, P0 audit). Channel-agnostic
embedding через linear projection или retrain с combined feature schema —
открытые опции. Решение откладывается до получения Phase 0.5 чисел и
обсуждения acceptance criteria.

## Команды для ручной проверки

```bash
# Список tmux workers
ssh a100-remote 'tmux ls'

# Логи отдельного worker'а
ssh a100-remote 'tail -f /root/projects/alma_servie/runs/3w_worker_1.log'

# Метрики класса
ssh a100-remote 'cat /root/projects/alma_servie/artifacts/3w/metrics/class_7_metrics.json'

# Просмотр HTML отчёта
scp a100-remote:/root/projects/alma_servie/artifacts/3w/reports/3w_class_7_SCALING_IN_PCK.html /tmp/
xdg-open /tmp/3w_class_7_SCALING_IN_PCK.html
```
