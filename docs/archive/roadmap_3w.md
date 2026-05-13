# 3W PaAno pipeline — roadmap

Дата старта: 2026-05-12
Baseline tag: `baseline-pre-3w-2026-05-12`

## Статус

| ID | Задача | Статус |
|----|--------|--------|
| P0.1 | Audit `alma_service/shared_encoder.py` | done |
| P0.2 | Clone Petrobras 3W на сервер | done |
| P1.1 | `configs/3w_paano.json` | done (drafted) |
| P1.2 | `docs/roadmap_3w.md` (этот файл) | done |
| P1.3 | `scripts/datasets/build_3w_dataset.py` | pending |
| P1.4 | `scripts/detection/detect_3w.py` | pending |
| P1.5 | `scripts/evaluation/evaluate_3w_onset.py` | pending |
| P1.6 | `scripts/reports/generate_3w_report.py` | pending |
| P2 | Build всех 9 классов на сервере | pending |
| P3 | Training 9 классов в tmux (GPU 1..5 параллельно) | pending |
| P4 | Evaluate + report | pending |
| P10 | Transfer 3W→ALMA encoder (P0 audit показал: нужен adapter) | research |

## P0 audit — findings (`alma_service/shared_encoder.py`)

1. `PatchEncoder(in_channels=N, use_revin=True)` — channel-count **запекается** в архитектуру.
2. `train_shared_encoder` принимает любое `prepared_wells` dict; работает с любым `in_channels` если все wells имеют один и тот же intersection of channels.
3. `fine_tune_shared_encoder` требует **точное совпадение** `shared_channels` между pretrained_state и target prepared_wells — иначе `state_dict` load падает.

**Следствия для 3W**:
- **Standalone 3W detector работает без правок** в `alma_service/shared_encoder.py`. Просто передаём 3W instances в формате `PreparedWellData`-like (нужно сделать тонкий adapter dataclass со совместимыми полями).
- **Transfer 3W→ALMA через существующий `fine_tune_shared_encoder` невозможен**: каналы (`P-PDG`, `P-TPT`, ...) ≠ ALMA каналы (давление приёма, токи фаз, ...). Это P10.

## P0 audit — findings (Petrobras 3W structure)

- HEAD: `227fce3`, 3.9 GB, 2228 instances в классах 1-9 + 594 NORMAL (класс 0).
- 1-секундная сетка. TRANSIENT_OFFSET=100 подтверждён (`class=107` для transient в folder 7).
- 29 columns, но в hand-drawn instances **большинство NaN** — реально 5 каналов `P-MON-CKP, P-PDG, P-TPT, T-JUS-CKP, T-TPT`. Real WELL-* могут иметь больше.
- `dataset/folds/` пуст → делаем свой instance-level split.
- Source types в filename: `WELL-NNNNN_*` (real), `DRAWN_*` (hand-drawn), `SIMULATED_*`.

Class distribution:

| Class | Name | Instances |
|------:|------|----------:|
| 0 | NORMAL | 594 |
| 1 | ABRUPT_INCREASE_OF_BSW | 128 |
| 2 | SPURIOUS_CLOSURE_OF_DHSV | 38 |
| 3 | SEVERE_SLUGGING | 106 |
| 4 | FLOW_INSTABILITY | 343 |
| 5 | RAPID_PRODUCTIVITY_LOSS | 450 |
| 6 | QUICK_RESTRICTION_IN_PCK | 221 |
| 7 | SCALING_IN_PCK | 46 |
| 8 | HYDRATE_IN_PRODUCTION_LINE | 95 |
| 9 | HYDRATE_IN_SERVICE_LINE | 207 |

## Архитектурные решения (зафиксированы)

1. **Раскладка скриптов**: per-anomaly в существующем стиле (`scripts/datasets/build_3w_dataset.py`, `scripts/detection/detect_3w.py`), а не `scripts/3w/` namespace.
2. **Config format**: JSON (`configs/3w_paano.json`), без PyYAML.
3. **Resample**: 1min median для numeric, mode для `class`/`state`.
4. **Channel set**: intersection non-NaN columns within each class (отдельный shared_channels per event class).
5. **Splits**: instance-level, stratified by (event_label, source_type), prefer real в test, seed=2027.
6. **Acceptance objective**: hard constraint (FAR/day ≤ 0.10, starts/event ≤ 3) + lexicographic max hit_rate → min median_delay → min p90_delay.
7. **Transfer 3W→ALMA**: отложен до P10. В Phase 0.5 — standalone 3W detector per class.
8. **Detector key**: `paano_shared` (один inference API), варьируется `encoder_training_mode` в metadata.

## Acceptance gate (для будущей оценки)

Production baseline `paano_shared` (см. `docs/baseline_state_2026-05-12.md`):

| Anomaly  | Hit-rate | FAR/day | Starts/interval | Median delay (ч) | P90 delay (ч) |
|----------|---------:|--------:|----------------:|-----------------:|--------------:|
| negermet | 1.00     | 0.250   | 1.80            | 0.00             | 0.61          |
| pritok   | 1.00     | 0.039   | 3.67            | 4.01             | 59.27         |
| salt     | 1.00     | 0.013   | 2.63            | 0.03             | 19.97         |

3W warm-start в production через transfer-learning принимается только если:
- non-deterioration по каждой строке выше (per-class);
- norm_work guard FAR не выше baseline.

## Что делать когда вернусь

Продолжить с P1.3:

1. `scripts/datasets/build_3w_dataset.py` — single CLI:
   - parse `configs/3w_paano.json`
   - walk `data/raw/3w/3W/dataset/[0-9]`, прочитать каждый parquet
   - extract source_type из filename prefix
   - resample to 1min, build labels (transient/event/undesirable_start)
   - per-class: drop high-NaN channels, intersect channel sets, save `features_class_<N>_1min.parquet`
   - build instance-level splits (stratified, real preferred for test)
   - save `manifest.parquet`, `intervals.parquet`, `splits.parquet`
2. `scripts/detection/detect_3w.py` — train PaAno encoder per class, score val/test через `_score_well_single_scale`.
3. `scripts/evaluation/evaluate_3w_onset.py` — переиспользовать `evaluate_onset_metrics.py` через создание 3W-совместимых intervals.
4. `scripts/reports/generate_3w_report.py` — Plotly offline (CDN-free), reuse `paano_report.py` patterns.
5. tmux launch на GPU 1..5 параллельно (2 волны: 5+4 классов).
6. Final commit + `docs/3w_phase05_status.md` с числами.

## Команда продолжения (когда вернёшься)

```
"продолжи 3W roadmap с P1.3, scripts всё пиши на сервере как source of truth,
training запускай в tmux на GPU 1..5 параллельно, к концу — финальный отчёт
в docs/3w_phase05_status.md"
```

Все важные context зафиксированы в этом файле, `configs/3w_paano.json` и
`docs/baseline_state_2026-05-12.md`. Новая сессия сможет продолжить без потери знаний.
