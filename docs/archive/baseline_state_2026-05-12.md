# Baseline state pre-3W (2026-05-12)

Зафиксированное состояние репозитория и production-baseline `paano_shared`
перед реализацией 3W PaAno pipeline.

Используется как точка отката (git tag `baseline-pre-3w-2026-05-12`)
и как acceptance reference для сравнения с 3W warm-start вариантами.

## Git

- HEAD: `36bd748` — "Document class specific versus global pretrain"
- Branch: `app/webapp`
- Tag: `baseline-pre-3w-2026-05-12`
- Submodule: `paano` modified (ожидаемо по `CLAUDE.md`)

## Окружение (a100, source of truth)

```
python 3.13.5
torch  2.11.0+cu130
CUDA   13.0
GPU    6× NVIDIA A100-SXM4-40GB
       GPU0 занят (~8.4 GiB), GPU1..5 свободны (~40 GiB каждая)
```

## Артефакты на сервере (`/root/projects/alma_servie`)

```
db                127 MB
artifacts         594 MB
models             13 MB
salym              67 GB
salym_prepared    32 GB
```

## Frozen encoders

Скопированы как точка отката (в `.gitignore`, защищены от перезаписи при следующем training run):

```
models/baseline_2026-05-12_negermet_paano_shared_encoder.pt
models/baseline_2026-05-12_pritok_paano_shared_encoder.pt
models/baseline_2026-05-12_salt_paano_shared_encoder.pt
```

## Frozen benchmark summaries

```
artifacts/results/baselines/2026-05-12/negermet_benchmark_summary.json
artifacts/results/baselines/2026-05-12/pritok_benchmark_summary.json
artifacts/results/baselines/2026-05-12/salt_benchmark_summary.json
artifacts/results/baselines/2026-05-12/negermet_paano_shared_results.summary.json
artifacts/results/baselines/2026-05-12/pritok_paano_shared_results.summary.json
artifacts/results/baselines/2026-05-12/salt_paano_shared_results.summary.json
```

## Baseline метрики (`paano_shared`, split `all`)

| Аномалия | Hit-rate | FAR/day | Starts/interval | Median delay (ч) | P90 delay (ч) | Дней наблюдения |
|----------|---------:|--------:|----------------:|-----------------:|--------------:|----------------:|
| negermet | 1.00     | 0.250   | 1.80            | 0.00             | 0.61          | 16              |
| pritok   | 1.00     | 0.039   | 3.67            | 4.01             | 59.27         | 691             |
| salt     | 1.00     | 0.013   | 2.63            | 0.03             | 19.97         | 707             |

**Acceptance reference:** любой 3W warm-start вариант проходит в production
только если не ухудшает ни одну из этих величин по каждому классу
(см. `ALMA_class_specific_vs_global_pretrain.md`, раздел 9).

## Detector configs (зафиксированы)

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

## Структура репо до 3W

### `alma_service/` (24 модуля)

Ключевые для 3W transfer:

```
shared_encoder.py         # encoder API, точка transfer для warm-start
generic_detection.py      # run_detection() unified entry
onset_detection.py        # detect_causal_onsets, CausalThresholds
paano_defaults.py
dataset_builder.py
dataset_config.py
engineered_features.py
```

### `scripts/`

```
scripts/datasets/build_{negermet,pritok,salt,norm_work}_dataset.py
scripts/detection/detect_{negermet,pritok,salt}.py
scripts/detection/screen_norm_work_false_alarms.py
scripts/detection/screen_salym_unlabeled.py
scripts/detection/build_shared_encoders.py
scripts/evaluation/evaluate_onset_metrics.py
scripts/evaluation/benchmark_global_pretrain_finetune.py
scripts/evaluation/tune_saved_onset.py
scripts/evaluation/tune_with_norm_work_guard.py
scripts/evaluation/apply_norm_work_guard_configs.py
scripts/reports/generate_{negermet,pritok,salt}_paano_report.py
scripts/reports/generate_feature_importance_report.py
scripts/reports/generate_salym_expert_package.py
```

### `docs/` (8 файлов)

включая `TASK_3W_PaAno_ALMA_detector.md` (untracked на сервере)
и `ALMA_class_specific_vs_global_pretrain.md`.

### Чего нет (создаётся 3W-задачей)

```
configs/                                  # директории нет
configs/3w_paano.yaml
scripts/datasets/build_3w_dataset.py
scripts/detection/detect_3w.py
scripts/reports/generate_3w_report.py
artifacts/3w/
data/raw/3w/
docs/roadmap_3w.md
```

## Untracked на сервере (не относятся к baseline)

```
data/raw/norm_work/                       # 20 normal-work скважин по экспертной оценке
docs/TASK_3W_PaAno_ALMA_detector.md       # ТЗ на 3W
runs/                                     # логи tmux-прогонов
uv.lock                                   # стоит закоммитить отдельным change
```

## Условия отката

```bash
# вернуться к baseline на сервере
ssh a100 'cd /root/projects/alma_servie && git checkout baseline-pre-3w-2026-05-12'

# восстановить production encoders из frozen-копий
ssh a100 'cd /root/projects/alma_servie && \
  cp models/baseline_2026-05-12_negermet_paano_shared_encoder.pt models/negermet_paano_shared_encoder.pt && \
  cp models/baseline_2026-05-12_pritok_paano_shared_encoder.pt   models/pritok_paano_shared_encoder.pt && \
  cp models/baseline_2026-05-12_salt_paano_shared_encoder.pt     models/salt_paano_shared_encoder.pt'
```

## Открытые архитектурные решения (зафиксированы для 3W roadmap)

1. **Channel mismatch (3W → ALMA)**: encoder работает на нормализованных
   patch-эмбеддингах безотносительно семантики каналов; adapter — channel-agnostic
   patch encoder + per-dataset normalizer head. Подлежит проверке при аудите
   `shared_encoder.py`.
2. **Раскладка скриптов**: per-anomaly в существующем стиле
   (`scripts/datasets/build_3w_dataset.py`, `scripts/detection/detect_3w.py`),
   а не отдельный `scripts/3w/` namespace.
3. **Acceptance objective**: hard constraint + lexicographic
   (`FAR/day <= 0.10`, `starts/event <= 3`, `norm_work_far <= baseline`;
   далее максимизация `hit_rate`, tie-break по `median_delay`, затем `p90_delay`).
4. **GPU budget**: Phase 0.5 на GPU1; full Optuna sweep на GPU1..4 параллельно
   через independent Optuna workers (`sqlite:///artifacts/3w/optuna.db`).

## Training mode taxonomy (для metadata всех будущих encoder-ов)

```
class_specific                          — текущий production baseline
alma_global_pretrain_class_finetune     — global pretrain на ALMA-only нормах
3w_pretrain_only                        — только 3W
3w_pretrain_alma_finetune               — 3W warm-start + ALMA fine-tune
```

Один inference API: `paano_shared`. Никаких новых detector keys.
