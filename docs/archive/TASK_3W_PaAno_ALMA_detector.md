# TASK: 3W-first PaAno detector for ALMA / нефтяная детекция аномалий

Дата: 2026-05-12  
Цель документа: передать ИИ-агенту или инженеру полное ТЗ на реализацию системы детекции аномалий на базе **Petrobras 3W Dataset 2.0.0** и **PaAno / Shared Encoder**.

---

## 0. Короткий вывод

У нас мало собственных промысловых данных, поэтому концентрируемся на одном внешнем доменном датасете: **Petrobras 3W Dataset 2.0.0**.

Нужно построить не “ещё один эксперимент”, а аккуратный production-like исследовательский пайплайн:

```text
3W raw Parquet
-> manifest + validation
-> interval extraction
-> resampling + cleaning
-> feature set
-> instance-level train/val/test split
-> PaAno shared encoder pretrain on normal patches
-> event-specific memory banks / calibration
-> onset detection
-> metrics + HTML reports
-> optional transfer of encoder weights into ALMA pipeline
```

Критическая мысль: **3W используем как доменный нефтяной pretrain/benchmark**, а не как окончательную замену данным заказчика. Если дальше появятся данные заказчика, то production memory bank и thresholds должны строиться только по данным заказчика.

---

## 1. Источники, которые надо использовать

### 1.1 Petrobras 3W

Основной датасет:

```text
GitHub:
https://github.com/petrobras/3W

Dataset folder:
https://github.com/petrobras/3W/tree/main/dataset

Dataset structure:
https://github.com/petrobras/3W/blob/main/3W_DATASET_STRUCTURE.md

dataset.ini:
https://github.com/petrobras/3W/blob/main/dataset/dataset.ini

Figshare 3W Dataset 2.0.0:
https://figshare.com/articles/dataset/3W_Dataset_2_0_0/29205836

DOI:
https://doi.org/10.6084/m9.figshare.29205836

Article:
https://arxiv.org/abs/2507.01048
```

### 1.2 PaAno

Базовый метод:

```text
PaAno GitHub:
https://github.com/jinnnju/PaAno

PaAno paper:
https://arxiv.org/abs/2602.01359

OpenReview:
https://openreview.net/forum?id=NXThkM7Iym
```

PaAno — representation-based метод для time-series anomaly detection. Он режет ряд на temporal patches, кодирует патчи через 1D-CNN, обучает embedding через triplet/pretext losses, а во время инференса считает anomaly score через сравнение embedding текущих патчей с embedding нормальных training-патчей.

---

## 2. Что именно надо сделать

Нужно реализовать в репозитории новый 3W-пайплайн, который:

1. Скачивает или использует локально скачанный 3W Dataset 2.0.0.
2. Валидирует структуру датасета.
3. Строит manifest всех Parquet-инстансов.
4. Извлекает интервалы нормы, transient-участки и интервалы нежелательных событий.
5. Готовит единый табличный формат для PaAno.
6. Обучает PaAno shared encoder на нормальных патчах 3W.
7. Строит memory bank нормы.
8. Считает anomaly score на validation/test инстансах.
9. Подбирает пороги только на validation/train.
10. Оценивает качество на test без leakage.
11. Генерирует отчёты и графики.
12. Сохраняет веса encoder-а так, чтобы их можно было использовать как `3w_pretrained` warm-start в ALMA.

---

## 3. Нельзя делать

Запрещено:

```text
- Делить данные построчно случайным train/test split.
- Использовать test-инстансы для pretrain, threshold tuning или feature selection.
- Обучать supervised classifier “class -> anomaly”.
- Подбирать пороги на test.
- Смешивать labels в признаки.
- Интерполировать class/state как обычные числовые признаки.
- Считать 3W production-заменой данным заказчика.
- Создавать новые production detector keys вроде paano_3w, paano_global, paano_magic.
```

Правильный detector key в ALMA-контексте должен оставаться:

```text
paano_shared
```

Разрешается добавить training mode / checkpoint name:

```text
encoder_training_mode = 3w_pretrain
encoder_training_mode = 3w_pretrain_alma_finetune
```

---

## 4. Факты о 3W Dataset 2.0.0, которые надо учитывать

### 4.1 Версия и формат

По `dataset.ini` версия датасета:

```text
DATASET = 2.0.0
```

3W Dataset 2.0.0 хранится в Parquet. Файлы читаются через:

```text
pandas + pyarrow engine + brotli compression
```

Структура:

```text
3W/dataset/
  folds/
  0/
  1/
  2/
  ...
  9/
```

Смысл:

```text
dataset/<label>/*.parquet
```

где подкаталог соответствует типу события / label.

Каждый Parquet-файл — один instance. Внутри:

```text
- timestamp хранится как index;
- каждая observation — строка;
- переменные — float columns;
- labels — Int64 columns;
- class — label наблюдения;
- state — operational status скважины.
```

### 4.2 Event classes

Из `dataset.ini`:

| Label | Internal name | Description |
|---:|---|---|
| 0 | NORMAL | Normal Operation |
| 1 | ABRUPT_INCREASE_OF_BSW | Abrupt Increase of BSW |
| 2 | SPURIOUS_CLOSURE_OF_DHSV | Spurious Closure of DHSV |
| 3 | SEVERE_SLUGGING | Severe Slugging |
| 4 | FLOW_INSTABILITY | Flow Instability |
| 5 | RAPID_PRODUCTIVITY_LOSS | Rapid Productivity Loss |
| 6 | QUICK_RESTRICTION_IN_PCK | Quick Restriction in PCK |
| 7 | SCALING_IN_PCK | Scaling in PCK |
| 8 | HYDRATE_IN_PRODUCTION_LINE | Hydrate in Production Line |
| 9 | HYDRATE_IN_SERVICE_LINE | Hydrate in Service Line |

В `dataset.ini` также указан:

```text
TRANSIENT_OFFSET = 100
```

Это значит, что при разборе labels надо отдельно проверять не только основной event label, но и transient labels. Нельзя просто делать `class == folder_label` и игнорировать `class >= 100`.

### 4.3 Переменные 3W

По `dataset.ini` в Parquet могут быть такие переменные:

| Column | Meaning |
|---|---|
| ABER-CKGL | Opening of gas lift choke |
| ABER-CKP | Opening of production choke |
| ESTADO-DHSV | State of downhole safety valve |
| ESTADO-M1 | State of production master valve |
| ESTADO-M2 | State of annulus master valve |
| ESTADO-PXO | State of pig-crossover valve |
| ESTADO-SDV-GL | State of gas lift shutdown valve |
| ESTADO-SDV-P | State of production shutdown valve |
| ESTADO-W1 | State of production wing valve |
| ESTADO-W2 | State of annulus wing valve |
| ESTADO-XO | State of crossover valve |
| P-ANULAR | Pressure in well annulus |
| P-JUS-BS | Downstream pressure of service pump |
| P-JUS-CKGL | Downstream pressure of gas lift choke |
| P-JUS-CKP | Downstream pressure of production choke |
| P-MON-CKGL | Upstream pressure of gas lift choke |
| P-MON-CKP | Upstream pressure of production choke |
| P-MON-SDV-P | Upstream pressure of production shutdown valve |
| P-PDG | Downhole pressure at permanent downhole gauge |
| PT-P | Subsea Xmas-tree pressure downstream of production wing valve |
| P-TPT | Subsea Xmas-tree pressure at TPT |
| QBS | Flow rate at service pump |
| QGL | Gas lift flow rate |
| T-JUS-CKP | Downstream temperature of production choke |
| T-MON-CKP | Upstream temperature of production choke |
| T-PDG | Downhole temperature at permanent downhole gauge |
| T-TPT | Subsea Xmas-tree temperature at TPT |
| class | Label of observation |
| state | Well operational status |

Важно: это не ЭЦН-датасет. Здесь нет токов фаз, ПЧ, вибраций XYZ и загрузки ПЭД. Поэтому для ALMA это **доменный нефтяной pretrain**, а не direct production replacement.

---

## 5. Архитектурное решение

### 5.1 Основная схема

```text
Stage 1. 3W standalone detector
  3W normal patches -> PaAno encoder pretrain
  3W train normal -> memory bank
  3W val -> threshold/onset tuning
  3W test -> final evaluation

Stage 2. Optional ALMA transfer
  load 3W-pretrained encoder
  fine-tune on ALMA reference
  build ALMA memory bank from ALMA reference only
  tune thresholds on ALMA train only
  validate on ALMA test only
```

### 5.2 Почему так

PaAno не требует много размеченных аномалий для обучения encoder-а. Ему критичны нормальные патчи, потому что anomaly score считается как удалённость текущего патча от memory bank нормальных патчей.

Поэтому 3W используем так:

```text
3W class 0 NORMAL
+ нормальные участки внутри event instances
-> clean-normal pool
-> pretrain shared encoder
```

А event intervals используем для:

```text
- threshold tuning;
- onset evaluation;
- per-class metrics;
- stress-test физически разных типов событий.
```

---

## 6. Целевая структура файлов в репозитории

Добавить примерно такую структуру:

```text
configs/
  3w_paano.yaml

scripts/
  3w/
    download_3w.py
    build_3w_manifest.py
    prepare_3w_dataset.py
    train_3w_paano.py
    detect_3w_paano.py
    evaluate_3w_onset.py
    generate_3w_report.py
    export_3w_encoder_for_alma.py

alma_service/
  datasets/
    three_w.py
  detectors/
    paano_shared.py              # если уже есть — не дублировать
  training/
    three_w_pretrain.py
  evaluation/
    onset_metrics.py
  reports/
    three_w_report.py

data/
  raw/
    3w/
  processed/
    3w/
      manifest.parquet
      intervals.parquet
      splits.parquet
      timeseries_1min.parquet
      features_1min.parquet

artifacts/
  3w/
    checkpoints/
    scores/
    metrics/
    reports/
```

Если в проекте уже есть иная структура, агент должен адаптироваться под неё, но сохранить смысл.

---

## 7. Конфиг `configs/3w_paano.yaml`

Создать конфиг:

```yaml
dataset:
  name: "3W"
  version: "2.0.0"
  raw_dir: "data/raw/3w"
  processed_dir: "data/processed/3w"
  manifest_path: "data/processed/3w/manifest.parquet"
  intervals_path: "data/processed/3w/intervals.parquet"
  splits_path: "data/processed/3w/splits.parquet"

  parquet_engine: "pyarrow"
  parquet_compression: "brotli"

  timestamp_index: true
  label_col: "class"
  state_col: "state"
  transient_offset: 100

  event_labels:
    0: "NORMAL"
    1: "ABRUPT_INCREASE_OF_BSW"
    2: "SPURIOUS_CLOSURE_OF_DHSV"
    3: "SEVERE_SLUGGING"
    4: "FLOW_INSTABILITY"
    5: "RAPID_PRODUCTIVITY_LOSS"
    6: "QUICK_RESTRICTION_IN_PCK"
    7: "SCALING_IN_PCK"
    8: "HYDRATE_IN_PRODUCTION_LINE"
    9: "HYDRATE_IN_SERVICE_LINE"

preprocess:
  resample_freq: "1min"
  min_points_per_instance: 128
  max_interpolate_gap: "15min"
  add_missingness_features: true
  drop_columns:
    - "class"
    - "state"
  keep_valve_state_columns: true

features:
  use_raw_channels: true
  use_rolling_features: true
  rolling_windows:
    - "5min"
    - "15min"
    - "60min"
  rolling_ops:
    - "mean"
    - "std"
    - "min"
    - "max"
    - "slope"
  add_pressure_diffs: true
  add_flow_pressure_ratios: true
  max_features_after_reduction: 80

splits:
  split_unit: "instance"
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  stratify_by:
    - "event_label"
    - "source_type"
  prefer_real_in_test: true
  seed: 2027

paano:
  patch_sizes:
    - 64
    - 96
  stride: 1
  embedding_dim: 128
  encoder: "1d_cnn"
  train_iters: 200
  batch_size: 256
  learning_rate: 0.001
  device: "cuda"
  seed: 2027

memory_bank:
  source: "train_normal_patches"
  max_patches_per_event_class: 200000
  sampling: "balanced_by_instance"
  distance: "cosine"
  knn_k: 5

thresholding:
  tune_on: "val"
  target_far_per_day_grid:
    - 0.01
    - 0.05
    - 0.10
    - 0.25
    - 0.50
  min_run_points_grid:
    - 2
    - 3
    - 4
    - 5
  ema_alpha_grid:
    - 0.04
    - 0.08
    - 0.12
    - 0.20
  cooldown_hours_grid:
    - 6
    - 12
    - 24
    - 48

evaluation:
  main_target: "undesirable_start"
  also_report:
    - "transient_start"
    - "event_start"
  lead_tolerance_hours: 24
  late_tolerance_hours: 24
  metrics:
    - "hit_rate"
    - "false_alarms_per_day"
    - "starts_per_event"
    - "median_delay_hours"
    - "p90_delay_hours"
    - "range_f1"
    - "auc_pr"
```

---

## 8. Скрипт 1: `download_3w.py`

Назначение: помочь пользователю получить данные.

Поведение:

1. Проверить, есть ли локальная директория `data/raw/3w/dataset`.
2. Если есть — ничего не скачивать.
3. Если нет — вывести инструкции.
4. Не делать скрытый download огромных файлов без явной команды.
5. Поддержать варианты:
   - clone GitHub repo;
   - download Figshare archive вручную;
   - путь к уже скачанному датасету.

CLI:

```bash
uv run python scripts/3w/download_3w.py \
  --raw-dir data/raw/3w \
  --source github
```

Допустимое поведение:

```bash
git clone https://github.com/petrobras/3W data/raw/3w/3W
```

После скачивания проверить:

```text
data/raw/3w/3W/dataset/dataset.ini
data/raw/3w/3W/3W_DATASET_STRUCTURE.md
data/raw/3w/3W/dataset/<0..9>/*.parquet
```

---

## 9. Скрипт 2: `build_3w_manifest.py`

Назначение: построить полный manifest всех Parquet-инстансов.

CLI:

```bash
uv run python scripts/3w/build_3w_manifest.py \
  --dataset-dir data/raw/3w/3W/dataset \
  --out data/processed/3w/manifest.parquet
```

Manifest columns:

```text
instance_id
path
folder_label
event_name
source_type
n_rows
start_ts
end_ts
duration_hours
columns
n_columns
has_class
has_state
class_values
state_values
missing_ratio_total
missing_ratio_by_column
is_valid
validation_errors
```

Как определить `source_type`:

```text
1. Сначала изучить реальные имена файлов в dataset/<label>/.
2. Если convention ясен — реализовать parser.
3. Если convention не ясен — поставить source_type="unknown" и не ломать pipeline.
4. Не придумывать source_type из головы.
```

Валидация каждого файла:

```python
import pandas as pd

df = pd.read_parquet(path, engine="pyarrow")

assert isinstance(df.index, pd.DatetimeIndex) or can_convert_index_to_datetime
assert "class" in df.columns
assert "state" in df.columns
assert len(df) > 0
```

Важно: `class` и `state` не должны попадать в feature columns.

---

## 10. Скрипт 3: `prepare_3w_dataset.py`

Назначение: подготовить единый датасет для PaAno и извлечь intervals.

CLI:

```bash
uv run python scripts/3w/prepare_3w_dataset.py \
  --config configs/3w_paano.yaml
```

Выходы:

```text
data/processed/3w/timeseries_1min.parquet
data/processed/3w/features_1min.parquet
data/processed/3w/intervals.parquet
data/processed/3w/splits.parquet
data/processed/3w/normal_pool.parquet
```

### 10.1 Извлечение интервалов

Для каждого instance:

```text
folder_label = int(parent directory name)
event_label = folder_label
transient_label = event_label + TRANSIENT_OFFSET
```

Маски:

```python
normal_mask = df["class"].eq(0)
event_mask = df["class"].eq(event_label)
transient_mask = df["class"].eq(event_label + transient_offset)
non_normal_mask = ~df["class"].eq(0)
```

Нужно сохранять три варианта старта:

```text
normal_end_ts
transient_start_ts
event_start_ts
undesirable_start_ts = min(first transient_start_ts, first event_start_ts, first non_normal_ts)
```

`intervals.parquet` columns:

```text
instance_id
event_label
event_name
source_type
start_ts
end_ts
target_type                # transient/event/undesirable
transient_start_ts
event_start_ts
undesirable_start_ts
normal_end_ts
duration_hours
n_points
has_transient
has_event
```

Главный target для onset detection:

```text
undesirable_start_ts
```

Но в отчёте обязательно показывать также:

```text
transient_start_ts
event_start_ts
```

### 10.2 Resampling

Default:

```text
resample_freq = 1min
```

Почему 1 минута:

```text
- 3W может быть достаточно плотным по времени;
- PaAno на raw-секундной сетке может стать тяжёлым;
- для нефтяных режимов 1 минута обычно сохраняет технологическую динамику;
- при необходимости можно добавить 10s и 5min как ablation.
```

Правила:

1. Numeric process columns агрегировать через median или mean.
2. `class` агрегировать отдельно:
   - если внутри окна есть non-zero label, окно получает non-zero label;
   - transient/event labels должны сохраняться;
   - нельзя усреднять labels.
3. `state` агрегировать через mode / last valid.
4. Интерполировать только process variables.
5. Не интерполировать `class` и `state`.
6. Не заполнять длинные gaps:
   - если gap > `max_interpolate_gap`, оставить NaN или разорвать segment.
7. Добавить missingness indicators:
   - `is_missing::<col>`
   - `gap_seconds`
   - `valid_observation_count`

### 10.3 Feature engineering

Начальный feature set:

```text
raw process variables
+ missingness indicators
+ rolling mean/std/min/max/slope
+ pressure differences
+ flow/pressure ratios
+ valve movement indicators
```

Не включать:

```text
class
state как target label
timestamp как числовой feature
future-looking rolling features
```

Можно включить valve state columns (`ESTADO-*`) как контекстные признаки, но:
- проверить, не доминируют ли они над score;
- сделать ablation `with_valves` vs `without_valves`;
- для leakage-sensitive оценки сохранить оба результата.

---

## 11. Splits без leakage

Сделать split на уровне instance, не на уровне rows.

Правильно:

```text
train instances
val instances
test instances
```

Неправильно:

```text
случайно перемешать строки одного и того же instance между train/test
```

Если в 3W есть готовые fold-конфиги в `dataset/folds`, сначала изучить их. Если они подходят, использовать их. Если нет — создать свой deterministic split.

Правила split:

```text
- stratify by event_label;
- если source_type известен, stratify by source_type;
- test должен содержать real instances, если такие есть;
- simulated/hand-drawn можно использовать в train/pretrain;
- seed фиксировать: 2027;
- сохранить splits.parquet;
- сохранить split summary JSON.
```

`splits.parquet`:

```text
instance_id
event_label
event_name
source_type
split        # train/val/test
reason
seed
```

---

## 12. PaAno training strategy

### 12.1 Минимальный baseline

```text
Train:
  train normal patches from all train instances

Validation:
  val instances for threshold/onset tuning

Test:
  test instances for final metrics
```

### 12.2 Более сильная схема

```text
Global 3W pretrain:
  all clean normal segments from train instances across all event classes
  + class 0 NORMAL train instances

Event-specific calibration:
  for each event label 1..9:
    memory bank = normal patches from train instances of this event + class 0 normal train
    thresholds = tuned on val instances
    test = held-out test instances of this event
```

### 12.3 Почему memory bank должен быть event-specific

Если сделать один слишком широкий memory bank на все режимы, можно получить “слишком широкую норму”, которая начнет считать ранний drift нормальным.

Поэтому сравнить два режима:

```text
A. global_memory_bank
B. event_specific_memory_bank
```

Default candidate:

```text
event_specific_memory_bank
```

### 12.4 Checkpoints

Сохранять:

```text
artifacts/3w/checkpoints/
  paano_3w_global_encoder.pt
  paano_3w_event_1_encoder.pt
  paano_3w_event_2_encoder.pt
  ...
  metadata.json
```

`metadata.json`:

```json
{
  "dataset": "3W",
  "dataset_version": "2.0.0",
  "training_mode": "3w_global_pretrain_event_specific_memory_bank",
  "seed": 2027,
  "resample_freq": "1min",
  "patch_sizes": [64, 96],
  "feature_columns_hash": "...",
  "train_instance_ids": ["..."],
  "val_instance_ids": ["..."],
  "test_instance_ids": ["..."],
  "created_at": "..."
}
```

---

## 13. Скрипт 4: `train_3w_paano.py`

CLI:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/3w/train_3w_paano.py \
  --config configs/3w_paano.yaml \
  --mode global_pretrain_event_specific \
  --out-dir artifacts/3w/checkpoints
```

Что сделать внутри:

1. Загрузить `features_1min.parquet`, `intervals.parquet`, `splits.parquet`.
2. Выбрать train instances.
3. Собрать clean normal pool:
   - `class == 0`;
   - до `undesirable_start_ts` для event instances;
   - исключить нестабильные start/end fragments;
   - исключить участки с большими gaps.
4. Нормализацию fit делать только на train normal pool.
5. Создать patches.
6. Обучить PaAno encoder.
7. Сохранить encoder.
8. Построить memory banks.
9. Сохранить memory banks.

Нормализация:

```python
x_scaled = (x - median_train) / iqr_train
```

Не использовать mean/std как единственный вариант: 3W содержит outliers и реальные industrial artifacts.

---

## 14. Скрипт 5: `detect_3w_paano.py`

CLI:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/3w/detect_3w_paano.py \
  --config configs/3w_paano.yaml \
  --checkpoint artifacts/3w/checkpoints/paano_3w_global_encoder.pt \
  --split val \
  --out artifacts/3w/scores/val_scores.parquet
```

И для test:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/3w/detect_3w_paano.py \
  --config configs/3w_paano.yaml \
  --checkpoint artifacts/3w/checkpoints/paano_3w_global_encoder.pt \
  --split test \
  --out artifacts/3w/scores/test_scores.parquet
```

`scores.parquet` columns:

```text
instance_id
timestamp
event_label
event_name
split
score_raw
score_ema
threshold
is_above_threshold
detected_start
class
state
is_reference
is_allowed_for_detection
paano_score_short
paano_score_long
memory_bank_distance
nearest_neighbor_distance
```

---

## 15. Onset detection

State machine:

```text
score_raw
-> EMA smoothing
-> threshold crossing
-> min_run_points
-> cooldown
-> detected_start
```

Pseudo:

```python
def detect_starts(score, threshold, timestamps, min_run_points, cooldown):
    starts = []
    run = 0
    last_start = None

    for t, s in zip(timestamps, score):
        above = s >= threshold

        if above:
            run += 1
        else:
            run = 0

        if run >= min_run_points:
            candidate = t

            if last_start is None or (candidate - last_start) >= cooldown:
                starts.append(candidate)
                last_start = candidate

            run = 0

    return starts
```

Важно:
- не разрешать detection внутри reference/normal training portion;
- detection_allowed_mask должен исключать warm-up до первого полного patch;
- first valid score появляется только после достаточной истории для patch.

---

## 16. Threshold tuning

Tuning только на validation.

Grid:

```text
target_far_per_day
min_run_points
ema_alpha
cooldown_hours
memory_bank_k
patch_size weights
```

Objective:

```text
maximize:
  hit_rate

minimize:
  false_alarms_per_day
  starts_per_event
  median_delay_hours
  p90_delay_hours
```

Не выбирать конфиг только по hit-rate. Иначе можно получить слишком шумный detector.

Suggested scoring:

```python
objective = (
    100.0 * hit_rate
    - 10.0 * false_alarms_per_day
    - 2.0 * starts_per_event
    - 0.2 * max(median_delay_hours, 0)
    - 0.1 * max(p90_delay_hours, 0)
)
```

Отрицательный delay допустим, если detector сработал до event label, но надо проверять, что это не false positive задолго до события.

---

## 17. Evaluation

### 17.1 Главные метрики

Для каждого event class:

```text
Hit-rate
False alarms per day on normal periods
Starts per event
Median delay hours
P90 delay hours
Range-F1
AUC-PR
```

Delay:

```text
delay_hours = detected_start - target_start
```

Main target:

```text
target_start = undesirable_start_ts
```

Дополнительно report:

```text
delay_to_transient_start
delay_to_event_start
```

### 17.2 Acceptance table

Сформировать таблицу:

| Event | Split | Instances | Hit-rate | FAR/day | Starts/event | Median delay | P90 delay | AUC-PR | Range-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAPID_PRODUCTIVITY_LOSS | test | ... | ... | ... | ... | ... | ... | ... | ... |
| SCALING_IN_PCK | test | ... | ... | ... | ... | ... | ... | ... | ... |
| FLOW_INSTABILITY | test | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | test | ... | ... | ... | ... | ... | ... | ... | ... |

### 17.3 Normal-only guard

Обязательно отдельно проверить class 0 NORMAL test instances:

```text
expected:
  no event hits
  low false alarm rate
```

Если модель сильно срабатывает на NORMAL, она непригодна независимо от hit-rate на event classes.

---

## 18. Физические ветки для 3W

Базово сделать чистый PaAno. Потом добавить физические diagnostic branches, но не смешивать их с PaAno до baseline.

Recommended branches:

### 18.1 Rapid Productivity Loss

Смысл: падение продуктивности должно отражаться в давлениях/расходах.

Signals:

```text
P-PDG
P-TPT
PT-P
P-MON-CKP
P-JUS-CKP
QGL
QBS
ABER-CKP
```

Branch:

```text
pressure/flow trend score
```

### 18.2 Scaling in PCK

Смысл: restriction/scaling near production choke.

Signals:

```text
P-MON-CKP
P-JUS-CKP
ABER-CKP
T-MON-CKP
T-JUS-CKP
```

Branch:

```text
choke differential pressure drift
delta_p_ckp = P-MON-CKP - P-JUS-CKP
```

### 18.3 Flow Instability / Severe Slugging

Смысл: oscillatory instability.

Signals:

```text
pressure channels
flow channels
temperature channels
```

Branch:

```text
rolling std / spectral roughness / oscillation score
```

### 18.4 Quick Restriction in PCK

Смысл: быстрый step-like event.

Branch:

```text
step detector on delta_p_ckp and flow
```

### 18.5 Hydrate events

Смысл: pressure/temperature/flow joint shift.

Branch:

```text
temperature-pressure-flow consistency score
```

Важно: physical branches должны быть train/val tuned. Если branch ухудшает test или NORMAL guard, отключить его.

---

## 19. Reports

Сгенерировать HTML-отчёты:

```text
artifacts/3w/reports/
  3w_overview.html
  3w_event_1_ABRUPT_INCREASE_OF_BSW.html
  3w_event_2_SPURIOUS_CLOSURE_OF_DHSV.html
  ...
  3w_event_9_HYDRATE_IN_SERVICE_LINE.html
```

Каждый event report:

1. Summary metrics.
2. Split table.
3. Per-instance plots:
   - score;
   - threshold;
   - true interval;
   - detected starts;
   - key physical variables.
4. Worst misses.
5. Worst false positives.
6. Top feature importance / channel contribution.
7. Notes on data gaps.

Plot panels:

```text
Panel 1: PaAno score + threshold + detections
Panel 2: pressures
Panel 3: flow / choke / temperature depending on event
Panel 4: class/state labels
```

---

## 20. Команды полного запуска

Ожидаемый full run:

```bash
# 0. Prepare dirs
mkdir -p data/raw/3w data/processed/3w artifacts/3w

# 1. Download or clone
uv run python scripts/3w/download_3w.py \
  --raw-dir data/raw/3w \
  --source github

# 2. Manifest
uv run python scripts/3w/build_3w_manifest.py \
  --dataset-dir data/raw/3w/3W/dataset \
  --out data/processed/3w/manifest.parquet

# 3. Prepare dataset
uv run python scripts/3w/prepare_3w_dataset.py \
  --config configs/3w_paano.yaml

# 4. Train PaAno
CUDA_VISIBLE_DEVICES=1 uv run python scripts/3w/train_3w_paano.py \
  --config configs/3w_paano.yaml \
  --mode global_pretrain_event_specific \
  --out-dir artifacts/3w/checkpoints

# 5. Detect validation
CUDA_VISIBLE_DEVICES=1 uv run python scripts/3w/detect_3w_paano.py \
  --config configs/3w_paano.yaml \
  --split val \
  --out artifacts/3w/scores/val_scores.parquet

# 6. Tune thresholds
uv run python scripts/3w/evaluate_3w_onset.py \
  --config configs/3w_paano.yaml \
  --scores artifacts/3w/scores/val_scores.parquet \
  --mode tune \
  --out artifacts/3w/metrics/selected_thresholds.json

# 7. Detect test
CUDA_VISIBLE_DEVICES=1 uv run python scripts/3w/detect_3w_paano.py \
  --config configs/3w_paano.yaml \
  --split test \
  --thresholds artifacts/3w/metrics/selected_thresholds.json \
  --out artifacts/3w/scores/test_scores.parquet

# 8. Evaluate test
uv run python scripts/3w/evaluate_3w_onset.py \
  --config configs/3w_paano.yaml \
  --scores artifacts/3w/scores/test_scores.parquet \
  --mode evaluate \
  --out artifacts/3w/metrics/test_metrics.json

# 9. Reports
uv run python scripts/3w/generate_3w_report.py \
  --config configs/3w_paano.yaml \
  --scores artifacts/3w/scores/test_scores.parquet \
  --metrics artifacts/3w/metrics/test_metrics.json \
  --out-dir artifacts/3w/reports

# 10. Export encoder for ALMA optional
uv run python scripts/3w/export_3w_encoder_for_alma.py \
  --checkpoint artifacts/3w/checkpoints/paano_3w_global_encoder.pt \
  --out artifacts/3w/checkpoints/alma_warm_start_encoder_3w.pt
```

---

## 21. Quality gates

Работа считается выполненной, если:

```text
1. 3W manifest построен и сохранён.
2. Все Parquet-инстансы валидированы.
3. intervals.parquet содержит starts для transient/event/undesirable.
4. splits.parquet сделан на уровне instance.
5. Есть доказательство отсутствия test leakage.
6. PaAno encoder обучен.
7. Scores сохранены в Parquet.
8. Пороги подобраны только на val.
9. Test metrics посчитаны отдельно.
10. NORMAL guard посчитан отдельно.
11. HTML reports сгенерированы.
12. Checkpoint encoder-а экспортирован.
13. README с командами запуска обновлён.
```

---

## 22. Что агент должен проверить в коде перед реализацией

Перед написанием новых файлов агент обязан найти в репозитории:

```text
- где сейчас реализован PaAno Shared Encoder;
- есть ли уже SharedPaAnoDetector;
- как сейчас сохраняются scores;
- как устроен onset detection;
- как устроен evaluate_onset_metrics.py;
- как устроены отчёты;
- есть ли общий config loader;
- есть ли принятый стиль scripts/detection и scripts/evaluation.
```

Если в репозитории уже есть аналоги, не дублировать логику. Нужно переиспользовать.

---

## 23. Минимальный кодовый API

Ожидаемые функции:

```python
def load_3w_instance(path: str) -> pd.DataFrame:
    ...

def build_3w_manifest(dataset_dir: str) -> pd.DataFrame:
    ...

def extract_3w_intervals(df: pd.DataFrame, instance_id: str, folder_label: int, transient_offset: int = 100) -> pd.DataFrame:
    ...

def resample_3w_instance(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    ...

def build_3w_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    ...

def make_instance_level_splits(manifest: pd.DataFrame, config: dict) -> pd.DataFrame:
    ...

def collect_clean_normal_patches(features: pd.DataFrame, intervals: pd.DataFrame, splits: pd.DataFrame) -> PatchDataset:
    ...

def train_paano_encoder(patches: PatchDataset, config: dict) -> torch.nn.Module:
    ...

def score_instances_with_paano(model, memory_bank, features: pd.DataFrame, config: dict) -> pd.DataFrame:
    ...

def tune_onset_thresholds(scores: pd.DataFrame, intervals: pd.DataFrame, config: dict) -> dict:
    ...

def evaluate_onset(scores: pd.DataFrame, intervals: pd.DataFrame, thresholds: dict) -> dict:
    ...
```

---

## 24. Особые риски

### 24.1 Label leakage через `class` и `state`

`class` точно нельзя использовать как feature.  
`state` тоже нельзя без проверки: это operational status, он может быть слишком близок к label. Default: не использовать `state` как feature, только как служебную колонку для анализа.

### 24.2 Leakage через нормальные участки event instance

Если event instance попадает в test, его нормальные участки нельзя использовать в train normal pool.

Правило:

```text
normal segments from test instances are test-only
```

### 24.3 Simulated / hand-drawn vs real

Если источник instance можно определить:
- train может включать simulated и hand-drawn;
- test должен обязательно показывать real-only metrics;
- финальный отчёт должен разделять all-test и real-test.

### 24.4 Слишком широкий memory bank

Если memory bank из всех событий ухудшает раннее обнаружение, перейти на event-specific memory banks.

### 24.5 Слишком агрессивная интерполяция

Нельзя заливать огромные gaps линейной интерполяцией. Для 3W реальные gaps и missing variables оставлены намеренно. Нужно сохранять missingness как сигнал качества данных.

---

## 25. Что написать в README после реализации

Добавить раздел:

```markdown
## 3W PaAno pretraining / evaluation

This pipeline trains and evaluates a PaAno-based anomaly detector on Petrobras 3W Dataset 2.0.0.

Sources:
- https://github.com/petrobras/3W
- https://doi.org/10.6084/m9.figshare.29205836
- https://github.com/jinnnju/PaAno

Run:
...
```

И объяснить:

```text
3W is used as a domain pretraining and public oil-well benchmark dataset.
It does not replace customer-specific production calibration.
```

---

## 26. Финальный deliverable для заказчика / отчёта

Нужно получить:

```text
artifacts/3w/metrics/test_metrics.json
artifacts/3w/reports/3w_overview.html
artifacts/3w/checkpoints/paano_3w_global_encoder.pt
artifacts/3w/checkpoints/alma_warm_start_encoder_3w.pt
data/processed/3w/manifest.parquet
data/processed/3w/intervals.parquet
data/processed/3w/splits.parquet
```

И короткое резюме:

```text
Был реализован и проверен PaAno-based detector на публичном нефтегазовом датасете Petrobras 3W Dataset 2.0.0. 
Датасет использовался как доменный источник нормальных и аномальных многомерных временных рядов нефтяных скважин. 
Обучение encoder-а проводилось на нормальных патчах train-инстансов, подбор порогов — на validation, итоговая оценка — на отложенном test split без утечки данных.
Полученный encoder может использоваться как warm-start для дальнейшей адаптации на телеметрии ALMA.
```

---

## 27. Приоритет реализации

Порядок работы для агента:

```text
P0. Найти существующие PaAno/onset/report utilities в репозитории.
P1. Реализовать manifest + validation.
P2. Реализовать intervals extraction.
P3. Реализовать resampling + feature table.
P4. Реализовать instance-level splits.
P5. Подключить существующий PaAno shared encoder.
P6. Получить первый scores.parquet на val/test.
P7. Настроить onset metrics.
P8. Добавить reports.
P9. Добавить physical branches только после чистого PaAno baseline.
P10. Экспортировать 3W-pretrained encoder для ALMA.
```

Не начинать с физической ветки и красивых отчётов. Сначала нужен честный baseline без leakage.

---

## 28. Критерий “классная система”

Система считается сильной, если она не только показывает высокий hit-rate, но и:

```text
- имеет низкий FAR на NORMAL;
- не дробит одно событие на десятки starts;
- показывает задержку detection относительно transient/event start;
- разделяет качество по event classes;
- показывает real-only metrics;
- умеет объяснить score через ключевые физические каналы;
- сохраняет воспроизводимые configs/checkpoints/splits;
- не загрязняет ALMA production API новыми detector keys.
```
