# ALMA: class-specific PaAno Shared vs global pretrain

Дата: 2026-05-12

Этот документ фиксирует текущую позицию по вопросу: нужно ли обучать
`paano_shared` отдельно по каждому классу аномалии или переходить к глобальному
pretrain на общей нормальной работе с последующим class fine-tune.

Цель документа - передать контекст другому ИИ-агенту или инженеру без истории
чата.

## 1. Текущий production-пайплайн

После рефакторинга production-путь в репозитории один:

```text
Excel/Parquet данные
-> build_*_dataset.py
-> db/*_anomaly_database_*.parquet + db/*_intervals.parquet
-> detect_*.py --detector paano_shared
-> PaAno Shared Encoder
-> физическая ветка конкретной аномалии
-> conformal/tail calibration + onset detection
-> results/scores/predicted_starts
-> HTML reports + feature importance
```

Публичный detector key теперь один:

```text
paano_shared
```

Удалены как production-ветки:

- `pca_spe`;
- `paano_feat`;
- `ensemble`;
- legacy `detect_*_paano.py`;
- legacy `alma_service/paano_pipeline.py`.

Важно: PCA/SPE не исчез как идея. Для соли его полезная часть перенесена внутрь
`salt_deposition` как PCA/SPE-like residual, но это больше не отдельный detector
key.

## 2. Что различается между классами сейчас

Единый detector key не означает один одинаковый физический сигнал. Сейчас общий
каркас один, а физическая ветка зависит от природы аномалии:

```text
negermet:
  PaAno Shared
  + negermet_signature
  pressure-step / load-response

pritok:
  PaAno Shared
  + pressure_trend
  тренд давления на приеме насоса

salt:
  PaAno Shared
  + salt_deposition
  grouped multichannel drift
  + PCA/SPE-like residual
  + KS/distribution shift
```

Это осознанное решение: унифицировать каркас, но не притворяться, что физика
притока, соли и негермета одинаковая.

## 3. Что означает class-specific обучение

Текущий production-вариант - class-specific shared encoder.

Это значит:

```text
negermet encoder обучается на нормальной работе train-скважин negermet
pritok encoder обучается на нормальной работе train-скважин pritok
salt encoder обучается на нормальной работе train-скважин salt
```

При этом внутри каждого класса:

- используется только reference/normal часть;
- разметка аномалий нужна не чтобы учить PaAno “аномалии”, а чтобы отделить
  норму от аномальных интервалов и настроить onset/tuning;
- test-скважины не используются для подбора параметров;
- физические ветки и веса тюнятся train-only.

Плюсы class-specific:

- модель видит норму именно в режиме, близком к данному классу датасета;
- меньше риск negative transfer между разными распределениями;
- текущие метрики уже подтверждены на размеченных наборах;
- проще объяснять результаты: “детектор класса обучался на нормальной работе
  скважин этого класса”;
- меньше архитектурных развилок.

Минусы class-specific:

- если в новом классе мало размеченных скважин, reference pool может быть
  узким;
- если эксперт прав, что “нормальная работа везде примерно одинаковая”, мы не
  используем весь доступный clean-normal материал;
- при появлении чисто нормальных скважин без класса их трудно напрямую
  использовать в class-specific обучении.

## 4. Что означает global pretrain

Global pretrain - это другой способ получить веса shared encoder, но не другой
detector key.

Идея:

```text
1. Собрать clean-normal pool из всех доступных нормальных участков:
   negermet normal + pritok normal + salt normal + norm_work wells

2. Обучить глобальный PaAno shared encoder на общей нормальной работе.

3. Для каждого класса сделать fine-tune на class-specific normal pool:
   global encoder -> fine-tune negermet
   global encoder -> fine-tune pritok
   global encoder -> fine-tune salt

4. Дальше использовать тот же detector key:
   detect_*.py --detector paano_shared
```

То есть правильная production-форма, если global докажет пользу:

```text
paano_shared
  training_mode = global_pretrain_class_finetune
```

А не:

```text
paano_shared
paano_global
paano_class
paano_ensemble
...
```

Добавлять новые detector keys нельзя, иначе снова начнется накопление моделей и
условий.

## 5. Почему global pretrain не включен в production прямо сейчас

Global pretrain уже представлен в коде как research/benchmark:

```text
scripts/evaluation/benchmark_global_pretrain_finetune.py
alma_service/shared_encoder.py::fine_tune_shared_encoder
```

Но он не включен в production по умолчанию.

Причина: пока нет достаточного доказательства, что global pretrain стабильно
лучше текущего class-specific варианта.

Ранее обсуждалось и проверялось, что global-подход может:

- уменьшать false alarms на одних данных;
- ухудшать latency на других;
- давать неоднозначный результат по `pritok`;
- усложнять объяснение модели без гарантированного выигрыша.

Поэтому текущая позиция:

```text
global pretrain = перспективный research-кандидат
class-specific = текущий production baseline
```

## 6. Аргумент эксперта про нормальную работу

Эксперт сказал, что нормальная работа “не сильно привязана к месторождению” и
масштаб значений “тоже одинаковый”, с оговоркой:

```text
есть нюансы: вязкость, давления, месторождение, режимы;
но с текущим датасетом мы это полноценно не покажем;
на production можно будет совершенствовать дальше.
```

Это важная предметная гипотеза.

Из нее следует:

- global pretrain имеет смысл исследовать;
- чистые normal-work скважины ценны;
- нельзя просто игнорировать 20 новых нормальных скважин;
- но нельзя автоматически считать, что global лучше, пока это не проверено
  метриками.

Корректная инженерная формулировка:

```text
Экспертная гипотеза: нормальная работа достаточно универсальна.
Инженерная проверка: global-pretrained encoder не должен ухудшить detection
latency, hit-rate и false alarms относительно class-specific baseline.
```

## 7. Почему нельзя просто обучить на 20 новых normal_work скважинах

20 новых скважин без аномалий полезны, но они не заменяют размеченные классы.

Они дают только negative/normal evidence:

```text
на этих скважинах не должно быть срабатываний
```

Они не дают positive evidence:

```text
где детектор обязан сработать на соли/притоке/негермет
```

Если обучить или настроить систему только так, чтобы молчать на 20 нормальных
скважинах, можно получить “тихий” детектор, который пропустит реальные
аномалии.

Поэтому normal_work надо использовать так:

- как guard против false alarms;
- как расширение clean-normal pretrain pool;
- как sanity-check распределения;
- но не как единственный критерий качества.

## 8. Что если появится датасет 5000 скважин

Если появится большой датасет, где есть:

- много нормальной работы;
- разные аномалии;
- размеченные интервалы;
- возможно новые классы;
- разные месторождения/режимы;

тогда global pretrain становится намного более сильной стратегией.

Но даже в этом случае не надо делать хаос из моделей.

Правильная схема:

```text
large clean-normal corpus
-> global self-supervised pretrain
-> class-specific adapter/fine-tune/calibration
-> same detector API: paano_shared
```

Если классы размечены:

- positive intervals используются для tuning/evaluation;
- normal intervals используются для pretrain/fine-tune;
- test wells остаются слепыми.

Если часть скважин не размечена:

- их можно использовать как normal pretrain только при достаточной уверенности,
  что они действительно clean;
- иначе лучше использовать их через robust/noisy pretraining, где возможные
  аномалии не ломают модель.

## 9. Критерии, по которым global может стать production

Global pretrain можно включать в production только если он проходит критерии
против class-specific baseline.

Минимальные критерии:

```text
1. Hit-rate не ниже class-specific по каждому классу.

2. Median delay и P90 delay не хуже или улучшаются.

3. False alarms per day не выше.

4. Avg starts per interval не выше.

5. На norm_work скважинах false alarms не растут.

6. На test split нет деградации.

7. Результат стабилен при повторном запуске / seed.
```

Если global выигрывает только по одному классу, а другой ухудшает, нельзя
делать его общим default.

Допустимые варианты в таком случае:

```text
training_mode по классу:
  salt -> global_pretrain_class_finetune
  pritok -> class_specific
  negermet -> class_specific
```

Но detector key все равно остается один:

```text
paano_shared
```

Отличается только способ подготовки весов.

## 10. План честного эксперимента

### 10.1 Baseline

Зафиксировать текущий class-specific baseline:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/detection/detect_negermet.py --detector paano_shared
CUDA_VISIBLE_DEVICES=1 uv run python scripts/detection/detect_pritok.py --detector paano_shared
CUDA_VISIBLE_DEVICES=1 uv run python scripts/detection/detect_salt.py --detector paano_shared
```

Сохранить:

- summary json;
- onset metrics;
- norm_work false alarm screening;
- selected configs;
- score distributions.

### 10.2 Global pretrain experiment

Запустить:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluation/benchmark_global_pretrain_finetune.py
```

Этот скрипт должен сравнить:

```text
class_specific_saved
vs
global_pretrain_finetune
```

по всем трем аномалиям.

### 10.3 Norm-work guard

После global experiment обязательно проверить normal-only wells:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/detection/screen_norm_work_false_alarms.py
```

Смысл проверки:

```text
если global pretrain дает больше срабатываний на заведомой норме,
это плохой production-кандидат даже при неплохих размеченных метриках.
```

### 10.4 Acceptance table

Нужна итоговая таблица:

| Класс | Variant | Hit-rate | FAR/day | Starts/interval | Median delay | P90 delay | Norm-work FAR |
|---|---|---:|---:|---:|---:|---:|---:|
| negermet | class_specific | ... | ... | ... | ... | ... | ... |
| negermet | global_ft | ... | ... | ... | ... | ... | ... |
| pritok | class_specific | ... | ... | ... | ... | ... | ... |
| pritok | global_ft | ... | ... | ... | ... | ... | ... |
| salt | class_specific | ... | ... | ... | ... | ... | ... |
| salt | global_ft | ... | ... | ... | ... | ... | ... |

Решение принимается не по одной красивой цифре, а по балансу всех метрик.

## 11. Как встроить global, если он выиграет

Если global pretrain пройдет acceptance criteria, production-пайплайн должен
измениться так:

```text
build datasets
-> train global shared encoder once
-> fine-tune per anomaly class
-> save final class encoder under current shared encoder path
-> detect_*.py --detector paano_shared
```

Важно:

- не добавлять новый detector key;
- не добавлять fallback на class-specific внутри runtime detection;
- не делать if-else на каждую скважину;
- не смешивать несколько моделей в inference;
- не возвращать `ensemble`.

Runtime detection должен оставаться простым:

```text
load prepared data
load saved shared encoder for anomaly
score with SharedPaAnoDetector
apply class physical branch
apply tuned onset config
write outputs
```

Разница должна жить в training/build stage:

```text
encoder_training_mode:
  class_specific
  или
  global_pretrain_class_finetune
```

## 12. Потенциальные риски global pretrain

### 12.1 Negative transfer

Если нормальная работа разных классов/месторождений не настолько одинаковая,
глобальная модель может усреднить разные режимы и хуже видеть тонкие сигналы.

### 12.2 Слишком широкая норма

Если global pretrain видит слишком много вариантов нормальной работы, она может
начать считать некоторые ранние аномальные drift нормальными.

Это особенно опасно для соли, где аномалия медленная.

### 12.3 Noisy normal pool

Если в “нормальную” глобальную выборку попадут неразмеченные скрытые аномалии,
модель может обучиться считать их нормой.

### 12.4 Усложнение эксплуатации

Если оставить и class-specific, и global как параллельные production-модели,
система снова превратится в набор условий и ручных исключений.

Именно поэтому global должен либо заменить training stage, либо остаться
research.

## 13. Рекомендуемая позиция на сейчас

Текущая рекомендация:

```text
1. Production оставить на class-specific paano_shared.

2. Global pretrain держать как research-кандидат.

3. Проверять global строго через benchmark + norm_work guard.

4. Если global выигрывает стабильно - заменить training stage, не добавляя
   новый detector key.

5. Если global не выигрывает - оставить как исследование и не усложнять
   production.
```

Главная архитектурная мысль:

```text
Нам нужен один inference pipeline.
Экспериментировать можно со способом обучения encoder.
Нельзя плодить detector keys и runtime-ветки.
```

## 14. Вопросы для ИИ-агента, который будет анализировать дальше

1. Достаточно ли текущий `benchmark_global_pretrain_finetune.py` честно
   сравнивает global fine-tune против class-specific baseline?

2. Нужно ли добавить в benchmark обязательный norm_work false alarm guard?

3. Не происходит ли leakage через общий normal pool, если в него попадают
   test-wells или интервалы после начала аномалии?

4. Правильно ли global pool выбирает общие feature channels между классами?

5. Может ли global pretrain ухудшить salt раннее обнаружение из-за слишком
   широкой нормы?

6. Нужно ли делать global pretrain не на всех классах сразу, а на clean-normal
   wells + class-balanced sampling?

7. Нужно ли использовать adapters/head/fine-tune только части encoder, чтобы
   снизить negative transfer?

8. Какие acceptance thresholds выбрать для production-перехода?

9. Нужно ли сохранять `training_mode` в model metadata и выводить его в
   отчетах?

10. Как сделать так, чтобы production CLI остался:

```bash
python scripts/detection/detect_salt.py --detector paano_shared
```

и при этом внутри можно было прозрачно использовать либо class-specific, либо
global-pretrained weights?

