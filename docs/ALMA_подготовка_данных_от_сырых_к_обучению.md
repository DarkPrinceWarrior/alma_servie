# ALMA: подготовка данных от сырых рядов до обучения

Дата фиксации: 2026-05-18.

Документ фиксирует текущие решения по preprocessing для размеченных ALMA-датасетов
`negermet`, `pritok`, `salt` и вспомогательной нормальной выборки `norm_work`.
Цель - не описание модели, а именно путь данных: от сырой телеметрии до окон,
которые попадают в обучение `paano_shared` и экспериментальных общих моделей.

## Scope

В активном scope:

- размеченные ALMA-датасеты `negermet`, `pritok`, `salt`;
- их train/test split;
- `norm_work` как вспомогательная чистая норма.

Не входит в этот документ:

- Salym;
- test35;
- экспертные пакеты для неразмеченных скважин;
- классификация типа аномалии после обнаружения инцидента.

## Базовый путь данных

Текущий production-путь:

```text
Excel / raw telemetry
-> parquet на регулярной временной сетке
-> per-well подготовка каналов
-> engineered features
-> выбор reference-нормы
-> общий train pool
-> feature intersection / reduction
-> PaAno Shared Encoder
```

Текущие production-сетки:

| Аномалия | Parquet | Частота | Текущие patch sizes | Физический смысл |
|---|---|---:|---:|---|
| `negermet` | `db/negermet_anomaly_database_2min.parquet` | 2 минуты | 64 / 128 | примерно 2.1 / 4.3 часа |
| `pritok` | `db/pritok_anomaly_database_10min.parquet` | 10 минут | 96 / 192 | примерно 16 / 32 часа |
| `salt` | `db/salt_anomaly_database_15min.parquet` | 15 минут | 64 / 128 | примерно 16 / 32 часа |

## 1. Сырые временные ряды и единая частота

Подтверждено по сырым данным: внутри одной скважины разные каналы могут иметь
разное количество точек и собственную временную шкалу. Это нормальное свойство
телеметрии, а не ошибка.

Инженерное объяснение: датчик фиксирует новое значение при изменении, а если
значение не меняется, оно может повторяться реже. Поэтому плотность сырых точек
не равна "качеству" канала.

Решение:

- сырые каналы нельзя сравнивать по количеству строк напрямую;
- сравнивать нужно по времени, покрытию и моменту первого валидного значения;
- для обучения pooled/global-модели лучше иметь одну общую частоту;
- смешивать в одном encoder pool разные частоты плохо, потому что один и тот же
  patch size начинает означать разную физическую длительность;
- текущие class-specific production-сетки пока допустимы, потому что patch sizes
  уже подобраны под физическую скорость конкретной аномалии;
- для следующего общего пайплайна нужно отдельно протестировать кандидаты
  общей частоты на размеченных `negermet/pritok/salt`.

Практический вывод: если строим одну общую модель нормальности, надо сначала
пересобрать все три класса на общей сетке и сравнить метрики. Если остаёмся в
class-specific режиме, текущие 2/10/15 минут можно сохранять.

## 2. Пустые каналы и `norm_work`

Пустые каналы нужно выкидывать. Если канал полностью отсутствует по конкретной
скважине, нельзя восстанавливать его искусственно: это будет синтетический
сигнал, а не измерение.

Как сейчас работает по размеченным данным:

- all-NaN канал исключается из `raw_columns` конкретной скважины;
- скважина не выбрасывается целиком, если остальные каналы пригодны;
- shared encoder строится по пересечению feature columns между train-скважинами;
- если канал отсутствует у части train-скважин, он не попадает в общий shared set.

Отдельный вывод по `norm_work`:

| Показатель | Значение |
|---|---:|
| Excel-файлов | 20 |
| Параметров в каждом файле | 26 |
| Полностью пустых каналов | 0 |
| Средний процент пропусков | 0.003% - 0.028% |

`norm_work` сам по себе хороший: каналы есть, пропусков мало. Проблема не в
качестве этих данных, а в наивном смешивании с ALMA train pool.

Факт проверки:

```text
ALMA labeled only:
  shared_before_reduction = 138
  shared_after_reduction = 80

ALMA labeled + norm_work:
  shared_before_reduction = 138
  shared_after_reduction = 1
  оставшийся признак = Дисбаланс токов::raw
```

Точная формулировка: пересечение признаков не схлопнулось. Оно осталось 138.
Схлопнулся текущий `reduce_features`, который после добавления `norm_work`
оставил только 1 признак.

Решение по `norm_work`:

- не удалять;
- не считать мусором;
- не смешивать напрямую в общий encoder pool старым способом;
- использовать как guard: детектор не должен массово срабатывать на экспертно
  подтвержденной нормальной работе;
- нормальный среднесрочный вариант - `fixed_feature_schema`: сначала выбрать
  канонический набор признаков на размеченных ALMA train-normal данных, затем
  добавлять `norm_work` уже в этот фиксированный feature space, не позволяя
  `norm_work` заново управлять feature reduction.

## 3. Пропуски и интерполяция

Для этой телеметрии основной корректный способ ресемплинга - zero-order hold,
то есть forward fill: последнее измеренное значение считается актуальным до
следующего изменения.

Почему:

- данные событийные, а не равномерный аналоговый сигнал;
- значение датчика физически держится до следующего записанного изменения;
- линейная, cubic, PCHIP или Kalman-интерполяция могут дорисовать плавный
  переход, которого в реальности не было;
- для детекции аномалий такие дорисованные slope/trend могут создать ложный
  сигнал.

Решение:

- базовый ресемплинг делать через causal forward fill;
- не использовать будущие точки для заполнения прошлого;
- advanced learned imputation типа SAITS/BRITS сейчас не нужна: пропусков мало,
  а природа данных событийная;
- если позже появятся тяжелые пропуски, advanced imputation рассматривать как
  отдельный research-эксперимент, не как default.

## 4. История канала, coverage и глубокий вывод

Проверялся вопрос: не выкидываем ли мы полезный канал только потому, что он
редко пишет значения в сырых данных.

Вывод: по текущим размеченным production-датасетам такой проблемы практически
нет. После ресемплинга `ffill` редкий, но рано появившийся канал получает
полноценную историю на регулярной сетке. Каналы выкидываются в основном потому,
что они реально пустые или имеют слишком низкое покрытие.

Сводка по production parquet:

| Класс | Пар скважина-канал | Взято в модель | Выкинуто | Главная причина |
|---|---:|---:|---:|---|
| `negermet` | 130 | 130 | 0 | проблем нет |
| `pritok` | 486 | 465 | 21 | канал полностью пустой по скважине |
| `salt` | 189 | 182 | 7 | 6 полностью пустые, 1 низкое покрытие |

Не найдено случаев, где канал был бы выкинут именно из-за недостаточной истории
до `actual_start`, хотя плотность сырых записей у каналов разная.

Примеры выкинутых каналов:

| Класс | Скважина | Канал | Причина |
|---|---|---|---|
| `pritok` | многие | `Выходное напряжение ПЧ` | полностью пустой |
| `pritok` | `1395` | `Активная выходная мощность`, `Вибрация ХY`, `Выходное напряжение ПЧ` | полностью пустые |
| `pritok` | `5144г` | `Напряжение в звене постоянного тока ПЧ` | полностью пустой |
| `pritok` | `790` | `Выходной ток ПЧ` | полностью пустой |
| `salt` | `149г`, `2991г`, `3269`, `4039`, `408` | `Выходное напряжение ПЧ` | полностью пустой |
| `salt` | `3245(2)` | `Cos Ф` | полностью пустой |
| `salt` | `3245` | `Напряжение в звене постоянного тока ПЧ` | низкое покрытие |

Запас истории у выбранных каналов:

| Класс | Минимальное покрытие до аномалии | Медианное покрытие до аномалии | Минимум истории до старта |
|---|---:|---:|---:|
| `negermet` | 98.69% | 99.81% | 226 точек |
| `pritok` | 99.83% | 100% | 994 точки |
| `salt` | 99.41% | 100% | 1343 точки |

Решение:

- фильтр достаточной истории и coverage оставить;
- не ослаблять фильтр ради количества каналов;
- добавить диагностическую таблицу доступности каналов: скважина, канал,
  выбран/выкинут, причина, coverage до аномалии;
- sparse raw channel не считать плохим, если он появился рано и после `ffill`
  имеет нормальное покрытие;
- канал без достаточной ранней истории не использовать для обучения, потому что
  модель не сможет честно выучить нормальное поведение этого канала.

## 5. Выбор нормальной зоны

Инженерное уточнение: нормальный период перед `actual_start` нужно брать целиком.
Нет необходимости искусственно сужать pre-anomaly участок, потому что
подозрительная зона уже должна входить в размеченный интервал аномалии.

Решение:

- default reference policy - `normal_windows`;
- для размеченной скважины нормальная зона - весь период `timestamp < actual_start`;
- не вырезать вручную "сомнительный хвост" перед стартом;
- если у скважины несколько интервалов, для обучения брать период до первого
  размеченного старта, если нет отдельной логики по нескольким инцидентам;
- `normal_window_selector` оставить как research-инструмент, а не как замену
  production default без отдельного benchmark.

Отдельное правило для blind/unlabeled скважин:

- если у скважины нет размеченного интервала и нет внешне подтвержденного
  периода нормы, начало её собственного ряда нельзя автоматически считать
  нормальной работой;
- для таких скважин `paano_global` не должен строить локальный reference из
  первых 20-40% ряда;
- текущий безопасный статус для такого случая - `Not assessed` /
  `unlabeled_no_reference`;
- `edge_hold_padded` разрешен только там, где reference уже честно задан:
  размеченной нормой до `actual_start` или будущим population memory bank;
- для полноценного blind-инференса нужен population memory bank: общий банк
  нормальных эмбеддингов, собранный из подтвержденной нормальной работы, а не
  из начала самой проверяемой скважины.

Это правило не отменяет `norm_work`: эти 20 скважин экспертно описаны как
чистая нормальная работа на всем периоде, поэтому они могут использоваться как
нормальный донорский pool/guard. Запрет касается именно неразмеченных
проверяемых скважин, где неизвестно, нормален ли старт ряда.

## 6. Окна feature engineering при общей частоте

Сейчас engineered features используют физические окна, например:

```text
уровни / z / std: 10, 60, 240 минут
slope: 10, 60 минут
```

Если частота у всех классов станет одинаковой, окна всё равно нужно задавать в
физическом времени, а не в "числе точек". Потом они переводятся в steps через
частоту датасета.

Решение:

- окна должны отражать физику аномалии;
- для быстрой негерметичности нужны короткие окна;
- для притока и соли нужны более длинные окна;
- при единой частоте можно держать общий набор окон, например короткие,
  средние и длинные, но оценивать их вклад через feature reduction/tuning;
- не делать разные смыслы одного и того же patch/window из-за разных частот.

## 7. Нормализация

Текущий принцип должен оставаться robust и per-well: модель сравнивает скважину
с её собственной нормой, а не только с абсолютными значениями по всему фонду.

Решение:

- сохранять per-well robust scaling относительно reference;
- отдельно проверять глобальную шкалу/калибровку, чтобы score между скважинами
  был сопоставим;
- не давать near-constant каналам создавать гигантские искусственные z-score;
- для отчетов и экспертных таблиц показывать физические значения каналов, а не
  только нормализованные score.

## 8. Режимные события

Режимные события пока не решаются автоматически до конца. Их должен помочь
разметить эксперт, потому что не всякое изменение параметров является аварией.

Примеры режимных событий:

- изменение выходной частоты;
- остановка/пуск скважины;
- штатное изменение тока после смены частоты;
- нормальная перестройка давления после режима управления.

Решение:

- этот слой оставить на потом;
- не смешивать режимные события с базовой подготовкой данных;
- в будущем делать отдельный `operational_state` / `telemetry_status` слой,
  который объясняет часть стартов как режимные, а не аварийные.

## 9. Калибровка score

Сырые score разных скважин и разных моделей не всегда сопоставимы. Поэтому
после стабилизации preprocessing нужна tail/conformal calibration.

Решение:

- сначала фиксировать чистую схему данных и feature space;
- потом калибровать score относительно reference-tail конкретной скважины;
- использовать калиброванный score для fusion/onset, а сырой score оставлять
  как диагностический;
- не подбирать калибровку на test-интервалах напрямую.

## 10. Patch size

Patch size должен задаваться через физическую длительность, а не как случайное
количество точек.

Текущий смысл:

- `negermet`: короткие патчи, потому что событие быстрое;
- `pritok`: длинные патчи, потому что изменение давления развивается медленно;
- `salt`: длинные патчи, потому что солеотложение похоже на медленный
  многоканальный drift.

Если будет единая частота, patch size нужно пересчитать так, чтобы сохранить
физическую длительность окон. Например, если целимся в 16 часов, то на 10
минутах это 96 точек, а на 5 минутах - уже 192 точки.

Решение:

- patch sizes выбирать по времени;
- для общего encoder не использовать один и тот же numeric patch size, если он
  означает разные часы на разных классах;
- при benchmark общей частоты сравнить несколько физических длительностей, а не
  только одну пару patch sizes.

## Итоговая позиция

Текущая правильная линия:

```text
event-like raw telemetry
-> causal ffill to regular grid
-> drop truly empty / low-coverage channels
-> full pre-anomaly normal reference
-> fixed or carefully controlled feature schema
-> robust per-well normalization
-> PaAno/global normality training
-> calibrated score and incident logic
```

Главные запреты:

- не дорисовывать пустые каналы продвинутой интерполяцией;
- не смешивать разные частоты в один encoder pool без пересчета patch/window;
- не добавлять `norm_work` напрямую так, чтобы он управлял feature reduction;
- не сужать нормальный период перед `actual_start` без фактической причины;
- не считать sparse raw channel плохим, если после `ffill` он имеет нормальную
  историю и покрытие.

## 11. Как собирать общий pool нормальной работы

Общая модель не должна превращать несколько скважин в один искусственный
временной ряд. Склейка делается только после того, как каждая скважина отдельно
приведена к общей частоте, общей схеме колонок и своей нормализации.

Правильная форма данных для одной скважины:

```text
well_i -> матрица T_i x C
```

где:

- `T_i` - количество временных точек конкретной скважины;
- `C` - одинаковый набор каналов/признаков для всех скважин в данном
  эксперименте;
- строки внутри `T_i` остаются временным рядом этой скважины;
- значения разных скважин не смешиваются внутри одной строки.

Общий train pool собирается вертикальной склейкой нормальных окон:

```text
X_train =
  concat(
    well_1_normal_windows,
    well_2_normal_windows,
    ...,
    well_n_normal_windows,
    axis=0
  )
```

Это значит, что если есть приток с `X` скважинами по 26 параметров и негермет с
5 скважинами по 26 параметров, то после приведения к одному `C` мы складываем
нормальные окна этих скважин как разные обучающие примеры. Мы не складываем
давление одной скважины с давлением другой по времени и не строим общий
календарный ряд.

Пример:

```text
pritok_1062:    7000 x 26
pritok_129л:    8900 x 26
negermet_1123л: 1200 x 26
negermet_524:   1800 x 26

pool:
  normal_windows(pritok_1062)
  + normal_windows(pritok_129л)
  + normal_windows(negermet_1123л)
  + normal_windows(negermet_524)
```

Если используется sliding-window/patch обучение, фактическая единица склейки -
не отдельная строка, а окно:

```text
well_i_normal -> patches: N_i x patch_size x C
global_pool -> concat(patches всех скважин, axis=0)
```

То есть склеиваются независимые окна нормальной работы разных скважин. Порядок
времени внутри каждого окна сохраняется.

Что нельзя делать:

- нельзя соединять конец ряда одной скважины с началом ряда другой и считать
  это непрерывным процессом;
- нельзя использовать разные частоты в одном pool без пересчета физической
  длительности patch/window;
- нельзя позволять длинной скважине или большому классу полностью доминировать
  в обучении;
- нельзя динамически менять набор колонок от скважины к скважине внутри одной
  модели.

Для общей модели нужен баланс:

- ограничивать максимальное число окон от одной скважины;
- балансировать вклад классов `negermet`, `pritok`, `salt` и чистого
  `norm_work`;
- длинные и короткие скважины должны давать сопоставимый вклад;
- `norm_work` использовать как дополнительную нормальную работу, но не давать
  ему управлять выбором каналов.

Итоговая логика:

```text
raw wells
-> per-well regular grid
-> per-well interpolation/ffill
-> fixed feature schema
-> per-well normal windows
-> balanced concat of windows
-> global normality encoder
```

Класс аномалии в этом подходе не является target для encoder. Он нужен для
балансировки train pool, оценки результатов и последующей интерпретации.

## Ближайшие практические шаги

1. Исправить stale defaults в full dataset build, чтобы они совпадали с
   production parquet или явно были помечены как research. Статус: сделано.
   Defaults приведены к production-источникам: `negermet=2min`,
   `pritok=10min`, `salt=15min`.
2. Сделать отчет доступности каналов по каждому датасету: selected/dropped,
   причина, coverage, first valid timestamp. Статус: сделано. Скрипт:
   `scripts/datasets/audit_preprocessing_defaults_and_channels.py`, результат:
   `artifacts/analysis/preprocessing_defaults_and_channels.md` и `.json`.
3. Подготовить benchmark единой частоты для `negermet/pritok/salt`. Статус:
   аналитический аудит сделан, обучение не запускалось. Скрипт:
   `scripts/datasets/audit_frequency_candidates.py`, результат:
   `artifacts/analysis/frequency_candidates_audit.md` и `.json`.
4. Реализовать `fixed_feature_schema` для безопасного подключения `norm_work`.
   Статус: сделано для global normality benchmark. Конфиг:
   `configs/alma_global_feature_schema.json`, код:
   `alma_service/feature_schema.py`, подключение:
   `scripts/evaluation/benchmark_global_normality_detector.py`.
5. Реализовать `balanced global normal pool`. Статус: сделано в
   `scripts/evaluation/benchmark_global_normality_detector.py`. По умолчанию
   используется `equal_min`: каждый источник дает одинаковое число reference
   rows, равное минимальному доступному источнику после per-well cap.
6. Пересобрать `negermet/pritok/salt/norm_work` на `5min` и сделать
   schema-only проверку уже на единой частоте. Статус: сделано. Parquet:
   `db/negermet_anomaly_database_5min.parquet`,
   `db/pritok_anomaly_database_5min.parquet`,
   `db/salt_anomaly_database_5min.parquet`,
   `db/norm_work_database_5min.parquet`. Schema-only результат:
   `artifacts/results/global_normality_detector/schema_preview_5min/global_fixed_feature_schema_preview.json`.
7. После этого повторить global normality benchmark и сравнить с текущим
   `paano_shared`. Статус: не запускать до разбора ограничения по `negermet`
   на 5min.

## Прогресс от 2026-05-18

Что проверено:

- `negermet` production source: `db/negermet_anomaly_database_2min.parquet`;
- `pritok` production source: `db/pritok_anomaly_database_10min.parquet`;
- `salt` production source: `db/salt_anomaly_database_15min.parquet`;
- `norm_work` для аудита общего набора каналов: `db/norm_work_database_10min.parquet`.

До исправления defaults были расхождения:

- `build_negermet_dataset.py` по умолчанию собирал `15s`, а production detection
  брал `2min`;
- `run_full_dataset_build.sh` по умолчанию собирал `pritok=2min`, а production
  detection брал `10min`;
- `build_salt_dataset.py` и `run_full_dataset_build.sh` по умолчанию собирали
  `2min`, а production detection брал `15min`.

После исправления:

```text
negermet -> 2min
pritok   -> 10min
salt     -> 15min
```

Отчет доступности каналов показал:

- primary-источники: `negermet` - 6 скважин / 26 каналов, `pritok` - 27
  скважин / 27 каналов, `salt` - 9 скважин / 27 каналов, `norm_work` - 20
  скважин / 26 каналов;
- в строгий общий набор для global normality сейчас проходят 21 канал;
- спорные/отброшенные каналы: `Cos Ф`, `Активная выходная мощность`,
  `Вибрация ХY`, `Выходное напряжение ПЧ`, `Выходной ток ПЧ`,
  `Напряжение в звене постоянного тока ПЧ`;
- причина спорности не в частоте, а в отсутствии канала хотя бы в одной
  скважине/датасете или недостаточном покрытии.

Вывод: общий encoder можно строить на фиксированном core из 21 канала. Если
хотим использовать остальные 6 каналов, их нельзя молча добавлять в общий
`C`: нужен отдельный режим `optional channels` или class-specific/diagnostic
ветки, иначе модель будет зависеть от того, у какой скважины какие параметры
случайно есть.

## Прогресс по частоте от 2026-05-18

Сделан отдельный аудит кандидатов общей частоты по сырым cache-файлам
`db/raw_cache/*/*.parquet`. Это не пересобранные регулярные parquet, а исходные
timestamp каждого параметра до `ffill` и регулярной сетки.

Проверены частоты:

- `2min`;
- `5min`;
- `10min`;
- `15min`.

Факты по сырым интервалам:

```text
negermet  median raw delta = 26s,  p90 = 2.72min, p95 = 15.07min
pritok    median raw delta = 36s,  p90 = 15.12min, p95 = 15.23min
salt      median raw delta = 44s,  p90 = 15.10min, p95 = 15.25min
norm_work median raw delta = 42s,  p90 = 15.12min, p95 = 15.25min
```

Важный вывод: сырые ряды действительно event-like. Часто параметр может
обновляться десятками секунд, но это не значит, что общую модель надо учить на
секундной сетке. Иначе сетка будет огромной, а значительная часть частых
обновлений будет от вибраций/служебных частых изменений, а не от масштаба
аномалии.

Размер общей сетки по всем четырем наборам:

```text
2min  -> 1 489 511 grid points
5min  ->   595 780 grid points
10min ->   297 862 grid points
15min ->   198 556 grid points
```

Рекомендация:

- первый общий benchmark делать на `5min`;
- `10min` оставить как дешевый контрольный baseline;
- `2min` не брать первым общим вариантом, потому что он сильно раздувает
  `salt/pritok/norm_work`;
- `15min` не брать как основной общий вариант, потому что он слишком грубый для
  коротких событий `negermet`.

Почему `5min`: это компромисс. Он не пытается сохранить каждый raw tick, но
оставляет больше временного разрешения для быстрых событий, чем `10min/15min`,
и при этом в 2.5 раза меньше по размеру, чем `2min`.

## Прогресс по fixed feature schema от 2026-05-18

Добавлен фиксированный raw-core schema для общей модели:

```text
configs/alma_global_feature_schema.json
```

В схему включен 21 канал, который прошел аудит доступности во всех primary-
источниках `negermet`, `pritok`, `salt` и `norm_work`.

Кодовая поддержка:

- `alma_service/feature_schema.py` - загрузка schema и фильтрация
  `PreparedWellData`;
- `scripts/evaluation/benchmark_global_normality_detector.py` - новые флаги
  `--feature-schema`, `--schema-only`, `--enable-feature-reduction`;
- `alma_service/shared_encoder.py` - добавлен необязательный параметр
  `enable_reduction`, старое поведение по умолчанию сохранено.

Проверка без обучения:

```bash
uv run python scripts/evaluation/benchmark_global_normality_detector.py \
  --include-norm-work \
  --schema-only \
  --no-retune \
  --output-dir artifacts/results/global_normality_detector/schema_preview
```

Результат:

```text
pool rows        = 210 474
train wells      = 95
shared features  = 126
```

Это означает: 21 raw-канал после engineered feature expansion дает 126 общих
feature columns, и strict schema не выбросила скважины.

Важная найденная проблема: старая `feature reduction` при смешанном global pool
схлопнула 126 features до 1 (`Дисбаланс токов::raw`). Для общего fixed-schema
режима это некорректно, поэтому reduction по умолчанию выключена в global
benchmark и может включаться только явно через `--enable-feature-reduction`.
Для production class-specific `paano_shared` старое поведение не менялось.

## Прогресс по balanced pool от 2026-05-18

Добавлена балансировка train reference pool для общей модели:

- `--disable-balanced-pool` - выключить балансировку;
- `--balance-source-policy equal_min|cap`;
- `--balance-rows-per-source`;
- `--balance-rows-per-well`;
- `--balance-seed`.

Default-режим: `equal_min`.

Смысл `equal_min`: сначала ограничиваем вклад одной скважины, потом смотрим,
сколько reference rows осталось у каждого источника (`negermet`, `pritok`,
`salt`, `norm_work`), и все источники режем до размера самого маленького. Так
модель не получает ситуацию, где `pritok` или `norm_work` в десятки раз
доминируют над `negermet`.

Dry-run без обучения на текущих production-источниках:

```bash
uv run python scripts/evaluation/benchmark_global_normality_detector.py \
  --include-norm-work \
  --schema-only \
  --no-retune \
  --output-dir artifacts/results/global_normality_detector/schema_preview
```

Результат:

```text
shared features = 126
train wells     = 95
pool rows       = 24 532

negermet  = 6 133 rows
pritok    = 6 133 rows
salt      = 6 133 rows
norm_work = 6 133 rows
```

Важное ограничение текущего dry-run: он проверяет механику fixed schema и
balanced pool на production-источниках с разной частотой. Это не финальный
global benchmark. Перед обучением нужно пересобрать все четыре источника на
единую частоту `5min` и повторить schema-only проверку.

## Прогресс по единой частоте 5min от 2026-05-18

Пересобраны регулярные parquet на единой сетке `5min`:

```text
db/negermet_anomaly_database_5min.parquet -> 9 145 rows, 6 wells
db/pritok_anomaly_database_5min.parquet   -> 284 172 rows, 27 wells
db/salt_anomaly_database_5min.parquet     -> 218 964 rows, 9 wells
db/norm_work_database_5min.parquet        -> 83 499 rows, 20 wells
```

Сборка запускалась на сервере без обучения. Лог сохранен в
`artifacts/logs/build_common_5min_*.log`.

Повторена strict schema-only проверка уже на `5min`:

```bash
uv run python scripts/evaluation/benchmark_global_normality_detector.py \
  --common-source-freq 5min \
  --include-norm-work \
  --schema-only \
  --no-retune \
  --output-dir artifacts/results/global_normality_detector/schema_preview_5min
```

Первый результат до исправления подготовки:

```text
shared features = 126
train wells     = 93
pool rows       = 7 904

negermet  = 1 976 rows, 2 train wells
pritok    = 1 976 rows, 25 train wells
salt      = 1 976 rows, 6 train wells
norm_work = 1 976 rows, 60 prepared entries
```

После разбора причины внесены две правки в
`scripts/evaluation/benchmark_global_normality_detector.py`:

- при `--common-source-freq` patch size для подготовки пересчитывается по
  физической длительности production-сетки;
- `norm_work` добавляется один раз через `--norm-work-profile`, а не три раза
  через все anomaly-profile.

Для `5min` получились такие prepare patch overrides:

```text
negermet -> 52   вместо 128 на 2min
pritok   -> 384  вместо 192 на 10min
salt     -> 384  вместо 128 на 15min
```

Повторный schema-only результат:

```text
shared features = 126
train wells     = 55
pool rows       = 9 816

negermet  = 2 454 rows, 4 train wells
pritok    = 2 454 rows, 25 train wells
salt      = 2 454 rows, 6 train wells
norm_work = 2 454 rows, 20 prepared entries
```

Фиксированная схема прошла чисто:

```text
negermet schema skips = 0
pritok schema skips   = 0
salt schema skips     = 0
```

Это значит, что выбранный core из 21 raw-канала совместим с пересобранными
`5min` источниками. После feature expansion он дает те же 126 общих признаков.

После пересчета patch size на `5min` в train reference pool для `negermet`
попали четыре train-скважины:

```text
1123л -> 364 reference rows
172г  -> 1 612 reference rows
524   -> 209 reference rows
5271г -> 269 reference rows
```

Из-за default-балансировки `equal_min` весь общий pool режется до размера
самого маленького источника, поэтому остальные источники тоже ограничены
`2 454` rows. Это ожидаемое поведение: `negermet` остается самым маленьким
источником, но теперь он представлен четырьмя train-скважинами, а не двумя.

`norm_work` теперь учитывается как 20 prepared entries: одна исходная скважина
= один элемент pool. Default profile для подготовки `norm_work` выбран
`pritok`, потому что это нейтральный профиль без salt soft-sensor веток; при
fixed schema soft features всё равно не попадают в общий `C`.

Вывод: пересборка и schema-only на `5min` выполнены, механические ограничения
по patch size и тройному учету `norm_work` исправлены.

## Global benchmark 5min no-retune от 2026-05-18

Проведен первый benchmark общего encoder на `5min` без retune и без генерации
отчетов. Цель была проверить, дает ли общий encoder полезный score сам по себе,
не пряча проблему за подбором onset-конфига.

Общие условия:

```text
common_source_freq = 5min
feature schema     = 21 raw channels -> 126 features
pool               = balanced equal_min
norm_work          = включен один раз, profile=pritok
retune             = false
global_iters       = 200
```

Проверенные patch pairs:

| Patch pair | Негермет | Приток | Соли | Вывод |
|---|---:|---:|---:|---|
| `52/104` | 4/5, p90=0.83h | 23/24, p90=27.01h | 7/8, p90=62.87h | слишком короткий, теряет slow/drift события |
| `96/192` | 5/5, p90=0.72h | 24/24, p90=7.43h | 8/8, p90=14.59h | рабочий компромисс |
| `192/384` | 5/5, p90=0.72h | 24/24, p90=5.33h | 8/8, p90=12.07h | лучший no-retune кандидат |

Выбранный research-кандидат:

```text
patch_short = 192
patch_long  = 384
status      = best_no_retune_candidate
```

Причина выбора: `192/384` не ухудшил быстрый `negermet`, сохранил `100%`
hit-rate на всех трех классах и дал лучший p90 delay на `pritok` и `salt`.

Конфиг кандидата сохранен:

```text
configs/alma_global_normality_5min.json
```

Важно: это не production default. Это кандидат для следующего шага:

```text
global normality 5min 192/384 -> retune -> сравнение с class-specific paano_shared
```

## Global benchmark 5min retune 192/384 от 2026-05-18

Retune для выбранного research-кандидата `192/384` выполнен на сервере в
`tmux`-сессии `alma_global5min_retune_192_384`.

Команда:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluation/benchmark_global_normality_detector.py \
  --common-source-freq 5min \
  --include-norm-work \
  --patch-short 192 \
  --patch-long 384 \
  --global-iters 200 \
  --output-dir artifacts/results/global_normality_detector/5min_retune_patch192_384
```

Факт по времени: обучение shared encoder заняло секунды на A100, а основное
время занял CPU-bound retune onset-логики. Полный прогон завершился примерно за
10-11 минут.

Финальный summary:

| Variant | Аномалия | Интервалы | Hit-rate | FAR/day | Старты | Median delay | P90 delay |
|---|---|---:|---:|---:|---:|---:|---:|
| `global_normality` | `negermet` | 5 | 5/5 = 1.0 | 0.0 | 5 | 0.05h | 0.72h |
| `global_normality` | `pritok` | 24 | 24/24 = 1.0 | 0.0 | 45 | 1.60h | 5.33h |
| `global_normality` | `salt` | 8 | 8/8 = 1.0 | 0.0 | 14 | 1.00h | 12.07h |

Сравнение с no-retune для того же `192/384`:

| Аномалия | No-retune старты | Retune старты | Изменение задержки |
|---|---:|---:|---|
| `negermet` | 5 | 5 | без изменений |
| `pritok` | 54 | 45 | p90 delay тот же, лишних стартов меньше |
| `salt` | 16 | 14 | p90 delay тот же, лишних стартов меньше |

Сравнение с сохраненным class-specific baseline в этом же benchmark:

| Аномалия | Class-specific saved | Global normality retuned | Вывод |
|---|---|---|---|
| `negermet` | 4/5, p90=0.70h | 5/5, p90=0.72h | global нашел пропущенный interval, задержка почти та же |
| `pritok` | 24/24, p90=8.83h | 24/24, p90=5.33h | global лучше по p90 delay и дает меньше стартов |
| `salt` | 8/8, p90=78.55h | 8/8, p90=12.07h | global сильно лучше по поздним salt-срабатываниям |

Файлы результата:

```text
artifacts/results/global_normality_detector/5min_retune_patch192_384/global_normality_benchmark_summary.csv
artifacts/results/global_normality_detector/5min_retune_patch192_384/global_normality_benchmark.json
artifacts/results/global_normality_detector/5min_retune_patch192_384/global_normality_negermet_intervals.csv
artifacts/results/global_normality_detector/5min_retune_patch192_384/global_normality_pritok_intervals.csv
artifacts/results/global_normality_detector/5min_retune_patch192_384/global_normality_salt_intervals.csv
```

Конфиг кандидата обновлен:

```text
configs/alma_global_normality_5min.json
status = best_retuned_candidate
```

Важно: даже после хорошего aggregate summary это всё ещё research-кандидат, а
не production default. Следующий обязательный шаг - проверить interval-level
старты и сравнить по скважинам с текущим production `paano_shared`, чтобы
исключить ситуацию "метрика хорошая, но отдельные старты физически странные".

### Interval-level review

Сформирован отдельный файл сравнения с текущими production parquet:

```text
artifacts/results/global_normality_detector/5min_retune_patch192_384/global_vs_production_interval_review.csv
artifacts/results/global_normality_detector/5min_retune_patch192_384/global_vs_production_interval_review.md
```

Сравнение по интервалам:

| Аномалия | Production hits | Global hits | Production starts | Global starts | Production p90 delay | Global p90 delay |
|---|---:|---:|---:|---:|---:|---:|
| `negermet` | 4/5 | 5/5 | 4 | 5 | 0.70h | 0.72h |
| `pritok` | 24/24 | 24/24 | 53 | 45 | 8.83h | 5.33h |
| `salt` | 8/8 | 8/8 | 22 | 14 | 78.55h | 12.07h |

Слабые места global-кандидата:

- `salt/3245(2)`: global delay `35.75h`. Это худший случай, хотя production на
  этом же интервале еще хуже.
- `pritok/5144г`: global delay `19.85h`. Это худший приточный случай; global
  здесь хуже production примерно на `15.17h`, поэтому этот интервал нужно
  отдельно проверить по физическим графикам.
- `pritok` всё еще имеет дубли внутри некоторых длинных интервалов, хотя после
  retune их стало меньше.

Положительные факты:

- `negermet` стал 5/5 вместо production 4/5, без дублей.
- `pritok` сохранил 24/24 и уменьшил p90 delay.
- `salt` сохранил 8/8 и резко уменьшил p90 delay относительно текущего
  production baseline.

Следующая проверка перед production-решением: открыть худшие интервалы
`salt/3245(2)` и `pritok/5144г` на графиках/физических каналах. Для
`salt/3245(2)` нужно понять, почему оба подхода поздние; для `pritok/5144г` -
почему global существенно позже текущего production.

### Диагностика слабых интервалов

Собран focused diagnostic package без нового обучения:

```text
artifacts/analysis/global_normality_5min_diagnostics/index.html
artifacts/analysis/global_normality_5min_diagnostics/pritok_5144g.html
artifacts/analysis/global_normality_5min_diagnostics/salt_3245_2.html
artifacts/analysis/global_normality_5min_diagnostics/diagnostic_notes.md
artifacts/analysis/global_normality_5min_diagnostics/diagnostic_parameter_values_all.csv
```

HTML сделан офлайн-совместимым: локальный Plotly лежит в
`artifacts/analysis/global_normality_5min_diagnostics/assets/plotly.min.js`.

Предварительный вывод по `pritok/5144г`:

- global позже production примерно на `15.17h`;
- физические изменения около стартов слабые: давление относительно медианы
  нормы меняется примерно на `-0.5%`, мощность/загрузка примерно на `+1.1%`,
  токи почти не меняются;
- это похоже не на потерю очевидного сигнала, а на тонкий/слабовыраженный
  интервал, где production срабатывает раньше, но физическое подтверждение
  нужно проверять по графику.

Предварительный вывод по `salt/3245(2)`:

- global delay `35.75h`, но production на этом же интервале еще позже;
- физический сигнал есть: давление ниже нормы примерно на `-16%`, мощность
  выше нормы примерно на `+5%`, токи/загрузка/температура масла выше нормы
  примерно на `+2-3%`;
- этот кейс подходит как targeted salt/onset case, если будем доводить global
  normality до production.

Следующий шаг: открыть оба HTML и подтвердить выводы визуально. Если графики
подтверждают эту картину, не запускать новый общий benchmark, а делать точечную
настройку против этих двух случаев.

### Реализация доменной диагностики без влияния на score

Добавлен первый безопасный слой доменных правил: не как новый детектор и не как
поправка к итоговому score, а как диагностические признаки в interval-level
выводе global benchmark.

Новый модуль:

```text
alma_service/domain_rule_diagnostics.py
```

Он считает вокруг `actual_start` локальные pre/post признаки:

- `pressure_pre_slope_per_day` - наклон давления до старта;
- `pressure_post_slope_per_day` - наклон давления после старта;
- `pressure_slope_change_per_day` - изменение наклона давления;
- `pressure_post_vs_pre_median_pct` - локальное изменение медианы давления;
- `pressure_pre_slope_direction` / `pressure_post_slope_direction` - направление
  локального тренда;
- `trend_reversal_score` - сила смены направления тренда;
- `frequency_post_vs_pre_median_pct` - локальное изменение выходной частоты;
- `frequency_stability_score` - диагностическая оценка стабильности частоты;
- `domain_rule_status`, `domain_rule_pre_points`, `domain_rule_post_points` -
  техническое качество расчета.

Почему это сделано именно так:

- для солей важна не грубая проверка "давление после выше старой медианы", а
  локальная смена поведения; пример `salt/3245(2)`: до `actual_start` давление
  снижалось, после старта начало восстанавливаться/расти;
- для притока важен тренд давления при стабильной частоте, причем направление
  давления может быть вверх или вниз;
- для негермета важен короткий резкий step-сигнал, поэтому окна диагностики
  короче, чем для солей/притока.

Подключение:

```text
scripts/evaluation/benchmark_global_normality_detector.py
```

Теперь будущие файлы:

```text
global_normality_negermet_intervals.csv
global_normality_pritok_intervals.csv
global_normality_salt_intervals.csv
global_normality_benchmark.json
```

будут содержать эти диагностические колонки. Текущие production-скрипты
`detect_negermet.py`, `detect_pritok.py`, `detect_salt.py` и итоговый скор
`paano_shared` не изменены.

Проверки:

```text
uv --directory /root/projects/alma_servie run python -m py_compile \
  alma_service/domain_rule_diagnostics.py \
  scripts/evaluation/benchmark_global_normality_detector.py \
  tests/test_domain_rule_diagnostics.py

uv --directory /root/projects/alma_servie run python -c "... synthetic tests ..."
```

Синтетический тест подтверждает:

- локальная смена давления `down -> up` дает положительный
  `pressure_slope_change_per_day`;
- `trend_reversal_score` становится высоким;
- стабильная частота дает `frequency_stability_score = 100`;
- при отсутствии канала давления возвращается статус
  `missing_pressure_channel`, а не исключение.

### Повторный global benchmark с доменной диагностикой

Запуск выполнен на сервере в `tmux`:

```text
CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluation/benchmark_global_normality_detector.py \
  --output-dir artifacts/results/global_normality_detector/5min_retune_patch192_384 \
  --patch-short 192 \
  --patch-long 384 \
  --common-source-freq 5min \
  --include-norm-work \
  --global-iters 200
```

Факты запуска:

- использована GPU1: `cuda (NVIDIA A100-SXM4-40GB)`;
- pool: `9816` reference points, `126` feature channels, `55` wells;
- patch `192` обучился примерно за `5.1s`;
- patch `384` обучился примерно за `13s`;
- основное время занял CPU-heavy retune/scoring.

Обновленные артефакты:

```text
artifacts/results/global_normality_detector/5min_retune_patch192_384/global_normality_benchmark.json
artifacts/results/global_normality_detector/5min_retune_patch192_384/global_normality_benchmark_summary.csv
artifacts/results/global_normality_detector/5min_retune_patch192_384/global_normality_negermet_intervals.csv
artifacts/results/global_normality_detector/5min_retune_patch192_384/global_normality_pritok_intervals.csv
artifacts/results/global_normality_detector/5min_retune_patch192_384/global_normality_salt_intervals.csv
```

Итоговые метрики не ухудшились:

| Variant | Anomaly | Hit-rate | FAR/day | Starts | Median delay | P90 delay |
|---|---|---:|---:|---:|---:|---:|
| class_specific_saved | negermet | 4/5 | 0.0 | 4 | 0.00h | 0.70h |
| class_specific_saved | pritok | 24/24 | 0.0 | 53 | 1.43h | 8.83h |
| class_specific_saved | salt | 8/8 | 0.0 | 22 | 1.44h | 78.55h |
| global_normality | negermet | 5/5 | 0.0 | 5 | 0.05h | 0.72h |
| global_normality | pritok | 24/24 | 0.0 | 45 | 1.60h | 5.33h |
| global_normality | salt | 8/8 | 0.0 | 14 | 1.00h | 12.07h |

Проверено, что во всех трех interval CSV появились диагностические поля:

```text
domain_rule_status
pressure_pre_slope_per_day
pressure_post_slope_per_day
pressure_slope_change_per_day
pressure_post_vs_pre_median_pct
pressure_pre_slope_direction
pressure_post_slope_direction
trend_reversal_score
frequency_post_vs_pre_median_pct
frequency_stability_score
```

Быстрые наблюдения по диагностике:

- `negermet`: у всех 5 интервалов давление после старта идет вверх; это
  согласуется с экспертным правилом резкого pressure-step.
- `pritok`: направление давления смешанное (`up` и `down`), при этом частота
  почти везде стабильна; это согласуется с экспертным правилом "давление
  трендово вверх/вниз при стабильной частоте".
- `salt`: у проблемного `3245(2)` диагностика показывает `down -> up` по
  локальному тренду давления: это совпадает с ручной проверкой, что перед
  `actual_start` давление падало, а после старта начало разворачиваться.

Следующий аккуратный шаг: не менять score, а сделать review-файл по
диагностике доменных правил:

```text
global_domain_rule_interval_review.csv
global_domain_rule_interval_review.md
```

В этом review нужно сгруппировать интервалы по аномалиям и явно показать, где
доменные признаки подтверждают/не подтверждают экспертное описание. Только
после этого можно обсуждать optional score-branch для global candidate.

Review-файлы созданы:

```text
artifacts/results/global_normality_detector/5min_retune_patch192_384/global_domain_rule_interval_review.csv
artifacts/results/global_normality_detector/5min_retune_patch192_384/global_domain_rule_interval_review.md
```

Итог review по текущим простым правилам:

| Аномалия | Интервалы | Детектировано | Доменные признаки совпали |
|---|---:|---:|---:|
| `negermet` | 5 | 5 | 5 |
| `pritok` | 24 | 24 | 24 |
| `salt` | 8 | 8 | 8 |

Это не означает, что доменные правила уже можно добавлять в score. Это означает
только, что диагностические признаки на размеченных интервалах согласуются с
экспертной физикой и пригодны для следующего этапа: отдельной offline-проверки
как optional branch с train-only tuning.

### Offline ablation optional domain branch

Проверен полный 7-шаговый сценарий optional branch:

1. Собран кандидат `global_normality + domain_rule_branch`.
2. Для score-ветки использованы только causal physical branches без
   `actual_start` leakage:
   - `negermet_signature`;
   - `pressure_trend`;
   - `salt_deposition`.
3. Вес ветки подбирался train-only через общий tuning.
4. Метрики сравнивались против `global_core`.
5. Отдельно проверены худшие интервалы `salt/3245(2)` и `pritok/5144г`.
6. Условие переноса в detector pipeline: улучшение без потери hit-rate и без
   роста FAR.
7. Если улучшения нет - ветка остается только diagnostics/offline.

Важно: interval diagnostics (`pressure_pre_slope`, `pressure_post_slope`,
`trend_reversal_score` вокруг `actual_start`) не использовались как score,
потому что это было бы подглядывание в разметку. Они остаются explainability
слоем.

Запуск:

```text
CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluation/benchmark_global_domain_branch_ablation.py \
  --output-dir artifacts/results/global_normality_detector/5min_domain_branch_ablation \
  --patch-short 192 \
  --patch-long 384 \
  --common-source-freq 5min \
  --include-norm-work \
  --global-iters 200
```

Артефакты:

```text
artifacts/results/global_normality_detector/5min_domain_branch_ablation/global_domain_branch_ablation.json
artifacts/results/global_normality_detector/5min_domain_branch_ablation/global_domain_branch_ablation_summary.csv
artifacts/results/global_normality_detector/5min_domain_branch_ablation/global_domain_branch_ablation.md
artifacts/results/global_normality_detector/5min_domain_branch_ablation/global_core_*_intervals.csv
artifacts/results/global_normality_detector/5min_domain_branch_ablation/global_domain_branch_*_intervals.csv
```

Результат:

| Variant | Anomaly | Effective branch weight | Hit-rate | FAR/day | Starts | Median delay | P90 delay |
|---|---|---:|---:|---:|---:|---:|---:|
| `global_core` | `negermet` | 0.0 | 5/5 | 0.0 | 5 | 0.05h | 0.72h |
| `global_domain_branch` | `negermet` | 0.0 | 5/5 | 0.0 | 5 | 0.05h | 0.72h |
| `global_core` | `pritok` | 0.0 | 24/24 | 0.0 | 45 | 1.60h | 5.33h |
| `global_domain_branch` | `pritok` | 0.0 | 24/24 | 0.0 | 145 | 1.60h | 5.33h |
| `global_core` | `salt` | 0.0 | 8/8 | 0.0 | 14 | 1.00h | 12.07h |
| `global_domain_branch` | `salt` | 0.0 | 8/8 | 0.0 | 13 | 1.00h | 12.07h |

Вывод:

- train-only tuning выбрал effective branch weight `0.0` для всех трех
  классов;
- `salt/3245(2)` не улучшился: задержка осталась `35.75h`;
- `pritok/5144г` не улучшился: задержка осталась `19.85h`;
- переносить optional domain branch в detector pipeline сейчас нельзя;
- доменная диагностика остается полезной для review/explainability, но не как
  часть score.

Следующий технический вывод: если хотим улучшать худшие кейсы, это нужно делать
не добавлением текущих physical branch weights, а отдельно разбирать onset
policy/thresholding для `5144г` и targeted salt onset для `3245(2)`.

---

# Приложение A. Экспертные правила по аномалиям

Источник до объединения: `docs/ALMA_domain_rules_by_anomaly.md`.

# ALMA: экспертные правила по аномалиям и проверка на данных

Дата фиксации: 2026-05-19.

Документ фиксирует доменные правила по трем типам аномалий ALMA:

- `negermet` - негерметичность НКТ;
- `pritok` - изменение притока;
- `salt` - солеотложение.

Основа документа:

- диалог Руслана с инженером-экспертом Эльдаром;
- проверка правил на текущих размеченных ALMA-данных, пересобранных на единой сетке `5min`;
- результаты global normality benchmark `5min 192/384`.

Проверка выполнена не через простое сравнение с медианой нормы, а через локальные тренды:

```text
pre-window before actual_start
post-window after actual_start
pressure_pre_slope
pressure_post_slope
pressure_slope_change = post_slope - pre_slope
post/pre median delta
```

Это важно: эксперт при просмотре графиков оценивает не только уровень относительно всей прошлой нормы, а форму ряда до и после `actual_start`.

Файл интервалов проверки:

```text
artifacts/analysis/domain_rule_trend_check/domain_rule_trend_check_intervals.csv
```

## Короткий вывод

| Аномалия | Проверенное правило | Статус |
|---|---|---|
| `negermet` | резкий скачок давления вверх, температура/токи как поддержка | подтверждено частично, pressure-up подтвержден сильно |
| `pritok` | тренд давления вверх или вниз при стабильной частоте и без сильного изменения остальных каналов | подтверждено хорошо |
| `salt` | после старта давление получает положительный slope или перелом тренда вверх; важна группа каналов | подтверждено, но не как `pressure > old median` |

Главное внедренческое правило после проверки ablation: эти знания нельзя превращать ни в жесткие `if/else` по сырым значениям, ни в простую добавку к anomaly score. Их нужно использовать как диагностический слой, guards, объяснения и следующий `domain decision layer` поверх уже найденных событий.

## Общие параметры

Базовый набор физических каналов:

- `Давление на приеме насоса кгс/см²`;
- `Выходная частота`;
- `Выходной ток ПЧ`;
- `Ток на фазе А`;
- `Ток на фазе В`;
- `Ток на фазе С`;
- `Коэффициент загрузки ПЭД`;
- `Полная выходная мощность`;
- `Активная выходная мощность`;
- `Температура на приёме насоса`;
- `Температура масла двигателя`;
- `Дисбаланс токов`;
- `Дисбаланс напряжений`;
- вибрации и напряжения как дополнительные диагностические группы.

`Давление на приеме насоса кгс/см²` - главный общий канал, но его смысл разный для разных аномалий:

| Аномалия | Как читать давление |
|---|---|
| `negermet` | короткий резкий скачок вверх |
| `pritok` | тренд вверх или вниз при стабильной частоте |
| `salt` | положительный post-slope или перелом тренда вверх, часто вместе с групповой деградацией режима |

## Негерметичность НКТ

### Экспертная формулировка

По словам эксперта, негермет легче всего определяется: моментально растет давление на приеме и температура; также могут начаться колебания тока.

### Проверка на данных

Проверено 5 размеченных интервалов.

Факты:

```text
pressure_post_slope: вверх 5/5
pressure_slope_change: вверх 5/5
pressure post/pre median delta: от +2% до +130%, медиана +52.8%
frequency median abs delta: 0% в большинстве случаев
```

По интервалам:

| Скважина | Pressure post/pre median | Комментарий |
|---|---:|---|
| `1123л` | +52.8% | сильный pressure-up |
| `172г` | +2.1% | давление слабее, зато токи растут примерно +22%, частота +5.8% |
| `3509г` | +6.8% | pressure-up умеренный |
| `524` | +130.1% | очень сильный pressure-up |
| `5271г` | +65.6% | сильный pressure-up, температура масла +17.4% |

Температура и токи подтверждают не все случаи одинаково:

- температура сильно растет у `1123л` и `5271г`, слабее у остальных;
- токи сильно растут у `172г` и `524`;
- у `1123л` и `5271г` токи могут снижаться.

### Решение для внедрения

Для `negermet` главный доменный признак для диагностики и `domain decision layer`:

```text
pressure_step_up_score
```

Поддерживающие компоненты:

```text
thermal_support_score
load_current_support_score
frequency_context_score
```

Правильно:

- усиливать срабатывание при резком росте давления;
- использовать температуру и токи как поддержку;
- хранить направление давления в диагностике.

Неправильно:

- требовать обязательный рост температуры во всех случаях;
- требовать обязательный рост токов во всех случаях;
- использовать только pressure level без оценки резкого step-change.

## Приток

### Экспертная формулировка

По словам эксперта, основной триггер притока - давление на приеме. Приток отличается тем, что давление может трендово идти как вверх, так и вниз. Частота используется как контрольный параметр: если давление меняется при неизменной частоте, а остальные параметры в норме, это сильный признак изменения притока.

### Проверка на данных

Проверено 24 размеченных интервала.

Факты:

```text
pressure_post_slope: вверх 11/24, вниз 13/24
pressure post/pre median delta: вверх 11/24, вниз 13/24
frequency median abs delta: 0%
```

Это хорошо подтверждает экспертную формулировку: для притока важен не знак давления, а сам устойчивый тренд давления при стабильной частоте.

Примеры давления после старта:

| Скважина | Pressure post/pre median | Частота | Интерпретация |
|---|---:|---:|---|
| `1062` | +5.87% | 0% | приток через рост давления |
| `1508` | +5.17% | 0% | приток через рост давления |
| `3261` | +3.96% | 0% | приток через рост давления |
| `3245` | -7.76% | 0% | приток через падение давления |
| `3027` | -4.38% | 0% | приток через падение давления |
| `790` | -4.25% | 0% | приток через падение давления |
| `5144г` | -1.04% | 0% | слабый/тонкий случай |

### Решение для внедрения

Для `pritok` главный доменный признак для диагностики и `domain decision layer`:

```text
pressure_abs_trend_score
```

Обязательный контекст:

```text
frequency_stability_guard
```

Дополнительная логика:

```text
other_channels_stability_score
```

Правильно:

- учитывать и рост, и падение давления;
- усиливать сигнал, если частота стабильна;
- ослаблять сигнал, если изменение давления объясняется изменением частоты;
- не требовать изменения токов/загрузки/температуры.

Неправильно:

- искать только рост давления;
- считать любое изменение давления аномалией без проверки частоты;
- классифицировать приток как деградацию насоса.

## Солеотложение

### Экспертная формулировка

По словам эксперта, соли - это плохая работа насоса. Давление на приеме участвует в сигнале, но в отличие от притока меняются также токи и другие параметры. То есть соли нужно читать как многоканальную деградацию режима, а не как одиночный канал.

### Важное уточнение после проверки

Исходную формулировку `при солях давление всегда растет` нельзя внедрять как `давление выше медианы нормы`.

Причина: эксперт смотрит график как форму ряда до и после `actual_start`, а не простое сравнение с общей медианой прошлого периода.

Ключевой пример - `3245(2)`:

| Дата | Давление |
|---|---:|
| 2025-09-16 | 38.92 |
| 2025-09-23 | 34.37 |
| 2025-09-30 | 31.59 |
| 2025-10-04 `actual_start` | 31.15 |
| 2025-10-05 | 30.26 |
| 2025-10-07 | 30.60 |
| 2025-10-11 | 31.83 |

С 16 сентября до старта давление падало примерно на 20%. После старта уровень еще ниже старой медианы, но локальный тренд меняется: после минимума начинается восстановление/рост.

### Проверка на данных

Проверено 8 размеченных интервалов.

Факты:

```text
pressure_post_slope: вверх 8/8
pressure_slope_change: вверх 8/8
pressure post/pre median delta: вверх 7/8, вниз 1/8
frequency median abs delta: 0% в большинстве случаев
```

По интервалам:

| Скважина | Pre slope | Post slope | Slope change | Pressure post/pre median | Комментарий |
|---|---:|---:|---:|---:|---|
| `149г` | -0.002 | +1.147 | +1.149 | +1.53% | явный post-slope вверх |
| `2991г` | +0.040 | +0.121 | +0.082 | +0.67% | слабый рост |
| `3244г` | -0.029 | +0.047 | +0.076 | +0.96% | есть режимный фактор: частота +4.63% |
| `3245` | -0.088 | +0.411 | +0.499 | +4.66% | сильный перелом вверх |
| `3245(2)` | -0.273 | +0.189 | +0.463 | -4.15% | уровень ниже старой нормы, но тренд переломился вверх |
| `3269` | +0.070 | +0.210 | +0.140 | +1.53% | рост ускорился |
| `4039` | +0.003 | +0.280 | +0.278 | +0.60% | рост ускорился |
| `408` | +0.035 | +0.195 | +0.160 | +1.84% | рост ускорился |

Групповые каналы ток/загрузка/мощность/температура подтверждают событие не одинаково во всех скважинах. Поэтому они должны быть support/residual-группой, а не жестким шаблоном.

### Решение для внедрения

Для `salt` главный pressure-признак для диагностики и `domain decision layer`:

```text
pressure_post_slope_positive_score
pressure_slope_change_up_score
trend_reversal_score
```

Групповая поддержка:

```text
load_current_drift_score
power_drift_score
thermal_drift_score
electrical_imbalance_score
multivariate_residual_score
ks_distribution_shift_score
```

Правильно:

- искать положительный post-slope давления;
- искать перелом давления вверх относительно pre-trend;
- не требовать, чтобы давление было выше всей прошлой нормы;
- использовать токи/нагрузку/мощность/температуру как многоканальное подтверждение;
- учитывать frequency guard, если частота менялась.

Неправильно:

- писать hard-rule `pressure > reference_median`;
- писать hard-rule `pressure must increase from actual_start immediately`;
- считать соли одноканальным pressure detector.

## Как это соотносится с текущим кодом

Текущие physical branches уже близки к правильной форме:

| Модуль | Текущее поведение | Оценка |
|---|---|---|
| `alma_service/pressure_trend.py` | считает абсолютное изменение давления и хранит направление | хорошо для притока |
| `alma_service/negermet_signature.py` | считает step-change давления/нагрузки и хранит direction | нужно усилить pressure-up prior, но не делать hard-only |
| `alma_service/salt_trend.py` | grouped drift + residual + conformal tail + distribution shift | правильная архитектура для солей |

Что стоит добавить дальше:

- `pressure_pre_slope`;
- `pressure_post_slope`;
- `pressure_slope_change`;
- `trend_reversal_score` для солей;
- `frequency_stability_guard` для притока и солей;
- диагностический вывод направлений в отчетах.

## Результат проверки additive domain branch

После первичной доменной проверки был отдельно протестирован вариант:

```text
final_score = PaAno/global score + tuned_weight * domain_physics_score
```

Вес физической ветки подбирался автоматически только на train-интервалах. Результат важный: tuning выбрал эффективный вес `0.0` для всех трех классов. То есть в текущей реализации физическая ветка полезна для объяснения, но не улучшает итоговый score как числовая прибавка.

Факты ablation:

| Класс | Что произошло при добавке физики в score | Вывод |
|---|---|---|
| `negermet` | метрики не улучшились, потому что core уже ловит интервалы хорошо | физика подтверждает событие, но не нужна как score-boost |
| `pritok` | задержка не улучшилась, число стартов выросло `45 -> 145` | простая добавка физики ухудшает чистоту событий |
| `salt` | худший кейс `3245(2)` не улучшился, delay остался `35.75h` | текущая ветка не решает поздний salt onset |

Поэтому текущий вывод такой:

```text
domain rules != additive score branch
domain rules = explainability + guards + post-detection decision layer
```

Это не означает, что экспертные правила бесполезны. Наоборот, они нужны, но в другой роли.

## Политика внедрения

Экспертные правила нужно внедрять как слой принятия решения после первичной детекции:

```text
1. PaAno/global модель находит подозрительные события.
2. Domain decision layer анализирует физику вокруг найденного события.
3. Слой принимает решение:
   - принять событие;
   - отклонить как остановку или режимное изменение;
   - уточнить тип: приток / соли / негермет;
   - сформировать объяснение для эксперта.
```

Роли доменных правил:

- `classification`: отличать тип аномалии после того, как событие найдено;
- `guard`: отбрасывать остановки, частотные переходы и штатные режимные изменения;
- `explainability`: показывать эксперту, какие параметры подтвердили событие;
- `diagnostics`: находить слабые места onset policy по конкретным кейсам.

Нельзя:

- добавлять физику в production score без доказанного выигрыша;
- подгонять правила под test-интервалы вручную;
- зашивать отдельные скважины;
- делать hard-coded thresholds по конкретным значениям давления;
- удалять важные каналы глобально из-за одного артефакта.

Можно:

- использовать экспертные правила как post-detection decision rules;
- добавлять диагностические компоненты в отчеты;
- обучать/подбирать параметры decision layer только на train-интервалах;
- добавлять guards для режимных событий и артефактов;
- отдельно тестировать domain score branches offline, не включая их в production score до доказанного выигрыша.

## Следующий практический шаг

Следующий правильный шаг - не продолжать прибавлять физику к score, а сделать offline `domain decision layer` поверх найденных событий.

План:

1. Сохранять все найденные старты/инциденты global core, а не только interval-level результат.
2. Для каждого старта считать локальные физические признаки вокруг `detected_time`, без использования `actual_start`.
3. Применить экспертные правила как post-detection verdict:
   - `accepted`;
   - `rejected_stop`;
   - `rejected_frequency_transition`;
   - `pritok_candidate`;
   - `salt_candidate`;
   - `negermet_candidate`;
   - `uncertain`.
4. Проверить на размеченных `negermet/pritok/salt`, не трогая Salym/test35.
5. Отдельно разобрать слабые кейсы:
   - `pritok/5144г`: onset/threshold policy;
   - `salt/3245(2)`: поздний старт и salt-specific onset.

Для runtime score текущая политика:

```text
production score = PaAno/global score
domain rules = post-detection decision/explanation layer
```

---

# Приложение B. Статистика длины сырых рядов

Источник до объединения: `docs/ALMA_raw_series_length_stats_2026-05-18.md`.

# ALMA: статистика сырых рядов по скважинам

Дата фиксации: 2026-05-18.

Статистика посчитана по `db/raw_cache`, то есть до пересборки на регулярную сетку, до `ffill`, до интерполяции и до feature engineering. Это именно исходные event-like ряды параметров из Excel-cache.

В таблице `Сырые точки всего` - это сумма исходных записей по всем параметрам скважины. Так как каждый канал пишет свою временную шкалу, дополнительно показаны `min/median/max` точек на один параметр и число уникальных timestamp по скважине.

## Сводка по источникам

| Источник | Скважин | Параметров, медиана | Сырых точек, сумма | Сырых точек на скважину, медиана | Длительность, медиана дней |
|---|---:|---:|---:|---:|---:|
| Негерметичность НКТ (`negermet`) | 6 | 26 | 715945 | 26002 | 2.00 |
| Нормальная работа (`norm_work`) | 20 | 26 | 2554284 | 71960 | 14.00 |
| Изменение притока (`pritok`) | 27 | 26 | 9854273 | 228010 | 32.00 |
| Солеотложение (`salt`) | 9 | 26 | 8784297 | 793397 | 84.00 |

## Сводка по скважинам

### Негерметичность НКТ (`negermet`)

| Скважина | Параметров | Сырые точки всего | Уникальных timestamp | Начало | Конец | Дней | min/median/max точек на параметр |
|---|---:|---:|---:|---|---|---:|---:|
| 1123л | 26 | 26013 | 11562 | 2025-06-20 02:00:11 | 2025-06-22 01:59:58 | 2.00 | 191 / 307 / 4630 |
| 172г | 26 | 392459 | 89834 | 2025-04-27 21:00:01 | 2025-05-06 20:59:49 | 9.00 | 861 / 12200 / 45785 |
| 3509г | 26 | 8379 | 3503 | 2025-08-10 02:00:52 | 2025-08-11 01:59:51 | 1.00 | 95 / 104 / 1564 |
| 524 | 26 | 25992 | 5674 | 2025-06-09 02:00:04 | 2025-06-10 20:31:53 | 1.77 | 168 / 314 / 3839 |
| 5271г | 26 | 17070 | 6547 | 2025-08-14 02:00:15 | 2025-08-16 01:58:52 | 2.00 | 189 / 210 / 2383 |
| ю-я 39-651 | 26 | 246032 | 46458 | 2026-03-01 00:00:26 | 2026-03-16 23:59:41 | 16.00 | 1513 / 1735 / 44797 |

### Изменение притока (`pritok`)

| Скважина | Параметров | Сырые точки всего | Уникальных timestamp | Начало | Конец | Дней | min/median/max точек на параметр |
|---|---:|---:|---:|---|---|---:|---:|
| 1062 | 26 | 374649 | 106625 | 2025-12-07 00:00:12 | 2026-01-26 23:59:26 | 51.00 | 4791 / 4836 / 61318 |
| 122 | 26 | 295661 | 161045 | 2026-02-20 02:00:12 | 2026-03-31 01:59:50 | 39.00 | 3709 / 3799 / 139263 |
| 129л | 26 | 826155 | 265910 | 2025-12-01 02:00:17 | 2026-02-01 01:59:51 | 62.00 | 5878 / 7844 / 138988 |
| 1395 | 24 | 146325 | 28806 | 2026-01-14 19:00:13 | 2026-02-28 18:59:12 | 45.00 | 4271 / 4271 / 12580 |
| 1442л | 26 | 203754 | 32272 | 2026-01-19 21:03:24 | 2026-03-16 20:56:35 | 56.00 | 5294 / 5294 / 15403 |
| 1449 | 26 | 206067 | 34460 | 2026-01-19 21:00:38 | 2026-03-16 20:59:19 | 56.00 | 5322 / 5322 / 15662 |
| 1508 | 26 | 29036 | 4525 | 2025-11-11 21:00:28 | 2025-11-19 20:57:36 | 8.00 | 755 / 755 / 2191 |
| 1756 | 26 | 90793 | 30732 | 2025-11-04 00:00:00 | 2025-11-19 23:59:38 | 16.00 | 1501 / 1502 / 23016 |
| 1809 | 26 | 228010 | 103044 | 2025-12-25 00:00:17 | 2026-01-24 23:59:47 | 31.00 | 2945 / 3050 / 66317 |
| 1995 | 26 | 1064027 | 126212 | 2026-03-14 00:01:07 | 2026-04-30 18:59:47 | 47.79 | 4415 / 21073 / 109390 |
| 1996л | 26 | 422066 | 72961 | 2026-03-16 19:01:13 | 2026-04-17 18:59:54 | 32.00 | 2461 / 10830 / 46968 |
| 3027 | 26 | 194014 | 47692 | 2026-02-14 00:00:17 | 2026-03-13 23:57:07 | 28.00 | 2630 / 3090 / 30443 |
| 305г | 26 | 39717 | 8131 | 2026-04-08 00:04:00 | 2026-04-18 23:59:50 | 11.00 | 1035 / 1035 / 2990 |
| 3245 | 26 | 337623 | 81670 | 2026-02-14 00:00:19 | 2026-03-13 23:59:51 | 28.00 | 2657 / 3154 / 60261 |
| 3261 | 26 | 400186 | 134252 | 2025-10-01 02:00:00 | 2025-11-05 01:59:53 | 35.00 | 3320 / 3566 / 89367 |
| 4203у | 26 | 114517 | 25594 | 2025-12-12 21:01:08 | 2026-01-12 20:59:17 | 31.00 | 2953 / 2954 / 8727 |
| 495 | 26 | 92809 | 21643 | 2025-09-21 21:03:08 | 2025-10-16 20:57:59 | 25.00 | 2388 / 2389 / 7100 |
| 5021 | 26 | 55604 | 9524 | 2026-03-01 21:00:39 | 2026-03-16 20:56:20 | 15.00 | 1430 / 1430 / 4253 |
| 5144г | 25 | 936592 | 557335 | 2025-12-07 00:00:07 | 2026-02-06 23:59:59 | 62.00 | 5938 / 6063 / 322824 |
| 602 | 26 | 753838 | 235797 | 2025-12-13 02:02:46 | 2026-03-13 01:59:59 | 90.00 | 8502 / 8636 / 137406 |
| 610 | 26 | 231060 | 113967 | 2026-02-18 02:00:00 | 2026-03-23 01:59:57 | 33.00 | 3059 / 3495 / 61803 |
| 691 | 26 | 973541 | 246301 | 2025-11-26 00:00:08 | 2026-01-06 23:59:53 | 42.00 | 3999 / 4048 / 179660 |
| 713 | 26 | 434436 | 96464 | 2025-12-21 00:00:03 | 2026-01-16 23:59:47 | 27.00 | 2559 / 2578 / 83417 |
| 790 | 26 | 100127 | 32491 | 2025-12-18 00:00:01 | 2026-01-16 23:59:14 | 30.00 | 2788 / 2840 / 13790 |
| 792 | 26 | 1001894 | 252154 | 2025-12-06 00:00:08 | 2026-01-16 23:59:55 | 42.00 | 4006 / 4256 / 201273 |
| 816 | 26 | 87978 | 21249 | 2026-01-01 00:00:20 | 2026-01-13 23:59:16 | 13.00 | 1209 / 1229 / 15583 |
| 902 | 26 | 213794 | 48047 | 2025-10-15 02:01:04 | 2025-11-15 01:59:32 | 31.00 | 2898 / 3012 / 34144 |

### Солеотложение (`salt`)

| Скважина | Параметров | Сырые точки всего | Уникальных timestamp | Начало | Конец | Дней | min/median/max точек на параметр |
|---|---:|---:|---:|---|---|---:|---:|
| 149г | 26 | 172770 | 46788 | 2025-06-01 00:01:10 | 2025-06-28 01:59:29 | 27.08 | 2526 / 2816 / 28133 |
| 1740 | 26 | 793397 | 160665 | 2026-01-01 02:00:57 | 2026-03-26 01:59:45 | 84.00 | 7814 / 11623 / 99104 |
| 2991г | 26 | 178441 | 35155 | 2025-11-27 02:00:35 | 2025-12-24 01:59:23 | 27.00 | 2499 / 2516 / 26903 |
| 3244г | 26 | 724710 | 83555 | 2025-11-01 02:01:00 | 2026-01-01 01:59:15 | 61.00 | 5609 / 12702 / 71905 |
| 3245 | 27 | 873789 | 335140 | 2025-02-01 00:00:20 | 2025-05-12 01:59:53 | 100.08 | 9459 / 9647 / 260647 |
| 3245(2) | 26 | 826849 | 357937 | 2025-08-01 00:08:59 | 2025-11-07 01:59:56 | 98.08 | 9100 / 9617 / 275430 |
| 3269 | 26 | 134969 | 60099 | 2025-10-01 00:13:09 | 2025-10-29 01:59:04 | 28.07 | 2629 / 2808 / 14828 |
| 4039 | 26 | 3256251 | 701241 | 2025-03-31 21:00:08 | 2025-09-30 20:59:50 | 183.00 | 17345 / 41770 / 567535 |
| 408 | 26 | 1823121 | 620281 | 2025-03-31 21:00:08 | 2025-08-30 20:59:39 | 152.00 | 14441 / 18075 / 365819 |

### Нормальная работа (`norm_work`)

| Скважина | Параметров | Сырые точки всего | Уникальных timestamp | Начало | Конец | Дней | min/median/max точек на параметр |
|---|---:|---:|---:|---|---|---:|---:|
| 1151 | 26 | 43605 | 5927 | 2026-04-25 02:03:48 | 2026-05-07 01:59:30 | 12.00 | 1127 / 1127 / 3318 |
| 1213л | 26 | 51182 | 7444 | 2026-04-13 02:00:45 | 2026-04-27 01:56:27 | 14.00 | 1324 / 1324 / 3892 |
| 1214у | 26 | 29248 | 5212 | 2026-04-19 02:02:05 | 2026-04-27 01:56:30 | 8.00 | 755 / 755 / 2224 |
| 1247 | 26 | 72967 | 9957 | 2026-04-17 02:01:09 | 2026-05-07 01:59:58 | 20.00 | 1888 / 1888 / 5549 |
| 126 | 26 | 36464 | 4927 | 2026-04-17 02:03:01 | 2026-04-27 01:58:52 | 10.00 | 944 / 944 / 2770 |
| 129л | 26 | 87897 | 12016 | 2026-04-13 02:03:55 | 2026-05-07 01:55:35 | 23.99 | 2270 / 2270 / 6699 |
| 1396у | 26 | 958324 | 189931 | 2026-03-31 21:00:19 | 2026-04-30 20:59:53 | 30.00 | 2695 / 24936 / 111411 |
| 144 | 26 | 29146 | 4823 | 2026-04-15 02:03:05 | 2026-04-23 01:55:48 | 7.99 | 755 / 755 / 2210 |
| 220 | 26 | 50497 | 6637 | 2026-04-25 02:03:29 | 2026-05-09 01:54:51 | 13.99 | 1310 / 1310 / 3827 |
| 220(2) | 26 | 72124 | 9535 | 2026-04-17 02:00:38 | 2026-05-07 01:59:20 | 20.00 | 1872 / 1872 / 5463 |
| 254 | 26 | 71795 | 19209 | 2026-04-15 02:02:15 | 2026-05-05 01:59:30 | 20.00 | 1864 / 1864 / 5437 |
| 2661 | 26 | 168999 | 42647 | 2026-04-13 02:00:17 | 2026-04-21 01:59:50 | 8.00 | 762 / 1040 / 29474 |
| 4649 | 26 | 74788 | 23354 | 2026-04-13 02:00:33 | 2026-04-27 01:58:16 | 14.00 | 1240 / 1244 / 13020 |
| 512 | 26 | 74566 | 23213 | 2026-04-13 02:00:12 | 2026-04-27 01:59:17 | 14.00 | 1240 / 1246 / 12769 |
| 5302 | 26 | 21689 | 4350 | 2026-04-15 02:00:55 | 2026-04-21 01:55:17 | 6.00 | 563 / 564 / 1639 |
| 5467 | 26 | 346644 | 127363 | 2026-04-15 02:00:04 | 2026-05-09 01:59:57 | 24.00 | 2280 / 2346 / 71515 |
| 610 | 26 | 63469 | 36567 | 2026-04-13 02:00:04 | 2026-04-21 01:59:46 | 8.00 | 761 / 788 / 23342 |
| 660 | 26 | 97473 | 33564 | 2026-04-27 02:00:06 | 2026-05-07 01:59:54 | 10.00 | 937 / 1104 / 21826 |
| 662 | 26 | 152822 | 32702 | 2026-04-13 02:00:14 | 2026-04-25 01:59:51 | 12.00 | 1127 / 1340 / 27538 |
| 8016 | 26 | 50585 | 9097 | 2026-04-25 02:00:03 | 2026-05-09 01:58:10 | 14.00 | 1315 / 1315 / 3839 |

## Самые короткие сырые ряды

| Источник | Скважина | Начало | Конец | Дней | Сырые точки всего | min/median/max на параметр |
|---|---|---|---|---:|---:|---:|
| Негерметичность НКТ | 3509г | 2025-08-10 02:00:52 | 2025-08-11 01:59:51 | 1.00 | 8379 | 95 / 104 / 1564 |
| Негерметичность НКТ | 524 | 2025-06-09 02:00:04 | 2025-06-10 20:31:53 | 1.77 | 25992 | 168 / 314 / 3839 |
| Негерметичность НКТ | 5271г | 2025-08-14 02:00:15 | 2025-08-16 01:58:52 | 2.00 | 17070 | 189 / 210 / 2383 |
| Негерметичность НКТ | 1123л | 2025-06-20 02:00:11 | 2025-06-22 01:59:58 | 2.00 | 26013 | 191 / 307 / 4630 |
| Нормальная работа | 5302 | 2026-04-15 02:00:55 | 2026-04-21 01:55:17 | 6.00 | 21689 | 563 / 564 / 1639 |
| Нормальная работа | 144 | 2026-04-15 02:03:05 | 2026-04-23 01:55:48 | 7.99 | 29146 | 755 / 755 / 2210 |
| Нормальная работа | 1214у | 2026-04-19 02:02:05 | 2026-04-27 01:56:30 | 8.00 | 29248 | 755 / 755 / 2224 |
| Изменение притока | 1508 | 2025-11-11 21:00:28 | 2025-11-19 20:57:36 | 8.00 | 29036 | 755 / 755 / 2191 |
| Нормальная работа | 2661 | 2026-04-13 02:00:17 | 2026-04-21 01:59:50 | 8.00 | 168999 | 762 / 1040 / 29474 |
| Нормальная работа | 610 | 2026-04-13 02:00:04 | 2026-04-21 01:59:46 | 8.00 | 63469 | 761 / 788 / 23342 |
| Негерметичность НКТ | 172г | 2025-04-27 21:00:01 | 2025-05-06 20:59:49 | 9.00 | 392459 | 861 / 12200 / 45785 |
| Нормальная работа | 126 | 2026-04-17 02:03:01 | 2026-04-27 01:58:52 | 10.00 | 36464 | 944 / 944 / 2770 |
| Нормальная работа | 660 | 2026-04-27 02:00:06 | 2026-05-07 01:59:54 | 10.00 | 97473 | 937 / 1104 / 21826 |
| Изменение притока | 305г | 2026-04-08 00:04:00 | 2026-04-18 23:59:50 | 11.00 | 39717 | 1035 / 1035 / 2990 |
| Нормальная работа | 1151 | 2026-04-25 02:03:48 | 2026-05-07 01:59:30 | 12.00 | 43605 | 1127 / 1127 / 3318 |

## Вывод

- Сырые ряды разных каналов внутри одной скважины имеют разное число точек; поэтому для обучения нельзя ориентироваться только на `raw_rows_total`.
- Для выбора общей частоты и patch size важнее календарная длительность ряда и длительность нормального pre-anomaly участка после приведения к сетке.
- `negermet` содержит самые короткие календарные ряды; поэтому при переходе на `5min` нельзя переносить numeric patch size из `2min` один-в-один. Patch size нужно пересчитывать по физической длительности.
- `norm_work` имеет полноценные сырые ряды по всем 20 скважинам и может использоваться как нормальный guard/pool, но в global-модели его нужно добавлять один раз на скважину, а не через три anomaly-profile.

## Машинно-читаемые артефакты

- `artifacts/analysis/raw_series_length_stats_wells.csv` - сводка по скважинам.
- `artifacts/analysis/raw_series_length_stats_parameters.csv` - детализация по каждому параметру.

---

## Прогресс 2026-05-19: domain decision layer

Принятое решение: экспертные правила не добавляются в `production score` как
`PaAno + weight * physics_score`. После ablation такая схема не дала выигрыша:
для `pritok` выросло число стартов, для `salt/3245(2)` задержка не улучшилась,
для `negermet` core уже достаточно сильный.

Что реализуется вместо этого:

```text
PaAno/global score -> найденные старты -> domain decision layer -> verdict/explanation
```

Роль `domain decision layer`:

- считать физические признаки вокруг `detected_time`, а не вокруг
  `actual_start`;
- не подглядывать в разметку при формировании признаков события;
- отличать доменные кандидаты `pritok/salt/negermet`;
- отклонять очевидные bad-data/regime/frequency-transition случаи;
- сохранять объяснение для review, не меняя core score.

Добавляемые артефакты global benchmark:

```text
global_normality_<anomaly>_starts.csv
global_normality_<anomaly>_incidents.csv
```

`*_starts.csv` должен содержать все найденные старты с полями:

- `domain_verdict`;
- `domain_action`;
- `domain_reason`;
- локальные pressure/frequency/load/temperature/imbalance признаки.

`*_incidents.csv` должен агрегировать verdict по incident-level событиям.

Текущая роль `norm_work`:

- в class-specific production CLI (`detect_negermet.py`, `detect_pritok.py`,
  `detect_salt.py`) `norm_work` не является частью обучения по умолчанию;
- в global-normality benchmark `norm_work` подключается флагом
  `--include-norm-work`;
- при global-подходе `norm_work` используется как дополнительные нормальные
  скважины в train/reference pool;
- после исправления подготовки `norm_work` добавляется один раз на скважину, а
  не как три копии под три anomaly-profile;
- последняя рабочая схема: общий pool `negermet + pritok + salt + norm_work`,
  fixed schema, `5min`, balanced reference pool.

Запуск ускоренного прогона:

```text
CUDA_VISIBLE_DEVICES=1 \
ALMA_RETUNE_MODE=fast \
ALMA_OPTUNA_N_JOBS=16 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
uv run python scripts/evaluation/benchmark_global_normality_detector.py \
  --output-dir artifacts/results/global_normality_detector/5min_domain_decision_layer \
  --patch-short 192 \
  --patch-long 384 \
  --common-source-freq 5min \
  --include-norm-work \
  --global-iters 200
```

Причина ускорения:

- PaAno training/scoring остается на GPU `CUDA_VISIBLE_DEVICES=1`;
- `retune` остается CPU-bound, поэтому включен `ALMA_OPTUNA_N_JOBS=16`;
- BLAS/NumExpr потоки ограничены до 1, чтобы параллельные trial не создавали
  oversubscription на CPU;
- режим `fast`, потому что `quality` в текущем коде принудительно ставит
  `n_jobs=1`.

Результат ускоренного прогона:

```text
artifacts/results/global_normality_detector/5min_domain_decision_layer/
```

Итоговые метрики global core:

| Anomaly | Hit-rate | FAR/day | Starts | Median delay | P90 delay |
|---|---:|---:|---:|---:|---:|
| `negermet` | 5/5 = 1.0 | 0.0 | 5 | 0.05h | 0.72h |
| `pritok` | 24/24 = 1.0 | 0.0 | 54 | 1.60h | 5.33h |
| `salt` | 8/8 = 1.0 | 0.0 | 14 | 1.00h | 12.07h |

По сравнению с сохраненным class-specific baseline:

- `negermet`: global нашел 5/5 против 4/5;
- `pritok`: hit-rate тот же 24/24, p90 delay лучше `8.83h -> 5.33h`;
- `salt`: hit-rate тот же 8/8, p90 delay сильно лучше `78.55h -> 12.07h`.

Первичная проверка `domain decision layer`:

| Anomaly | Starts rows | Accepted | Rejected | Uncertain | Основной вывод |
|---|---:|---:|---:|---:|---|
| `negermet` | 6 | 0 | 5 | 1 | слой слишком агрессивно доверяет `bad_data/regime_event` в момент старта |
| `pritok` | 63 | 23 | 40 | 0 | часть стартов корректно принята как stable-frequency pressure trend, но много raw starts помечено bad/regime |
| `salt` | 22 | 7 | 10 | 5 | есть salt candidates, но слабые pressure patterns требуют доработки |

Вывод: global core сейчас выглядит сильным, а `domain decision layer` пока не
готов как фильтр. Его текущая версия годится для диагностики/review, но не
должна менять метрики и не должна подавлять production alerts.

Что нужно поправить в следующей итерации:

- не отклонять старт только потому, что `telemetry_status` в точке старта
  пометил `bad_data/regime_event`;
- использовать `bad_data/regime_event` как контекстный признак окна, а не как
  hard reject;
- делать hard reject только для устойчивых остановок/частотных переходов,
  подтвержденных окном до/после старта;
- отдельно проверить `negermet`, потому что резкий реальный скачок может быть
  похож на regime event по текущей эвристике.

### Файлы, которые нужно сохранить и развивать

Эти untracked Python-файлы относятся к текущему направлению и не являются
мусором:

- `alma_service/domain_decision_layer.py` - post-detection verdict по найденным
  стартам: pressure/frequency/load/temperature/imbalance признаки вокруг
  `detected_time`;
- `alma_service/domain_rule_diagnostics.py` - interval-level диагностика вокруг
  `actual_start` для проверки экспертных правил на размеченных данных;
- `tests/test_domain_decision_layer.py` - unit-тесты логики verdict;
- `tests/test_domain_rule_diagnostics.py` - unit-тесты диагностики тренда и
  missing pressure-channel cases.

Текущая связь с benchmark:

- `scripts/evaluation/benchmark_global_normality_detector.py` уже импортирует
  `attach_domain_decisions_to_starts`,
  `attach_domain_decisions_to_incidents` и
  `attach_domain_rule_diagnostics`;
- поэтому эти файлы нужно либо коммитить вместе с изменением benchmark, либо
  откатывать benchmark. Удалять их отдельно нельзя, иначе benchmark станет
  неработоспособным.

Правильное развитие:

1. Оставить слой как offline/review layer, не как production score.
2. Убрать hard reject по одиночному `telemetry_status` в одной точке старта.
3. Считать режимные события по окну до/после старта: устойчивость частоты,
   остановка, восстановление, длительность переходного режима.
4. Для `pritok` подтверждать stable-frequency pressure trend.
5. Для `salt` подтверждать local trend reversal / pressure-up pattern и
   многоканальную поддержку токов/нагрузки.
6. Для `negermet` подтверждать быстрый pressure/temperature step и реакцию
   токов, но не отклонять реальный скачок только потому, что он похож на
   regime event.
7. Добавить в выходные CSV/HTML понятные поля для эксперта: `verdict`,
   `action`, `reason`, ключевые изменения давления, частоты, нагрузки,
   температуры.

До отдельной валидации этот слой не должен подавлять alerts. Его задача -
объяснить и разметить кандидаты для review.

### Прогресс 2026-05-20: первый `negermet` domain guard

Выполнена первая правка `domain decision layer` для `negermet`.

Проблема предыдущей версии: слой делал hard reject, если точка старта была
помечена как `bad_data` или `regime_event`. Для ННКТ это оказалось неверно:
реальный резкий скачок давления может выглядеть как режимное/качество-событие
по простой эвристике telemetry status. Из-за этого реальные интервалы
`1123л`, `172г`, `3509г`, `524`, `5271г` в diagnostic layer уходили в reject,
хотя global core их нашел.

Новое правило:

```text
bad_data/regime_event в точке старта = контекстный флаг, не hard reject

для negermet принимаем candidate, если есть:
  pressure step up
  и либо сильный рост давления, либо поддержка температуры/тока/нагрузки

если pressure/temperature/load response нет:
  uncertain/review, не уверенный ННКТ
```

Проверка выполнена без переобучения: сохраненные старты последнего
`global_normality` benchmark были пропущены через новый classifier.

Результат по `negermet`:

| Скважина | Было | Стало | Причина |
|---|---|---|---|
| `1123л` | `rejected_bad_data` | `negermet_candidate / accept` | сильный pressure step + temperature/load support |
| `172г` | `rejected_regime_event` | `negermet_candidate / accept` | pressure step + temperature/load support |
| `3509г` | `rejected_bad_data` | `negermet_candidate / accept` | pressure step + load support |
| `524` | `rejected_regime_event` | `negermet_candidate / accept` | сильный pressure step + load/imbalance support |
| `5271г` | `rejected_regime_event` | `negermet_candidate / accept` | сильный pressure step + temperature/load support |
| `ю-я 39-651` | `uncertain` | `uncertain` | нет pressure/temperature/load response |

Итог: первый guard-шаг для ННКТ работает в нужную сторону. Он не подавляет
реальные резкие ННКТ из-за одиночного bad/regime-флага и одновременно не
принимает blind-кандидат `ю-я 39-651`, где экспертная физика ННКТ отсутствует.

Ограничение: это пока диагностический слой. Он не меняет raw PaAno score и не
должен менять production alerts до отдельной валидации.

Следующий шаг: аналогично доработать `pritok` guard на hard-negative нормальных
frequency-transition случаях `305г`, `1996л`, `1995`, потому что текущий
простоватый `pritok` verdict слишком легко принимает слабые pressure trends.

### Уточнение 2026-05-20: комментарии эксперта из сводной таблицы

Источник данных не расширяется из сводной таблицы. Для обучения, оценки и
benchmark используются только фактические файлы, которые есть в `data/` и уже
собраны в `db/*.parquet`. Если в сводной таблице есть скважина или категория,
но соответствующего файла нет в `data/`, она не становится новой меткой и не
попадает в расчет.

Комментарии эксперта из сводной таблицы используются только как доменная
логика для `domain decision layer`: объяснение, guard, классификация кандидата
и разбор ложных срабатываний. Это не дополнительные labels.

Что нужно обязательно учесть в правилах:

1. Нормальное изменение частоты не является аномалией само по себе.
   Если после увеличения частоты давление на приеме немного снижается, а токи
   растут, это ожидаемая режимная реакция. Если после снижения частоты давление
   немного растет, а токовые нагрузки снижаются, это тоже ожидаемая режимная
   реакция. Такие события должны попадать в `rejected_frequency_transition` или
   `review`, а не в уверенную аномалию.

2. Для `pritok` основной сигнал - тренд давления на приеме вверх или вниз при
   стабильной частоте. Если изменение давления хорошо объясняется изменением
   частоты, это слабый кандидат на приток. Если частота стабильна, остальные
   каналы не показывают деградации насоса, а давление устойчиво меняет тренд,
   это сильный `pritok_candidate`.

3. Для `salt` важно не простое превышение давления над старой медианой, а
   форма процесса: рост давления, ускорение роста или разворот локального
   тренда вверх после периода снижения. Отдельный важный экспертный паттерн:
   если частота трендово увеличивается, но давление на приеме не снижается, а
   продолжает расти, это признак ухудшения работы ГНО и сильный кандидат на
   соли.

4. Для `negermet` ключевой паттерн - быстрый скачок давления вверх с
   поддержкой температуры и/или токов. Температура и токи не должны быть
   жесткими обязательными условиями для всех скважин, но отсутствие реакции
   давления, температуры и нагрузки одновременно делает candidate слабым и
   должно отправлять событие в `review/suppressed`.

5. Остановки скважины и восстановление после остановки нельзя считать
   аномалией без отдельного подтверждения физики события. Для `pritok/610` в
   сводной отдельно зафиксирована остановка, которую нужно игнорировать как
   аномальный старт.

6. `Нормальная работа` из `data/raw/norm_work` - это чистая нормальная работа
   за весь период. Она полезна как normal pool/guard, но не должна
   дублироваться по трем anomaly-profile.

   Отдельное уточнение: `305г`, `1996л`, `1995` сейчас физически лежат в
   `data/raw/pritok`, потому что исторически добавлялись рядом с притоком, но
   по экспертной разметке это не приток и не аномалия. Это чистое нормальное
   поведение с изменением частоты за весь период выгрузки. Комментарий эксперта
   нужен именно потому, что визуально такие случаи можно спутать с аномалией:
   при изменении частоты давление и токи меняются ожидаемо. В `db/pritok_intervals`
   для этих скважин нет anomaly-intervals, и это правильно. Для `domain decision
   layer` они должны использоваться как hard-negative normal/frequency-transition
   cases. Для global-normality обучения/калибровки их нужно использовать как
   `norm_work`-подобные normal entries за весь период, но с отдельной
   диагностической пометкой `frequency_transition_normal`, чтобы не смешивать
   их с "ровной" нормой при анализе ошибок.

7. Категории вроде `Нехватка напора`, `Ухудшение работы ГНО`, `Забитый штуцер`
   и прочие единичные типы не смешиваются с `negermet`, `pritok`, `salt`, если
   под них нет фактических файлов и утвержденной схемы разметки. Для текущего
   пайплайна это будущие `other/review` классы, а не training labels.

Практический контракт для следующей итерации:

```text
PaAno/global candidate
  -> проверить остановку и частотный переход
  -> проверить pressure/frequency/load/temperature pattern
  -> выдать verdict:
       accepted
       rejected_frequency_transition
       rejected_stop
       negermet_candidate
       pritok_candidate
       salt_candidate
       review
```

Этот слой не меняет raw PaAno score. Он объясняет и фильтрует кандидаты после
детекции, чтобы эксперт видел, почему событие принято, отклонено или отправлено
на ручной разбор.

## 13. Короткие ряды, нулевой score и правильный fallback для global PaAno

Дата фиксации вывода: 2026-05-19.

Контекст: после добавления `paano_global` выяснилось, что часть скважин в
отчете имела нулевое `Отклонение от нормы`, но раньше при этом могла получать
детекцию. Это не означает, что скважина идеально нормальная. Это означает, что
конкретная конфигурация PaAno не смогла корректно посчитать score для этого
ряда.

### Подтвержденные факты из кода

- `paano_global` сейчас глобален по encoder, но не полностью глобален по
  reference. Encoder обучается на общем normal pool, а memory bank для scoring
  строится из локального reference конкретной скважины.
- В `global_normality.py` используется единая частота `5min`, fixed schema,
  balanced pool и patch scale `192/384`.
- В `SharedPaAnoDetector.score_stream` есть ограничение: если длина ряда или
  локального reference меньше `patch_long * 2`, метод возвращает нулевые score
  и `detail.reason = "not_enough_points"`.
- Для `patch_long=384` это означает практический минимум `768` точек, то есть
  около `64` часов на сетке `5min`.
- После исправления `score_valid=False` такие нулевые score больше не должны
  превращаться в старты детекции.

Текущий результат после guard:

- `pritok` и `salt` на текущем global scale `192/384` оцениваются нормально;
- `negermet` частично не оценивается из-за коротких рядов и/или короткого
  локального reference;
- это не провал global-подхода, а несовпадение масштаба окна с длительностью
  данных по коротким негерметам.

### Что говорит оригинальный PaAno

По оригинальной статье PaAno и реализации авторов:

- PaAno обучается на нормальных patch-окнах training time-series;
- во время inference score считается сравнением patch embeddings с memory bank
  нормальных patch embeddings;
- memory bank строится из нормальных patch-окон training/reference участка;
- качество score зависит не только от encoder, но и от того, есть ли достаточно
  нормальных patch-окон для memory bank.

Вывод для ALMA: одного глобального encoder недостаточно, если локальный
reference слишком короткий. Нужно отдельно решать, какой reference использовать
для memory bank и как честно сообщать, что ряд не был оценен.

Источники:

- PaAno arXiv/OpenReview: https://arxiv.org/html/2602.01359v2,
  https://openreview.net/forum?id=NXThkM7Iym
- PaAno GitHub: https://github.com/jinnnju/PaAno

### Почему старое поведение было ошибкой

Старое поведение:

```text
not_enough_points -> score = 0 -> threshold тоже около 0 -> onset logic может
считать score валидным -> появляются старты
```

Это технически неверно. Нулевой score в таком случае не является измерением
нормальности. Это sentinel-значение для "score не рассчитан".

Правильный контракт:

```text
score_valid = True  -> score можно использовать для детекции
score_valid = False -> score нельзя использовать для детекции
```

Если `score_valid=False`, результат должен быть не `Not found`, а отдельный
статус `Not assessed` / `Не оценено`.

### Корректировка решения после экспертной проверки 2026-05-20

Первичная идея `long -> medium -> short` признана рискованной как основной путь.
Причина: разные `patch_size` меняют физический масштаб анализа, частотную
характеристику encoder и распределение anomaly score. Это создает риск
`score shattering`: одна и та же нормальная динамика может выглядеть по-разному
на `long` и `short` окнах.

Особенно опасный сценарий:

```text
ряд физически стабилен
-> локальной истории не хватает для long window
-> система переключается на short window
-> short encoder видит только локальный кусок суточной/режимной динамики
-> score скачет из-за смены масштаба, а не из-за изменения состояния скважины
```

Поэтому каскад разных scale не является первым production-планом. Он остается
только резервным research-вариантом для быстрых `negermet`-событий, если более
консервативный invariant-scale подход не сработает.

Новый основной принцип:

```text
сначала сохранить один физический масштаб анализа,
потом честно обработать короткую историю,
и только потом думать о short-scale fallback.
```

Целевая архитектура первого этапа:

```text
global_long encoder
  -> local reference если достаточно истории
  -> иначе edge/hold padded long-window contract
  -> population memory bank если локальный memory bank слабый
  -> calibrated score под конкретный input contract
  -> onset logic
  -> domain decision layer / explanation
```

То есть основной путь теперь не `multi-scale cascade`, а `invariant-scale
global_long` с аккуратным fallback по reference/memory bank.

### Почему не reflective padding

`Reflective padding` выглядит математически аккуратно, но физически опасен для
нефтяной телеметрии.

Пример риска:

```text
в начале доступного ряда давление снижалось
reflective padding дорисовывает в прошлом рост давления
получается искусственный V-образный разворот тренда
PaAno может воспринять этот излом как abnormal local pattern
```

Для наших рядов это особенно плохо, потому что телеметрия часто event-like:
значение держится до изменения, а не является гладким лабораторным сигналом.

Более безопасный первый вариант: `Edge/Hold Padding`, то есть ZOH-экстраполяция
первого валидного значения назад во времени.

Физический смысл:

```text
до начала наблюдений скважина находилась в стабильном установившемся режиме
```

Это не доказывает, что padding всегда будет корректен, но это более
консервативный prior, чем зеркальное рисование обратного тренда.

### Почему не attention masking и не global pooling сейчас

`Temporal Attention Masking` и `Global Temporal Pooling` теоретически подходят
для variable-length time series, но для текущего PaAno это уже изменение
архитектуры.

Текущая реализация PaAno в ALMA:

```text
CNN/patch encoder -> patch embeddings -> local/population memory bank -> distance score
```

Поэтому:

- attention masking не добавляется напрямую, потому что текущий encoder не
  Transformer с attention mask;
- masked convolution потребует отдельной архитектурной разработки;
- adaptive/global pooling изменит embedding contract и может сломать
  сопоставимость с уже обученным memory bank;
- это не короткий фикс, а отдельный research-проект.

Решение: не раздувать архитектуру на этом этапе. Сначала проверить более
локальные изменения: честный статус, edge/hold padding, population memory bank,
раздельная калибровка.

### Что делать с memory bank

Нужно разделить два уровня:

- encoder training: общий normal pool;
- memory bank / reference for scoring: источник нормальных patch-окон, с
  которым сравнивается конкретная скважина.

Подтвержденный факт: в текущем `paano_global` encoder уже глобальный, но memory
bank при scoring остается локальным, потому что строится из
`prepared.reference_mask` конкретной скважины.

Это хорошо для длинных рядов: локальный memory bank учитывает индивидуальный
режим скважины. Но для коротких `negermet`-рядов это слабое место: reference
может быть слишком коротким, вырожденным или вообще непригодным для long patch.

Приоритет reference/memory bank:

1. Локальный reference конкретной скважины, если он достаточно длинный и
   качественный.
2. Population/global memory bank из balanced normal pool, если локальный
   reference короткий.
3. `Не оценено`, если не хватает данных даже для короткого масштаба или слишком
   слабое покрытие каналов.

Для production это значит: global model должен уметь работать в двух режимах:

```text
shared encoder + local memory bank
shared encoder + population normal memory bank
```

Population memory bank не должен заменять локальный режим всегда. Он нужен как
fallback для коротких рядов и cold-start случаев.

### Калибровка score

Калибровка нужна не только для разных `patch_size`. Даже при одном `global_long`
scale разные способы подачи входа дают разные распределения score:

- реальные long windows;
- edge/hold padded long windows;
- local memory bank;
- population memory bank.

Поэтому нельзя смешивать все raw distance в одну таблицу порогов.

Минимальные input contracts для калибровки:

```text
real_long_local_memory
padded_long_local_memory
real_long_population_memory
padded_long_population_memory
```

На первом этапе можно начать проще:

```text
real_long_windows
padded_long_windows
```

Но в любом случае score должен быть интерпретируемым как tail score /
p-value-like score внутри своего контракта, а не как сырое расстояние в
embedding space.

Это совпадает с современной практикой anomaly detection: score должен быть
интерпретируемым и калиброванным, особенно при переносе между рядами и
масштабами.

Источники по calibration / FPR control:

- Adaptive conformal anomaly detection:
  https://arxiv.org/html/2604.20122v1
- CADES, conformal anomaly detection with FPR control:
  https://proceedings.mlr.press/v267/zhang25dn.html

### Роль экспертных правил

Экспертные правила не нужно добавлять как `score = PaAno + physics_score`.
Проверка уже показала, что такая добавка может не улучшать метрики и создавать
лишние старты.

Правильная роль правил:

- объяснить событие;
- отфильтровать остановки и частотные переходы;
- классифицировать тип события после того, как core model нашла кандидат;
- помочь отличить `pritok`, `salt`, `negermet`.

Для `negermet` короткая физическая ветка может быть fallback-кандидатом, но
только с отдельной валидацией. Она должна ловить резкий pressure/temperature
step и реакцию токов, а не заменять PaAno score без проверки.

### Что нужно сделать следующим

1. Ввести полноценный result status:
   `Detected`, `Not detected`, `Not assessed`.
2. В onset/evaluation добавить coverage-aware метрики:
   `assessed_interval_count`, `not_assessed_interval_count`,
   `coverage_rate`, `hit_rate_on_assessed`.
3. Зафиксировать отказ от cascade scale как первого решения.
4. Реализовать и проверить `global_long + edge/hold padding` на коротких
   `negermet`-рядах.
5. Добавить population/global memory bank fallback для короткого локального
   reference.
6. Разделить калибровку минимум на `real_long_windows` и
   `padded_long_windows`.
7. Прогнать benchmark только на размеченных `negermet`, `pritok`, `salt` и
   `norm_work`; `Salym` и `test35` в этот этап не входят.
8. Рассматривать `global_short` только как резервный validated fallback для
   быстрых `negermet`, если invariant-scale подход не закроет короткие ряды.
9. Не делать `paano_global` default, пока новый подход не пройдет сравнение с
   текущим `paano_shared` по coverage-aware метрикам.

Критерий готовности:

```text
global invariant-scale подход считается production-кандидатом только если:
- нет silent zero-score;
- все invalid случаи явно видны как "Не оценено";
- coverage достаточно высокий;
- FAR/day не хуже текущего baseline;
- hit-rate на assessed intervals не хуже baseline;
- задержки по salt/pritok не деградируют;
- negermet короткие ряды либо оцениваются через hold padding/population memory,
  либо честно помечаются как "Не оценено";
- padded-window контракты калиброваны отдельно от real-window контрактов.
```

### Итоговое решение

Самое зрелое решение для текущей ситуации:

```text
не скрывать невозможность оценки,
не считать нулевой score нормой,
не переключать scale без необходимости,
сохранить global_long как основной физический масштаб,
для короткой истории проверить edge/hold padding,
для короткого reference добавить population memory bank,
калибровать padded и real контракты отдельно.
```

Это сохраняет сильную сторону global model и одновременно убирает главный риск:
ложную уверенность на коротких рядах.

### Прогресс реализации 2026-05-20: `Not assessed`

Шаг 1 выполнен кодово:

- `evaluate_predictions` теперь различает `Not found` и `Not assessed`;
- если в scores есть `score_valid=False` по скважине, интервалы этой скважины
  получают статус `Not assessed`, а не `Not found`;
- summary/evaluation теперь содержит coverage-aware поля:
  `assessed_interval_count`, `not_assessed_interval_count`, `coverage_rate`,
  `hit_rate_on_assessed`;
- HTML-отчет показывает статус `Не оценено`, `Оценено моделью`, `Покрытие
  оценки`, `Доля найденных среди оценённых`;
- старты по invalid score по-прежнему подавляются раньше, на уровне
  `_detect_starts_for_run`.

Проверка на сохраненном `negermet/paano_global` после перезапуска detector на
GPU:

```text
interval_count              = 5
assessed_interval_count     = 1
not_assessed_interval_count = 4
coverage_rate               = 0.20
hit_count                   = 1
hit_rate_on_assessed        = 1.00
```

Фактический смысл результата: `paano_global` с текущим `global_long=192/384`
честно оценивает только длинную скважину `172г`; короткие `1123л`, `3509г`,
`524`, `5271г` теперь не маскируются как "аномалии нет", а помечаются как
`Not assessed`.

Сгенерированный отчет:

```text
artifacts/reports/negermet/negermet_paano_global_report.html
```

### Прогресс реализации 2026-05-20: coverage-аудит `paano_global`

Шаг 2 выполнен: текущий `paano_global` прогнан без изменения алгоритма на
размеченных `negermet`, `pritok`, `salt`. `Salym` и `test35` не использовались.
Цель шага была не улучшить метрики, а честно измерить, где модель реально
оценивает интервалы, а где должна писать `Not assessed`.

Команды выполнялись на сервере с GPU:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/detection/detect_negermet.py --detector paano_global
CUDA_VISIBLE_DEVICES=1 uv run python scripts/detection/detect_pritok.py --detector paano_global
CUDA_VISIBLE_DEVICES=1 uv run python scripts/detection/detect_salt.py --detector paano_global
```

Итог по coverage-aware метрикам:

```text
negermet / paano_global
  interval_count              = 5
  assessed_interval_count     = 1
  not_assessed_interval_count = 4
  coverage_rate               = 0.20
  hit_count                   = 1
  hit_rate_on_assessed        = 1.00
  false_alarms_per_day        = 0.00

pritok / paano_global
  interval_count              = 27
  assessed_interval_count     = 27
  not_assessed_interval_count = 0
  coverage_rate               = 1.00
  hit_count                   = 25
  hit_rate_on_assessed        = 0.926
  false_alarms_per_day        = 0.00
  median_abs_delay_hours      = 1.67

salt / paano_global
  interval_count              = 9
  assessed_interval_count     = 9
  not_assessed_interval_count = 0
  coverage_rate               = 1.00
  hit_count                   = 8
  hit_rate_on_assessed        = 0.889
  false_alarms_per_day        = 0.00
  median_abs_delay_hours      = 1.00
```

Вывод:

- для `pritok` и `salt` текущий `global_long=192/384` применим по coverage;
- для `negermet` текущий `global_long=192/384` применим только к длинной
  `172г`;
- `pritok` не нашел вторые интервалы `602` и `691`; это не coverage-проблема,
  а именно `Not found` на оцененных рядах;
- `salt` не нашел второй интервал `408`; это тоже `Not found` при валидном
  score;
- короткие `negermet`-ряды требуют следующего шага: `global_long +
  edge/hold padding` и/или population memory bank;
- cascade scale по-прежнему не внедряем, пока не доказана необходимость.

Сгенерированные отчеты:

```text
artifacts/reports/negermet/negermet_paano_global_report.html
artifacts/reports/pritok/pritok_paano_global_report.html
artifacts/reports/salt/salt_paano_global_report.html
```

Следующий шаг по плану: эксперимент `global_long + edge/hold padding` для
коротких `negermet`-рядов, без включения `Salym/test35` и без cascade scale.

### Прогресс реализации 2026-05-20: edge/hold padding для коротких рядов

Шаг 3 выполнен как контролируемый эксперимент, а не как безусловная замена
production-поведения.

В код добавлен input-contract:

```text
ALMA_PAANO_INPUT_PADDING=none       # поведение по умолчанию: короткие ряды -> Not assessed
ALMA_PAANO_INPUT_PADDING=edge_hold  # эксперимент: дополнить начало ряда первым валидным значением
```

Математически это не cascade scale: модель остается той же
`global_long=192/384`, encoder и memory-bank размерность не меняются. Если
истории меньше, чем требуется для патчей, в начало ряда добавляются копии
первого валидного состояния. После scoring искусственный префикс отбрасывается,
и в artifacts остаются score только исходных реальных timestamp.

Проверка:

```bash
ALMA_PAANO_INPUT_PADDING=edge_hold \
CUDA_VISIBLE_DEVICES=1 \
uv run python scripts/detection/detect_negermet.py --detector paano_global

uv run python scripts/evaluation/evaluate_onset_metrics.py \
  --anomaly negermet \
  --detector paano_global \
  --name negermet_paano_global_edge_hold_padding
```

Результат на размеченном `negermet`:

```text
interval_count              = 5
assessed_interval_count     = 5
not_assessed_interval_count = 0
coverage_rate               = 1.00
hit_count                   = 5
hit_rate_on_assessed        = 1.00
false_alarms_per_day        = 0.00
median_abs_delay_hours      = 0.05
p90_abs_delay_hours         = 0.72
```

По интервалам:

```text
1123л  -> detected 2025-06-21 08:35:00, actual_start 2025-06-21 08:32:00
172г   -> detected 2025-05-03 11:30:00, actual_start 2025-05-03 11:30:00
3509г  -> detected 2025-08-10 10:50:00, actual_start 2025-08-10 09:40:00
524    -> detected 2025-06-09 19:40:00, actual_start 2025-06-09 19:38:00
5271г  -> detected 2025-08-15 00:40:00, actual_start 2025-08-15 00:37:00
```

Важная оговорка: blind/unlabeled скважина `ю-я 39-651` после включения
`edge_hold` получила 1 candidate-start:

```text
ю-я 39-651 -> 2026-03-04 11:55:00
```

Это не входит в interval hit-rate, потому что у нее нет разметки. Но для
production-решения это обязательно надо разобрать как потенциальный false
positive на чистом/слепом ряду.

Вывод по шагу:

- `edge_hold` решает проблему silent zero-score и резко повышает coverage на
  коротком `negermet`;
- метрики на размеченных интервалах стали сильными: `5/5`, задержки минуты,
  FAR/day по размеченной оценке `0.0`;
- включать `edge_hold` по умолчанию рано: сначала нужно визуально и доменно
  разобрать `ю-я 39-651`, а также проверить padded-контракт отдельно от
  обычного real-window-контракта;
- следующий технический шаг: записывать в отчеты/score detail, что конкретная
  скважина оценена по `edge_hold_padded` контракту, и затем сравнить padded vs
  non-padded на чистом пуле.

### Прогресс реализации 2026-05-20: `input_contract` и разбор `ю-я 39-651`

Шаг 4 выполнен.

В artifacts добавлен явный `input_contract`:

```text
real_window        = обычная оценка без искусственного дополнения
edge_hold_padded   = ряд/reference дополнялись первым валидным значением
```

Теперь `input_contract` записывается в:

- `db/*_scores.parquet`;
- detail-map detector run;
- HTML-отчет в виде русской поясняющей плашки.

После пересоздания `negermet/paano_global` с `ALMA_PAANO_INPUT_PADDING=edge_hold`
получилась такая картина:

```text
1123л       -> edge_hold_padded
172г        -> real_window
3509г       -> edge_hold_padded
524         -> edge_hold_padded
5271г       -> edge_hold_padded
ю-я 39-651  -> edge_hold_padded
```

То есть длинная `172г` оценивается обычным контрактом, а короткие ряды явно
помечены как padded. Это важно: метрики `5/5` по `negermet` больше не выглядят
как обычный PaAno-score без оговорок.

Разбор blind/unlabeled `ю-я 39-651`:

```text
candidate-start = 2026-03-04 11:55:00
input_contract  = edge_hold_padded
quality_status  = ok
regime_status   = normal
event_class     = normal_context
score           = 0.002696
paano_short     = 0.003103
paano_long      = 0.002086
```

Физические параметры вокруг старта:

```text
Параметр                         prev 24h median   next 2h median   изменение
Давление на приеме насоса        57.300            57.380           +0.14%
Температура на приёме насоса     79.080            79.080            0.00%
Температура масла двигателя      85.510            85.590           +0.09%
Выходная частота                 203.300           203.300           0.00%
Выходной ток ПЧ                  65.100            65.600           +0.77%
Ток на фазе А                    15.300            15.300            0.00%
Активная выходная мощность       49.000            49.000            0.00%
Коэффициент загрузки ПЭД         66.000            66.000            0.00%
Дисбаланс токов                  0.700             0.700             0.00%
Вибрация ХY                      1.490             1.500            +0.67%
```

Вывод по `ю-я 39-651`: физически это слабый candidate. Нет ключевой картины
негермета: нет резкого роста давления, нет роста температуры, нет заметной
реакции фазного тока. На текущем этапе считаем это потенциальным false positive
от padded-контракта/пороговой логики, а не подтвержденной аномалией.

Практический вывод по шагу:

- `edge_hold` полезен для покрытия коротких размеченных `negermet`;
- `input_contract` обязателен в отчетах, иначе результат нельзя корректно
  интерпретировать;
- перед включением `edge_hold` по умолчанию нужен отдельный guard по доменной
  физике для blind/clean рядов: если нет pressure/temperature/load response,
  candidate должен уходить в review/suppressed, а не становиться уверенной
  детекцией.

### Прогресс реализации 2026-05-20: population memory bank fallback

Шаг 5 реализован кодово как opt-in эксперимент, а не как новый default.

Режим управления:

```text
ALMA_GLOBAL_MEMORY_BANK_MODE=local                # default
ALMA_GLOBAL_MEMORY_BANK_MODE=population_fallback  # эксперимент
```

До этого `paano_global` был глобальным только по encoder, но memory bank для
scoring строился из локального reference конкретной скважины. Это давало две
проблемы:

- короткий `negermet` мог иметь слишком мало локальных нормальных patch-окон;
- blind/unlabeled скважина могла быть оценена только ценой ошибочного
  предположения, что начало её собственного ряда - это норма.

Новая экспериментальная логика:

```text
если локальный reference достаточно длинный:
    использовать local memory bank
иначе:
    использовать population memory bank из balanced normal pool
если population memory bank недоступен:
    Not assessed
```

Для blind/unlabeled скважины локальный reference не используется вообще. Если
есть population memory bank, такая скважина сравнивается с донорским банком
подтвержденной нормы. Если population memory bank недоступен, скважина остается
`Not assessed`.

Почему это не default: первый контрольный прогон `negermet/paano_global` с
автоматическим population fallback показал деградацию размеченных интервалов:
короткие `1123л`, `3509г`, `524`, `5271г` перестали находиться при старом
onset config. Это ожидаемый риск: population memory bank меняет распределение
score, поэтому старые пороги, подобранные для local memory bank, нельзя
считать валидными.

Population memory bank строится из уже подготовленного `global_pool`:

- размеченная нормальная работа train-скважин `negermet`, `pritok`, `salt`;
- `norm_work`, где весь период экспертно считается нормальной работой;
- fixed feature schema;
- balanced source/well sampling.

При scoring текущая скважина исключается из population reference по `well_id`,
чтобы не сравнивать ряд с самим собой.

В `input_contract` теперь отражаются два независимых факта: был ли padding и
какой memory bank использован:

```text
real_long_local_memory
padded_long_local_memory
real_long_population_memory
padded_long_population_memory
no_local_reference
no_population_reference
```

Практический смысл:

- `real_long_local_memory` - обычный и самый чистый контракт;
- `padded_long_local_memory` - короткая история, но локальная норма есть;
- `real_long_population_memory` - локальный reference не используется, сравнение
  идет с фондовой нормой;
- `padded_long_population_memory` - самый осторожный fallback: короткая история
  плюс population memory bank;
- `no_*` - оценка не выполнена, нулевой score не означает норму.

Ограничение: population memory bank решает проблему "с чем сравнить короткий
ряд", но не решает сам по себе проблему ложных кандидатов на blind/clean рядах.
Для production-пути после этого всё равно нужен domain decision layer/guard:
если у candidate нет pressure/temperature/load response, он должен уходить в
review/suppressed, а не считаться уверенной аномалией.

Следствие: следующий обязательный шаг после реализации fallback - отдельная
калибровка/retune для контрактов `*_population_memory`, минимум отдельно от
`*_local_memory`.

### Проверка 2026-05-20: отдельная калибровка population memory bank

Отдельная калибровка была выполнена как изолированный эксперимент, без
перезаписи production-артефактов. После запуска экспериментальные результаты
были сохранены отдельно, а текущие default config/results восстановлены.

Условия запуска:

```text
ALMA_GLOBAL_MEMORY_BANK_MODE=population_fallback
ALMA_PAANO_INPUT_PADDING=edge_hold
ALMA_RETUNE_MODE=fast
ALMA_OPTUNA_N_JOBS=8
CUDA_VISIBLE_DEVICES=1
```

Папка эксперимента на сервере:

```text
artifacts/results/global_population_calibration_20260520_103510
```

Смысл проверки: не сравнивать population fallback со старыми local-порогами, а
дать ему честную отдельную калибровку через `--retune`. Это важно, потому что
population memory bank меняет распределение score: старые пороги local memory
bank математически не обязаны подходить.

Результат отдельной population-калибровки:

| Аномалия | Population hit-rate | Coverage | Starts | FAR/day | Median delay | P90 delay | Вывод |
|---|---:|---:|---:|---:|---:|---:|---|
| `negermet` | `1/5 = 0.200` | `1.000` | `1` | `0.000` | `0.00h` | `0.00h` | провал |
| `pritok` | `25/27 = 0.926` | `1.000` | `45` | `0.000` | `1.67h` | `7.21h` | рабоче, но не лучше default |
| `salt` | `8/9 = 0.889` | `1.000` | `14` | `0.000` | `1.00h` | `12.07h` | рабоче, но не лучше default |

Контрольный current default после восстановления артефактов:

| Аномалия | Current default hit-rate | Starts | FAR/day | Median delay | P90 delay |
|---|---:|---:|---:|---:|---:|
| `negermet` | `5/5 = 1.000` | `5` | `0.000` | `0.05h` | `0.72h` |
| `pritok` | `24/24 = 1.000` | `45` | `0.000` | `1.60h` | `5.33h` |
| `salt` | `8/8 = 1.000` | `14` | `0.000` | `1.00h` | `12.07h` |

Главный факт: отдельный retune не спас `negermet`. Population fallback нашёл
только `172г`; короткие размеченные интервалы `1123л`, `3509г`, `524`,
`5271г` не были найдены. Значит, текущий population memory bank нельзя
включать как production fallback для всех аномалий.

Интерпретация:

- Population memory bank технически полезен как способ оценить blind/unlabeled
  ряд без ложного предположения "первые точки - это норма".
- Но текущий общий donor-pool не является корректной заменой локального memory
  bank для короткого `negermet`.
- Для `pritok` и `salt` population fallback работает приемлемо, но не даёт
  доказанного выигрыша относительно current default.
- Поэтому доказательство production-пользы не получено: режим остаётся
  research/diagnostic opt-in, а default должен оставаться
  `ALMA_GLOBAL_MEMORY_BANK_MODE=local`.

Решение по архитектуре:

```text
default:
    local memory bank

research only:
    population_fallback
```

Что можно развивать дальше, если population всё же нужен:

- делать не один общий population bank, а отдельные donor banks по
  технологическому профилю/классу события;
- калибровать onset отдельно по `real_long_population_memory` и
  `padded_long_population_memory`;
- отдельно проверять короткие `negermet`-ряды, потому что именно там общий
  population bank сейчас ломает hit-rate;
- добавлять domain decision layer после candidate detection, чтобы blind
  candidate без pressure/temperature/load response уходили в `review` или
  `suppressed`, а не считались уверенной аномалией.

Текущий вывод: population отдельной калибровкой проверен и **не доказан как
production default**. Его нельзя включать автоматически. Безопасный путь -
оставить `local` default, а population держать как диагностический режим для
следующих controlled experiments.
