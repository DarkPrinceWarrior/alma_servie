# notes_from_pro5.5: scoped version for ALMA

Дата обновления: 2026-05-18.

Этот файл больше не рассматривает неразмеченные пакеты. Текущий scope ограничен
только размеченными ALMA-датасетами:

- `negermet`;
- `pritok`;
- `salt`;
- их текущие train/test split и тестовые скважины.

## Вне scope

До отдельного разрешения не трогать:

- `Salym`;
- `test35`;
- любые новые неразмеченные скважины;
- экспертные feedback-таблицы по неразмеченным пакетам.

`norm_work` сохраняется как вспомогательная нормальная выборка. Её можно
использовать для дополнительного обучения нормы, reference-library или guard
проверки ложных тревог. Но она не заменяет размеченные `negermet/pritok/salt`
и не должна попадать в основной benchmark как размеченный класс.

## Главная проверяемая гипотеза

Для размеченных ALMA-датасетов нужно понять, можно ли улучшить текущий
production `paano_shared` без расширения данных за пределы разметки.

Проверяются только такие идеи:

1. Более честный выбор reference-нормы внутри размеченной скважины.
2. Явные статусы качества данных и режимных событий.
3. Постобработка стартов в инциденты.
4. Внешние модели как benchmark baseline на тех же размеченных wells.
5. Общая модель + индивидуальная библиотека нормы, но только если это лучше
   текущего `paano_shared` на размеченном test split.
6. `norm_work` как отдельная вспомогательная проверка чистой нормы, если
   эксперимент требует дополнительного normal-only источника.

## Уточнение после комментария инженера

Инженерная позиция: `salt`, `pritok`, `negermet` не должны рассматриваться как
три разные физические сущности. Это одни и те же нефтяные скважины, на которых
могут возникать разные типы аномалий. Поэтому стратегически более правильный
пайплайн:

```text
единая модель нормальности по скважинам
-> обнаружение отклонения / инцидента
-> отдельная диагностика типа аномалии
```

Это отличается от текущего production-исторического подхода:

```text
salt model
pritok model
negermet model
```

Проверочный benchmark реализован в:

```text
scripts/evaluation/benchmark_global_normality_detector.py
```

Что именно проверялось:

- один global `PaAno Shared Encoder`;
- train pool из всех train-нормальных участков размеченных `salt/pritok/negermet`;
- без class-specific fine-tune;
- без физических веток `pressure_trend`, `salt_deposition`, `negermet_signature`;
- per-class остался только слой оценки качества onset на существующей разметке.

Результат на A100 (`CUDA_VISIBLE_DEVICES=1`, `--global-iters 200`):

| Вариант | `negermet` | `pritok` | `salt` |
|---|---:|---:|---:|
| class-specific saved hit-rate | 0.800 | 1.000 | 1.000 |
| global normality hit-rate | 1.000 | 1.000 | 1.000 |
| class-specific starts | 4 | 53 | 22 |
| global normality starts | 7 | 47 | 13 |
| class-specific p90 delay | 0.700h | 8.825h | 78.550h |
| global normality p90 delay | 0.607h | 8.825h | 98.150h |

Вывод по гипотезе:

- единая модель нормальности технически жизнеспособна;
- она не ломает `pritok`, улучшает `negermet`, сохраняет hit-rate по `salt`;
- соль остаётся самым сложным случаем по задержке, поэтому diagnosis layer для
  соли нужен отдельно;
- развивать дальше стоит не как "три независимых модели", а как общий
  normality detector плюс классификатор/физические signatures поверх инцидента.

Проверка `norm_work`:

- прямое добавление `norm_work` в global train pool сейчас невалидно;
- пересечение признаков сжалось с `80` до `1` канала;
- из-за этого `pritok/salt` ухудшились;
- добавлена защита `--min-shared-channels 40`.

Следующий правильный шаг по `norm_work`: сначала привести схему параметров и
feature coverage к совместимой форме, либо использовать `norm_work` как
отдельный guard/reference-library, а не как прямую примесь в encoder pool.

## Что не является целью

Не является целью сейчас:

- делать screening неразмеченных пакетов;
- превращать экспертные комментарии по test35/Salym в constraints;
- строить отдельный pipeline для новых 35/547 скважин.

## Критерий качества

Каждый эксперимент сравнивается с текущим production `paano_shared`:

- `hit_rate`;
- `false_alarms_per_day`;
- `avg_starts_per_interval`;
- `median_abs_delay_hours`;
- `p90_abs_delay_hours`;
- поведение на test split.

Если эксперимент хуже production по размеченным `negermet/pritok/salt`, он
остаётся research-заметкой и не входит в default pipeline.

## Минимальный следующий план

1. Полный benchmark `reference_policy=default` против
   `reference_policy=normal_windows` на размеченных ALMA datasets.
2. Проверка `telemetry_status` и `prediction_postprocess` на тех же outputs.
3. Сводный отчёт по внешним моделям только на размеченных ALMA wells.
4. Решение: что из этого внедрять в production `paano_shared`, а что удалить
   как research dead-end.
