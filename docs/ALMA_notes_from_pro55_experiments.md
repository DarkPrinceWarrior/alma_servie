# ALMA: актуальный план экспериментов после уточнения scope

Дата обновления: 2026-05-18.

## Scope

Активный scope теперь только такой:

- размеченный `negermet`;
- размеченный `pritok`;
- размеченный `salt`;
- их текущие train/test split и тестовые скважины из ALMA-разметки.

Не входят в активный scope:

- `Salym`;
- `test35`;
- новые неразмеченные пакеты;
- экспертные таблицы по неразмеченным пакетам.

`norm_work` не удаляется и не считается мусором. Это отдельная вспомогательная
нормальная выборка: её можно использовать как дополнительный источник чистой
нормы для pretrain/reference-library или как guard-проверку "не ловить норму".
Но `norm_work` нельзя смешивать с основным размеченным benchmark как будто там
есть классы `salt/negermet/pritok`.

Соответственно, все выводы ниже относятся только к размеченным ALMA-датасетам.
Если направление требует неразмеченных данных, оно снято с активного плана.

## Что удалено из активной ветки

Удалены рабочие файлы, которые были сделаны под старый расширенный scope:

- `alma_service/expert_feedback.py`;
- `scripts/evaluation/evaluate_expert_feedback.py`;
- `tests/test_expert_feedback.py`;
- `docs/ALMA_notes_completion_audit_2026-05-18.md`;
- `docs/ALMA_журнал_внешних_материалов.md`.

`norm_work`-скрипты восстановлены и сохраняются как вспомогательный контур.

Существующие Salym/test35 исходные пайплайны и данные этой правкой не
анализируются и не развиваются.

## Актуальный checklist

| № | Пункт из исходного плана | Текущий scoped-статус |
|---:|---|---|
| 1 | `normal_window_selector v2`: `reference_before_first_anomaly`, bias к ранним окнам, штраф за физический сдвиг | Оставить как research baseline на размеченных `negermet/pritok/salt`. Production default менять только если benchmark на размеченных test split лучше текущего `paano_shared`. |
| 2 | Встроить `normal_window_selector` в production `paano_shared` | Встроено только как опция `--reference-policy normal_windows`. Следующий шаг: честный benchmark `default` vs `normal_windows` только на размеченных ALMA train/test. |
| 3 | Multi-grid inference 2min/10min/15min + 1h/6h/12h/1d | Оставить только как aggregation layer для уже размеченных ALMA outputs. Orchestrator для новых неразмеченных пакетов сейчас вне scope. |
| 4 | Явный слой качества данных | Оставить для размеченных ALMA scores/predicted starts: пропуски, залипания, остановы, пуски, смены частоты. |
| 5 | Классификатор `аномалия / режимное событие / плохие данные / преданомальная зона` | Оставить только для стартов на размеченных ALMA-датасетах. Цель: не считать режимные события полноценными авариями в onset-metrics. |
| 6 | Преданомальные зоны через журналы ремонтов/остановов/экспертные события | Реальные внешние журналы сейчас вне scope. Для размеченных ALMA использовать только имеющиеся интервалы `start/end` и class-specific prestart logic. |
| 7 | Внешние модели | Сравнивать только на размеченных `negermet/pritok/salt`: `MTGFlow`, `DCdetector`, `MTAD-GAT`, `TranAD`, `D3R`, `TimesNet`; `GDN` отдельно только при готовом PyG-окружении; `PANF/tcNF/H-PAD` только при найденном runnable-коде. |
| 8 | Единый benchmark-протокол для внешних моделей | Оставить формат export/evaluation по размеченным ALMA wells. `norm_work` можно использовать отдельно как guard, но не как размеченный класс. |
| 9 | Общая модель на очищенных окнах + индивидуальная библиотека нормы | Проверять на размеченных ALMA wells; `norm_work` допустим как дополнительная чистая норма или sanity-check. Если хуже текущего `paano_shared`, не внедрять в default. |
| 10 | Экспертная обратная связь Salym/test35 в formal constraints | Снято с активного плана. Не трогать. |
| 11 | Incident lifecycle `open / continues / closed` | Оставить для размеченных ALMA predictions: объединять повторные старты в один инцидент и оценивать onset по actionable starts. |
| 12 | Контрольные нормальные скважины без аварий | `norm_work` сохранить. Использовать только как вспомогательную проверку или дополнительную норму, не как основной размеченный benchmark. |

## 2026-05-18: проверка гипотезы "одна модель нормальности"

Экспертная гипотеза: делить задачу на три независимые модели по классам может
быть методологически неправильно, потому что это не три разные сущности, а одни
и те же скважины. Скважина может иметь разные типы аномалий, поэтому базовый
слой должен сначала отвечать на вопрос "есть отклонение от нормальной работы
или нет", а уже после этого отдельный слой должен объяснять/классифицировать
тип аномалии.

Для проверки добавлен отдельный research benchmark:

- `scripts/evaluation/benchmark_global_normality_detector.py`;
- режим: один `PaAno Shared Encoder` на все train-нормальные участки
  размеченных `negermet`, `pritok`, `salt`;
- без class fine-tune;
- без class-specific physical branches;
- оценка по каждому классу через тот же production onset/postprocess;
- detector key для production не менялся.

Запуск на сервере:

```bash
cd /root/projects/alma_servie
ALMA_RETUNE_MODE=fast CUDA_VISIBLE_DEVICES=1 uv run python \
  scripts/evaluation/benchmark_global_normality_detector.py \
  --output-dir artifacts/results/global_normality_detector \
  --global-iters 200
```

Фактический train pool:

- `104402` точки;
- `80` общих каналов после feature reduction;
- `35` train-скважин;
- устройство: `cuda`, `NVIDIA A100-SXM4-40GB`.

Результаты:

| Вариант | Класс | Hit-rate | FAR/day | Старты | Median delay | P90 delay |
|---|---:|---:|---:|---:|---:|---:|
| class-specific saved | `negermet` | 4/5 = 0.800 | 0.000 | 4 | 0.000h | 0.700h |
| global normality | `negermet` | 5/5 = 1.000 | 0.000 | 7 | 0.000h | 0.607h |
| class-specific saved | `pritok` | 24/24 = 1.000 | 0.000 | 53 | 1.429h | 8.825h |
| global normality | `pritok` | 24/24 = 1.000 | 0.000 | 47 | 1.429h | 8.825h |
| class-specific saved | `salt` | 8/8 = 1.000 | 0.000 | 22 | 1.442h | 78.550h |
| global normality | `salt` | 8/8 = 1.000 | 0.000 | 13 | 1.442h | 98.150h |

Вывод:

- гипотеза о едином базовом детекторе нормальности выглядит рабочей;
- `negermet` улучшился по hit-rate;
- `pritok` остался на уровне class-specific, но с меньшим числом стартов;
- `salt` сохранил hit-rate, но ухудшил дальний хвост задержки, поэтому для соли
  нужен отдельный downstream diagnosis/signature layer, а не возврат к трём
  независимым encoder-моделям;
- правильная целевая архитектура: `global normality score -> incident/onset ->
  anomaly diagnosis/classification`.

Дополнительно проверен наивный вариант `--include-norm-work`.

Факт: прямое добавление `norm_work` в общий train pool сжало пересечение
признаков с `80` до `1` канала:

```text
Feature reduction: 138 -> 1 channels
Shared encoder pool: 210474 points, 1 channels, 95 wells
```

Такой вариант не является валидным рабочим решением: он ухудшил `pritok` и
`salt`. В скрипт добавлена защита `--min-shared-channels 40`, чтобы подобное
смешивание данных больше не проходило молча. `norm_work` можно использовать
только после нормализации схемы каналов/feature coverage либо как отдельный
guard/reference-library, но не как прямую примесь в общий encoder pool.

## Что делать дальше

Ближайшая работа должна идти в таком порядке:

1. Запустить и зафиксировать `default` vs `normal_windows` только на размеченных
   `negermet/pritok/salt`.
2. Проверить, что `telemetry_status` и `prediction_postprocess` не ухудшают
   текущие production metrics на этих же размеченных test split.
3. Довести external quick benchmark до аккуратного отчёта только по размеченным
   ALMA wells.
4. После этого решать, что реально внедрять в `paano_shared` default.

## Принцип принятия изменений

Изменение можно считать полезным только если оно проходит проверку на
размеченных ALMA test split:

- не снижает hit-rate без явной причины;
- не ухудшает задержку критично;
- снижает ложные старты или делает их объяснимыми;
- не использует данные Salym/test35;
- если использует `norm_work`, то только как дополнительную норму или guard,
  отдельно от размеченных ALMA metrics;
- не подбирает параметры по test split напрямую.
