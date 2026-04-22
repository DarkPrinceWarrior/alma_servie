# Исследование рынка: weak-supervised onset discovery для Salym

> **Дата исследования:** 2026-04-22
> **Источник:** Tavily MCP search (2023–2026, RU+EN)
> **Контекст задачи:** применение к Salym-датасету (568 скважин × 15 параметров × 4 года + ~3800 циклов ЭЦН с точками `Failed`) методов для извлечения onset деградации из слабой разметки (точка отказа + тип отказа).

---

## 1. Релевантные научные работы

### 1.1. ESP-специфика (электроцентробежные насосы)

| # | Источник | Год | Суть | Почему важно для нас |
|---|---|---|---|---|
| 1 | **Time-aware predictive maintenance of ESP using CatBoost + trend-based labeling** — Springer J. Pet. Explor. Prod. Technol. | 2025 | Rolling averages + pressure slopes для "захвата постепенной деградации", Isolation Forest + PCA, CatBoost/ExtraTrees сверху | ⭐ Готовый рецепт proxy-labeling окна деградации из точки `Failed`. Применимо 1:1 |
| 2 | **Hybrid LSTM-CNN for ESP Condition Prediction and Diagnosis** — SPE Journal | 2024 | Гибрид LSTM+CNN на сенсорах ESP | Бенчмарк архитектурных решений для сравнения с PaAno |
| 3 | **ESP failure diagnosis via LSTM-AE + PCA** — Geoenergy Sci. Eng. | 2024 | LSTM-autoencoder + PCA | Архитектура близка к нашему PaAno + PCA/SPE стеку |
| 4 | **RUL Prediction and Operation Optimization of ESPs** — MDPI JMSE (712 скважин Bohai Oilfield) | 2024 | Детальный анализ ключевых факторов RUL для ESP | Масштабно сопоставимо с Salym |
| 5 | **Data-Driven Fault Prediction for ESPCP Wells** — MDPI Processes | 2025 | PCA + LSTM vs ARIMA/GBDT; упоминает Hamedi Shokrlu & Bazile (2024) "Real-Time ML-based changepoint detection" для ESP | ⭐ Прямо наш кейс онлайн CPD для ESP |
| 6 | **Enhancing ESP reliability (Egypt case)** — Results in Engineering (231 скважина, 676 установок, 14 лет) | 2026 | MAE=17 дней по runlife, precision=96% по типу отказа; статические признаки пласта + оборудования | Показывает, что даже без временной разметки статический RUL-регрессор даёт разумный MAE |

**Ссылки:**
1. https://link.springer.com/article/10.1007/s13202-025-02070-z
2. https://www.sciencedirect.com/science/article/abs/pii/S0952197626005361
3. https://www.sciencedirect.com/science/article/abs/pii/S2949891024006493
4. https://www.mdpi.com/2077-1312/14/1/75
5. https://www.mdpi.com/2227-9717/13/9/2890
6. https://www.sciencedirect.com/science/article/pii/S2590123026014969

### 1.2. Weak/Self-supervised для Time Series

| # | Источник | Год | Суть | Почему важно |
|---|---|---|---|---|
| 1 | **WMAD: Weakly-supervised Multi-sensor Anomaly Detection with Time-series Foundation Models** — NeurIPS | 2024 | Data-enclosing hypersphere + two-level importance sampling + meta-learning. Валидация на Amazon индустриальном датасете (>700K часов) | ⭐ Ровно про "мало экспертной разметки + гетерогенные сенсоры разных машин" — это наша Salym |
| 2 | **CARLA: Self-supervised Contrastive Representation Learning for TS Anomaly Detection** — Pattern Recognition | 2025 | Контрастивное SSL для временных рядов | Первый этап представлений перед PCA/SPE |
| 3 | **AnomalyBERT / SPT-AD** — MDPI Appl. Sci. | 2025 | BERT-encoder с data-degradation вместо разметки | Pretraining-стратегия для PaAno на всём Salym |
| 4 | **PatchTrAD: A Patch-Based Transformer for Anomaly Detection** — EUSIPCO | 2025 | Расширение PatchTST именно под AD | Прямой наследник нашей PaAno-архитектуры |
| 5 | **Change-Point Detection in Industrial Data Streams via Online DMD with Control** — arXiv 2407.05976 | 2024 | Обзор + реализация онлайн-CPD для industrial streams, включая contrastive CoCPD для subtle changepoints | Второй слой поверх proxy-labeling для точного onset |
| 6 | **Change-point detection with deep learning: A review** — Frontiers of Eng. Mgmt. | 2025 | Обзор CPD + DL для predictive maintenance | Литературная база |
| 7 | **Survey: RUL Prediction Methods based on Deep Learning** — Cambridge AI EDAM | ~2024 | Self-supervised + pseudo-label для RUL: классификаторная итерация, псевдо-метки, монотонные health indicators | ⭐ Максимально близко к нашей weak-supervised задаче |
| 8 | **Oil and gas flow anomaly detection on offshore wells using DNN** — Geoenergy SE (3W Petrobras dataset 2012-2018, 21 скважина) | 2024 | LSTM+GRU с GA-тюнингом | Публичный датасет, структурно похожий на Salym — годится для предварительной отработки метода |

**Ссылки:**
1. https://neurips.cc/virtual/2024/103033
2. https://arxiv.org/abs/2308.09296
3. https://www.mdpi.com/2076-3417/15/9/5185
4. https://eusipco2025.org/wp-content/uploads/pdfs/0001104.pdf
5. https://arxiv.org/html/2407.05976v1
6. https://journal.hep.com.cn/fem/EN/10.1007/s42524-025-4109-z
7. https://www.cambridge.org/core/journals/ai-edam/article/remaining-useful-life-prediction-methods-of-equipment-components-based-on-deep-learning-for-sustainable-manufacturing-a-literature-review/C3FFF4402D1EF1EC9BDD1F5C84198BC1
8. https://www.sciencedirect.com/science/article/pii/S2949891024006109

---

## 2. Русскоязычные статьи и кейсы

| # | Источник | Кто / Когда | Суть | Почему важно |
|---|---|---|---|---|
| 1 | **Habr: "Предсказание выбытия насосов"** | habr.com/ru/articles/827242 | Оператор размечает временное окно 15→30 дней **до отказа** и ищет в нём аномалии | ⭐⭐ Прямой RU-кейс weak-supervised onset discovery на Salym-подобной задаче. Проверить подход первым |
| 2 | **Neftegaz.RU №1/2025**: "Прогнозирование отказов УЭЦН методами..." | Отраслевой журнал, 2025 | Обзор методов прогноза отказов именно УЭЦН | Отраслевая ситуация в РФ |
| 3 | **Обзор нейросетевых моделей для прогнозирования отказов оборудования нефтяных скважин** | Омский ГТУ + Уфимский, 2024 | Систематический обзор. Фиксируют проблему: **"отсутствие данных, описывающих предаварийные ситуации"** | Прямо признают нашу проблему |
| 4 | **Habr (Factory5): "ML в помощь диагностам и инженерам по надёжности"** | 2023 | Продакшн-грейд PMM-система с автоэнкодерами на эталонном периоде | Продакшн-образец архитектуры |
| 5 | **Habr: "Предиктивная аналитика в промышленности"** | 2024 | Кейсы Лукойл-Пермнефтеоргсинтез (предсказание за 50 дней); Газпром нефть (−30% простоев) | Российские бизнес-кейсы в цифрах |
| 6 | **Habr: "ML и инфобез: три подхода для поиска аномалий во временных рядах"** | 2025 | Разбор автоэнкодеров, CatBoost-прогнозов, статистических методов | Референсная база методов |
| 7 | **Kaspersky MLAD** | 2024+ | Техстраница продакшн-подхода: ~100k отсчётов для обучения, все режимы и сезоны | Indirect эталон требуемого объёма обучающих данных |
| 8 | **CyberLeninka: "Нейросетевой метод обнаружения аномалий в многомерных потоковых временных рядах"** | Академика | Теоретическая база на русском | Русская терминология и методология |

**Ссылки:**
1. https://habr.com/ru/articles/827242/
2. https://magazine.neftegaz.ru/upload/iblock/3e4/rc044tgb066tvu3n3en3iwe48i5p41o9/NeftegazRU_01_2025.pdf
3. https://moitvivt.ru/ru/journal/pdf?id=1472
4. https://habr.com/ru/companies/factory5/articles/699496/
5. https://habr.com/ru/articles/849364/
6. https://habr.com/ru/articles/1018204/
7. https://mlad.kaspersky.ru/technologies/
8. https://cyberleninka.ru/article/n/neyrosetevoy-metod-obnaruzheniya-anomaliy-v-mnogomernyh-potokovyh-vremennyh-ryadah

---

## 3. Англоязычные блог-статьи

| # | Источник | Суть | Ссылка |
|---|---|---|---|
| 1 | **Grid Dynamics: "Anomaly detection in IoT data for smart manufacturing"** | Разбор supervised vs unsupervised + one-class подхода | https://www.griddynamics.com/blog/detecting-anomalies |
| 2 | **TDS: "5 Unexplored Python Libraries for Time Series Analysis"** | Darts, Prophet, AutoTS и др. | https://towardsdatascience.com/5-unexplored-python-libraries-for-time-series-analysis-e9375962fbb2/ |

Более специфичные материалы именно по ESP/onset discovery в Medium/TDS за 2024-2025 Tavily не вернул.

---

## 4. Продуктовые / open-source решения

| # | Решение | Год | Особенности | Ссылка |
|---|---|---|---|---|
| 1 | **dtaianomaly** (arXiv 2502.14381) | 2025 | sklearn-подобная Python-библиотека для TSAD, продакшн-ориентированная. Стандартный BaseDetector, визуализация, runtime/memory profiling | https://arxiv.org/html/2502.14381v1 |
| 2 | **Darts** (Unit8) | 2024-25 | Multivariate TS + модуль `darts.ad`: AnomalyScorer / AnomalyModel / AnomalyDetector / Aggregator, PyODScorer | https://github.com/unit8co/darts |
| 3 | **PyOD 3.x** | 2025 | 38M downloads, явная поддержка TS-детекторов, embedding-based детектор на foundation-model эмбеддингах | https://pyod.readthedocs.io/ |
| 4 | **TSB-AD** (NeurIPS) | 2024 | Бенчмарк: 1070 TS из 40 датасетов + надёжные метрики | https://thedatumorg.github.io/TSB-AD/ |
| 5 | **Aeon** (Middlehurst et al.) | 2024 | Общий TS-ML фреймворк, TSAD экспериментально | — |
| 6 | **Foundation models**: MOMENT, Chronos-2, Toto, TimesFM-2.5, Moirai-2 | 2024-25 | Open-weights foundation models для TS. Годятся как backbone вместо обучения энкодера с нуля | — |

---

## 5. Специфика ESP / нефтегаз

Сводно (детали в разделе 1):

- **Hamedi Shokrlu & Bazile (2024)** "Improving ESP production through Real-Time ML-based changepoint detection" — явно CPD для старта деградации ESP.
- **Trend-based labeling + CatBoost** (Springer 2025) — готовый рецепт proxy-labeling через скользящие средние и pressure slopes.
- **3W Petrobras dataset + LSTM/GRU+GA** (Geoenergy SE 2024) — публичный близкий-по-структуре бенчмарк для предварительной отработки метода перед Salym.
- **PCA-based ESP failure patterns** — ACS Omega 2022: **за 7 дней до workover** данные сенсоров ESP демонстрируют отчётливые паттерны конкретных режимов отказов (**MDHF** = Mechanical Downhole Failure, **EDHF** = Electrical Downhole Failure). Физически подтверждает детектируемость onset. https://pmc.ncbi.nlm.nih.gov/articles/PMC9161246/

---

## 6. Вывод и рекомендации

Задача **решаемая**, но акцент смещается: с чистого onset detection на **weakly-supervised onset discovery + RUL/Failure Prediction**.

Для Salym сочетание "известен `Failed` timestamp + тип отказа" — это классическая постановка **weak supervision в predictive maintenance**. Это подтверждается и общим обзором (Cambridge AI EDAM: pseudo-label + monotonic health indicator + semi-supervised RUL), и отраслевым кейсом (Springer 2025 на ESP).

### Пять практических векторов применения

1. **Proxy-labeling по тренду** — повторить подход Springer 2025: от `Failed` отмотать окно (15–60 дней), сгенерировать rolling-features (pressure slopes, скользящие std), запустить Isolation Forest / ваш PaAno-энкодер, кластеризовать ответ и взять точку "срыва" как актуальный `actual_start`. Это же подтверждено русским Habr-кейсом "Предсказание выбытия насосов".

2. **Self-supervised pretraining** вашего PaAno на всём Salym (masked-patch reconstruction в духе PatchTST/AnomalyBERT), затем fine-tune на 16 размеченных скважинах. Это даст большой датасет представлений и попадёт в тренд 2024-2025.

3. **CPD + contrastive как второй слой**: CoCPD (Bao 2024) или BOCPD на остатках даст инженерно-интерпретируемую точку старта, когда proxy-label слишком шумный.

4. **Разделение по типу отказа** — `R=0`/`Клин`/`Не герметичность` имеют разные сигнатуры (MDHF vs EDHF паттерны в ACS Omega 2022). Это позволяет обучать отдельные head'ы/детекторы под каждый класс, а не одну универсальную модель.

5. **Static features fallback** (Results in Eng. 2026): если временная разметка совсем не даётся — RUL-регрессор на статических признаках (пласт, оборудование, траектория) даёт MAE ≈ 17 дней. Это baseline, от которого можно стартовать.

### Итоговая рекомендация

Не менять задачу полностью на Failure Prediction/RUL, а **расширить её до двухуровневой**:
- **Уровень 1:** weak supervision через proxy-onset (Springer 2025 рецепт).
- **Уровень 2:** supervised RUL-регрессор сверху (Egypt 2026 рецепт).

Все кирпичики уже есть в публикациях 2024–2025 — это не R&D-риск, а инженерная интеграция известных методов.

### Приоритет для реализации

| # | Шаг | Оценка трудозатрат | Ожидаемый результат |
|---|---|---|---|
| 1 | Воспроизвести Springer 2025 рецепт (trend-based proxy-labeling) на **одном типе отказа** (`Не герметичность лифта` — 109 циклов, ближайший аналог `negermet`) | 1–2 дня | Первая полноразмерная разметка на ~100 циклов, быстрая проверка hypothesis |
| 2 | Self-supervised pretraining PaAno на всём Salym | 3–5 дней | Универсальный энкодер для 568 скважин |
| 3 | CPD/BOCPD валидация onset на 2–3 примерах | 1 день | Оценка честности proxy-разметки |
| 4 | Переобучение PaAno_shared на 15 каналах под proxy-метки | 2–3 дня | Расширенная версия детектора, обученная на порядок большем датасете |
| 5 | Benchmark на hold-out скважинах + сравнение с текущей системой | 2 дня | Финальное решение, оставлять ли подход в продакшене |

**Итого:** ~2 недели на MVP weak-supervised расширения ALMA.

---

## 7. Что осталось непокрыто исследованием

- Специфики **солеотложения** (наш `salt`-класс) в Salym stop_reason нет отдельным классом, соли размазаны по лабораторным блокам Excel (Calcium carbonate, Adjournment of salt). Отдельный поиск под эту задачу не проводился.
- **Hamedi Shokrlu & Bazile (2024)** — упомянута во вторичном источнике, прямой полный текст статьи не извлечён, стоит отдельно найти и прочитать.
- Вопросы **метрик для weak-supervised setting** (когда нет truth onset) покрыты поверхностно — стоит отдельно проработать (hit rate near Failed, согласованность методов, onset stability).
