Как устроена работа через API

  - Инициализация. Команда python -m src.pipeline anomalies --config config/pipeline.yaml загружает YAML и вызывает
    resolve_workbook_source (src/pipeline/anomalies/workbook.py:156).
      - Если в конфиге anomalies.input.mode: "api", берётся секция anomalies.input.api.
      - Из неё читаются url, method (GET/POST и т.д.), headers, params, опциональные data или json, timeout, флаг verify.
      - Выполняется HTTP-запрос: requests.request(method, url, ...). Ответ должен быть JSON. При ошибке статус-кода бросится исключение с
        текстом ответа — это сразу видно в консоли.
  - Парсинг JSON. После response.json() данные проходят через _payload_to_frames (src/pipeline/anomalies/workbook.py:106):
      - По умолчанию ожидается словарь вида

        {
          "svod": [...],
          "1128г": [...],
          "5271г": [...]
        }
        где ключ = имя листа, значение = список строк, каждая строка — словарь с колонками.
      - Альтернативно можно прислать

          ]
        }
        тогда задаём sheets_key, name_key, records_key.
      - Если нужные данные глубже, используем payload_path (строка или массив ключей) — код пройдёт по указанному пути внутри ответа.
    (timestamp_Intake_Pressure, Intake_Pressure, …).
  - Предобработка.
      - load_svod_sheet (preprocessing.py:94) прочитает svod, заполнит пропуски в колонках Скв, ПричОст, приведёт даты в datetime.
      - load_well_series (preprocessing.py:110) обходит все остальные “листы”, строит часы времени, делает resample до сетки
        alignment.frequency, добавляет min/max для давления, собирает быстрые ряды.
      - Далее всё стандартно: Hampel-фильтр, ffill, вычисление признаков, baseline, пороги и т.д.
  - Выход. По завершении увидите сообщение вида Using workbook source: api:https://example.com/.... Все артефакты (reports/anomalies/
    anomaly_analysis.*, и т.д.) формируются так же, как при Excel.

  Что нужно подготовить на стороне API

  1. Реализовать endpoint, который выдаёт JSON со всеми листами в договорённом формате.
  2. Передать координаты в конфиг (url, возможно headers/params). Если требуется POST с телом — добавить method: "POST" и положить JSON
     в anomalies.input.api.json.
  3. Убедиться, что временные метки в ответе парсятся pandas.to_datetime(..., dayfirst=True) (как сейчас). При необходимости поменять
     формат или параметры в коде.


Модуль WorkbookSource, который умеет брать данные либо из
  Excel, либо из HTTP API. В config/pipeline.yaml достаточно поставить

  anomalies:
    input:
      mode: "api"
      api:
        url: "https://..."
        method: "GET"
        # при необходимости headers/params/json/payload_path

  После этого запуск python -m src.pipeline anomalies --config config/pipeline.yaml подтянет данные по сети (видно по сообщению Using
  workbook source: api:...), а дальше весь детектор отработает без доработок: очистка, агрегирование, признаки, отчёты.

  По формату JSON: по умолчанию ждём:

  {
    "svod": [...],
    "1128г": [...],
    "5271г": [...]
  }

  либо

  {
    "sheets": [
      {"name": "svod", "records": [...]},
      {"name": "1128г", "records": [...]}
    ]
  }

  Если структура другая — можно перенастроить ключи sheets_key, name_key, records_key, а также payload_path, чтобы добраться до нужного узла. 