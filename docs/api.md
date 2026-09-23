# Интерфейсы решения

Два способа обратиться к агенту: командная строка (`app/cli.py`) и HTTP-сервис (`app/serve.py`). Оба используют один и тот же класс `ForecastAgent`, поэтому результат выпуска не зависит от способа вызова.

Все примеры ниже выполнены на эталонном окружении (Python 3.14, обученные модели в `models/`) и приведены по фактическому выводу.

---

## HTTP-сервис

### Запуск

```bash
python -m app.serve                                   # host и port из .env (APP_HOST, APP_PORT, по умолчанию 0.0.0.0:8000)
python -m uvicorn app.serve:app --port 8010           # произвольный порт
python -m uvicorn app.serve:app --reload              # режим разработки
```

В Docker сервис поднимается командой по умолчанию: `docker compose up` (порт 8000). Этот путь не проверялся — Docker на машине сборки не устанавливался.

Интерактивная документация FastAPI доступна на `/docs`, схема OpenAPI — на `/openapi.json`.

### `GET /health`

Проверка готовности: поднят ли сервис, обучены ли модели, какой провайдер LLM настроен. Ключ не раскрывается.

```bash
curl -s http://127.0.0.1:8010/health
```

```json
{"status": "ok", "models_trained": true, "llm_provider": "openai"}
```

| Поле | Тип | Смысл |
|---|---|---|
| `status` | string | всегда `ok`, если процесс отвечает |
| `models_trained` | bool | найден ли `models/turbine_1.joblib`; при `false` выпуск прогноза вернёт 503 |
| `llm_provider` | string | `openai`, `nvidia` или `demo` из `LLM_PROVIDER` |

### `GET /forecast/{issue_date}`

Полный цикл агента на одну дату выпуска: получение архивного прогноза погоды, доступного на эту дату, расчёт выработки, анализ, при необходимости пересчёт, сохранение.

| Параметр | Где | Тип | По умолчанию | Значения |
|---|---|---|---|---|
| `issue_date` | путь | string | — | дата выпуска `YYYY-MM-DD` |
| `horizon` | запрос | int | `48` | только `24` или `48` |
| `llm` | запрос | bool | `false` | `true` — порядок инструментов выбирает модель, `false` — детерминированный план |

```bash
curl -s "http://127.0.0.1:8010/forecast/2026-02-10?horizon=24&llm=false"
```

```json
{
  "issue_date": "2026-02-10",
  "horizon_hours": 24,
  "tools": ["get_weather", "check_input_quality", "prepare_and_predict", "analyze", "save_forecast"],
  "analysis": {
    "calm_hours": 5,
    "rated_hours": 3,
    "storm_hours": 0,
    "turbine_spread_max": 0.065,
    "mean_cf": {"1": 0.536, "2": 0.521},
    "previous_issue_available": false,
    "previous_issue_source": "отсутствует",
    "input_update_vs_previous_issue": null,
    "input_change_vs_previous_issue": null,
    "update_threshold": 0.15,
    "needs_recalculation": false
  },
  "recalculated": false,
  "conclusion": "Прогноз на 24 ч от 2026-02-10: средняя нормализованная мощность T1 0.54, T2 0.52; ожидаемая выработка 12.9 и 12.5 норм·ч…",
  "forecast": [
    {"ts": "2026-02-11 00:00", "lead_day": 1, "turbine": 1, "power_norm_pred": 0.9683, "power_p10": 0.8611, "power_p90": 1.0}
  ]
}
```

Поля ответа: `tools` — фактическая трасса вызванных инструментов; `analysis` — результат инструмента `analyze` (пороги: штиль < 3 м/с, номинал ≥ 11.5 м/с, шторм ≥ 22 м/с, порог пересчёта 0.15); `recalculated` — выполнялся ли повторный расчёт; `conclusion` — заключение модели или детерминированный текст; `forecast` — почасовые строки по турбинам, значения 0..1, округление до 4 знаков.

В примере `previous_issue_available: false`: выпуска за предыдущую дату нет ни в памяти, ни на диске, поэтому сравнить с ним нечего. Это явный признак, а не молчаливое отсутствие поля — иначе отсутствие сравнения выглядело бы как «изменений нет». Если контроль входа даст вердикт `fail`, выпуск не публикуется: в трассе будет `check_input_quality`, а `save_forecast` вернёт `published: false` с перечнем причин.

Побочный эффект вызова: как и в CLI, выпуск сохраняется на диск — `outputs/forecasts/<issue_date>.csv` и `outputs/agent_logs/<issue_date>.json`. Запрос с `horizon=24` перезапишет выпуск этой даты 48 строками вместо 96 — учитывайте это, если артефакты бэктеста нужны в исходном виде.

| Код | Когда |
|---|---|
| 200 | выпуск состоялся |
| 400 | `issue_date` не разбирается как дата, либо `horizon` не 24 и не 48 |
| 503 | модели не обучены — выполните `python -m app.cli train` |

### `GET /submission`

Итоговый почасовой ряд февраля 2026 из `outputs/forecast_feb2026.csv`, если бэктест уже выполнен.

```bash
curl -s http://127.0.0.1:8010/submission
```

```json
{
  "rows": 672,
  "data": [
    {"ts": "2026-02-01 00:00:00", "turbine_1_lead1": 0.0287, "turbine_2_lead1": 0.0267,
     "turbine_1_lead2": 0.0325, "turbine_2_lead2": 0.0375,
     "turbine_1_p10": 0.0, "turbine_2_p10": 0.0, "turbine_1_p90": 0.2425, "turbine_2_p90": 0.2249,
     "farm_mean_lead1": 0.0277, "farm_mean_lead2": 0.035, "farm_p10_lead1": 0.0, "farm_p90_lead1": 0.2337}
  ]
}
```

Пропусков в файле нет: выпуски начинаются с 29 января, поэтому каждый час февраля закрыт и прогнозом на сутки (`*_lead1`, выпуск накануне), и прогнозом на двое суток (`*_lead2`, выпуск за день до того). Колонки `farm_*` — станционный уровень: среднее долей номинала двух турбин. Код 404 — файла нет, сначала выполните `python -m app.cli backtest --no-llm`.

---

## Командная строка

```bash
python -m app.cli <команда> [аргументы]
python -m app <команда>           # то же самое
```

Режим LLM общий для команд `forecast` и `backtest`: без флагов агент использует модель, если задан ключ провайдера, иначе идёт детерминированным планом. `--no-llm` выключает обращения к LLM принудительно (так воспроизводят результат без ключей), `--llm` требует включения.

### `check`

Проверка окружения: версия Python, наличие данных организаторов, обученных моделей, объём кэша прогнозов, состояние ключа LLM (печатается только «задан / не задан»), версии библиотек. Возвращает 1, если зависимости не установлены.

```bash
python -m app.cli check
```

### `fetch-weather --start ДАТА --end ДАТА`

Прогрев кэша архивных прогнозов Open-Meteo за период. Единственная команда, которой нужна сеть; для дат тестового периода кэш уже лежит в репозитории.

```bash
python -m app.cli fetch-weather --start 2026-01-29 --end 2026-02-27
```

Печатает число часов в ответе источника и число файлов в `data/cache/weather/`.

### `train`

Обучение моделей обеих турбин, около 15 секунд. Создаёт `models/turbine_1.joblib`, `models/turbine_2.joblib` и перезаписывает `docs/metrics.json`. Печатает подобранные веса смеси и ошибку на честной валидации января 2026.

### `forecast --date ДАТА [--horizon 48] [--llm | --no-llm]`

Один выпуск агента. Печатает трассу инструментов, анализ, заключение и пути файлов.

```bash
python -m app.cli forecast --date 2026-02-15 --llm
```

Создаёт `outputs/forecasts/<дата>.csv` (96 строк при горизонте 48: 48 часов × 2 турбины, 13 колонок) и `outputs/agent_logs/<дата>.json` (трасса, вердикт контроля входа, анализ, заключение, признаки `llm_used`, `published`, `confidence`).

Если контроль входа дал вердикт `fail`, расчёт не запускается и файл выпуска не создаётся; журнал пишется всё равно — отказ фиксируется так же, как состоявшийся прогноз.

### `backtest [--start 2026-01-29] [--end 2026-02-27] [--horizon 48] [--llm | --no-llm]`

Последовательные выпуски за весь тестовый период: выпуск дня N получает результат дня N−1 и сравнивает пересекающиеся часы. Печатает ход по дням и итоговые числа.

```bash
python -m app.cli backtest --no-llm
```

Создаёт `outputs/forecast_all_issues.csv` (2 880 строк, 30 выпусков), `outputs/forecast_feb2026.csv` (672 часа), `outputs/backtest_summary.csv` и `outputs/backtest_summary.json`, плюс пофайловые выпуски и журналы.

### `report`

Сводка по уже посчитанным результатам: метрики из `docs/metrics.json`, число выпусков и дни с пересчётом, число часов февраля и пропусков, средняя ожидаемая мощность по турбинам, суточный профиль по ВЭС. Сеть и ключи не нужны.

---

## Форматы выходных файлов

| Файл | Строк | Ключевые колонки |
|---|---|---|
| `outputs/forecasts/<дата>.csv` | 96 | `issue_date, ts, horizon_hour, lead_day, turbine, wind_speed_100m, power_norm_pred, power_p10, power_p90, power_gbm_pred, power_two_stage_pred, power_curve_pred, wind_nacelle_pred` |
| `outputs/forecast_all_issues.csv` | 2 880 | те же поля по всем 30 выпускам |
| `outputs/forecast_feb2026.csv` | 672 | `ts, turbine_1_lead1, turbine_2_lead1, turbine_1_lead2, turbine_2_lead2, turbine_1_p10, turbine_2_p10, turbine_1_p90, turbine_2_p90, farm_mean_lead1, farm_mean_lead2, farm_p10_lead1, farm_p90_lead1` |
| `outputs/backtest_summary.csv` | 30 | `issue_date, hours, recomputed, llm_used, storm_hours, calm_hours, rated_hours, mean_cf_t1, mean_cf_t2, mean_abs_change_vs_prev` |
| `outputs/agent_logs/<дата>.json` | — | `issue_date, llm_used, published, confidence, quality, recalculated, analysis, conclusion, trace` |

Мощность везде нормирована на 1 (доля установленной), время — местное, Алматы.
