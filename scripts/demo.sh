#!/usr/bin/env bash
# Контрольный сценарий для жюри (Linux/macOS/Git Bash).
# Проверка окружения -> обучение (только если моделей нет) -> выпуск прогноза -> бэктест февраля -> сводка -> тесты.
# Сеть и ключи не нужны: архивные прогнозы погоды лежат в data/cache/weather.
set -euo pipefail
cd "$(dirname "$0")/.."

# интерпретатор: переменная PYTHON, затем виртуальное окружение проекта, затем python из PATH
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

echo "================================================================"
echo " Контрольный сценарий WindAgent"
echo " Интерпретатор: $PY  ($("$PY" -V 2>&1))"
if [ ! -f "models/turbine_1.joblib" ]; then
  echo " Модели не обучены: шаг 2 выполнит обучение, это займёт около 4 минут."
else
  echo " Модели уже обучены: шаг 2 будет пропущен."
fi
echo " Сеть не требуется, ключи LLM не требуются."
echo "================================================================"

echo
echo "[1/7] Проверка окружения"
"$PY" scripts/check_env.py

echo
echo "[2/7] Обучение моделей"
if [ -f "models/turbine_1.joblib" ] && [ -f "models/turbine_2.joblib" ]; then
  echo "Модели уже обучены, шаг пропущен (удалите папку models, чтобы переобучить)."
else
  echo "Моделей нет, запускаю обучение (около 4 минут)."
  "$PY" -m app.cli train
fi

echo
echo "[3/7] Выпуск прогноза на 2026-01-31 (детерминированный план, без LLM)"
"$PY" -m app.cli forecast --date 2026-01-31 --no-llm

echo
echo "[4/7] Бэктест февраля 2026: 28 последовательных выпусков"
"$PY" -m app.cli backtest --no-llm

echo
echo "[5/7] Сводка по результатам"
"$PY" -m app.cli report

echo
echo "[6/7] Автотесты"
"$PY" -m pytest -q tests

echo
if [ -n "${OPENAI_API_KEY:-}" ] || grep -qE "^OPENAI_API_KEY=.+" .env 2>/dev/null; then
  echo "[7/7] Живая оркестрация: порядок инструментов выбирает LLM"
  "$PY" -m app.cli forecast --date 2026-02-15 --llm
else
  echo "[7/7] Шаг с LLM пропущен: ключ провайдера не задан."
  echo "      Решение полностью работает без него, цикл выполняет детерминированный планировщик."
  echo "      Журналы выпусков с живой оркестрацией приложены в outputs/agent_logs_llm/."
fi

echo
echo "================================================================"
echo " Контрольный сценарий пройден. Созданные артефакты:"
echo "   outputs/forecast_feb2026.csv      — почасовой прогноз февраля, 672 часа"
echo "   outputs/forecast_all_issues.csv   — все 28 выпусков по часам"
echo "   outputs/backtest_summary.csv      — сводка по выпускам, отметки пересчёта"
echo "   outputs/forecasts/<дата>.csv      — выпуск отдельного дня, 96 строк"
echo "   outputs/agent_logs/<дата>.json    — журнал действий агента с трассой инструментов"
echo "   docs/metrics.json                 — метрики честной валидации января 2026"
echo "================================================================"
