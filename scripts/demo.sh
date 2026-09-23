#!/usr/bin/env bash
# Контрольный сценарий для жюри: обучение, прогноз на одну дату, бэктест февраля, сводка.
set -e
cd "$(dirname "$0")/.."
python scripts/check_env.py
python -m app.cli train
python -m app.cli forecast --date 2026-01-31 --no-llm
python -m app.cli backtest --no-llm
python -m app.cli report
