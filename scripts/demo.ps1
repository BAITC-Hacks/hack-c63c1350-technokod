# Контрольный сценарий для жюри (Windows PowerShell)
Set-Location (Join-Path $PSScriptRoot "..")
python scripts/check_env.py
python -m app.cli train
python -m app.cli forecast --date 2026-01-31 --no-llm
python -m app.cli backtest --no-llm
python -m app.cli report
