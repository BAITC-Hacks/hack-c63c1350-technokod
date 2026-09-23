# Контрольный сценарий для жюри (Windows PowerShell).
# Проверка окружения -> обучение (только если моделей нет) -> выпуск прогноза -> бэктест февраля -> сводка -> тесты.
# Сеть и ключи не нужны: архивные прогнозы погоды лежат в data/cache/weather.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

# интерпретатор: переменная PYTHON, затем виртуальное окружение проекта, затем python из PATH
if ($env:PYTHON) {
    $PY = $env:PYTHON
} elseif (Test-Path '.venv\Scripts\python.exe') {
    $PY = '.venv\Scripts\python.exe'
} elseif (Test-Path '.venv/bin/python') {
    $PY = '.venv/bin/python'
} else {
    $PY = 'python'
}

function Invoke-Step {
    param([string]$Title, [string[]]$Arguments)
    Write-Host ''
    Write-Host $Title
    & $PY @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Шаг завершился с ошибкой (код $LASTEXITCODE): $Title"
        exit $LASTEXITCODE
    }
}

$version = & $PY -V 2>&1
Write-Host '================================================================'
Write-Host ' Контрольный сценарий WindAgent'
Write-Host " Интерпретатор: $PY  ($version)"
if (-not (Test-Path 'models\turbine_1.joblib')) {
    Write-Host ' Модели не обучены: шаг 2 выполнит обучение, это займёт около 15 секунд.'
} else {
    Write-Host ' Модели уже обучены: шаг 2 будет пропущен.'
}
Write-Host ' Сеть не требуется, ключи LLM не требуются.'
Write-Host '================================================================'

Invoke-Step '[1/7] Проверка окружения' @('scripts/check_env.py')

Write-Host ''
Write-Host '[2/7] Обучение моделей'
if ((Test-Path 'models\turbine_1.joblib') -and (Test-Path 'models\turbine_2.joblib')) {
    Write-Host 'Модели уже обучены, шаг пропущен (удалите папку models, чтобы переобучить).'
} else {
    Write-Host 'Моделей нет, запускаю обучение (около 15 секунд).'
    & $PY -m app.cli train
    if ($LASTEXITCODE -ne 0) { Write-Host "Обучение завершилось с ошибкой (код $LASTEXITCODE)"; exit $LASTEXITCODE }
}

Invoke-Step '[3/7] Выпуск прогноза на 2026-01-31 (детерминированный план, без LLM)' @('-m', 'app.cli', 'forecast', '--date', '2026-01-31', '--no-llm')
Invoke-Step '[4/7] Бэктест февраля 2026: 30 последовательных выпусков' @('-m', 'app.cli', 'backtest', '--no-llm')
Invoke-Step '[5/7] Сводка по результатам' @('-m', 'app.cli', 'report')
# --basetemp внутри .pytest_cache: системный временный каталог на машине жюри
# может быть недоступен по правам, и шаг падал бы не из-за кода
if (-not (Test-Path '.pytest_cache')) { New-Item -ItemType Directory '.pytest_cache' | Out-Null }   # pytest не создаёт родителя для --basetemp
Invoke-Step '[6/7] Автотесты (22 теста)' @('-m', 'pytest', '-q', 'tests', '--basetemp=.pytest_cache/tmp')

$hasKey = $false
if ($env:OPENAI_API_KEY) { $hasKey = $true }
elseif ((Test-Path '.env') -and (Select-String -Path '.env' -Pattern '^OPENAI_API_KEY=.+' -Quiet)) { $hasKey = $true }
if ($hasKey) {
    Invoke-Step '[7/7] Живая оркестрация: порядок инструментов выбирает LLM' @('-m', 'app.cli', 'forecast', '--date', '2026-02-15', '--llm')
    # канонические артефакты — детерминированные: возвращаем журнал и выпуск после прогона с LLM
    & $PY -m app.cli forecast --date 2026-02-15 --no-llm | Out-Null
    Write-Host '      Канонический журнал 2026-02-15 восстановлен детерминированным прогоном.'
    Write-Host '      Журнал прогона с LLM по этой дате: outputs/agent_logs_llm/2026-02-15.json'
} else {
    Write-Host ''
    Write-Host '[7/7] Шаг с LLM пропущен: ключ провайдера не задан.'
    Write-Host '      Решение полностью работает без него, цикл выполняет детерминированный планировщик.'
    Write-Host '      Журналы выпусков с живой оркестрацией приложены в outputs/agent_logs_llm/.'
}

Write-Host ''
Write-Host '================================================================'
Write-Host ' Контрольный сценарий пройден. Созданные артефакты:'
Write-Host '   outputs/forecast_feb2026.csv      — почасовой прогноз февраля, 672 часа'
Write-Host '   outputs/forecast_all_issues.csv   — все 30 выпусков по часам, 2880 строк'
Write-Host '   outputs/backtest_summary.csv      — сводка по 30 выпускам, отметки пересчёта'
Write-Host '   outputs/forecasts/<дата>.csv      — выпуск отдельного дня, 96 строк'
Write-Host '   outputs/agent_logs/<дата>.json    — журнал действий агента с трассой инструментов'
Write-Host '   docs/metrics.json                 — метрики честной валидации января 2026'
Write-Host '================================================================'
