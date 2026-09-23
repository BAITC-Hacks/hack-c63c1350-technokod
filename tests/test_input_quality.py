"""Контроль качества входных данных: вердикты и поведение выпуска.

Сценарии повторяют `scripts/degradation_experiment.py`. Смысл проверок: на исправных данных
поведение прежнее, на испорченных выпуск не публикуется молча.
Все прогоны идут во временный каталог через фикстуру isolated_outputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import app.agent as agent
from app.agent import VERDICT_FAIL, VERDICT_OK, VERDICT_WARN, AgentState, ForecastAgent
from app.weather import FARM, forecast_available_at

DAY = "2026-02-10"
MODELS_READY = Path("models/turbine_1.joblib").exists() and Path("models/turbine_2.joblib").exists()
needs_models = pytest.mark.skipif(not MODELS_READY, reason="модели не обучены: python -m app.cli train")


@pytest.fixture(scope="module")
def clean_weather() -> pd.DataFrame:
    return forecast_available_at(FARM, DAY, 48)


def quality_of(weather: pd.DataFrame, horizon: int = 48) -> dict:
    """Вердикт контроля качества для подготовленного кадра, без запуска модели."""
    a = ForecastAgent.__new__(ForecastAgent)  # модель не нужна: проверяется только вход
    a.state = AgentState(issue_date=DAY, horizon_hours=horizon)
    a.state.weather = weather
    return a.check_input_quality()


def test_clean_input_is_ok(clean_weather):
    q = quality_of(clean_weather)
    assert q["verdict"] == VERDICT_OK
    assert q["findings"] == []
    assert q["publish_allowed"] is True


@pytest.mark.parametrize("name,code", [
    ("missing", "missing_values"),
    ("stuck", "stuck_sensor"),
    ("negative", "negative_wind"),
    ("truncated", "incomplete_horizon"),
    ("over_range", "wind_out_of_range"),
])
def test_degradations_are_detected(clean_weather, name, code):
    w = clean_weather.copy()
    if name == "missing":
        w.loc[10:22, "wind_speed_100m"] = np.nan
    elif name == "stuck":
        w["wind_speed_100m"] = float(w["wind_speed_100m"].iloc[0])
    elif name == "negative":
        w["wind_speed_100m"] = -w["wind_speed_100m"]
    elif name == "truncated":
        w = w.iloc[:20]
    elif name == "over_range":
        w.loc[5, "wind_speed_100m"] = 120.0

    q = quality_of(w)
    assert q["verdict"] == VERDICT_FAIL
    assert code in [f["code"] for f in q["findings"]]
    assert q["publish_allowed"] is False


def test_low_variability_is_warning_not_failure(clean_weather):
    """Ровный, но не застрявший ряд: выпуск возможен с пониженной достоверностью."""
    w = clean_weather.copy()
    base = float(w["wind_speed_100m"].mean())
    # мелкий шум: разброс ниже порога, но одинаковых значений подряд нет
    w["wind_speed_100m"] = base + np.linspace(0, 0.1, len(w))
    q = quality_of(w)
    assert q["verdict"] == VERDICT_WARN
    assert "low_variability" in [f["code"] for f in q["findings"]]
    assert q["publish_allowed"] is True


@needs_models
def test_clean_day_publishes_as_before(isolated_outputs, clean_weather):
    """Исправные данные: поведение прежнее — выпуск на 96 строк и файл на диске."""
    s = ForecastAgent().run_day(DAY, use_llm=False)
    assert s.quality["verdict"] == VERDICT_OK
    assert s.published is True
    assert s.confidence == "обычная"
    assert len(s.forecast) == 96
    assert (agent.OUT_DIR / f"{DAY}.csv").exists()
    assert [t["tool"] for t in s.trace][:3] == ["get_weather", "check_input_quality", "prepare_and_predict"]


@needs_models
def test_failed_input_blocks_publication(isolated_outputs, clean_weather, monkeypatch):
    """Вердикт fail: файла выпуска нет, отказ зафиксирован в журнале."""
    broken = clean_weather.copy()
    broken["wind_speed_100m"] = float(broken["wind_speed_100m"].iloc[0])  # застрявший датчик
    monkeypatch.setattr(agent, "forecast_available_at", lambda *a, **k: broken.copy())

    s = ForecastAgent().run_day(DAY, use_llm=False)

    assert s.quality["verdict"] == VERDICT_FAIL
    assert s.published is False
    assert not (agent.OUT_DIR / f"{DAY}.csv").exists(), "выпуск по негодным данным не должен публиковаться"
    assert "не состоялся" in s.conclusion
    log = agent.LOG_DIR / f"{DAY}.json"
    assert log.exists(), "отказ обязан попасть в журнал наравне с состоявшимся выпуском"


@needs_models
def test_nan_forecast_is_never_saved(isolated_outputs, clean_weather, monkeypatch):
    """Прогноз с пропусками мощности не сохраняется даже при обходе контроля входа."""
    gapped = clean_weather.copy()
    gapped.loc[10:22, "wind_speed_100m"] = np.nan
    monkeypatch.setattr(agent, "forecast_available_at", lambda *a, **k: gapped.copy())

    a = ForecastAgent()
    a.state = AgentState(issue_date=DAY, horizon_hours=48)
    a.get_weather()
    a.prepare_and_predict()  # контроль качества сознательно пропущен
    result = a.save_forecast()

    assert result["published"] is False
    assert result["nan_rows"] > 0
    assert not (agent.OUT_DIR / f"{DAY}.csv").exists()


@needs_models
def test_missing_previous_issue_is_explicit(isolated_outputs):
    """Отсутствие вчерашнего выпуска — явный признак в анализе, а не пропущенное поле."""
    s = ForecastAgent().run_day(DAY, previous=None, use_llm=False)
    assert s.analysis["previous_issue_available"] is False
    assert s.analysis["previous_issue_source"] == "отсутствует"
    assert "не удалось" in s.conclusion


@needs_models
def test_recalculation_records_numeric_trigger(isolated_outputs):
    """Причина пересчёта фиксируется числами, а не общей фразой."""
    a = ForecastAgent(use_llm=False)
    prev = a.run_day("2026-02-14", use_llm=False).forecast.copy()
    s = a.run_day("2026-02-15", previous=prev, use_llm=False)

    rec = [t["result"] for t in s.trace if t["tool"] == "recalculate"]
    assert rec, "на 15 февраля расхождение с прошлым выпуском выше порога, пересчёт обязан сработать"
    r = rec[0]
    assert r["trigger"] == "input_update"
    assert "при пороге" in r["trigger_detail"]
    assert r["trigger_values"]["mean_abs_change"] > r["trigger_values"]["threshold"]
    assert r["input_changed"] is False and "архив" in r["note"]
