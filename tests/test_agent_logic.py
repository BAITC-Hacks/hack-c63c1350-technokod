"""Проверки логики, которую правили после первой сборки: сборка итогового ряда,
буревая защита, откат при отказе LLM и границы интервала неопределённости.

Все проверки работают без сети и без ключей: архивные прогнозы берутся из кэша
репозитория. Выпуски пишутся во временный каталог, артефакты в outputs не трогаются.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from app.weather import FARM, forecast_available_at

MODELS_READY = Path("models/turbine_1.joblib").exists() and Path("models/turbine_2.joblib").exists()
needs_models = pytest.mark.skipif(not MODELS_READY, reason="модели не обучены: python -m app.cli train")


def test_submission_covers_february_grid():
    """Итоговый ряд февраля — ровно 672 часа по явной сетке; для 1 февраля выпуска
    за двое суток не существует, он потребовал бы прогноза от 30 января."""
    from app.backtest import build_submission

    hours = pd.date_range("2026-02-01 00:00", "2026-02-02 23:00", freq="h")
    rows = []
    for t in (1, 2):
        for i, ts in enumerate(hours):
            lead = 1 if ts < pd.Timestamp("2026-02-02") else 2
            rows.append({"issue_date": "2026-01-31", "ts": ts, "lead_day": lead, "turbine": t,
                         "power_norm_pred": 0.4 + 0.01 * t, "power_p10": 0.2, "power_p90": 0.6})
    out = build_submission(pd.DataFrame(rows))

    assert len(out) == 672
    assert out["ts"].iloc[0] == pd.Timestamp("2026-02-01 00:00")
    assert out["ts"].iloc[-1] == pd.Timestamp("2026-02-28 23:00")
    assert out["ts"].is_monotonic_increasing and not out["ts"].duplicated().any()
    first_day = out[out["ts"] < "2026-02-02"]
    assert first_day["turbine_1_lead1"].notna().all(), "часы первых суток закрыты выпуском за сутки"
    assert first_day["turbine_1_lead2"].isna().all(), "выпуска за двое суток на 1 февраля быть не может"


@needs_models
def test_storm_zeroes_every_power_column(isolated_outputs, monkeypatch):
    """При штормовом ветре обнуляется весь прогноз, а не только итоговая смесь:
    иначе компоненты и интервал противоречат выпущенному числу."""
    import app.agent as agent

    storm = forecast_available_at(FARM, "2026-02-10", 48).copy()
    # выше порога отключения 22 м/с, но переменный: константа была бы отбракована
    # контролем качества как застрявшее значение датчика
    storm["wind_speed_100m"] = 25.0 + np.linspace(0, 3, len(storm))
    monkeypatch.setattr(agent, "forecast_available_at", lambda *a, **k: storm.copy())

    s = agent.ForecastAgent().run_day("2026-02-10", use_llm=False)

    assert s.analysis["storm_hours"] == 48
    assert s.recalculated is True
    power_cols = [c for c in s.forecast.columns if c.startswith("power_")]
    assert len(power_cols) >= 5
    assert (s.forecast[power_cols] == 0.0).all().all(), "остались ненулевые компоненты прогноза"


@needs_models
def test_llm_failure_falls_back_to_deterministic_plan(isolated_outputs, monkeypatch):
    """Отказ провайдера не срывает выпуск: цикл доводит детерминированный планировщик,
    факт отката виден в трассе."""
    import app.llm as llm
    from app.agent import ForecastAgent

    monkeypatch.setattr(llm, "available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("429 rate limit")

    monkeypatch.setattr(llm, "chat_tools", boom)

    s = ForecastAgent().run_day("2026-02-10", use_llm=True)

    tools = [t["tool"] for t in s.trace]
    assert tools[0] == "llm_fallback" and "429" in s.trace[0]["result"]["error"]
    assert tools[1:] == ["get_weather", "check_input_quality", "prepare_and_predict", "analyze", "save_forecast"]
    assert s.llm_used is False
    assert len(s.forecast) == 96 and s.conclusion


@needs_models
def test_interval_bounds_and_widens_with_lead(isolated_outputs):
    """P10 не выше прогноза, P90 не ниже, и на вторые сутки интервал шире:
    квантили берутся по фактическому лиду, а не по суточному для всего горизонта."""
    from app.agent import ForecastAgent

    s = ForecastAgent().run_day("2026-02-10", use_llm=False)
    f = s.forecast

    assert (f["power_p10"] <= f["power_norm_pred"] + 1e-9).all()
    assert (f["power_p90"] >= f["power_norm_pred"] - 1e-9).all()
    bounds = f[["power_p10", "power_p90"]]
    assert ((bounds >= 0) & (bounds <= 1)).all().all()

    width = (f["power_p90"] - f["power_p10"]).groupby(f["lead_day"]).mean()
    assert width[2] > width[1], f"интервал на 48 часах не шире суточного: {width.to_dict()}"
