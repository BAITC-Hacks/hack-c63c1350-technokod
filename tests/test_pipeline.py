"""Проверки основного сценария на кэшированных данных: без сети, без ключей.

Пять проверок здесь плюс `test_smoke.py::test_imports` — весь набор из шести тестов,
который обещан в README.
"""
from pathlib import Path

import pandas as pd
import pytest

from app.backtest import build_submission
from app.data import to_hourly
from app.features import FEATURES, build_features
from app.weather import FARM, forecast_available_at

MODELS_READY = Path("models/turbine_1.joblib").exists() and Path("models/turbine_2.joblib").exists()


def test_hourly_aggregation_marks_gaps():
    """Час с двумя десятиминутками — это пропуск, а не среднее по двум точкам."""
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:10", "2025-01-01 01:00"]),
            "wind_ms": [5, 6, 7],
            "power_norm": [0.1, 0.2, 0.3],
            "temp_c": [1, 1, 1],
        }
    )
    h = to_hourly(df, min_points=3)
    assert len(h) == 2
    assert h["power_norm"].isna().all()


def test_forecast_available_at_uses_only_past_runs():
    """Главное требование кейса: на дату выпуска берётся прогноз, а не факт.

    Первые сутки горизонта — прогноз, выпущенный за день (lead 1), вторые — за два дня (lead 2).
    """
    w = forecast_available_at(FARM, "2026-01-31", 48)
    assert len(w) == 48
    assert w["ts"].min() == pd.Timestamp("2026-02-01 00:00")
    assert w["ts"].max() == pd.Timestamp("2026-02-02 23:00")
    assert set(w["lead_day"]) == {1, 2}
    assert (w[w["ts"] < "2026-02-02"]["lead_day"] == 1).all()
    assert (w[w["ts"] >= "2026-02-02"]["lead_day"] == 2).all()


def test_features_complete():
    """Все признаки модели считаются из прогноза и не содержат пропусков."""
    f = build_features(forecast_available_at(FARM, "2026-01-31", 24))
    assert all(c in f.columns for c in FEATURES)
    assert f[FEATURES].isna().sum().sum() == 0


@pytest.mark.skipif(not MODELS_READY, reason="модели не обучены: python -m app.cli train")
def test_agent_day_deterministic():
    """Выпуск агента без LLM: полный цикл инструментов, 96 строк (48 часов × 2 турбины), значения 0..1."""
    from app.agent import ForecastAgent

    s = ForecastAgent().run_day("2026-02-10", use_llm=False)
    tools = [t["tool"] for t in s.trace]
    assert tools[:3] == ["get_weather", "prepare_and_predict", "analyze"]
    assert tools[-1] == "save_forecast"
    assert len(s.forecast) == 96
    assert s.forecast["ts"].nunique() == 48
    assert set(s.forecast["turbine"]) == {1, 2}
    for col in ("power_norm_pred", "power_gbm_pred", "power_curve_pred"):
        assert s.forecast[col].between(0, 1).all()
    saved = pd.read_csv(Path("outputs/forecasts/2026-02-10.csv"), parse_dates=["ts"])
    assert len(saved) == 96
    assert list(saved.columns)[:5] == ["issue_date", "ts", "horizon_hour", "lead_day", "turbine"]


def test_build_submission_covers_february_without_gaps():
    """Итоговый ряд: 672 часа февраля, ни одной пропущенной строки, lead 1 заполнен везде."""
    rows = []
    for issue in pd.date_range("2026-01-31", "2026-02-27", freq="D"):
        start = issue + pd.Timedelta(days=1)
        for h in range(48):
            ts = start + pd.Timedelta(hours=h)
            for turbine in (1, 2):
                rows.append(
                    {
                        "issue_date": issue.strftime("%Y-%m-%d"),
                        "ts": ts,
                        "horizon_hour": h + 1,
                        "lead_day": 1 if h < 24 else 2,
                        "turbine": turbine,
                        "power_norm_pred": 0.5,
                        "power_gbm_pred": 0.5,
                        "power_curve_pred": 0.5,
                    }
                )
    final = build_submission(pd.DataFrame(rows))
    assert len(final) == 672
    assert final["ts"].min() == pd.Timestamp("2026-02-01 00:00")
    assert final["ts"].max() == pd.Timestamp("2026-02-28 23:00")
    assert final["ts"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert final[["turbine_1_lead1", "turbine_2_lead1", "farm_mean_lead1"]].notna().all().all()
    # прогноза за двое суток на 1 февраля быть не может: он потребовал бы выпуск 30 января
    assert final.loc[final["ts"] < "2026-02-02", "turbine_1_lead2"].isna().all()
