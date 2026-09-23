"""Проверки основного сценария на кэшированных данных (без сети и ключей)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from app.data import load_hourly, to_hourly
from app.features import build_features, FEATURES
from app.weather import FARM, forecast_available_at


def test_hourly_aggregation_marks_gaps():
    df = pd.DataFrame({"ts": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:10", "2025-01-01 01:00"]),
                       "wind_ms": [5, 6, 7], "power_norm": [0.1, 0.2, 0.3], "temp_c": [1, 1, 1]})
    h = to_hourly(df, min_points=3)
    assert len(h) == 2 and h["power_norm"].isna().all()


def test_forecast_available_at_uses_only_past_runs():
    w = forecast_available_at(FARM, "2026-01-31", 48)
    assert len(w) == 48
    assert w["ts"].min() == pd.Timestamp("2026-02-01 00:00")
    assert set(w["lead_day"]) == {1, 2}
    assert (w[w["ts"] < "2026-02-02"]["lead_day"] == 1).all()


def test_features_complete():
    w = forecast_available_at(FARM, "2026-01-31", 24)
    f = build_features(w)
    assert all(c in f.columns for c in FEATURES)
    assert f[FEATURES].isna().sum().sum() == 0


@pytest.mark.skipif(not Path("models/turbine_1.joblib").exists(), reason="модели не обучены: python -m app.cli train")
def test_agent_day_deterministic(isolated_outputs):
    from app.agent import ForecastAgent
    s = ForecastAgent().run_day("2026-02-10", use_llm=False)
    assert [t["tool"] for t in s.trace][:3] == ["get_weather", "prepare_and_predict", "analyze"]
    assert s.forecast["power_norm_pred"].between(0, 1).all()
    assert s.forecast["ts"].nunique() == 48


def test_bad_date_rejected():
    from app.agent import ForecastAgent
    with pytest.raises(Exception):
        forecast_available_at(FARM, "не дата", 24)
