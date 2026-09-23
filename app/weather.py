"""Инструмент погоды: архивные прогнозы Open-Meteo по координатам ВЭС.

Previous Runs API: для каждого часа отдаёт значение прогноза, выданного за N дней до него
(поля *_previous_dayN). Это прогноз, который был доступен на момент прогнозирования.
Historical Forecast API: непрерывный архив прогнозных моделей, используется для обучения
на периоде, который Previous Runs не покрывает.
Все ответы кэшируются в data/cache/weather/, повторные запуски работают без сети.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

CACHE_DIR = Path("data/cache/weather")
TZ = "Asia/Almaty"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

BASE_VARS = [
    "wind_speed_100m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_100m",
    "temperature_2m",
    "surface_pressure",
]
KMH_VARS = {"wind_speed_100m", "wind_speed_10m", "wind_gusts_10m"}


@dataclass(frozen=True)
class Site:
    name: str
    lat: float
    lon: float


# Координаты из ТЗ. Турбины в 400 м друг от друга, для погоды используется середина.
TURBINE_1 = Site("turbine_1", 43.645150, 78.535604)
TURBINE_2 = Site("turbine_2", 43.643198, 78.538828)
FARM = Site("farm", (TURBINE_1.lat + TURBINE_2.lat) / 2, (TURBINE_1.lon + TURBINE_2.lon) / 2)


def _cached_get(url: str, params: dict) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps({"u": url, "p": params}, sort_keys=True).encode()).hexdigest()[:24]
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with httpx.Client(timeout=60) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    if "error" in data and data.get("error"):
        raise RuntimeError(f"Open-Meteo: {data.get('reason')}")
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _to_frame(data: dict) -> pd.DataFrame:
    h = data["hourly"]
    df = pd.DataFrame(h)
    df["ts"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    for col in df.columns:
        base = col.split("_previous_day")[0]
        if base in KMH_VARS:
            df[col] = df[col] / 3.6  # км/ч -> м/с
    return df


def previous_runs(site: Site, start: str, end: str, lead_days: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    """Архивные прогнозы, доступные за lead_days дней до каждого часа. Один запрос до 1 года."""
    hourly = [f"{v}_previous_day{d}" for d in lead_days for v in BASE_VARS]
    params = {
        "latitude": site.lat,
        "longitude": site.lon,
        "hourly": ",".join(hourly),
        "start_date": start,
        "end_date": end,
        "timezone": TZ,
    }
    return _to_frame(_cached_get(PREVIOUS_RUNS_URL, params))


def historical_forecast(site: Site, start: str, end: str) -> pd.DataFrame:
    """Архив прогнозных моделей (лид 0–24 ч) для обучения на периоде без Previous Runs."""
    params = {
        "latitude": site.lat,
        "longitude": site.lon,
        "hourly": ",".join(BASE_VARS),
        "start_date": start,
        "end_date": end,
        "timezone": TZ,
    }
    return _to_frame(_cached_get(HISTORICAL_FORECAST_URL, params))


def forecast_available_at(site: Site, issue_date: str, horizon_hours: int = 48) -> pd.DataFrame:
    """Прогноз на следующие horizon_hours часов, каким он был на дату issue_date.

    Часы 0–24 после issue_date берутся из *_previous_day1, часы 24–48 из *_previous_day2:
    оба слоя содержат значения из прогнозов, выпущенных не позже issue_date.
    """
    issue = pd.Timestamp(issue_date).normalize()
    start = issue + pd.Timedelta(days=1)
    end = start + pd.Timedelta(hours=horizon_hours - 1)
    raw = previous_runs(site, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    raw = raw[(raw["ts"] >= start) & (raw["ts"] <= end)].copy()
    rows = []
    for _, r in raw.iterrows():
        lead = 1 if r["ts"] < start + pd.Timedelta(hours=24) else 2
        row = {"ts": r["ts"], "issue_date": issue, "lead_day": lead}
        for v in BASE_VARS:
            row[v] = r[f"{v}_previous_day{lead}"]
        rows.append(row)
    return pd.DataFrame(rows)


def training_weather(site: Site, start: str, end: str) -> pd.DataFrame:
    """Погодные признаки для обучения по годам (кэшируются по одному запросу на год)."""
    frames = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        y0 = max(start, f"{year}-01-01")
        y1 = min(end, f"{year}-12-31")
        frames.append(historical_forecast(site, y0, y1))
    return pd.concat(frames, ignore_index=True).drop_duplicates("ts")
