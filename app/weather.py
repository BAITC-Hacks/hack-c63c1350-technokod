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
    """Прогноз на следующие horizon_hours часов по состоянию на КОНЕЦ суток issue_date.

    Момент выпуска — операционный дедлайн в конце дня D (23:59 местного времени),
    когда диспетчер готовит заявку на следующие сутки. Покрываются дни D+1 и D+2.

    Часы дня D+1 берутся из *_previous_day1, часы дня D+2 из *_previous_day2.
    В Previous Runs API слой previous_dayN для часа T содержит значение из прогноза,
    выпущенного за N суток до T. Отсюда: для часа T дня D+1 это прогноз от того же
    часа дня D, для часа T дня D+2 — тоже прогноз от того же часа дня D. То есть все
    использованные прогнозы выпущены в пределах суток D и известны к 23:59 этого дня.

    Важно: это НЕ состояние на полночь дня D. Прогноз на 23:00 дня D+1 опирается на
    выпуск примерно 23:00 дня D. Поэтому момент выпуска определён как конец суток,
    а не их начало — иначе часть данных оказалась бы из будущего относительно метки.
    """
    issue = pd.Timestamp(issue_date).normalize()
    start = issue + pd.Timedelta(days=1)
    end = start + pd.Timedelta(hours=horizon_hours - 1)
    raw = previous_runs(site, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    raw = raw[(raw["ts"] >= start) & (raw["ts"] <= end)].copy()
    rows = []
    for _, r in raw.iterrows():
        lead = 1 if r["ts"] < start + pd.Timedelta(hours=24) else 2
        row = {"ts": r["ts"], "issue_date": issue, "lead_day": lead}  # issue_date — сутки выпуска, момент = их конец
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


def previous_runs_training(site: Site, start: str = "2024-02-16", end: str = "2026-01-31") -> pd.DataFrame:
    """Обучающая погода из Previous Runs: на каждый час два примера, прогноз за 1 и за 2 дня.
    Распределение признаков совпадает с тем, что модель увидит в тесте."""
    frames = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        y0 = max(start, f"{year}-01-01")
        y1 = min(end, f"{year}-12-31")
        frames.append(previous_runs(site, y0, y1))
    pr = pd.concat(frames, ignore_index=True).drop_duplicates("ts")
    rows = []
    for lead in (1, 2):
        d = pr[["ts"]].copy()
        for v in BASE_VARS:
            d[v] = pr[f"{v}_previous_day{lead}"]
        d["lead_day"] = lead
        rows.append(d)
    return pd.concat(rows, ignore_index=True).dropna(subset=["wind_speed_100m"]).reset_index(drop=True)
