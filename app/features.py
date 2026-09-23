"""Признаки для модели выработки из прогноза погоды и календаря."""
from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER_COLS = [
    "wind_speed_100m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_100m",
    "temperature_2m",
    "surface_pressure",
]

FEATURES = [
    "wind_speed_100m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "temperature_2m",
    "surface_pressure",
    "ws100_sq",
    "ws100_cube",
    "shear",
    "dir_sin",
    "dir_cos",
    "air_density",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "lead_day",
]


def build_features(weather: pd.DataFrame) -> pd.DataFrame:
    """weather: колонки ts + WEATHER_COLS (+ lead_day опционально)."""
    df = weather.copy()
    ws = df["wind_speed_100m"].clip(lower=0)
    df["ws100_sq"] = ws**2
    df["ws100_cube"] = ws**3
    df["shear"] = (df["wind_speed_100m"] - df["wind_speed_10m"]).fillna(0)
    rad = np.deg2rad(df["wind_direction_100m"].fillna(0))
    df["dir_sin"] = np.sin(rad)
    df["dir_cos"] = np.cos(rad)
    # плотность воздуха по давлению и температуре (идеальный газ), кг/м3
    p_pa = df["surface_pressure"].fillna(950) * 100
    t_k = df["temperature_2m"].fillna(10) + 273.15
    df["air_density"] = p_pa / (287.05 * t_k)
    hour = df["ts"].dt.hour
    doy = df["ts"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    if "lead_day" not in df.columns:
        df["lead_day"] = 1
    return df


def make_training_set(hourly: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Стыковка факта выработки с прогнозной погодой по часу. Часы без факта отбрасываются."""
    df = hourly.merge(weather, on="ts", how="inner")
    df = df.dropna(subset=["power_norm", "wind_speed_100m"])
    return build_features(df)
