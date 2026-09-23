"""Загрузка данных ВЭС организаторов и агрегация в часовой ряд."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
COLUMNS = ["id", "ts", "wind_ms", "power_norm", "temp_c"]


def load_turbine(turbine: int) -> pd.DataFrame:
    """Прочитать 10-минутный CSV турбины (1 или 2) в стандартные колонки."""
    path = RAW_DIR / f"turbine_{turbine}.csv"
    df = pd.read_csv(path)
    df.columns = COLUMNS
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.drop(columns=["id"]).sort_values("ts").drop_duplicates("ts")
    return df.reset_index(drop=True)


def to_hourly(df: pd.DataFrame, min_points: int = 3) -> pd.DataFrame:
    """Среднее за час. Часы, где меньше min_points десятиминуток, помечаются как пропуск."""
    g = df.set_index("ts").resample("1h")
    out = g[["wind_ms", "power_norm", "temp_c"]].mean()
    cnt = g["power_norm"].count()
    out.loc[cnt < min_points, ["wind_ms", "power_norm", "temp_c"]] = float("nan")
    out["n_points"] = cnt
    return out.reset_index()


def load_hourly(turbine: int) -> pd.DataFrame:
    return to_hourly(load_turbine(turbine))
