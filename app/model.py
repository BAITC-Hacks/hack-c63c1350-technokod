"""Модель выработки: градиентный бустинг по турбине плюс кривая мощности как запасной вариант."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from app.features import FEATURES, build_features

MODEL_DIR = Path("models")


@dataclass
class Metrics:
    mae: float
    rmse: float
    nmae_pct: float
    n: int

    def as_dict(self) -> dict:
        return {"mae": round(self.mae, 4), "rmse": round(self.rmse, 4), "nmae_pct": round(self.nmae_pct, 2), "n": self.n}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return Metrics(mae=mae, rmse=rmse, nmae_pct=mae * 100, n=len(y_true))  # мощность нормирована на 1


class PowerCurve:
    """Эмпирическая кривая мощности по прогнозной скорости ветра на 100 м (бины по 0.5 м/с)."""

    def __init__(self, step: float = 0.5):
        self.step = step
        self.table: dict[float, float] = {}

    def fit(self, ws: pd.Series, power: pd.Series) -> "PowerCurve":
        bins = (ws / self.step).round() * self.step
        self.table = power.groupby(bins).median().to_dict()
        return self

    def predict(self, ws: pd.Series) -> np.ndarray:
        keys = np.array(sorted(self.table))
        vals = np.array([self.table[k] for k in keys])
        return np.interp(ws.clip(lower=0), keys, vals)


class GenerationModel:
    def __init__(self, turbine: int):
        self.turbine = turbine
        self.gbm = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=42
        )
        self.curve = PowerCurve()
        self.blend_w = 0.5  # вес GBM в смеси, подбирается на честной валидации

    def fit(self, train: pd.DataFrame) -> "GenerationModel":
        self.gbm.fit(train[FEATURES], train["power_norm"])
        self.curve.fit(train["wind_speed_100m"], train["power_norm"])
        return self

    def predict(self, weather: pd.DataFrame) -> pd.DataFrame:
        df = build_features(weather)
        pred = self.gbm.predict(df[FEATURES])
        curve = self.curve.predict(df["wind_speed_100m"])
        out = df[["ts"]].copy()
        out["power_gbm_pred"] = np.clip(pred, 0, 1)
        out["power_curve_pred"] = np.clip(curve, 0, 1)
        out["power_norm_pred"] = self.blend_w * out["power_gbm_pred"] + (1 - self.blend_w) * out["power_curve_pred"]
        if "lead_day" in df.columns:
            out["lead_day"] = df["lead_day"].values
        return out

    def save(self) -> Path:
        MODEL_DIR.mkdir(exist_ok=True)
        path = MODEL_DIR / f"turbine_{self.turbine}.joblib"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, turbine: int) -> "GenerationModel":
        return joblib.load(MODEL_DIR / f"turbine_{turbine}.joblib")


def save_metrics(name: str, payload: dict) -> None:
    path = Path("docs/metrics.json")
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[name] = payload
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
