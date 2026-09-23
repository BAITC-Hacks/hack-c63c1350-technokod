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


def level_bin(pred: float) -> int:
    """Уровень прогноза для таблицы остатков: 0 (<0.2), 1 (<0.5), 2 (<0.8), 3 (>=0.8)."""
    return 0 if pred < 0.2 else 1 if pred < 0.5 else 2 if pred < 0.8 else 3


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
    """Три компонента и их смесь:
    - curve: кривая мощности по прогнозной скорости ветра на 100 м;
    - gbm: бустинг признаки прогноза -> мощность;
    - two_stage: бустинг признаки прогноза -> скорость ветра на гондоле, затем измеренная кривая
      мощности турбины (ветер гондолы -> мощность), которая почти детерминирована.
    Веса смеси подбираются на честной валидации (scripts/train.py)."""

    def __init__(self, turbine: int):
        self.turbine = turbine
        self.gbm = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=42
        )
        self.wind_gbm = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, random_state=42
        )
        self.curve = PowerCurve()
        self.nacelle_curve = PowerCurve(step=0.25)
        self.weights = {"gbm": 0.0, "two_stage": 0.5, "curve": 0.5}
        # квантили остатков (факт - прогноз) по лиду и уровню прогноза, из честной валидации
        self.residual_quantiles: dict[tuple[int, int], tuple[float, float]] = {}

    def fit(self, train: pd.DataFrame) -> "GenerationModel":
        self.gbm.fit(train[FEATURES], train["power_norm"])
        self.curve.fit(train["wind_speed_100m"], train["power_norm"])
        ok = train.dropna(subset=["wind_ms"])
        self.wind_gbm.fit(ok[FEATURES], ok["wind_ms"])
        self.nacelle_curve.fit(ok["wind_ms"], ok["power_norm"])
        return self

    def predict(self, weather: pd.DataFrame) -> pd.DataFrame:
        df = build_features(weather)
        pred = self.gbm.predict(df[FEATURES])
        curve = self.curve.predict(df["wind_speed_100m"])
        wind_nac = pd.Series(self.wind_gbm.predict(df[FEATURES])).clip(lower=0)
        two_stage = self.nacelle_curve.predict(wind_nac)
        out = df[["ts"]].copy()
        out["power_gbm_pred"] = np.clip(pred, 0, 1)
        out["power_curve_pred"] = np.clip(curve, 0, 1)
        out["power_two_stage_pred"] = np.clip(two_stage, 0, 1)
        out["wind_nacelle_pred"] = wind_nac.values
        w = self.weights
        out["power_norm_pred"] = (
            w["gbm"] * out["power_gbm_pred"] + w["two_stage"] * out["power_two_stage_pred"] + w["curve"] * out["power_curve_pred"]
        )
        lead = out["lead_day"] if "lead_day" in out.columns else pd.Series(1, index=out.index)
        lo, hi = [], []
        for pred, ld in zip(out["power_norm_pred"], lead):
            q10, q90 = self.residual_quantiles.get((int(ld), level_bin(pred)), (-0.25, 0.25))
            lo.append(min(max(pred + q10, 0.0), 1.0))
            hi.append(min(max(pred + q90, 0.0), 1.0))
        out["power_p10"], out["power_p90"] = lo, hi
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
