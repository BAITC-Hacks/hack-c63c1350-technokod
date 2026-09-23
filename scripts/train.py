"""Обучение моделей по турбинам.

Обучающая выборка: факт выработки (часы) + архивные прогнозы Previous Runs, выпущенные за 1 и 2 дня
(с 16.02.2024, раньше архива нет). Признаки в обучении распределены так же, как в тесте.
Честная валидация: январь 2026, прогнозы «как в прошлом» на 48 часов каждый день.
Вес смеси GBM и кривой мощности подбирается на валидации, затем модель дообучается на всех данных.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from app.data import load_hourly
from app.features import make_training_set
from app.model import GenerationModel, evaluate, save_metrics
from app.weather import FARM, forecast_available_at, previous_runs_training

TRAIN_END = "2025-12-31 23:00:00"
VAL_DAYS = pd.date_range("2025-12-31", "2026-01-29", freq="D")


def honest_predictions(model: GenerationModel, hourly: pd.DataFrame) -> pd.DataFrame:
    preds = [model.predict(forecast_available_at(FARM, d.strftime("%Y-%m-%d"), 48)) for d in VAL_DAYS]
    p = pd.concat(preds, ignore_index=True)
    return p.merge(hourly[["ts", "power_norm"]], on="ts").dropna(subset=["power_norm"])


def main() -> None:
    weather = previous_runs_training(FARM)
    for t in (1, 2):
        hourly = load_hourly(t)
        ds = make_training_set(hourly, weather)
        train = ds[ds["ts"] <= TRAIN_END]
        model = GenerationModel(t).fit(train)
        m = honest_predictions(model, hourly)
        # подбор веса смеси
        grid = {}
        for w in np.linspace(0, 1, 11):
            blend = w * m["power_gbm_pred"] + (1 - w) * m["power_curve_pred"]
            grid[round(float(w), 1)] = round(evaluate(m["power_norm"].values, blend.values).mae, 4)
        best_w = min(grid, key=grid.get)
        model.blend_w = best_w
        res = {"train_rows": int(len(train)), "blend_w_gbm": best_w, "val_mae_by_w": grid, "honest_jan2026": {}}
        for lead in (1, 2):
            s = m[m["lead_day"] == lead]
            blend = best_w * s["power_gbm_pred"] + (1 - best_w) * s["power_curve_pred"]
            res["honest_jan2026"][f"lead{lead}"] = {
                "blend": evaluate(s["power_norm"].values, blend.values).as_dict(),
                "gbm": evaluate(s["power_norm"].values, s["power_gbm_pred"].values).as_dict(),
                "curve": evaluate(s["power_norm"].values, s["power_curve_pred"].values).as_dict(),
            }
        final = GenerationModel(t).fit(ds)
        final.blend_w = best_w
        res["model_path"] = str(final.save())
        res["train_rows_final"] = int(len(ds))
        save_metrics(f"turbine_{t}", res)
        print(f"turbine {t}: w_gbm={best_w} lead1={res['honest_jan2026']['lead1']['blend']} lead2={res['honest_jan2026']['lead2']['blend']}")


if __name__ == "__main__":
    main()
