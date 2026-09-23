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
        # подбор весов смеси трёх компонентов на сетке с шагом 0.1
        grid = {}
        for wg in np.arange(0, 1.01, 0.1):
            for wt in np.arange(0, 1.01 - wg, 0.1):
                wc = 1 - wg - wt
                blend = wg * m["power_gbm_pred"] + wt * m["power_two_stage_pred"] + wc * m["power_curve_pred"]
                grid[(round(float(wg), 1), round(float(wt), 1))] = evaluate(m["power_norm"].values, blend.values).mae
        (bg, bt) = min(grid, key=grid.get)
        weights = {"gbm": bg, "two_stage": bt, "curve": round(1 - bg - bt, 1)}
        model.weights = weights
        res = {"train_rows": int(len(train)), "weights": weights, "honest_jan2026": {}}
        for lead in (1, 2):
            s = m[m["lead_day"] == lead]
            blend = bg * s["power_gbm_pred"] + bt * s["power_two_stage_pred"] + (1 - bg - bt) * s["power_curve_pred"]
            res["honest_jan2026"][f"lead{lead}"] = {
                "blend": evaluate(s["power_norm"].values, blend.values).as_dict(),
                "gbm": evaluate(s["power_norm"].values, s["power_gbm_pred"].values).as_dict(),
                "two_stage": evaluate(s["power_norm"].values, s["power_two_stage_pred"].values).as_dict(),
                "curve": evaluate(s["power_norm"].values, s["power_curve_pred"].values).as_dict(),
            }
        # квантили остатков для интервала P10–P90
        from app.model import level_bin
        blend_all = bg * m["power_gbm_pred"] + bt * m["power_two_stage_pred"] + (1 - bg - bt) * m["power_curve_pred"]
        resid = pd.DataFrame({"lead": m["lead_day"].astype(int), "lvl": [level_bin(v) for v in blend_all], "r": m["power_norm"] - blend_all})
        rq = {}
        for (ld, lvl), g in resid.groupby(["lead", "lvl"]):
            if len(g) >= 20:
                rq[(int(ld), int(lvl))] = (float(g["r"].quantile(0.1)), float(g["r"].quantile(0.9)))
        res["residual_quantiles"] = {f"lead{k[0]}_lvl{k[1]}": [round(v[0], 3), round(v[1], 3)] for k, v in rq.items()}
        cover = float(((resid["r"] >= resid.apply(lambda r: rq.get((r["lead"], r["lvl"]), (-0.25, 0.25))[0], axis=1)) &
                       (resid["r"] <= resid.apply(lambda r: rq.get((r["lead"], r["lvl"]), (-0.25, 0.25))[1], axis=1))).mean())
        res["p10_p90_coverage_jan2026"] = round(cover, 3)
        final = GenerationModel(t).fit(ds)
        final.weights = weights
        final.residual_quantiles = rq
        res["model_path"] = str(final.save())
        res["train_rows_final"] = int(len(ds))
        save_metrics(f"turbine_{t}", res)
        print(f"turbine {t}: weights={weights} lead1={res['honest_jan2026']['lead1']['blend']} lead2={res['honest_jan2026']['lead2']['blend']}")


if __name__ == "__main__":
    main()
