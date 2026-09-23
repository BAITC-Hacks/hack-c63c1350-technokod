"""Данные блока доверия: факт января 2026 против прогноза, доступного накануне.

Для каждой даты выпуска 31.12.2025 … 29.01.2026 берётся архивный прогноз, который
существовал на эту дату, из него — первые сутки горизонта (лид 1), и сопоставляется
с фактической выработкой. Факт используется только для сравнения постфактум,
в прогноз он не входит.

ВАЖНО про модель. Сохранённая в models/ модель дообучена на всех данных, включая
январь 2026 (так делает scripts/train.py перед сохранением), поэтому предсказывать
ею январь — значит мерить на своих же данных: выходит 11.7 % вместо честных 14.31 %.
Здесь, как и при подсчёте метрик, обучается отдельная отложенная модель на данных
до 31.12.2025, а веса смеси и квантили интервала берутся из сохранённой модели —
они и были подобраны на этой же честной валидации.

Результат — docs/demo/validation-data.js (window.VALID), подключается обычным
<script>, поэтому страница работает и по file://.

Сеть не нужна: архивные прогнозы лежат в data/cache/weather.
Запуск: python scripts/make_validation_data.py  (около 20 секунд)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from app.data import load_hourly
from app.features import make_training_set
from app.model import GenerationModel
from app.weather import FARM, forecast_available_at, previous_runs_training

OUT = Path("docs/demo/validation-data.js")
DAYS = pd.date_range("2025-12-31", "2026-01-29", freq="D")
TRAIN_END = "2025-12-31 23:00:00"
TURBINES = (1, 2)


def holdout_models() -> dict[int, GenerationModel]:
    """Модели, не видевшие января 2026: обучение до 31.12.2025, как в scripts/train.py."""
    weather = previous_runs_training(FARM)
    out = {}
    for t in TURBINES:
        ds = make_training_set(load_hourly(t), weather)
        m = GenerationModel(t).fit(ds[ds["ts"] <= TRAIN_END])
        final = GenerationModel.load(t)  # веса и квантили подобраны на этой же валидации
        m.weights = final.weights
        m.residual_quantiles = final.residual_quantiles
        out[t] = m
    return out


def _round(v, n=4):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), n)


def main() -> None:
    models = holdout_models()
    fact = {t: load_hourly(t)[["ts", "power_norm"]].rename(columns={"power_norm": f"fact_{t}"}) for t in TURBINES}

    frames = []
    for d in DAYS:
        day = d.strftime("%Y-%m-%d")
        weather = forecast_available_at(FARM, day, 48)
        for t in TURBINES:
            p = models[t].predict(weather)
            p["turbine"] = t
            p["issue_date"] = day
            frames.append(p[["ts", "lead_day", "turbine", "issue_date", "power_norm_pred", "power_p10", "power_p90"]])
    pred = pd.concat(frames, ignore_index=True)

    # nMAE по обоим горизонтам — для сверки с docs/metrics.json
    check = {}
    for t in TURBINES:
        sub = pred[pred["turbine"] == t].merge(fact[t], on="ts").dropna(subset=[f"fact_{t}"])
        for lead in (1, 2):
            s = sub[sub["lead_day"] == lead]
            mae = float((s["power_norm_pred"] - s[f"fact_{t}"]).abs().mean())
            check[f"turbine_{t}_lead{lead}"] = round(mae * 100, 2)

    # ряд по станции: только лид 1, среднее двух турбин
    lead1 = pred[pred["lead_day"] == 1]
    wide = lead1.pivot_table(index="ts", columns="turbine", values="power_norm_pred", aggfunc="last")
    lo = lead1.pivot_table(index="ts", columns="turbine", values="power_p10", aggfunc="last")
    hi = lead1.pivot_table(index="ts", columns="turbine", values="power_p90", aggfunc="last")
    f = fact[1].merge(fact[2], on="ts").set_index("ts")
    idx = wide.index.intersection(f.index)
    farm_pred = ((wide.loc[idx, 1] + wide.loc[idx, 2]) / 2)
    farm_fact = ((f.loc[idx, "fact_1"] + f.loc[idx, "fact_2"]) / 2)
    ok = farm_fact.notna()
    idx, farm_pred, farm_fact = idx[ok], farm_pred[ok], farm_fact[ok]

    payload = {
        "ts": [pd.Timestamp(v).strftime("%Y-%m-%d %H:%M") for v in idx],
        "fact": [_round(v) for v in farm_fact],
        "pred": [_round(v) for v in farm_pred],
        "p10": [_round(v) for v in ((lo.loc[idx, 1] + lo.loc[idx, 2]) / 2)],
        "p90": [_round(v) for v in ((hi.loc[idx, 1] + hi.loc[idx, 2]) / 2)],
        "nmae_check": check,
        "period": "2026-01-01 … 2026-01-30",
        "note": "прогноз на сутки вперёд, выпуск накануне; факт используется только для сравнения постфактум",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.VALID = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")

    metrics = json.loads(Path("docs/metrics.json").read_text(encoding="utf-8"))
    print(f"{OUT}: {len(payload['ts'])} часов, {OUT.stat().st_size // 1024} КБ")
    print("nMAE, посчитанный здесь / записанный в docs/metrics.json:")
    for t in TURBINES:
        for lead in (1, 2):
            mine = check[f"turbine_{t}_lead{lead}"]
            theirs = metrics[f"turbine_{t}"]["honest_jan2026"][f"lead{lead}"]["blend"]["nmae_pct"]
            mark = "совпало" if abs(mine - theirs) < 0.02 else "РАСХОЖДЕНИЕ"
            print(f"  турбина {t}, лид {lead}: {mine:.2f} / {theirs:.2f} — {mark}")


if __name__ == "__main__":
    main()
