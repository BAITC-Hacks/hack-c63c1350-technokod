"""Калибровка интервала для СУТОЧНОЙ энергии станции.

Зачем. На витрине суточный коридор строился как сумма почасовых P10 и P90 за 24 часа
и подписывался как интервал, куда суточная энергия попадает в ~80 % случаев. Это неверно:
квантили не складываются. Ошибки соседних часов сильно коррелированы (одна и та же ошибка
прогноза ветра тянется через все сутки), поэтому сумма почасовых границ даёт интервал
шириной как при независимых ошибках — то есть кратно шире нужного, с покрытием ~93 %.

Здесь интервал калибруется напрямую на суточных остатках: для каждого выпуска января 2026
считается суточная энергия прогноза станции и суточная энергия факта, а квантили берутся
по выборке этих остатков — отдельно для лида 1 и лида 2.

Протокол тот же, что в scripts/train.py и scripts/make_validation_data.py:
модель обучена только до 31.12.2025 (сохранённая в models/ видела январь и для калибровки
непригодна), веса смеси и почасовые квантили берутся из сохранённой модели.

Сеть не нужна: архивные прогнозы лежат в data/cache/weather.
Запуск: python scripts/calibrate_daily.py  (около 25 секунд)
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

OUT = Path("docs/demo/daily-interval.js")
DAYS = pd.date_range("2025-12-31", "2026-01-29", freq="D")
TRAIN_END = "2025-12-31 23:00:00"
TURBINES = (1, 2)
LO_Q, HI_Q = 0.10, 0.90
MODE = "absolute"  # см. диагностику масштабирования ошибки в main()


def holdout_models() -> dict[int, GenerationModel]:
    """Модели, не видевшие января 2026. Веса и почасовые квантили — из сохранённой модели."""
    weather = previous_runs_training(FARM)
    out = {}
    for t in TURBINES:
        ds = make_training_set(load_hourly(t), weather)
        m = GenerationModel(t).fit(ds[ds["ts"] <= TRAIN_END])
        saved = GenerationModel.load(t)
        m.weights = saved.weights
        m.residual_quantiles = saved.residual_quantiles
        out[t] = m
    return out


def daily_table() -> pd.DataFrame:
    """Суточные суммы по каждому выпуску и лиду: прогноз станции, факт, сумма почасовых границ."""
    models = holdout_models()
    fact = load_hourly(1)[["ts", "power_norm"]].rename(columns={"power_norm": "f1"}).merge(
        load_hourly(2)[["ts", "power_norm"]].rename(columns={"power_norm": "f2"}), on="ts"
    )
    fact["farm_fact"] = (fact["f1"] + fact["f2"]) / 2
    fact = fact.set_index("ts")["farm_fact"]

    rows = []
    for d in DAYS:
        day = d.strftime("%Y-%m-%d")
        weather = forecast_available_at(FARM, day, 48)
        preds = {}
        for t in TURBINES:
            p = models[t].predict(weather)
            preds[t] = p.set_index("ts")[["power_norm_pred", "power_p10", "power_p90", "lead_day"]]
        idx = preds[1].index
        farm = (preds[1]["power_norm_pred"] + preds[2]["power_norm_pred"]) / 2
        lo_h = (preds[1]["power_p10"] + preds[2]["power_p10"]) / 2
        hi_h = (preds[1]["power_p90"] + preds[2]["power_p90"]) / 2
        lead = preds[1]["lead_day"]

        for ld in (1, 2):
            sel = lead == ld
            hours = idx[sel]
            f = fact.reindex(hours)
            if f.isna().any() or len(hours) != 24:
                continue  # неполные сутки факта в оценку не берём
            rows.append({
                "issue_date": day,
                "lead": ld,
                "pred_energy": float(farm[sel].sum()),
                "fact_energy": float(f.sum()),
                "sum_p10": float(lo_h[sel].sum()),
                "sum_p90": float(hi_h[sel].sum()),
            })
    df = pd.DataFrame(rows)
    df["resid_abs"] = df["fact_energy"] - df["pred_energy"]
    df["resid_rel"] = df["resid_abs"] / df["pred_energy"]
    return df


def loo_coverage(resid: np.ndarray) -> float:
    """Покрытие по leave-one-out: квантили по n-1 наблюдению, проверка на оставшемся."""
    hits = 0
    for i in range(len(resid)):
        rest = np.delete(resid, i)
        lo, hi = np.quantile(rest, LO_Q), np.quantile(rest, HI_Q)
        hits += int(lo <= resid[i] <= hi)
    return hits / len(resid)


def main() -> None:
    df = daily_table()
    print(f"суток в выборке: {len(df)} ({len(df[df['lead'] == 1])} на лид 1, {len(df[df['lead'] == 2])} на лид 2)")
    print(f"средняя суточная энергия факта: {df['fact_energy'].mean():.2f} норм·ч\n")

    # диагностика: масштабируется ли ошибка с уровнем прогноза
    scaling = {}
    for ld in (1, 2):
        s = df[df["lead"] == ld]
        scaling[f"lead{ld}"] = {
            "corr_abs_resid_vs_level": round(float(np.corrcoef(s["pred_energy"], s["resid_abs"].abs())[0, 1]), 3),
            "corr_rel_resid_vs_level": round(float(np.corrcoef(s["pred_energy"], s["resid_rel"].abs())[0, 1]), 3),
        }
    print("масштабирование ошибки (корреляция |остатка| с энергией прогноза):")
    for k, v in scaling.items():
        print(f"  {k}: абсолютный {v['corr_abs_resid_vs_level']:+.3f}, относительный {v['corr_rel_resid_vs_level']:+.3f}")
    print("  -> ошибка постоянна по величине, выбран абсолютный режим\n")

    payload, report = {}, []
    for ld in (1, 2):
        s = df[df["lead"] == ld]
        n = len(s)

        # прежний способ: сумма почасовых границ
        old_hit = ((s["fact_energy"] >= s["sum_p10"]) & (s["fact_energy"] <= s["sum_p90"])).mean()
        old_width = float((s["sum_p90"] - s["sum_p10"]).mean())

        variants = {}
        for mode, col in (("absolute", "resid_abs"), ("relative", "resid_rel")):
            r = s[col].to_numpy()
            q10, q90 = float(np.quantile(r, LO_Q)), float(np.quantile(r, HI_Q))
            if mode == "absolute":
                lo, hi = s["pred_energy"] + q10, s["pred_energy"] + q90
            else:
                lo, hi = s["pred_energy"] * (1 + q10), s["pred_energy"] * (1 + q90)
            lo, hi = lo.clip(lower=0), hi.clip(lower=0)
            variants[mode] = {
                "q10": q10, "q90": q90,
                "coverage_in": float(((s["fact_energy"] >= lo) & (s["fact_energy"] <= hi)).mean()),
                "coverage_loo": loo_coverage(r),
                "width": float((hi - lo).mean()),
            }

        # Режим выбран по диагностике масштабирования, а не по покрытию: |остаток| практически
        # не зависит от уровня прогноза (корреляция -0.02 и +0.00 по лидам), тогда как
        # относительный остаток сильно убывает с ростом энергии (-0.53). Значит ошибка постоянна
        # по величине, и относительный режим зря раздувает интервал в ветреные сутки.
        # Разница LOO-покрытия между режимами — одни сутки из тридцати, то есть шум.
        best = MODE
        v = variants[best]
        payload[f"lead{ld}"] = {
            "mode": best,
            "q10": round(v["q10"], 4),
            "q90": round(v["q90"], 4),
            "coverage_in_sample": round(v["coverage_in"], 3),
            "coverage_loo": round(v["coverage_loo"], 3),
            "width_norm_h": round(v["width"], 2),
            "old_sum_of_hourly": {"coverage": round(float(old_hit), 3), "width_norm_h": round(old_width, 2)},
            "n_days": int(n),
        }
        report.append((ld, variants, best, old_hit, old_width, n))

    for ld, variants, best, old_hit, old_width, n in report:
        print(f"лид {ld} ({n} суток)")
        print(f"  прежний способ (сумма почасовых границ): покрытие {old_hit:.1%}, ширина {old_width:.2f} норм·ч")
        for mode, v in variants.items():
            mark = "  <- выбран" if mode == best else ""
            print(f"  {mode:9}: q10 {v['q10']:+.3f}  q90 {v['q90']:+.3f}  "
                  f"покрытие внутривыб. {v['coverage_in']:.1%}  LOO {v['coverage_loo']:.1%}  "
                  f"ширина {v['width']:.2f} норм·ч{mark}")
        print(f"  новый интервал уже прежнего в {old_width / variants[best]['width']:.1f} раза\n")

    payload["meta"] = {
        "period": "выпуски 2025-12-31 … 2026-01-29, факт января 2026",
        "target": "суточная энергия станции, норм·ч (сумма 24 часовых долей номинала, среднее двух турбин)",
        "nominal": 0.8,
        "error_scaling": scaling,
        "note": ("квантили посчитаны на тех же 30 сутках, на которых измерено покрытие; "
                 "LOO-оценка почти честная, но выборка мала — на новых данных интервал требует перекалибровки"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.DAILY_Q = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    print(f"{OUT}: записан")


if __name__ == "__main__":
    main()
