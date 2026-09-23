"""Бэктест «как в прошлом»: последовательные выпуски прогноза на каждый день тестового периода.

Для каждой даты выпуска агент берёт архивный прогноз погоды, доступный именно на эту дату,
считает 48 часов вперёд, сохраняет выпуск и журнал. Факт выработки нигде не используется.
Выпуск дня N получает выпуск дня N−1 и сравнивает пересекающиеся часы — это детектор
обновления входных данных, по которому агент решает о пересчёте.
Из 28 выпусков собирается итоговый почасовой ряд февраля 2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.agent import ForecastAgent

RESULT_DIR = Path("outputs")
# Обязательные колонки таблицы выпусков; необязательные (интервалы, компоненты смеси)
# дописываются справа, если модель их отдаёт.
ISSUE_COLS = [
    "issue_date", "ts", "horizon_hour", "lead_day", "turbine",
    "power_norm_pred", "power_gbm_pred", "power_curve_pred",
]
EXTRA_COLS = ["power_p10", "power_p90", "power_two_stage_pred", "wind_speed_100m", "wind_nacelle_pred"]
FEB_START = "2026-02-01 00:00"
FEB_END = "2026-02-28 23:00"


def _round(x, n: int = 3):
    return None if x is None or pd.isna(x) else round(float(x), n)


def _summary_row(r) -> dict:
    """Строка сводки по одному выпуску. Ключи анализа читаются мягко: набор может меняться."""
    a = r.analysis or {}
    cf = a.get("mean_cf", {})
    f = r.forecast
    return {
        "issue_date": r.issue_date,
        "hours": int(f["ts"].nunique()) if len(f) else 0,
        "recomputed": bool(r.recomputed),
        "llm_used": bool(r.llm_used),
        "storm_hours": a.get("storm_hours"),
        "calm_hours": a.get("calm_hours"),
        "rated_hours": a.get("rated_hours"),
        "mean_cf_t1": _round(cf.get("1")),
        "mean_cf_t2": _round(cf.get("2")),
        "mean_abs_change_vs_prev": _round(a.get("input_change_vs_previous_issue")),
    }


def backtest(
    start: str = "2026-01-29",
    end: str = "2026-02-27",
    use_llm: bool | None = False,
    turbines: tuple[int, ...] = (1, 2),
    horizon_hours: int = 48,
    progress: bool = False,
) -> pd.DataFrame:
    """Все выпуски подряд. Возвращает объединённую таблицу выпусков.

    Сводка по дням и журналы кладутся в `attrs`, чтобы run() записал их без повторного прогона.
    """
    agent = ForecastAgent(turbines=turbines, use_llm=use_llm, horizon_hours=horizon_hours)
    frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    logs: list[dict] = []
    previous: pd.DataFrame | None = None
    for d in pd.date_range(start, end, freq="D"):
        day = d.strftime("%Y-%m-%d")
        r = agent.run_day(day, previous=previous)
        row = _summary_row(r)
        summaries.append(row)
        logs.append({**row, "conclusion": r.narrative, "tools": [t.get("tool") for t in r.tool_trace]})
        frames.append(r.forecast.copy())
        previous = r.forecast  # следующий выпуск сравнивает себя с этим
        if progress:
            mark = ", пересчёт" if row["recomputed"] else ""
            llm = ", LLM" if row["llm_used"] else ""
            print(f"  {day}: часов {row['hours']}, T1 {row['mean_cf_t1']}, T2 {row['mean_cf_t2']}{mark}{llm}")
    allf = pd.concat(frames, ignore_index=True)
    allf.attrs["summary"] = summaries
    allf.attrs["logs"] = logs
    return allf


def build_submission(all_issues: pd.DataFrame) -> pd.DataFrame:
    """Итоговый почасовой ряд февраля: на каждый час свежайший выпуск (lead 1) и выпуск за двое суток (lead 2).

    Сетка часов задаётся явно, поэтому строка за каждый час февраля есть заведомо.
    Станция считается как среднее долей номинала двух турбин, то есть в тех же единицах,
    что и отдельная турбина.
    """
    out = pd.DataFrame({"ts": pd.date_range(FEB_START, FEB_END, freq="h")})
    for lead in (1, 2):
        sub = all_issues[all_issues["lead_day"] == lead]
        wide = sub.pivot_table(index="ts", columns="turbine", values="power_norm_pred", aggfunc="last")
        for t in (1, 2):
            col = f"turbine_{t}_lead{lead}"
            out[col] = out["ts"].map(wide[t]) if t in wide.columns else pd.NA
    # интервал неопределённости P10–P90 для свежайшего выпуска, если модель его считает
    fresh = all_issues[all_issues["lead_day"] == 1]
    for q in ("power_p10", "power_p90"):
        if q in fresh.columns:
            wide = fresh.pivot_table(index="ts", columns="turbine", values=q, aggfunc="last")
            for t in (1, 2):
                out[f"turbine_{t}_{q[-3:]}"] = out["ts"].map(wide[t]) if t in wide.columns else pd.NA
    # станционный уровень: прогноз на сутки и на двое суток плюс станционный интервал
    out["farm_mean_lead1"] = out[["turbine_1_lead1", "turbine_2_lead1"]].mean(axis=1)
    out["farm_mean_lead2"] = out[["turbine_1_lead2", "turbine_2_lead2"]].mean(axis=1)
    if "turbine_1_p10" in out.columns:
        out["farm_p10_lead1"] = out[["turbine_1_p10", "turbine_2_p10"]].mean(axis=1)
        out["farm_p90_lead1"] = out[["turbine_1_p90", "turbine_2_p90"]].mean(axis=1)
    return out


def run(
    start: str = "2026-01-29",
    end: str = "2026-02-27",
    use_llm: bool | None = False,
    horizon_hours: int = 48,
    progress: bool = False,
) -> dict:
    """Полный прогон с записью всех файлов. Возвращает числа прогона."""
    allf = backtest(start, end, use_llm=use_llm, horizon_hours=horizon_hours, progress=progress)
    summaries, logs = allf.attrs["summary"], allf.attrs["logs"]
    final = build_submission(allf)
    RESULT_DIR.mkdir(exist_ok=True)
    cols = ISSUE_COLS + [c for c in EXTRA_COLS if c in allf.columns]
    # округление до 4 знаков: значения нормированы на 1, дальше точности нет физического смысла
    issues_out = allf[cols].copy()
    issues_out[issues_out.select_dtypes("float").columns] = issues_out.select_dtypes("float").round(4)
    issues_out.to_csv(RESULT_DIR / "forecast_all_issues.csv", index=False)
    final_out = final.copy()
    final_out[final_out.select_dtypes("float").columns] = final_out.select_dtypes("float").round(4)
    final_out.to_csv(RESULT_DIR / "forecast_feb2026.csv", index=False)
    pd.DataFrame(summaries).to_csv(RESULT_DIR / "backtest_summary.csv", index=False)
    (RESULT_DIR / "backtest_summary.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    lead1 = ["turbine_1_lead1", "turbine_2_lead1"]
    lead2 = ["turbine_1_lead2", "turbine_2_lead2"]
    return {
        "issues": len(summaries),
        "rows_all_issues": int(len(allf)),
        "feb_hours": int(len(final)),
        "missing_lead1": int(final[lead1].isna().any(axis=1).sum()),
        "missing_lead2": int(final[lead2].isna().any(axis=1).sum()),
        "recomputed_days": [s["issue_date"] for s in summaries if s["recomputed"]],
        "llm_days": [s["issue_date"] for s in summaries if s["llm_used"]],
        "mean_cf_t1": _round(final["turbine_1_lead1"].mean()),
        "mean_cf_t2": _round(final["turbine_2_lead1"].mean()),
        "paths": {
            "all_issues": str(RESULT_DIR / "forecast_all_issues.csv"),
            "submission": str(RESULT_DIR / "forecast_feb2026.csv"),
            "summary": str(RESULT_DIR / "backtest_summary.csv"),
        },
    }
