"""Бэктест «как в прошлом»: последовательные выпуски прогноза на каждый день тестового периода.

Для каждой даты выпуска агент берёт архивный прогноз погоды, доступный именно на эту дату,
считает 48 часов вперёд, сохраняет выпуск и журнал. Факт выработки нигде не используется.
Из 28 выпусков собирается итоговый почасовой ряд февраля 2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.agent import OUT_DIR, ForecastAgent

RESULT_DIR = Path("outputs")
ISSUE_COLS = [
    "issue_date", "ts", "horizon_hour", "lead_day", "turbine",
    "power_norm_pred", "power_p10", "power_p90",
    "power_gbm_pred", "power_two_stage_pred", "power_curve_pred", "wind_nacelle_pred",
]
FEB_START = "2026-02-01 00:00"
FEB_END = "2026-02-28 23:00"


def _round(x: float | None, n: int = 4) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), n)


def _summary_row(state, llm_requested: bool) -> dict:
    """Строка сводки по одному выпуску. Ключи анализа берутся мягко: агент может их поменять."""
    a = state.analysis or {}
    upd = a.get("input_update_vs_previous_issue") or {}
    f = state.forecast
    means = {int(t): float(g["power_norm_pred"].mean()) for t, g in f.groupby("turbine")} if f is not None else {}
    fallback = any(t.get("tool") == "llm_fallback" for t in state.trace)
    return {
        "issue_date": state.issue_date,
        "hours": int(f["ts"].nunique()) if f is not None else 0,
        "recomputed": bool(state.recalculated),
        "llm_used": bool(llm_requested and not fallback),
        "storm_hours": a.get("storm_hours"),
        "calm_hours": a.get("calm_hours"),
        "rated_hours": a.get("rated_hours"),
        "mean_cf_t1": _round(means.get(1), 3),
        "mean_cf_t2": _round(means.get(2), 3),
        "mean_abs_change_vs_prev": upd.get("mean_abs_change"),
    }


def backtest(
    start: str = "2026-01-31",
    end: str = "2026-02-27",
    use_llm: bool | None = False,
    turbines: tuple[int, ...] = (1, 2),
    horizon_hours: int = 48,
    progress: bool = False,
) -> pd.DataFrame:
    """Прогон всех выпусков подряд. Возвращает объединённую таблицу выпусков.

    Выпуск дня N видит выпуск дня N−1 (агент сравнивает пересекающиеся часы и решает о пересчёте),
    поэтому дни идут строго по порядку и результат каждого дня сохраняется до начала следующего.
    """
    agent = ForecastAgent()
    frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    logs: list[dict] = []
    for d in pd.date_range(start, end, freq="D"):
        day = d.strftime("%Y-%m-%d")
        state = agent.run_day(day, horizon_hours=horizon_hours, use_llm=use_llm)
        row = _summary_row(state, llm_requested=use_llm is not False)
        summaries.append(row)
        logs.append({**row, "conclusion": state.conclusion, "tools": [t.get("tool") for t in state.trace]})
        frames.append(pd.read_csv(OUT_DIR / f"{day}.csv", parse_dates=["ts"]))
        if progress:
            mark = " пересчёт" if row["recomputed"] else ""
            print(f"  {day}: часов {row['hours']}, T1 {row['mean_cf_t1']}, T2 {row['mean_cf_t2']}{mark}")
    allf = pd.concat(frames, ignore_index=True)
    allf = allf[allf["turbine"].isin(turbines)]
    allf.attrs["summary"] = summaries
    allf.attrs["logs"] = logs
    return allf


def build_submission(all_issues: pd.DataFrame) -> pd.DataFrame:
    """Итоговый почасовой ряд февраля: на каждый час свежайший выпуск (lead 1) и выпуск за двое суток (lead 2).

    Сетка часов задаётся явно, поэтому пропусков строк быть не может: час без прогноза остался бы пустым.
    """
    out = pd.DataFrame({"ts": pd.date_range(FEB_START, FEB_END, freq="h")})
    for lead in (1, 2):
        sub = all_issues[all_issues["lead_day"] == lead]
        wide = sub.pivot_table(index="ts", columns="turbine", values="power_norm_pred", aggfunc="last")
        for t in (1, 2):
            col = f"turbine_{t}_lead{lead}"
            out[col] = out["ts"].map(wide[t]) if t in wide.columns else pd.NA
    # интервал P10–P90 для свежайшего выпуска
    sub = all_issues[all_issues["lead_day"] == 1]
    for q in ("power_p10", "power_p90"):
        if q in sub.columns:
            wide = sub.pivot_table(index="ts", columns="turbine", values=q, aggfunc="last")
            for t in (1, 2):
                out[f"turbine_{t}_{q[-3:]}"] = out["ts"].map(wide[t]) if t in wide.columns else pd.NA
    out["farm_mean_lead1"] = out[["turbine_1_lead1", "turbine_2_lead1"]].mean(axis=1)
    return out


def run(
    start: str = "2026-01-31",
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
    allf[[c for c in ISSUE_COLS if c in allf.columns]].to_csv(RESULT_DIR / "forecast_all_issues.csv", index=False)
    final.to_csv(RESULT_DIR / "forecast_feb2026.csv", index=False)
    pd.DataFrame(summaries).to_csv(RESULT_DIR / "backtest_summary.csv", index=False)
    (RESULT_DIR / "backtest_summary.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    lead1 = ["turbine_1_lead1", "turbine_2_lead1"]
    return {
        "issues": len(summaries),
        "rows_all_issues": int(len(allf)),
        "feb_hours": int(len(final)),
        "missing_lead1": int(final[lead1].isna().any(axis=1).sum()),
        "missing_lead2": int(final[["turbine_1_lead2", "turbine_2_lead2"]].isna().any(axis=1).sum()),
        "recomputed_days": [s["issue_date"] for s in summaries if s["recomputed"]],
        "llm_days": [s["issue_date"] for s in summaries if s["llm_used"]],
        "mean_cf_t1": _round(final["turbine_1_lead1"].mean(), 3),
        "mean_cf_t2": _round(final["turbine_2_lead1"].mean(), 3),
        "paths": {
            "all_issues": str(RESULT_DIR / "forecast_all_issues.csv"),
            "submission": str(RESULT_DIR / "forecast_feb2026.csv"),
            "summary": str(RESULT_DIR / "backtest_summary.csv"),
        },
    }
