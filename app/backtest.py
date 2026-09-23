"""Воспроизведение прошлого: последовательные запуски агента на каждый день тестового периода."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.agent import OUT_DIR, ForecastAgent

RESULT_DIR = Path("outputs")


def run_backtest(start: str = "2026-01-31", end: str = "2026-02-27", horizon_hours: int = 48, use_llm: bool | None = None) -> pd.DataFrame:
    """Даты выпуска прогноза: с start по end включительно, каждый день прогноз на следующие horizon_hours."""
    agent = ForecastAgent()
    summaries = []
    for d in pd.date_range(start, end, freq="D"):
        s = agent.run_day(d.strftime("%Y-%m-%d"), horizon_hours=horizon_hours, use_llm=use_llm)
        summaries.append({"issue_date": s.issue_date, "recalculated": s.recalculated, "conclusion": s.conclusion, **{k: v for k, v in s.analysis.items() if k != "input_update_vs_previous_issue"}})
    frames = [pd.read_csv(OUT_DIR / f"{d.date()}.csv", parse_dates=["ts"]) for d in pd.date_range(start, end, freq="D")]
    allf = pd.concat(frames, ignore_index=True)
    RESULT_DIR.mkdir(exist_ok=True)
    allf.to_csv(RESULT_DIR / "forecast_all_issues.csv", index=False)
    # Итоговый почасовой ряд за тест: для каждого часа берётся самый свежий выпуск (lead 1),
    # плюс отдельный ряд lead 2 для оценки 48-часового горизонта.
    wide = []
    for lead in (1, 2):
        sub = allf[allf["lead_day"] == lead].pivot_table(index="ts", columns="turbine", values="power_norm_pred")
        sub.columns = [f"turbine_{c}_lead{lead}" for c in sub.columns]
        wide.append(sub)
    final = pd.concat(wide, axis=1).reset_index()
    final = final[(final["ts"] >= "2026-02-01") & (final["ts"] < "2026-03-01")]
    final["farm_mean_lead1"] = final[["turbine_1_lead1", "turbine_2_lead1"]].mean(axis=1)
    final.to_csv(RESULT_DIR / "forecast_feb2026.csv", index=False)
    pd.DataFrame(summaries).to_csv(RESULT_DIR / "backtest_summary.csv", index=False)
    (RESULT_DIR / "backtest_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return final
