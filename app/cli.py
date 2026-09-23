"""Командная строка проекта.

python -m app.cli train                      обучить модели
python -m app.cli forecast --date 2026-01-31 прогноз агентом на одну дату
python -m app.cli backtest                   прогон по всем дням теста (31.01–27.02.2026)
python -m app.cli report                     сводка по результатам
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wind-agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    f = sub.add_parser("forecast")
    f.add_argument("--date", required=True, help="дата выпуска прогноза, YYYY-MM-DD")
    f.add_argument("--horizon", type=int, default=48)
    f.add_argument("--no-llm", action="store_true")
    b = sub.add_parser("backtest")
    b.add_argument("--start", default="2026-01-31")
    b.add_argument("--end", default="2026-02-27")
    b.add_argument("--horizon", type=int, default=48)
    b.add_argument("--no-llm", action="store_true")
    sub.add_parser("report")
    a = p.parse_args(argv)

    if a.cmd == "train":
        from scripts.train import main as train_main
        train_main()
    elif a.cmd == "forecast":
        from app.agent import ForecastAgent
        s = ForecastAgent().run_day(a.date, horizon_hours=a.horizon, use_llm=False if a.no_llm else None)
        print(json.dumps({"issue_date": s.issue_date, "tools": [t["tool"] for t in s.trace], "analysis": s.analysis, "conclusion": s.conclusion}, ensure_ascii=False, indent=2, default=str))
    elif a.cmd == "backtest":
        from app.backtest import run_backtest
        final = run_backtest(a.start, a.end, a.horizon, use_llm=False if a.no_llm else None)
        print(f"часов в итоговом ряду: {len(final)}, файл outputs/forecast_feb2026.csv")
        print(final.head(5).round(3).to_string(index=False))
    elif a.cmd == "report":
        import pandas as pd
        s = pd.read_csv("outputs/backtest_summary.csv")
        f = pd.read_csv("outputs/forecast_feb2026.csv", parse_dates=["ts"])
        print("выпусков прогноза:", len(s), "пересчётов:", int(s["recalculated"].sum()))
        print("средняя ожидаемая мощность по ВЭС за февраль (lead 1):", round(float(f["farm_mean_lead1"].mean()), 3))
        print(f.set_index("ts")["farm_mean_lead1"].resample("D").mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
