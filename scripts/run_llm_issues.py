"""Прогон всех выпусков с живой LLM-оркестрацией.

Порядок инструментов на каждом выпуске выбирает модель через function calling.
Журналы складываются в outputs/agent_logs_llm/, после чего канонические журналы
в outputs/agent_logs/ восстанавливаются детерминированным прогоном: числовой
результат от режима не зависит, а сдача должна воспроизводиться без ключей.

Запуск: python scripts/run_llm_issues.py [--start 2026-01-29] [--end 2026-02-27]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.agent import LOG_DIR, ForecastAgent

LLM_DIR = Path("outputs/agent_logs_llm")


def main() -> None:
    ap = argparse.ArgumentParser(description="Выпуски с LLM-оркестрацией")
    ap.add_argument("--start", default="2026-01-29")
    ap.add_argument("--end", default="2026-02-27")
    a = ap.parse_args()

    from app.llm import USAGE, available

    if not available():
        print("Ключ провайдера не задан: нечего прогонять. Задайте OPENAI_API_KEY в .env")
        raise SystemExit(1)

    LLM_DIR.mkdir(parents=True, exist_ok=True)
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(a.start, a.end, freq="D")]

    agent = ForecastAgent(use_llm=True)
    previous, done, fell_back = None, [], []
    for day in days:
        s = agent.run_day(day, previous=previous)
        shutil.copy(LOG_DIR / f"{day}.json", LLM_DIR / f"{day}.json")
        previous = s.forecast
        tools = " → ".join(t["tool"] for t in s.trace)
        mark = "LLM" if s.llm_used else "откат"
        if not s.llm_used:
            fell_back.append(day)
        done.append(day)
        print(f"{day}  {mark:5}  пересчёт: {'да' if s.recomputed else 'нет':3}  {tools}")

    print(f"\nвыпусков с LLM: {len(done) - len(fell_back)} из {len(done)}")
    if fell_back:
        print("откат на детерминированный план:", ", ".join(fell_back))
    print("расход:", USAGE)

    # канонические журналы возвращаем к детерминированному прогону
    det = ForecastAgent(use_llm=False)
    previous = None
    for day in days:
        s = det.run_day(day, previous=previous)
        previous = s.forecast
    print("канонические журналы восстановлены детерминированным прогоном")


if __name__ == "__main__":
    main()
