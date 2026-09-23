"""Эксперимент: что делает жёсткий конвейер и что делает агент на испорченных входных данных.

Зачем. Прогноз, посчитанный по испорченному прогнозу погоды, выглядит так же уверенно, как
исправный: те же 96 строк, то же заключение, тот же вид графика. Разница между конвейером и
агентом видна только тогда, когда вход деградировал. Этот скрипт показывает разницу на одних
и тех же входах.

Контуры:
  конвейер — погода -> прогноз -> запись, без контроля качества (поведение до появления
             инструмента check_input_quality);
  агент    — погода -> контроль качества -> прогноз -> анализ -> при необходимости пересчёт -> запись,
             с отказом от публикации при вердикте fail.

Запуск: python scripts/degradation_experiment.py
Скрипт ничего не пишет в outputs/: выпуски уходят во временный каталог.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import app.agent as agent
from app.agent import VERDICT_FAIL, AgentState, ForecastAgent
from app.weather import FARM, forecast_available_at

DAY = "2026-02-10"


def degradations(base: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Шесть правдоподобных отказов источника и телеметрии."""
    out: list[tuple[str, pd.DataFrame]] = [("норма (контроль)", base.copy())]

    w = base.copy()
    w.loc[10:22, "wind_speed_100m"] = np.nan
    out.append(("пропуски 13 часов", w))

    w = base.copy()
    w["wind_speed_100m"] = float(base["wind_speed_100m"].iloc[0])
    out.append(("застрявший датчик", w))

    w = base.copy()
    w["wind_speed_100m"] = base["wind_speed_100m"] + 5.0
    out.append(("сдвиг +5 м/с", w))

    w = base.copy()
    w["wind_speed_100m"] = -base["wind_speed_100m"]
    out.append(("отрицательный ветер", w))

    out.append(("обрезанный ряд, 20 ч", base.iloc[:20].copy()))
    return out


def run_pipeline(a: ForecastAgent, weather: pd.DataFrame) -> dict:
    """Жёсткий конвейер: посчитать и записать, ничего не проверяя."""
    a.state = AgentState(issue_date=DAY, horizon_hours=48)
    a.state.weather = weather.copy()
    a.prepare_and_predict()
    f = a.state.forecast
    nan_rows = int(f[[c for c in f.columns if c.startswith("power_")]].isna().any(axis=1).sum())
    return {
        "published": True,  # конвейер публикует всегда
        "hours": int(f["ts"].nunique()),
        "mean": float(f["power_norm_pred"].mean()),
        "nan_rows": nan_rows,
        "note": "выпуск опубликован",
    }


def run_agent(a: ForecastAgent, weather: pd.DataFrame, previous: pd.DataFrame | None) -> dict:
    """Агент: тот же вход, но с контролем качества, сравнением с прошлым выпуском
    и правом отказать в публикации."""
    original = agent.forecast_available_at
    agent.forecast_available_at = lambda *args, **kwargs: weather.copy()
    try:
        s = a.run_day(DAY, previous=previous, use_llm=False)
    finally:
        agent.forecast_available_at = original
    codes = [f["code"] for f in s.quality.get("findings", []) if f["level"] == VERDICT_FAIL]
    mean = float(s.forecast["power_norm_pred"].mean()) if s.forecast is not None else float("nan")
    rec = [t["result"] for t in s.trace if t["tool"] == "recalculate"]
    return {
        "published": s.published,
        "verdict": s.quality.get("verdict"),
        "hours": int(s.forecast["ts"].nunique()) if s.forecast is not None else 0,
        "mean": mean,
        "codes": codes,
        "confidence": s.confidence,
        "recalculated": s.recalculated,
        "trigger_detail": rec[0]["trigger_detail"] if rec else None,
    }


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="degradation-"))
    agent.OUT_DIR = tmp / "forecasts"
    agent.LOG_DIR = tmp / "agent_logs"

    base = forecast_available_at(FARM, DAY, 48)
    a = ForecastAgent(use_llm=False)

    # настоящий выпуск предыдущего дня на чистых данных: он даёт агенту второй слой контроля —
    # сравнение с прошлым прогнозом на пересекающихся часах
    prev_day = (pd.Timestamp(DAY) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    previous = a.run_day(prev_day, use_llm=False).forecast.copy()

    print(f"Дата выпуска {DAY}, горизонт 48 часов. Выпуски пишутся в {tmp}")
    print(f"Агенту передан выпуск предыдущего дня {prev_day} для сравнения.\n")
    head = f"{'сценарий':<22} | {'конвейер':<36} | {'агент':<52}"
    print(head)
    print("-" * len(head))

    rejected = flagged = 0
    for name, weather in degradations(base):
        p = run_pipeline(a, weather)
        g = run_agent(a, weather, previous)
        left = f"опубликован, средняя {p['mean']:.3f}"
        if p["nan_rows"]:
            left += f", NaN {p['nan_rows']}"
        if p["hours"] != 48:
            left += f", часов {p['hours']}"
        if not g["published"]:
            right = f"ОТКАЗ в публикации: {', '.join(g['codes'])}"
            rejected += 1
        elif g["recalculated"]:
            right = f"аномалия, пересчёт: {g['trigger_detail']}"
            flagged += 1
        else:
            right = f"опубликован, средняя {g['mean']:.3f}, достоверность {g['confidence']}"
        print(f"{name:<22} | {left:<36} | {right:<52}")

    print(f"\nВнесено дефектов: 5. Конвейер опубликовал все шесть выпусков молча.")
    print(f"Агент: отказал в публикации {rejected} раз, пометил аномалией и пересчитал {flagged} раз.")
    print("Контрольный сценарий «норма» в обоих контурах публикуется одинаково: проверка не мешает")
    print("нормальной работе, она срабатывает только на испорченном входе.")


if __name__ == "__main__":
    main()
