"""Agentic AI: агент прогнозирования выработки ВЭС.

Цикл на каждую дату прогноза (issue_date):
  1. get_weather        — архивный прогноз погоды, доступный на issue_date (Open-Meteo Previous Runs)
  2. prepare_and_predict — признаки + модель по каждой турбине, почасовой прогноз на 24–48 ч
  3. analyze            — проверка прогноза: диапазоны, штили, сильный ветер, сравнение с прогнозом
                          предыдущего дня на пересекающиеся часы (детектор обновления входных данных)
  4. recalculate        — повторный расчёт, если входные данные обновились или найдены аномалии
  5. save_forecast      — запись результата и журнала действий агента

Оркестрация: LLM (OpenAI, резерв NVIDIA NIM) вызывает инструменты через function calling и пишет
заключение. Без ключа или в DEMO_MODE тот же цикл выполняет детерминированный планировщик,
а заключение берётся из кэша ответов. Инструменты одинаковы в обоих режимах.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.config import settings
from app.model import GenerationModel
from app.weather import FARM, forecast_available_at

OUT_DIR = Path("outputs/forecasts")
LOG_DIR = Path("outputs/agent_logs")
CUT_IN = 3.0
RATED = 11.5
STORM = 22.0


@dataclass
class AgentState:
    issue_date: str
    horizon_hours: int = 48
    weather: pd.DataFrame | None = None
    forecast: pd.DataFrame | None = None
    analysis: dict = field(default_factory=dict)
    recalculated: bool = False
    trace: list[dict] = field(default_factory=list)
    conclusion: str = ""


class ForecastAgent:
    def __init__(self, models: dict[int, GenerationModel] | None = None):
        if models is None:
            missing = [t for t in (1, 2) if not (Path("models") / f"turbine_{t}.joblib").exists()]
            if missing:
                raise RuntimeError("Модели не обучены. Выполните: python -m app.cli train")
            models = {t: GenerationModel.load(t) for t in (1, 2)}
        self.models = models
        self.tools: dict[str, Callable[..., dict]] = {
            "get_weather": self.get_weather,
            "prepare_and_predict": self.prepare_and_predict,
            "analyze": self.analyze,
            "recalculate": self.recalculate,
            "save_forecast": self.save_forecast,
        }
        self.state: AgentState | None = None

    # ---------- инструменты ----------
    def _log(self, tool: str, result: dict) -> dict:
        assert self.state is not None
        self.state.trace.append({"tool": tool, "result": result})
        return result

    def get_weather(self) -> dict:
        s = self.state
        s.weather = forecast_available_at(FARM, s.issue_date, s.horizon_hours)
        w = s.weather
        return self._log("get_weather", {
            "hours": int(len(w)), "from": str(w["ts"].min()), "to": str(w["ts"].max()),
            "ws100_mean": round(float(w["wind_speed_100m"].mean()), 2),
            "ws100_max": round(float(w["wind_speed_100m"].max()), 2),
            "lead_days": sorted(w["lead_day"].unique().tolist()),
        })

    def prepare_and_predict(self) -> dict:
        s = self.state
        frames = []
        for t, model in self.models.items():
            p = model.predict(s.weather)
            p["turbine"] = t
            frames.append(p)
        s.forecast = pd.concat(frames, ignore_index=True)
        summary = {
            f"turbine_{t}_mean": round(float(g["power_norm_pred"].mean()), 3)
            for t, g in s.forecast.groupby("turbine")
        }
        summary["hours"] = int(s.forecast["ts"].nunique())
        return self._log("prepare_and_predict", summary)

    def analyze(self) -> dict:
        s = self.state
        w, f = s.weather, s.forecast
        calm = int((w["wind_speed_100m"] < CUT_IN).sum())
        rated = int((w["wind_speed_100m"] >= RATED).sum())
        storm = int((w["wind_speed_100m"] >= STORM).sum())
        spread = float((f.groupby("ts")["power_norm_pred"].max() - f.groupby("ts")["power_norm_pred"].min()).max())
        # сравнение с прогнозом предыдущего дня на пересекающиеся часы (часы 24–48 вчера = часы 0–24 сегодня)
        prev_path = OUT_DIR / f"{(pd.Timestamp(s.issue_date) - pd.Timedelta(days=1)).date()}.csv"
        update = None
        if prev_path.exists():
            prev = pd.read_csv(prev_path, parse_dates=["ts"])
            merged = f.merge(prev, on=["ts", "turbine"], suffixes=("", "_prev"))
            if len(merged):
                diff = (merged["power_norm_pred"] - merged["power_norm_pred_prev"]).abs()
                update = {"overlap_hours": int(merged["ts"].nunique()), "mean_abs_change": round(float(diff.mean()), 3),
                          "max_abs_change": round(float(diff.max()), 3)}
        s.analysis = {
            "calm_hours": calm, "rated_hours": rated, "storm_hours": storm,
            "turbine_spread_max": round(spread, 3), "input_update_vs_previous_issue": update,
            "needs_recalculation": bool(storm > 0 or (update and update["mean_abs_change"] > 0.15)),
        }
        return self._log("analyze", s.analysis)

    def recalculate(self, reason: str = "") -> dict:
        """Повторный расчёт: заново берём погоду (кэш обновится, если источник изменился) и пересчитываем.
        При штормовых часах применяем ограничение по отключению турбины."""
        s = self.state
        s.weather = forecast_available_at(FARM, s.issue_date, s.horizon_hours)
        self.prepare_and_predict()
        storm_mask = s.weather.set_index("ts")["wind_speed_100m"] >= STORM
        if storm_mask.any():
            idx = s.forecast["ts"].map(storm_mask).fillna(False).astype(bool)
            s.forecast.loc[idx, "power_norm_pred"] = 0.0
        s.recalculated = True
        return self._log("recalculate", {"reason": reason, "storm_hours_zeroed": int(storm_mask.sum())})

    def save_forecast(self) -> dict:
        s = self.state
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = s.forecast.copy()
        out["issue_date"] = s.issue_date
        out["horizon_hour"] = ((out["ts"] - pd.Timestamp(s.issue_date).normalize() - pd.Timedelta(days=1)).dt.total_seconds() // 3600 + 1).astype(int)
        cols = ["issue_date", "ts", "horizon_hour", "lead_day", "turbine", "power_norm_pred", "power_gbm_pred", "power_curve_pred"]
        path = OUT_DIR / f"{s.issue_date}.csv"
        out[cols].sort_values(["turbine", "ts"]).to_csv(path, index=False)
        log = {"issue_date": s.issue_date, "recalculated": s.recalculated, "analysis": s.analysis,
               "conclusion": s.conclusion, "trace": s.trace}
        (LOG_DIR / f"{s.issue_date}.json").write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return self._log("save_forecast", {"path": str(path), "rows": int(len(out))})

    # ---------- оркестрация ----------
    def run_day(self, issue_date: str, horizon_hours: int = 48, use_llm: bool | None = None) -> AgentState:
        self.state = AgentState(issue_date=issue_date, horizon_hours=horizon_hours)
        llm_ok = use_llm if use_llm is not None else bool(settings.openai_api_key or settings.nvidia_api_key) and not settings.demo_mode
        if llm_ok:
            try:
                self._run_with_llm()
            except Exception as e:  # сеть, лимиты, ключ: откат на детерминированный план
                self.state.trace.append({"tool": "llm_fallback", "result": {"error": str(e)[:200]}})
                self._run_deterministic()
        else:
            self._run_deterministic()
        if self.state.forecast is not None and not any(t["tool"] == "save_forecast" for t in self.state.trace):
            self.save_forecast()
        return self.state

    def _run_deterministic(self) -> None:
        self.get_weather()
        self.prepare_and_predict()
        a = self.analyze()
        if a["needs_recalculation"]:
            self.recalculate(reason="обновление входных данных или штормовые часы")
        self.state.conclusion = self._template_conclusion()
        self.save_forecast()

    def _template_conclusion(self) -> str:
        s, a = self.state, self.state.analysis
        means = {t: round(float(g["power_norm_pred"].mean()), 2) for t, g in s.forecast.groupby("turbine")}
        upd = a.get("input_update_vs_previous_issue")
        txt = (f"Прогноз на {s.horizon_hours} ч от {s.issue_date}: средняя нормализованная мощность "
               f"T1 {means.get(1)}, T2 {means.get(2)}. Часов штиля {a['calm_hours']}, часов на номинале {a['rated_hours']}, "
               f"штормовых {a['storm_hours']}.")
        if upd:
            txt += f" Относительно прогноза предыдущего дня входные данные изменились на {upd['mean_abs_change']} в среднем."
        if s.recalculated:
            txt += " Выполнен повторный расчёт."
        return txt

    def _run_with_llm(self) -> None:
        from app.llm import _client, _model  # noqa: WPS450

        client = _client()
        model = _model(reasoning=False)
        tool_specs = [
            {"type": "function", "function": {"name": "get_weather", "description": "Получить архивный прогноз погоды по координатам ВЭС, доступный на дату прогноза", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "prepare_and_predict", "description": "Подготовить признаки и запустить модель выработки для обеих турбин", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "analyze", "description": "Проанализировать прогноз: штиль, номинал, шторм, изменение входных данных относительно прошлого выпуска", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "recalculate", "description": "Повторный расчёт при обновлении входных данных или аномалиях", "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}}}},
            {"type": "function", "function": {"name": "save_forecast", "description": "Сохранить итоговый прогноз и журнал", "parameters": {"type": "object", "properties": {}}}},
        ]
        messages = [
            {"role": "system", "content": "Ты агент прогнозирования выработки ветроэлектростанции. Выполни полный цикл инструментами: погода, прогноз, анализ, при необходимости пересчёт, сохранение. В конце дай короткое заключение на русском: ожидаемая выработка, риски, изменилось ли что-то относительно прошлого выпуска."},
            {"role": "user", "content": f"Дата прогноза {self.state.issue_date}, горизонт {self.state.horizon_hours} часов."},
        ]
        for _ in range(10):
            resp = client.chat.completions.create(model=model, messages=messages, tools=tool_specs, temperature=0)
            msg = resp.choices[0].message
            messages.append(msg)
            if not msg.tool_calls:
                self.state.conclusion = msg.content or self._template_conclusion()
                break
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = self.tools[call.function.name](**args)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False, default=str)})
        if not self.state.conclusion:
            self.state.conclusion = self._template_conclusion()
