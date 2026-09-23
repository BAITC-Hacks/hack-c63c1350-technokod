"""Agentic AI: агент прогнозирования выработки ВЭС.

Цикл на каждую дату прогноза (issue_date):
  1. get_weather          — архивный прогноз погоды, доступный на issue_date (Open-Meteo Previous Runs)
  2. check_input_quality  — контроль входа: пропуски, неполный горизонт, застрявший датчик,
                            физически невозможные значения. Вердикт fail запрещает выпуск
  3. prepare_and_predict  — признаки + модель по каждой турбине, почасовой прогноз на 24–48 ч
  4. analyze              — проверка прогноза: диапазоны, штили, сильный ветер, сравнение с прогнозом
                            предыдущего дня на пересекающиеся часы (детектор обновления входных данных)
  5. recalculate          — повторный расчёт, если входные данные обновились или найдены аномалии
  6. save_forecast        — запись результата и журнала действий агента

Оркестрация: LLM (OpenAI, резерв NVIDIA NIM) вызывает инструменты через function calling и пишет
заключение. Без ключа, при ошибке провайдера и в DEMO_MODE тот же цикл выполняет детерминированный
планировщик, а заключение собирается по числам анализа шаблоном. Инструменты в обоих режимах одни
и те же, поэтому числовой результат выпуска от наличия ключа не зависит.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from app.model import GenerationModel
from app.weather import FARM, forecast_available_at, forecast_live

OUT_DIR = Path("outputs/forecasts")
LOG_DIR = Path("outputs/agent_logs")
CUT_IN = 3.0  # м/с, ниже — турбина не запускается
RATED = 11.5  # м/с, выше — полка номинальной мощности
STORM = 22.0  # м/с, выше — отключение по буревой защите
UPDATE_THRESHOLD = 0.15  # среднее изменение прогноза против вчерашнего выпуска, выше — пересчёт

# --- пороги контроля качества входных данных ---
# Прогноз, посчитанный по испорченному входу, выглядит так же уверенно, как исправный,
# поэтому вход проверяется до расчёта, а не после.
MAX_WIND = 40.0  # м/с, выше — неисправность источника: максимум за всю историю площадки ~23 м/с
MIN_TEMP = -60.0  # °C, ниже — неисправность: абсолютный минимум Казахстана около -57 °C
MAX_TEMP = 60.0  # °C, выше — неисправность: абсолютный максимум около 49 °C
STUCK_HOURS = 6  # часов подряд с точно одинаковой скоростью — застрявшее значение датчика или источника
MIN_WIND_STD = 0.3  # м/с, разброс скорости за горизонт ниже — подозрительно ровный ряд, вероятна подмена

VERDICT_OK, VERDICT_WARN, VERDICT_FAIL = "ok", "warn", "fail"

# Колонки выпуска. Первые пять — ключ, дальше прогноз смеси, компоненты и интервал.
FORECAST_COLUMNS = [
    "issue_date", "ts", "horizon_hour", "lead_day", "turbine",
    "wind_speed_100m", "power_norm_pred", "power_p10", "power_p90",
    "power_gbm_pred", "power_two_stage_pred", "power_curve_pred", "wind_nacelle_pred",
]

_UNSET = object()


@dataclass
class AgentState:
    issue_date: str
    horizon_hours: int = 48
    weather: pd.DataFrame | None = None
    forecast: pd.DataFrame | None = None
    analysis: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)  # вердикт контроля качества входа
    recalculated: bool = False
    published: bool = False  # выпуск записан на диск; при вердикте fail остаётся False
    trace: list[dict] = field(default_factory=list)
    conclusion: str = ""
    previous: pd.DataFrame | None = None  # выпуск предыдущего дня для детектора обновления
    llm_used: bool = False  # порядок инструментов выбрала модель, отката не было

    # синонимы: бэктест и сервис читают результат как tool_trace / narrative / recomputed
    @property
    def tool_trace(self) -> list[dict]:
        return self.trace

    @property
    def narrative(self) -> str:
        return self.conclusion

    @property
    def recomputed(self) -> bool:
        return self.recalculated

    @property
    def confidence(self) -> str:
        """Достоверность выпуска по вердикту контроля входа."""
        verdict = self.quality.get("verdict", VERDICT_OK)
        return {VERDICT_OK: "обычная", VERDICT_WARN: "пониженная", VERDICT_FAIL: "выпуск не состоялся"}[verdict]


class ForecastAgent:
    """Агент выпуска прогноза. Один экземпляр обслуживает любое число дат.

    Параметры задаются при создании, отдельный выпуск может их переопределить:
        agent = ForecastAgent(use_llm=False)
        state = agent.run_day("2026-02-05", previous=вчерашний_выпуск)
    """

    def __init__(
        self,
        turbines: tuple[int, ...] = (1, 2),
        use_llm: bool | None = None,
        horizon_hours: int = 48,
        max_steps: int = 10,
        models: dict[int, GenerationModel] | None = None,
        live: bool = False,
    ) -> None:
        self.live = live  # оперативный выпуск на завтра вместо воспроизведения прошлого
        self.turbines = tuple(turbines)
        if models is None:
            missing = [t for t in self.turbines if not (Path("models") / f"turbine_{t}.joblib").exists()]
            if missing:
                raise RuntimeError("Модели не обучены. Выполните: python -m app.cli train")
            models = {t: GenerationModel.load(t) for t in self.turbines}
        self.models = {t: m for t, m in models.items() if t in self.turbines}
        self.use_llm = use_llm
        self.horizon_hours = horizon_hours
        self.max_steps = max_steps
        self.tools: dict[str, Callable[..., dict]] = {
            "get_weather": self.get_weather,
            "check_input_quality": self.check_input_quality,
            "prepare_and_predict": self.prepare_and_predict,
            "analyze": self.analyze,
            "recalculate": self.recalculate,
            "save_forecast": self.save_forecast,
        }
        self.state: AgentState | None = None

    # ---------- инструменты ----------
    def _log(self, tool: str, result: dict) -> dict:
        self.state.trace.append({"tool": tool, "result": result})
        return result

    def get_weather(self, issue_date: str | None = None) -> dict:
        """Прогноз на 24–48 часов по координатам ВЭС. Факт погоды не используется.

        Ретроспектива (по умолчанию): архивный срез Previous Runs на дату выпуска.
        Оперативный режим (`live=True` при создании агента): текущий прогон Forecast API —
        так решение выпускает прогноз на завтра, а не только воспроизводит прошлое.
        """
        s = self.state
        if self.live:
            s.weather = forecast_live(FARM, s.horizon_hours)
        else:
            s.weather = forecast_available_at(FARM, issue_date or s.issue_date, s.horizon_hours)
        w = s.weather
        return self._log("get_weather", {
            "hours": int(len(w)), "from": str(w["ts"].min()), "to": str(w["ts"].max()),
            "lead1_hours": int((w["lead_day"] == 1).sum()), "lead2_hours": int((w["lead_day"] == 2).sum()),
            "ws100_min": round(float(w["wind_speed_100m"].min()), 1),
            "ws100_mean": round(float(w["wind_speed_100m"].mean()), 1),
            "ws100_max": round(float(w["wind_speed_100m"].max()), 1),
            "temp_min": round(float(w["temperature_2m"].min()), 1),
            "temp_max": round(float(w["temperature_2m"].max()), 1),
            "source": ("Open-Meteo Forecast API, текущий прогон (оперативный выпуск)" if self.live
                       else "Open-Meteo Previous Runs, прогнозы суток выпуска"),
        })

    def check_input_quality(self) -> dict:
        """Контроль качества погодного кадра до расчёта.

        Прогноз по испорченному входу неотличим по виду от исправного, поэтому вход проверяется
        отдельным шагом. Вердикт:
          fail — данные использовать нельзя, выпуск не публикуется;
          warn — выпуск возможен, но с пониженной достоверностью;
          ok   — ограничений нет.
        """
        s = self.state
        if s.weather is None:
            return self._log("check_input_quality", {"error": "погода не получена, сначала вызовите get_weather"})
        w = s.weather
        findings: list[dict] = []

        def add(level: str, code: str, message: str, **extra):
            findings.append({"level": level, "code": code, "message": message, **extra})

        # 1. полнота горизонта и непрерывность часовой сетки
        if len(w) < s.horizon_hours:
            add(VERDICT_FAIL, "incomplete_horizon",
                f"получено {len(w)} часов из запрошенных {s.horizon_hours}", hours=int(len(w)))
        gaps = w["ts"].sort_values().diff().dropna()
        bad_steps = int((gaps != pd.Timedelta(hours=1)).sum())
        if bad_steps:
            add(VERDICT_FAIL, "grid_gaps", f"разрывов часовой сетки: {bad_steps}", gaps=bad_steps)

        # 2. пропуски в ключевых полях
        for col in ("wind_speed_100m", "temperature_2m"):
            if col in w.columns:
                n = int(w[col].isna().sum())
                if n:
                    add(VERDICT_FAIL, "missing_values", f"пропусков в поле {col}: {n}", field=col, count=n)

        ws = w["wind_speed_100m"].dropna()
        if ws.empty:
            add(VERDICT_FAIL, "no_wind_data", "скорость ветра отсутствует целиком")
        else:
            # 3. физически невозможные значения
            neg = int((ws < 0).sum())
            if neg:
                add(VERDICT_FAIL, "negative_wind", f"отрицательная скорость ветра в {neg} часах", count=neg)
            over = int((ws > MAX_WIND).sum())
            if over:
                add(VERDICT_FAIL, "wind_out_of_range",
                    f"скорость выше физического предела {MAX_WIND} м/с в {over} часах", count=over)

            # 4. застрявшее значение: серия точно одинаковых отсчётов
            runs = (ws != ws.shift()).cumsum()
            longest = int(runs.value_counts().max())
            if longest >= STUCK_HOURS:
                add(VERDICT_FAIL, "stuck_sensor",
                    f"одинаковая скорость ветра {longest} часов подряд при пороге {STUCK_HOURS}", hours=longest)

            # 5. подозрительно ровный ряд
            std = float(ws.std()) if len(ws) > 1 else 0.0
            if std < MIN_WIND_STD:
                add(VERDICT_WARN, "low_variability",
                    f"разброс скорости за горизонт {std:.2f} м/с ниже {MIN_WIND_STD}", std=round(std, 3))

        if "temperature_2m" in w.columns:
            t = w["temperature_2m"].dropna()
            out_of_range = int(((t < MIN_TEMP) | (t > MAX_TEMP)).sum())
            if out_of_range:
                add(VERDICT_FAIL, "temp_out_of_range",
                    f"температура вне диапазона {MIN_TEMP}..{MAX_TEMP} °C в {out_of_range} часах", count=out_of_range)

        levels = {f["level"] for f in findings}
        verdict = VERDICT_FAIL if VERDICT_FAIL in levels else (VERDICT_WARN if VERDICT_WARN in levels else VERDICT_OK)
        s.quality = {
            "verdict": verdict,
            "hours": int(len(w)),
            "expected_hours": int(s.horizon_hours),
            "findings": findings,
            "publish_allowed": verdict != VERDICT_FAIL,
        }
        return self._log("check_input_quality", s.quality)

    def prepare_and_predict(self) -> dict:
        """Признаки и прогноз выработки по каждой турбине на весь горизонт."""
        s = self.state
        if s.weather is None:
            return self._log("prepare_and_predict", {"error": "погода не получена, сначала вызовите get_weather"})
        frames = []
        for t, model in self.models.items():
            p = model.predict(s.weather)
            p["turbine"] = t
            frames.append(p)
        f = pd.concat(frames, ignore_index=True)
        # ключевые колонки выпуска проставляются сразу: их читает и бэктест, и запись на диск
        issue = pd.Timestamp(s.issue_date).normalize()
        f["issue_date"] = s.issue_date
        f["horizon_hour"] = ((f["ts"] - issue - pd.Timedelta(days=1)).dt.total_seconds() // 3600 + 1).astype(int)
        f = f.merge(s.weather[["ts", "wind_speed_100m"]], on="ts", how="left")
        s.forecast = f
        mean_cf = {str(t): round(float(g["power_norm_pred"].mean()), 3) for t, g in f.groupby("turbine")}
        return self._log("prepare_and_predict", {
            "hours": int(f["ts"].nunique()),
            "rows": int(len(f)),
            "mean_cf": mean_cf,
            "energy_norm_h": {str(t): round(float(g["power_norm_pred"].sum()), 1) for t, g in f.groupby("turbine")},
        })

    def analyze(self) -> dict:
        """Проверка выпуска и детектор обновления входных данных относительно вчерашнего прогноза."""
        s = self.state
        if s.weather is None or s.forecast is None:
            return self._log("analyze", {"error": "нечего анализировать, сначала get_weather и prepare_and_predict"})
        w, f = s.weather, s.forecast
        calm = int((w["wind_speed_100m"] < CUT_IN).sum())
        rated = int((w["wind_speed_100m"] >= RATED).sum())
        storm = int((w["wind_speed_100m"] >= STORM).sum())
        by_ts = f.groupby("ts")["power_norm_pred"]
        spread = float((by_ts.max() - by_ts.min()).max())
        # сравнение с выпуском предыдущего дня на пересекающиеся часы:
        # часы 25–48 вчерашнего выпуска — это часы 1–24 сегодняшнего
        prev = s.previous
        prev_source = "передан вызывающим" if prev is not None else "отсутствует"
        if prev is None:
            prev_path = OUT_DIR / f"{(pd.Timestamp(s.issue_date) - pd.Timedelta(days=1)).date()}.csv"
            if prev_path.exists():
                prev = pd.read_csv(prev_path, parse_dates=["ts"])
                prev_source = f"прочитан с диска: {prev_path.name}"
        update = None
        if prev is not None and len(prev):
            merged = f.merge(prev[["ts", "turbine", "power_norm_pred"]], on=["ts", "turbine"], suffixes=("", "_prev"))
            if len(merged):
                diff = (merged["power_norm_pred"] - merged["power_norm_pred_prev"]).abs()
                update = {
                    "overlap_hours": int(merged["ts"].nunique()),
                    "mean_abs_change": round(float(diff.mean()), 3),
                    "max_abs_change": round(float(diff.max()), 3),
                }
        mean_change = update["mean_abs_change"] if update else None
        s.analysis = {
            "calm_hours": calm,
            "rated_hours": rated,
            "storm_hours": storm,
            "turbine_spread_max": round(spread, 3),
            "mean_cf": {str(t): round(float(g["power_norm_pred"].mean()), 3) for t, g in f.groupby("turbine")},
            # сравнение возможно только при наличии вчерашнего выпуска: если его нет, это явный
            # признак, а не молчаливое отсутствие поля
            "previous_issue_available": update is not None,
            "previous_issue_source": prev_source,
            "input_update_vs_previous_issue": update,
            "input_change_vs_previous_issue": mean_change,
            "update_threshold": UPDATE_THRESHOLD,
            "needs_recalculation": bool(storm > 0 or (mean_change is not None and mean_change > UPDATE_THRESHOLD)),
        }
        return self._log("analyze", s.analysis)

    def _recalculation_trigger(self) -> dict:
        """Какой именно критерий сработал и с какими числами. Считается из анализа, а не со слов
        вызывающего: в журнале должно быть видно основание решения, а не его пересказ."""
        s = self.state
        a = s.analysis or {}
        storm = a.get("storm_hours") or 0
        change = a.get("input_change_vs_previous_issue")
        threshold = a.get("update_threshold", UPDATE_THRESHOLD)
        if storm:
            return {"trigger": "storm_hours",
                    "detail": f"штормовых часов {storm} при пороге отключения {STORM} м/с",
                    "values": {"storm_hours": storm, "storm_threshold_ms": STORM}}
        if change is not None and change > threshold:
            prev_day = (pd.Timestamp(s.issue_date) - pd.Timedelta(days=1)).date()
            return {"trigger": "input_update",
                    "detail": f"расхождение с выпуском {prev_day} равно {change} при пороге {threshold}",
                    "values": {"mean_abs_change": change, "threshold": threshold, "previous_issue": str(prev_day)}}
        if s.quality.get("verdict") == VERDICT_WARN:
            return {"trigger": "quality_warn",
                    "detail": "контроль входа дал предупреждение",
                    "values": {"verdict": VERDICT_WARN}}
        return {"trigger": "manual", "detail": "вызов без сработавшего критерия", "values": {}}

    def recalculate(self, reason: str = "") -> dict:
        """Повторный расчёт выпуска: источник прогноза опрашивается заново, модель считается заново,
        штормовые часы обнуляются (буревая защита турбины).

        В бэктесте по архиву повторный запрос возвращает тот же срез архива, поэтому фиксируется,
        изменились ли входные данные фактически: это видно в поле input_changed журнала.
        """
        s = self.state
        if s.weather is None:
            return self._log("recalculate", {"error": "нечего пересчитывать, сначала get_weather"})
        before = s.weather["wind_speed_100m"].to_numpy(copy=True)
        s.weather = forecast_available_at(FARM, s.issue_date, s.horizon_hours)
        after = s.weather["wind_speed_100m"].to_numpy()
        input_changed = bool(len(before) != len(after) or (abs(before - after) > 1e-9).any())
        self.prepare_and_predict()
        storm_mask = s.weather.set_index("ts")["wind_speed_100m"] >= STORM
        zeroed = 0
        if storm_mask.any():
            idx = s.forecast["ts"].map(storm_mask).fillna(False).astype(bool)
            # обнуляем прогноз целиком, иначе компоненты смеси противоречат итогу
            power_cols = [c for c in s.forecast.columns if c.startswith("power_")]
            s.forecast.loc[idx, power_cols] = 0.0
            zeroed = int(idx.sum())
        s.recalculated = True
        trigger = self._recalculation_trigger()
        return self._log("recalculate", {
            "trigger": trigger["trigger"],
            "trigger_detail": trigger["detail"],
            "trigger_values": trigger["values"],
            # формулировка вызывающего: планировщика или модели в LLM-режиме. Хранится отдельно
            # от вычисленного триггера, чтобы было видно, на те же ли числа опиралось решение
            "reason_passed": reason or None,
            "input_changed": input_changed,
            "note": ("повторный опрос архива на прошедшую дату возвращает тот же срез, поэтому "
                     "input_changed=false; в промышленном режиме здесь пришёл бы новый выпуск прогноза погоды"),
            "storm_hours": int(storm_mask.sum()),
            "rows_zeroed": zeroed,
        })

    def save_forecast(self) -> dict:
        """Запись выпуска на диск. До этого вызова прогноз считается невыпущенным.

        Публикация запрещена, если контроль входа дал вердикт fail или если в прогнозе остались
        пропуски: выпуск с пустыми значениями мощности хуже отсутствия выпуска — диспетчер примет
        его за исправный.
        """
        s = self.state
        if s.forecast is None:
            return self._log("save_forecast", {"error": "прогноза нет, сначала prepare_and_predict"})
        if s.quality.get("verdict") == VERDICT_FAIL:
            codes = [f["code"] for f in s.quality.get("findings", []) if f["level"] == VERDICT_FAIL]
            return self._log("save_forecast", {
                "published": False,
                "error": "выпуск отклонён: контроль входных данных дал вердикт fail",
                "reasons": codes,
            })
        power_cols = [c for c in s.forecast.columns if c.startswith("power_")]
        nan_rows = int(s.forecast[power_cols].isna().any(axis=1).sum())
        if nan_rows:
            return self._log("save_forecast", {
                "published": False,
                "error": f"выпуск отклонён: в прогнозе {nan_rows} строк с пропусками мощности",
                "nan_rows": nan_rows,
            })
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        cols = [c for c in FORECAST_COLUMNS if c in s.forecast.columns]
        path = OUT_DIR / f"{s.issue_date}.csv"
        out = s.forecast[cols].sort_values(["turbine", "ts"]).copy()
        out[out.select_dtypes("float").columns] = out.select_dtypes("float").round(4)
        out.to_csv(path, index=False)
        s.published = True
        return self._log("save_forecast", {
            "published": True,
            "path": str(path),
            "rows": int(len(s.forecast)),
            "confidence": s.confidence,
        })

    # ---------- оркестрация ----------
    def run_day(
        self,
        issue_date: str,
        previous: pd.DataFrame | None = None,
        horizon_hours: int | None = None,
        use_llm: bool | None = _UNSET,  # type: ignore[assignment]
    ) -> AgentState:
        """Один выпуск: полный цикл инструментов, журнал и сохранение.

        use_llm=None — автоматически: оркеструет модель, если задан ключ провайдера.
        """
        from app.llm import available

        self.state = AgentState(
            issue_date=issue_date,
            horizon_hours=horizon_hours or self.horizon_hours,
            previous=previous,
        )
        want_llm = self.use_llm if use_llm is _UNSET else use_llm
        llm_ok = available() if want_llm is None else (bool(want_llm) and available())
        if llm_ok:
            try:
                self._run_with_llm()
                self.state.llm_used = True
            except Exception as e:  # сеть, лимиты, ключ: откат на детерминированный план
                self.state.trace.append({"tool": "llm_fallback", "result": {"error": str(e)[:200]}})
                self._run_deterministic()
        else:
            self._run_deterministic()
        # страховка: модель могла пропустить шаг — контроль входа, анализ и сохранение
        # выполняются принудительно, иначе выпуск окажется неполным или непроверенным
        if self.state.weather is not None and not self.state.quality:
            self.check_input_quality()
        if self.state.forecast is not None and not self.state.analysis:
            self.analyze()
        if self.state.forecast is not None and not any(t["tool"] == "save_forecast" for t in self.state.trace):
            self.save_forecast()
        if not self.state.conclusion:
            self.state.conclusion = self._template_conclusion()
        self._write_log()  # заключение модели появляется после инструментов, журнал пишем в конце
        return self.state

    def _write_log(self) -> None:
        """Журнал выпуска целиком: контроль входа, анализ, полная трасса и заключение.

        Журнал пишется и тогда, когда выпуск отклонён: отказ должен быть зафиксирован так же,
        как состоявшийся прогноз.
        """
        s = self.state
        if s is None:
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = {
            "issue_date": s.issue_date,
            "llm_used": s.llm_used,
            "published": s.published,
            "confidence": s.confidence,
            "quality": s.quality,
            "recalculated": s.recalculated,
            "analysis": s.analysis,
            "conclusion": s.conclusion,
            "trace": s.trace,
        }
        (LOG_DIR / f"{s.issue_date}.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    def _run_deterministic(self) -> None:
        """Тот же цикл без модели: порядок инструментов фиксирован."""
        self.get_weather()
        q = self.check_input_quality()
        if q.get("verdict") == VERDICT_FAIL:
            # расчёт по заведомо негодному входу не запускается: выпуска не будет
            self.state.conclusion = self._refusal_conclusion()
            return
        self.prepare_and_predict()
        a = self.analyze()
        if a.get("needs_recalculation"):
            self.recalculate(reason=self._recalculation_trigger()["detail"])
        self.state.conclusion = self._template_conclusion()
        self.save_forecast()

    def _refusal_conclusion(self) -> str:
        """Заключение при отказе от выпуска: что именно не так со входом."""
        s = self.state
        problems = "; ".join(f["message"] for f in s.quality.get("findings", []) if f["level"] == VERDICT_FAIL)
        return (f"Выпуск {s.issue_date} не состоялся: входные данные не прошли контроль качества. "
                f"Обнаружено: {problems}. Прогноз по таким данным не публикуется, требуется повторный "
                f"запрос источника или ручная проверка.")

    def _template_conclusion(self) -> str:
        s = self.state
        a = s.analysis or {}
        if s.quality.get("verdict") == VERDICT_FAIL:
            return self._refusal_conclusion()
        if s.forecast is None:
            return f"Выпуск {s.issue_date} не состоялся: прогноз не рассчитан."
        means = {t: round(float(g["power_norm_pred"].mean()), 2) for t, g in s.forecast.groupby("turbine")}
        energy = {t: round(float(g["power_norm_pred"].sum()), 1) for t, g in s.forecast.groupby("turbine")}
        txt = (f"Прогноз на {s.horizon_hours} ч от {s.issue_date}: средняя нормализованная мощность "
               f"T1 {means.get(1)}, T2 {means.get(2)}; ожидаемая выработка {energy.get(1)} и {energy.get(2)} норм·ч.")
        if a:
            txt += (f" Часов штиля {a.get('calm_hours')}, часов на номинале {a.get('rated_hours')}, "
                    f"штормовых {a.get('storm_hours')}.")
        upd = a.get("input_update_vs_previous_issue")
        if upd:
            txt += (f" Относительно прогноза предыдущего дня входные данные изменились в среднем на "
                    f"{upd['mean_abs_change']} при пороге пересчёта {UPDATE_THRESHOLD}.")
        txt += " Выполнен повторный расчёт." if s.recalculated else " Пересчёт не потребовался."
        if s.quality.get("verdict") == VERDICT_WARN:
            notes = "; ".join(f["message"] for f in s.quality.get("findings", []) if f["level"] == VERDICT_WARN)
            txt += f" Достоверность понижена: {notes}."
        if not a.get("previous_issue_available", True):
            txt += " Сравнить с предыдущим выпуском не удалось: вчерашнего прогноза нет."
        return txt

    def _run_with_llm(self) -> None:
        """Оркестрация моделью через function calling (app.llm.chat_tools).

        Порядок инструментов выбирает модель. Ошибка инструмента возвращается модели как результат,
        чтобы она исправилась сама, а не роняла весь выпуск в откат.
        """
        from app.llm import chat_tools

        tool_specs = [
            {"type": "function", "function": {
                "name": "get_weather",
                "description": "Получить архивный прогноз погоды по координатам ВЭС, доступный на дату выпуска. Вызывать первым.",
                "parameters": {"type": "object", "properties": {"issue_date": {"type": "string", "description": "дата выпуска, по умолчанию текущая"}}}}},
            {"type": "function", "function": {
                "name": "check_input_quality",
                "description": ("Проверить качество полученного прогноза погоды до расчёта: пропуски, неполный "
                                "горизонт, разрывы часовой сетки, застрявшее значение датчика, физически "
                                "невозможные величины. Возвращает verdict: ok — считать можно; warn — считать "
                                "можно, но выпуск пометить пониженной достоверностью; fail — данные негодны, "
                                "расчёт и публикация запрещены. Вызывать сразу после get_weather."),
                "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {
                "name": "prepare_and_predict",
                "description": "Подготовить признаки и запустить модель выработки для обеих турбин. Требует выполненного get_weather.",
                "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {
                "name": "analyze",
                "description": "Проанализировать выпуск: часы штиля, номинала, шторма, расхождение с прогнозом предыдущего дня. Возвращает needs_recalculation.",
                "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {
                "name": "recalculate",
                "description": "Повторно опросить источник прогноза, пересчитать выработку и применить буревую защиту. Вызывать при needs_recalculation.",
                "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}}}},
            {"type": "function", "function": {
                "name": "save_forecast",
                "description": "Сохранить выпуск и журнал. Обязателен: без него прогноз не выпущен.",
                "parameters": {"type": "object", "properties": {}}}},
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": (
                "Ты агент прогнозирования выработки ветроэлектростанции из двух турбин в Шелекском коридоре. "
                "Выполни полный цикл выпуска инструментами: получить прогноз погоды, проверить качество "
                "входных данных, рассчитать выработку, проанализировать результат, при необходимости "
                "пересчитать, обязательно сохранить. "
                "Проверка входа обязательна и идёт сразу после получения погоды. "
                "Если check_input_quality вернул verdict=fail — выпускать прогноз запрещено: не вызывай "
                "prepare_and_predict и save_forecast, вместо этого объясни в заключении, что именно не так "
                "со входными данными и почему выпуск не состоялся. "
                "Если verdict=warn — выпускай, но в заключении укажи пониженную достоверность и причину. "
                "Жёсткое правило задачи: разрешено использовать только прогноз погоды, доступный на дату выпуска; "
                "фактическая погода и фактическая выработка за прогнозируемые сутки недоступны и использовать их нельзя. "
                "Решение о пересчёте принимай по полю needs_recalculation и по числам анализа. "
                "Пока не вызван save_forecast, прогноз не выпущен. "
                "В конце дай короткое заключение на русском (3–5 предложений): ожидаемая выработка по турбинам, "
                "риски (штиль, шторм), изменилось ли что-то относительно прошлого выпуска и потребовался ли пересчёт."
            )},
            {"role": "user", "content": (
                f"Дата выпуска {self.state.issue_date}, горизонт {self.state.horizon_hours} часов, "
                f"турбины {', '.join(str(t) for t in self.turbines)}."
            )},
        ]
        for _ in range(self.max_steps):
            step = chat_tools(messages, tool_specs, temperature=0)
            if not step["tool_calls"]:
                self.state.conclusion = step["content"] or self._template_conclusion()
                break
            messages.append({"role": "assistant", "content": step["content"], "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                for c in step["tool_calls"]
            ]})
            for c in step["tool_calls"]:
                fn = self.tools.get(c["name"])
                try:
                    args = json.loads(c["arguments"] or "{}")
                    result = fn(**args) if fn else {"error": f"неизвестный инструмент {c['name']}"}
                except Exception as e:  # ошибку инструмента модель видит и исправляет сама
                    result = {"error": str(e)[:200]}
                    self.state.trace.append({"tool": c["name"], "result": result})
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": json.dumps(result, ensure_ascii=False, default=str)})
        if not self.state.conclusion:
            self.state.conclusion = self._template_conclusion()
