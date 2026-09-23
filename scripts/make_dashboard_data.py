"""Сборка данных дашборда в один JS-файл.

Дашборд открывается и по http, и напрямую из файла, поэтому данные подключаются
обычным <script>, а не загружаются fetch: с file:// запрос не проходит.

Источники — только артефакты репозитория:
  outputs/forecast_all_issues.csv  — выпуски по часам
  архив Open-Meteo через app.weather — направление ветра и порывы
  outputs/backtest_summary.csv     — сводка по выпускам
  outputs/agent_logs/<дата>.json   — журнал детерминированного выпуска
  outputs/agent_logs_llm/<дата>.json — журнал выпуска с LLM-оркестрацией (есть не для всех дат)
  docs/metrics.json                — метрики честной валидации января 2026

Запуск: python scripts/make_dashboard_data.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.weather import FARM, forecast_available_at

OUT = Path("docs/demo/dashboard-data.js")
ISSUES_CSV = Path("outputs/forecast_all_issues.csv")
SUMMARY_CSV = Path("outputs/backtest_summary.csv")
LOG_DIR = Path("outputs/agent_logs")
LOG_LLM_DIR = Path("outputs/agent_logs_llm")
METRICS = Path("docs/metrics.json")


def _round(v, n=4):
    return None if pd.isna(v) else round(float(v), n)


def _series(g: pd.DataFrame, col: str) -> list:
    return [_round(v) for v in g[col].tolist()]


def build_issue(day: str, g: pd.DataFrame, prev: pd.DataFrame | None, weather: pd.DataFrame | None = None) -> dict:
    """Один выпуск: 48 часов, обе турбины, коридор по станции и вчерашний выпуск.

    weather — кадр архивного прогноза на ту же дату: из него берутся направление ветра
    и порывы, которых нет в таблице выпусков, но которые нужны витрине.
    """
    g = g.sort_values(["ts", "turbine"])
    t1 = g[g["turbine"] == 1].sort_values("ts")
    t2 = g[g["turbine"] == 2].sort_values("ts")
    ts = [pd.Timestamp(v).strftime("%Y-%m-%d %H:%M") for v in t1["ts"]]
    farm = (t1["power_norm_pred"].values + t2["power_norm_pred"].values) / 2
    p10 = (t1["power_p10"].values + t2["power_p10"].values) / 2
    p90 = (t1["power_p90"].values + t2["power_p90"].values) / 2
    # вчерашний выпуск на пересекающихся часах: у него это часы лида 2
    prev_farm = [None] * len(ts)
    if prev is not None and len(prev):
        pv = prev.pivot_table(index="ts", columns="turbine", values="power_norm_pred", aggfunc="last")
        if 1 in pv.columns and 2 in pv.columns:
            pv = ((pv[1] + pv[2]) / 2).to_dict()
            prev_farm = [_round(pv.get(pd.Timestamp(v))) for v in t1["ts"]]
    direction, gusts = [None] * len(ts), [None] * len(ts)
    if weather is not None and len(weather):
        w = weather.set_index("ts")
        direction = [_round(w["wind_direction_100m"].get(pd.Timestamp(v)), 0) for v in t1["ts"]]
        gusts = [_round(w["wind_gusts_10m"].get(pd.Timestamp(v)), 1) for v in t1["ts"]]
    return {
        "issue_date": day,
        "ts": ts,
        "lead_day": [int(v) for v in t1["lead_day"]],
        "wind": _series(t1, "wind_speed_100m"),
        "wind_dir": direction,
        "gust": gusts,
        "t1": _series(t1, "power_norm_pred"),
        "t2": _series(t2, "power_norm_pred"),
        "farm": [_round(v) for v in farm],
        "p10": [_round(v) for v in p10],
        "p90": [_round(v) for v in p90],
        "prev_farm": prev_farm,
    }


def read_log(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return {
        "llm_used": bool(d.get("llm_used")),
        "recalculated": bool(d.get("recalculated")),
        "analysis": d.get("analysis", {}),
        "conclusion": d.get("conclusion", ""),
        "trace": [{"tool": t.get("tool"), "result": t.get("result", {})} for t in d.get("trace", [])],
    }


def model_version() -> dict:
    """Версия модели: коммит репозитория, время обучения и веса смеси."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        commit = ""
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    joblib = Path("models/turbine_1.joblib")
    trained = datetime.fromtimestamp(joblib.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if joblib.exists() else None
    return {
        "commit": commit,
        "trained_at": trained,
        "weights": {t: m[t]["weights"] for t in m},
        "metrics": {t: m[t]["honest_jan2026"] for t in m},
        "coverage": {t: m[t].get("p10_p90_coverage_jan2026") for t in m},
        "train_rows": {t: m[t].get("train_rows_final") for t in m},
    }


def main() -> None:
    issues_df = pd.read_csv(ISSUES_CSV, parse_dates=["ts"])
    summary = pd.read_csv(SUMMARY_CSV)
    days = sorted(issues_df["issue_date"].astype(str).unique())
    by_day = {d: g for d, g in issues_df.groupby(issues_df["issue_date"].astype(str))}

    issues = []
    for i, day in enumerate(days):
        prev = by_day.get(days[i - 1]) if i else None
        try:
            wx = forecast_available_at(FARM, day, 48)
        except Exception:  # кэш не покрывает дату — витрина обойдётся без направления
            wx = None
        item = build_issue(day, by_day[day], prev, wx)
        row = summary[summary["issue_date"].astype(str) == day]
        item["summary"] = {k: (None if pd.isna(v) else (bool(v) if isinstance(v, bool) else v))
                           for k, v in (row.iloc[0].to_dict().items() if len(row) else [])}
        item["agent"] = read_log(LOG_DIR / f"{day}.json")
        item["agent_llm"] = read_log(LOG_LLM_DIR / f"{day}.json")
        issues.append(item)

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": {
            "weather": "Open-Meteo Previous Runs API",
            "note": "для каждого часа берётся значение из прогноза, выпущенного за 1 сутки (часы 1–24) и за 2 суток (часы 25–48) до него",
            "timezone": "Asia/Almaty",
        },
        "model": model_version(),
        "thresholds": {"cut_in": 3.0, "rated": 11.5, "storm": 22.0},
        "issues": issues,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    js = "window.DASH = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n"
    OUT.write_text(js, encoding="utf-8")
    print(f"{OUT}: {len(issues)} выпусков, {OUT.stat().st_size // 1024} КБ")


if __name__ == "__main__":
    main()
