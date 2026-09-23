"""HTTP-сервис для интеграции с системами диспетчера: прогноз агентом по дате выпуска.

python -m app.serve            (порт из APP_PORT, по умолчанию 8000)
GET /health
GET /forecast/{issue_date}?horizon=48&llm=false   -> почасовой прогноз по турбинам, анализ, заключение
GET /submission                                   -> итоговый ряд февраля 2026, если бэктест уже выполнен
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.config import settings

app = FastAPI(title="WindAgent", version="0.1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_trained": Path("models/turbine_1.joblib").exists(), "llm_provider": settings.llm_provider}


@app.get("/forecast/{issue_date}")
def forecast(issue_date: str, horizon: int = 48, llm: bool = False) -> dict:
    from app.agent import ForecastAgent

    try:
        pd.Timestamp(issue_date)
    except Exception:
        raise HTTPException(400, "issue_date должна быть в формате YYYY-MM-DD")
    if horizon not in (24, 48):
        raise HTTPException(400, "horizon: 24 или 48")
    try:
        s = ForecastAgent().run_day(issue_date, horizon_hours=horizon, use_llm=llm)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    f = s.forecast.copy()
    f["ts"] = f["ts"].dt.strftime("%Y-%m-%d %H:%M")
    cols = ["ts", "lead_day", "turbine", "power_norm_pred", "power_p10", "power_p90"]
    return {
        "issue_date": s.issue_date,
        "horizon_hours": horizon,
        "tools": [t["tool"] for t in s.trace],
        "analysis": s.analysis,
        "recalculated": s.recalculated,
        "conclusion": s.conclusion,
        "forecast": f[cols].round(4).to_dict(orient="records"),
    }


@app.get("/submission")
def submission() -> dict:
    p = Path("outputs/forecast_feb2026.csv")
    if not p.exists():
        raise HTTPException(404, "нет outputs/forecast_feb2026.csv, выполните python -m app.cli backtest")
    df = pd.read_csv(p)
    return {"rows": len(df), "data": df.round(4).to_dict(orient="records")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.serve:app", host=settings.app_host, port=settings.app_port)
