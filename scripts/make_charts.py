"""Построение иллюстраций для README из артефактов репозитория.

Строит две картинки:
- assets/validation_jan2026.png — факт выработки против честного прогноза за показательную неделю января 2026;
- assets/forecast_feb2026.png — итоговый почасовой прогноз февраля 2026 с полосой P10–P90.

Сеть и ключи не нужны: погода берётся из кэша data/cache/weather, модели из models/,
факт из data/raw, числа ошибки из docs/metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from app.data import load_hourly
from app.model import GenerationModel
from app.weather import FARM, forecast_available_at

DOCS = Path("docs")
ASSETS = Path("assets")  # иллюстрации репозитория, см. CONTRIBUTING.md
OUTPUTS = Path("outputs")
FIGSIZE = (14.5, 5.5)  # при dpi=110 это примерно 1600x600 точек
DPI = 110
# неделя января 2026, на которой видны и штиль, и работа на номинале
VAL_START = pd.Timestamp("2026-01-12")
VAL_DAYS = 7

plt.rcParams["font.family"] = "DejaVu Sans"  # шрифт с кириллицей, идёт в комплекте matplotlib
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def honest_forecast(turbine: int, start: pd.Timestamp, days: int) -> pd.DataFrame:
    """Прогноз «как в прошлом»: на каждые сутки берётся выпуск предыдущего дня, часы 1–24."""
    model = GenerationModel.load(turbine)
    frames = []
    for i in range(days):
        issue = (start + pd.Timedelta(days=i) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        weather = forecast_available_at(FARM, issue, 48)
        pred = model.predict(weather)
        frames.append(pred[pred["lead_day"] == 1][["ts", "power_norm_pred"]])
    return pd.concat(frames, ignore_index=True).sort_values("ts")


def chart_validation() -> Path:
    metrics = json.loads((DOCS / "metrics.json").read_text(encoding="utf-8"))
    nmae = metrics["turbine_1"]["honest_jan2026"]["lead1"]["blend"]["nmae_pct"]

    pred = honest_forecast(1, VAL_START, VAL_DAYS)
    fact = load_hourly(1)
    end = VAL_START + pd.Timedelta(days=VAL_DAYS)
    fact = fact[(fact["ts"] >= VAL_START) & (fact["ts"] < end)]
    df = pred.merge(fact[["ts", "power_norm"]], on="ts", how="inner")

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(df["ts"], df["power_norm"], color="#1f3b73", linewidth=1.8, label="Факт выработки")
    ax.plot(df["ts"], df["power_norm_pred"], color="#d1495b", linewidth=1.8, label="Прогноз за сутки вперёд")
    ax.set_title(
        f"Турбина 1, неделя {VAL_START:%d.%m.%Y} – {end - pd.Timedelta(hours=1):%d.%m.%Y}: "
        f"факт против прогноза, доступного накануне (nMAE января {nmae} %)"
    )
    ax.set_xlabel("Местное время (Алматы)")
    ax.set_ylabel("Нормированная мощность, 0..1")
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = ASSETS / "validation_jan2026.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path}: часов на графике {len(df)}, факт {df['power_norm'].mean():.3f}, прогноз {df['power_norm_pred'].mean():.3f}")
    return path


def chart_february() -> Path:
    df = pd.read_csv(OUTPUTS / "forecast_feb2026.csv", parse_dates=["ts"])

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.fill_between(
        df["ts"], df["turbine_1_p10"], df["turbine_1_p90"],
        color="#1f3b73", alpha=0.15, label="Интервал P10–P90, турбина 1",
    )
    ax.plot(df["ts"], df["turbine_1_lead1"], color="#1f3b73", linewidth=1.3, label="Турбина 1")
    ax.plot(df["ts"], df["turbine_2_lead1"], color="#e07a1f", linewidth=1.3, alpha=0.85, label="Турбина 2")
    mean_1, mean_2 = df["turbine_1_lead1"].mean(), df["turbine_2_lead1"].mean()
    ax.set_title(
        f"Прогноз выработки на февраль 2026, {len(df)} часов: "
        f"средняя мощность турбина 1 — {mean_1:.3f}, турбина 2 — {mean_2:.3f}"
    )
    ax.set_xlabel("Местное время (Алматы)")
    ax.set_ylabel("Нормированная мощность, 0..1")
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    path = ASSETS / "forecast_feb2026.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path}: часов {len(df)}, пропусков lead1 {int(df['turbine_1_lead1'].isna().sum())}")
    return path


def main() -> None:
    if not (Path("models") / "turbine_1.joblib").exists():
        raise SystemExit("Модели не обучены. Выполните: python -m app.cli train")
    ASSETS.mkdir(exist_ok=True)
    chart_validation()
    chart_february()


if __name__ == "__main__":
    main()
