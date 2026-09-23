"""Командная строка проекта.

  python -m app.cli check                        проверка окружения и данных
  python -m app.cli fetch-weather --start --end  прогрев кэша архивных прогнозов
  python -m app.cli train                        обучение моделей
  python -m app.cli forecast --date 2026-01-31   прогноз агентом на одну дату
  python -m app.cli backtest                     все выпуски теста (31.01–27.02.2026)
  python -m app.cli report                       сводка по посчитанным результатам

Режим LLM: без флагов агент использует модель, если задан ключ, иначе детерминированный план.
Флаг --no-llm выключает LLM принудительно (воспроизведение без ключей), --llm требует его включения.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Windows-консоль по умолчанию не в UTF-8, русский вывод иначе ломается.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def _llm_flag(args: argparse.Namespace) -> bool | None:
    """None — решает наличие ключа, True — требуем LLM, False — детерминированный план."""
    if getattr(args, "no_llm", False):
        return False
    if getattr(args, "llm", False):
        return True
    return None


def _add_llm_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--llm", action="store_true", help="оркестрация через LLM")
    g.add_argument("--no-llm", action="store_true", help="детерминированный план, без обращений к LLM")


def cmd_check() -> int:
    from app.config import settings

    print(f"python           {sys.version.split()[0]}")
    print(f"каталог проекта  {ROOT}")
    for t in (1, 2):
        raw = ROOT / "data" / "raw" / f"turbine_{t}.csv"
        size = f"{raw.stat().st_size / 1e6:.1f} МБ" if raw.exists() else "НЕТ"
        print(f"данные турбины {t} data/raw/turbine_{t}.csv: {size}")
    models = [t for t in (1, 2) if (ROOT / "models" / f"turbine_{t}.joblib").exists()]
    print(f"модели           обучены: {models if models else 'нет, выполните python -m app.cli train'}")
    cache = list((ROOT / "data" / "cache" / "weather").glob("*.json"))
    offline = "да" if cache else "нет"
    print(f"кэш погоды       файлов: {len(cache)}, прогон без сети: {offline}")
    key = bool(settings.openai_api_key or settings.nvidia_api_key)
    print(
        f"LLM              провайдер {settings.llm_provider}, "
        f"ключ {'задан' if key else 'не задан'}, DEMO_MODE={settings.demo_mode}"
    )
    try:
        import httpx  # noqa: F401
        import joblib  # noqa: F401
        import pandas
        import sklearn

        print(f"библиотеки       pandas {pandas.__version__}, scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"библиотеки       НЕ УСТАНОВЛЕНЫ: {e}")
        return 1
    return 0


def cmd_fetch_weather(start: str, end: str) -> int:
    from app.weather import FARM, previous_runs

    df = previous_runs(FARM, start, end)
    cache = list((ROOT / "data" / "cache" / "weather").glob("*.json"))
    print(f"архивные прогнозы {start} — {end}: часов {len(df)}, файлов в кэше {len(cache)}")
    return 0


def cmd_train() -> int:
    from scripts.train import main as train_main

    train_main()
    return 0


def cmd_forecast(date: str, horizon: int, use_llm: bool | None) -> int:
    from app.agent import ForecastAgent

    s = ForecastAgent().run_day(date, horizon_hours=horizon, use_llm=use_llm)
    tools = " → ".join(t.get("tool", "?") for t in s.trace)
    print(f"выпуск прогноза {s.issue_date}, горизонт {s.horizon_hours} ч")
    print(f"трасса инструментов: {tools}")
    print("анализ: " + json.dumps(s.analysis, ensure_ascii=False, default=str))
    print("заключение: " + s.conclusion)
    print(f"файлы: outputs/forecasts/{s.issue_date}.csv, outputs/agent_logs/{s.issue_date}.json")
    return 0


def cmd_backtest(start: str, end: str, horizon: int, use_llm: bool | None) -> int:
    from app.backtest import run

    print(f"бэктест выпусков {start} — {end}")
    res = run(start, end, use_llm=use_llm, horizon_hours=horizon, progress=True)
    recomputed = ", ".join(res["recomputed_days"]) or "нет"
    print(f"выпусков {res['issues']}, строк по всем выпускам {res['rows_all_issues']}")
    print(
        f"часов февраля в итоговом ряду {res['feb_hours']}, "
        f"пропусков lead1 {res['missing_lead1']}, lead2 {res['missing_lead2']}"
    )
    print(f"средняя мощность за февраль: T1 {res['mean_cf_t1']}, T2 {res['mean_cf_t2']}")
    print(f"дни с пересчётом: {recomputed}")
    print(f"итог: {res['paths']['submission']}")
    return 0


def cmd_report() -> int:
    import pandas as pd

    metrics_path = ROOT / "docs" / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        print("качество на честной валидации (январь 2026, прогноз как в прошлом):")
        for name, payload in m.items():
            h = payload.get("honest_jan2026", {})
            for lead, label in (("lead1", "24 ч"), ("lead2", "48 ч")):
                b = h.get(lead, {}).get("blend", {})
                if b:
                    print(f"  {name} горизонт {label}: MAE {b['mae']}, RMSE {b['rmse']}, nMAE {b['nmae_pct']}%")
            weights = payload.get("weights") or payload.get("blend_w_gbm")
            print(f"  {name} веса смеси: {weights}")
    summary_path = ROOT / "outputs" / "backtest_summary.csv"
    final_path = ROOT / "outputs" / "forecast_feb2026.csv"
    if not summary_path.exists() or not final_path.exists():
        print("результаты бэктеста не найдены, выполните: python -m app.cli backtest --no-llm")
        return 0
    s = pd.read_csv(summary_path)
    f = pd.read_csv(final_path, parse_dates=["ts"])
    recomputed = s.loc[s["recomputed"].astype(bool), "issue_date"].tolist()
    missing = int(f[["turbine_1_lead1", "turbine_2_lead1"]].isna().any(axis=1).sum())
    print(f"\nвыпусков прогноза: {len(s)}, из них с пересчётом: {len(recomputed)}")
    print(f"дни с пересчётом: {', '.join(recomputed) or 'нет'}")
    print(f"часов февраля: {len(f)}, пропусков lead1: {missing}")
    print(
        f"средняя ожидаемая мощность (lead 1): "
        f"T1 {f['turbine_1_lead1'].mean():.3f}, T2 {f['turbine_2_lead1'].mean():.3f}"
    )
    daily = f.set_index("ts")["farm_mean_lead1"].resample("D").mean().round(3)
    print("\nсуточная средняя по ВЭС (доля номинала):")
    print(daily.to_string())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.cli", description="WindAgent: прогноз выработки ВЭС")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="проверка окружения")
    w = sub.add_parser("fetch-weather", help="прогрев кэша архивных прогнозов")
    w.add_argument("--start", required=True)
    w.add_argument("--end", required=True)
    sub.add_parser("train", help="обучение моделей")
    f = sub.add_parser("forecast", help="прогноз агентом на одну дату")
    f.add_argument("--date", required=True, help="дата выпуска прогноза, YYYY-MM-DD")
    f.add_argument("--horizon", type=int, default=48)
    _add_llm_flags(f)
    b = sub.add_parser("backtest", help="все выпуски тестового периода")
    b.add_argument("--start", default="2026-01-29")
    b.add_argument("--end", default="2026-02-27")
    b.add_argument("--horizon", type=int, default=48)
    _add_llm_flags(b)
    sub.add_parser("report", help="сводка по результатам")
    a = p.parse_args(argv)

    if a.cmd == "check":
        return cmd_check()
    if a.cmd == "fetch-weather":
        return cmd_fetch_weather(a.start, a.end)
    if a.cmd == "train":
        return cmd_train()
    if a.cmd == "forecast":
        return cmd_forecast(a.date, a.horizon, _llm_flag(a))
    if a.cmd == "backtest":
        return cmd_backtest(a.start, a.end, a.horizon, _llm_flag(a))
    if a.cmd == "report":
        return cmd_report()
    return 1


if __name__ == "__main__":
    sys.exit(main())
