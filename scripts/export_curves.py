"""Экспорт кривых мощности и квантилей в JS, чтобы витрина считала прогноз на любую дату.

Полная модель — Python (sklearn), в браузере её не выполнить. Но компонент «кривая
мощности по прогнозной скорости ветра» это таблица интерполяции, и она переносится
как есть. На честной валидации января кривая даёт nMAE 14.71 и 14.80 % против 14.31
и 14.55 % у полной смеси: разница 0.4 пункта при стандартной ошибке около 1 пункта.

Поэтому витрина считает произвольную дату упрощённой моделью и помечает это явно,
а сдаваемые 30 выпусков остаются результатом полной модели.

Запуск: python scripts/export_curves.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model import GenerationModel

OUT = Path("docs/demo/model-curves.js")


def main() -> None:
    payload = {"turbines": {}, "note": "кривая мощности по прогнозной скорости ветра на 100 м"}
    for t in (1, 2):
        m = GenerationModel.load(t)
        table = sorted((float(k), float(v)) for k, v in m.curve.table.items())
        payload["turbines"][str(t)] = {
            "ws": [round(k, 3) for k, _ in table],
            "power": [round(v, 5) for _, v in table],
            "weights": m.weights,
        }
        # квантили остатков по лиду и уровню прогноза — для коридора P10–P90
        q = {}
        for (lead, level), (lo, hi) in getattr(m, "residual_quantiles", {}).items():
            q[f"{lead}_{level}"] = [round(float(lo), 5), round(float(hi), 5)]
        payload["turbines"][str(t)]["quantiles"] = q

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.CURVES = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    n = len(payload["turbines"]["1"]["ws"])
    print(f"{OUT}: {n} точек кривой на турбину, {OUT.stat().st_size // 1024} КБ")


if __name__ == "__main__":
    main()
