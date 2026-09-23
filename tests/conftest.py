"""Корень проекта в sys.path, чтобы тесты запускались и как `pytest`, и как `python -m pytest`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest


@pytest.fixture
def isolated_outputs(tmp_path, monkeypatch):
    """Выпуск и журнал агента пишутся во временный каталог: прогон тестов не должен
    перезаписывать канонические артефакты в outputs/."""
    import app.agent as agent

    monkeypatch.setattr(agent, "OUT_DIR", tmp_path / "forecasts")
    monkeypatch.setattr(agent, "LOG_DIR", tmp_path / "logs")
    return tmp_path
