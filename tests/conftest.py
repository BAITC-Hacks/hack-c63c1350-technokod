"""Корень проекта в sys.path, чтобы тесты запускались и как `pytest`, и как `python -m pytest`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
