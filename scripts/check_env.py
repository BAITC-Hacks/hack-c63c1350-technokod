"""Проверка окружения: версии, зависимости, переменные, доступность провайдера."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REQUIRED = ["openai", "dotenv", "pydantic", "duckdb", "pandas", "fastapi", "uvicorn", "httpx", "pypdf"]


def main() -> int:
    ok = True
    print(f"Python {sys.version.split()[0]}")
    for name in REQUIRED:
        try:
            importlib.import_module(name)
            print(f"  [ok] {name}")
        except ImportError:
            ok = False
            print(f"  [нет] {name}  -> pip install -r requirements.txt")
    from app.config import settings

    print(f"LLM_PROVIDER={settings.llm_provider} DEMO_MODE={settings.demo_mode}")
    has_key = (settings.llm_provider == "openai" and settings.openai_api_key) or (
        settings.llm_provider == "nvidia" and settings.nvidia_api_key
    )
    if has_key:
        print("  [ok] ключ LLM задан, доступна оркестрация агента моделью")
    else:
        print("  [инфо] ключ LLM не задан: агент работает по детерминированному плану, это не ошибка")
    print("Итог:", "готово" if ok else "есть проблемы")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
