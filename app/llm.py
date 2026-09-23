"""Единый клиент LLM с переключением провайдера и режимом проверки без ключей.

Провайдеры:
- openai  : OpenAI API (основной)
- nvidia  : NVIDIA NIM, совместимый с OpenAI SDK (резервный)
- demo    : ответы берутся из data/cache/demo_responses.json (проверка без ключей)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.config import settings

DEMO_CACHE = Path("data/cache/demo_responses.json")


def _client() -> OpenAI:
    if settings.llm_provider == "nvidia":
        if not settings.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY не задан")
        return OpenAI(base_url=settings.nvidia_base_url, api_key=settings.nvidia_api_key)
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY не задан")
    return OpenAI(api_key=settings.openai_api_key)


def _model(reasoning: bool) -> str:
    if settings.llm_provider == "nvidia":
        return settings.nvidia_model
    return settings.openai_reasoning_model if reasoning else settings.openai_model


def _cache_key(messages: list[dict[str, Any]], model: str) -> str:
    raw = json.dumps({"m": messages, "model": model}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, str]:
    if DEMO_CACHE.exists():
        return json.loads(DEMO_CACHE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    DEMO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DEMO_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def chat(messages: list[dict[str, Any]], reasoning: bool = False, temperature: float = 0.1) -> str:
    """Вернуть текст ответа модели. В DEMO_MODE ответ берётся из кэша, иначе вызывается API
    и ответ дописывается в кэш, чтобы контрольный сценарий воспроизводился без ключей."""
    model = _model(reasoning)
    key = _cache_key(messages, model)
    cache = _load_cache()
    if settings.demo_mode or settings.llm_provider == "demo":
        if key in cache:
            return cache[key]
        raise RuntimeError("DEMO_MODE: ответа для этого запроса нет в кэше")
    resp = _client().chat.completions.create(model=model, messages=messages, temperature=temperature)
    text = resp.choices[0].message.content or ""
    cache[key] = text
    _save_cache(cache)
    return text


def available() -> bool:
    """Есть ли живой провайдер: ключ задан и режим проверки выключен."""
    if settings.demo_mode or settings.llm_provider == "demo":
        return False
    if settings.llm_provider == "nvidia":
        return bool(settings.nvidia_api_key)
    return bool(settings.openai_api_key)


def chat_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    reasoning: bool = False,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Один шаг диалога с инструментами (function calling).

    Возвращает {"content": текст или None, "tool_calls": [{"id", "name", "arguments"}]}.
    Форма ответа не зависит от версии SDK, оркестратор работает со словарями.
    """
    if settings.demo_mode or settings.llm_provider == "demo":
        raise RuntimeError("DEMO_MODE: оркестрация LLM недоступна")
    resp = _client().chat.completions.create(
        model=_model(reasoning), messages=messages, tools=tools, temperature=temperature
    )
    msg = resp.choices[0].message
    calls = [
        {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
        for tc in (msg.tool_calls or [])
    ]
    return {"content": msg.content, "tool_calls": calls}
