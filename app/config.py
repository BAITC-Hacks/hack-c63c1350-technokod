"""Настройки из переменных окружения. Ключи никогда не хранятся в коде."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    openai_reasoning_model: str = os.getenv("OPENAI_REASONING_MODEL", "gpt-4.1")
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    nvidia_base_url: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    nvidia_model: str = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    demo_mode: bool = os.getenv("DEMO_MODE", "false").lower() in {"1", "true", "yes"}
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))


settings = Settings()
