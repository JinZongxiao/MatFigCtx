"""Utilities for configuring OpenAI-compatible text backends across apps."""

from __future__ import annotations

from typing import Any

from .config import get_secret


PROVIDER_PREFIXES = {
    "ernie": "ERNIE",
    "deepseek": "DEEPSEEK",
    "dashscope": "DASHSCOPE",
}


def normalize_provider(provider: str | None, default: str = "ernie") -> str:
    value = (provider or default).strip().lower()
    if value not in PROVIDER_PREFIXES:
        raise ValueError(f"Unsupported provider: {provider}")
    return value


def resolve_provider_config(cfg: dict[str, Any], provider: str | None = None, *, default: str = "ernie") -> dict[str, str]:
    """Resolve model/base-url/api-key settings for an OpenAI-compatible provider."""
    chosen = normalize_provider(provider or cfg.get("DEFAULT_LLM_PROVIDER"), default=default)
    prefix = PROVIDER_PREFIXES[chosen]
    api_key = get_secret(cfg, f"{prefix}_API_KEY_ENV")
    base_url = cfg.get(f"{prefix}_BASE_URL", "")
    model = cfg.get(f"{prefix}_MODEL", "")
    if not api_key:
        env_name = cfg.get(f"{prefix}_API_KEY_ENV", f"{prefix}_API_KEY")
        raise RuntimeError(f"{env_name} env var not set")
    if not model:
        raise RuntimeError(f"{prefix}_MODEL is not configured")
    return {
        "provider": chosen,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


def make_openai_client(cfg: dict[str, Any], provider: str | None = None, *, default: str = "ernie"):
    """Create an OpenAI-compatible client lazily to keep imports optional."""
    from openai import OpenAI

    resolved = resolve_provider_config(cfg, provider=provider, default=default)
    client = OpenAI(api_key=resolved["api_key"], base_url=resolved["base_url"])
    return client, resolved


def make_langchain_chat_model(cfg: dict[str, Any], provider: str | None = None, *, default: str = "ernie", **kwargs):
    """Create a ChatOpenAI instance using the selected OpenAI-compatible backend."""
    from langchain_community.chat_models import ChatOpenAI

    resolved = resolve_provider_config(cfg, provider=provider, default=default)
    return ChatOpenAI(
        model=resolved["model"],
        api_key=resolved["api_key"],
        base_url=resolved["base_url"],
        **kwargs,
    )
