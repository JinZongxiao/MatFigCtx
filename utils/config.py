"""Helpers for loading MatFigCtx YAML config and secrets from .env/env vars."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in some environments
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_config_path(config_path: str | os.PathLike[str] = "configs/config.yaml") -> Path:
    """Resolve config paths relative to the repository root when needed."""
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_env_file(env_path: str | os.PathLike[str] | None = None, override: bool = False) -> Path | None:
    """Load .env if it exists and return the resolved path."""
    candidate = Path(env_path).expanduser() if env_path else (REPO_ROOT / ".env")
    if candidate.exists() and load_dotenv is not None:
        load_dotenv(candidate, override=override)
        return candidate.resolve()
    return None


def load_config(
    config_path: str | os.PathLike[str] = "configs/config.yaml",
    *,
    env_path: str | os.PathLike[str] | None = None,
    load_env: bool = True,
) -> dict[str, Any]:
    """Load YAML config and annotate it with resolved helper metadata."""
    if load_env:
        _load_env_file(env_path)

    resolved_path = resolve_config_path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {resolved_path}")

    data.setdefault("_config_path", str(resolved_path))
    data.setdefault("_repo_root", str(REPO_ROOT))
    return data


def get_secret(cfg: dict[str, Any] | None, env_key_field: str, default: str | None = None) -> str | None:
    """
    Read a secret from environment variables.

    `env_key_field` can be either:
    - a config key whose value is the env var name, e.g. "DEEPSEEK_API_KEY_ENV"
    - a direct env var name, e.g. "DEEPSEEK_API_KEY"
    """
    cfg = cfg or {}

    env_var_name = cfg.get(env_key_field, env_key_field)
    if not isinstance(env_var_name, str) or not env_var_name.strip():
        return default

    value = os.getenv(env_var_name.strip())
    if value is None or value == "":
        return default
    return value
