from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | os.PathLike[str] = "configs/config.yaml", load_env: bool = True) -> dict[str, Any]:
    if load_env and load_dotenv is not None:
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)

    p = Path(config_path).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("_repo_root", str(REPO_ROOT))
    return data
