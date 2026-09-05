"""Shared configuration loaded from the project's optional .env file."""

from __future__ import annotations

import os
from pathlib import Path


ENV_PATH = Path(__file__).with_name(".env")


def load_project_env(path: Path = ENV_PATH) -> None:
    """Load simple KEY=VALUE entries without overwriting shell environment variables."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and value:
            os.environ.setdefault(key, value)


load_project_env()

DEFAULT_MODEL = os.getenv(
    "TINKER_MODEL",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
)


def require_tinker_api_key() -> str:
    """Return the configured API key or explain how to configure one."""
    api_key = os.getenv("TINKER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "TINKER_API_KEY is not set. Paste it after TINKER_API_KEY= in the project's .env file."
        )
    return api_key
