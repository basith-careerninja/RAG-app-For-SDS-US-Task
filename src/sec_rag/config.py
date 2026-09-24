"""Env settings and YAML experiment-config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(REPO_ROOT / ".env"), extra="ignore")

    openai_api_key: str = ""
    gemini_api_key: str = ""


def get_settings() -> Settings:
    return Settings()


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(path.relative_to(REPO_ROOT))
    return config
