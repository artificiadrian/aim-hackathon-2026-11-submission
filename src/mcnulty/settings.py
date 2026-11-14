"""Configuration from the environment (the organisers' container) or the repo's ``.env`` (local dev)."""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, SecretStr

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    arena_base_url: str
    arena_api_key: SecretStr
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: SecretStr
    run_kind: Literal["dev", "final"] | None = None
    blob_cache: Path = REPO_ROOT / ".cache" / "blobs"

    @property
    def secrets(self):
        return (
            self.arena_api_key.get_secret_value(),
            self.openrouter_api_key.get_secret_value(),
        )


def load_settings():
    """Read the environment; the repo's ``.env`` fills in what is not set."""
    load_dotenv(REPO_ROOT / ".env")
    env = {
        "arena_base_url": os.environ.get("ARENA_BASE_URL"),
        "arena_api_key": os.environ.get("ARENA_API_KEY"),
        "openrouter_base_url": os.environ.get("OPENROUTER_BASE_URL"),
        "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY"),
        "run_kind": os.environ.get("ARENA_RUN_KIND"),
        "blob_cache": os.environ.get("ARENA_BLOB_CACHE"),
    }
    return Settings.model_validate({k: v for k, v in env.items() if v})
