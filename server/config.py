"""Server settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
_DEFAULT_STORAGE = _SERVER_DIR / "storage"
_DEFAULT_DATABASE_URL = (
    "postgresql+psycopg2://examguard:examguard@localhost:5432/examguard"
)
_DEFAULT_SECRET_KEY = "dev-secret-change-me"
_DEFAULT_VERSION = "0.1.0"


@dataclass(frozen=True)
class Settings:
    """Runtime server configuration."""

    database_url: str
    storage_path: Path
    secret_key: str
    version: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once from the process environment."""
    storage_raw = os.environ.get("STORAGE_PATH", "").strip()
    storage_path = Path(storage_raw) if storage_raw else _DEFAULT_STORAGE

    return Settings(
        database_url=os.environ.get("DATABASE_URL", _DEFAULT_DATABASE_URL).strip()
        or _DEFAULT_DATABASE_URL,
        storage_path=storage_path,
        secret_key=os.environ.get("SECRET_KEY", _DEFAULT_SECRET_KEY).strip()
        or _DEFAULT_SECRET_KEY,
        version=os.environ.get("SERVER_VERSION", _DEFAULT_VERSION).strip()
        or _DEFAULT_VERSION,
    )
