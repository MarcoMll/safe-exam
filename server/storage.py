"""Filesystem layout under STORAGE_PATH (Phase A/B local storage)."""

from __future__ import annotations

from pathlib import Path

from server.config import get_settings


def storage_root() -> Path:
    """Return the configured storage root directory."""
    return get_settings().storage_path


def clips_root() -> Path:
    """Directory for uploaded MP4 + sidecar pairs."""
    return storage_root() / "clips"


def sessions_root() -> Path:
    """Directory for session JSON records."""
    return storage_root() / "sessions"


def metadata_root() -> Path:
    """Directory for metadata JSONL batches."""
    return storage_root() / "metadata"
