"""Shared logging helpers for the agent and demos."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
    level: int = logging.INFO,
    *,
    log_dir: Path | None = None,
) -> None:
    """Configure root logging to stdout, and optionally to a file under log_dir."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "agent.log", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )

    json_formatter = JsonFormatter()
    file_handler.setFormatter(json_formatter)

    root_logger.addHandler(file_handler)
