"""Entry point: python -m safe_exam.agent. Owned by #39."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from safe_exam.agent.config import ConfigError, load_config
from safe_exam.agent.session import (
    ChecklistError,
    ExamSession,
    SessionError,
)
from safe_exam.utils.logging_utils import configure_logging

DEFAULT_CONFIG_PATH = Path("config/ex.config.yml")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExamGuard proctoring agent")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to agent YAML config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default: 0)",
    )
    return parser.parse_args()


def _configure_logging_from_config(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    configure_logging(level=level)


def main() -> int:
    args = _parse_args()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    _configure_logging_from_config(config.logging.level)

    session = ExamSession(config, camera_index=args.camera)

    def _handle_stop(signum: int, _frame) -> None:
        logger.info("Stop signal received (%s)", signum)
        session.request_stop()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        session.start()
        session.run()
    except (ChecklistError, SessionError) as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
        session.request_stop()
    finally:
        session.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
