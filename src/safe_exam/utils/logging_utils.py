import dataclasses
import json
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from typing import Any


class DataclassJSONEncoder(json.JSONEncoder):
    """JSON encoder that can handle dataclasses."""

    def default(self, o: Any) -> Any:
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        return super().default(o)


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the LogRecord.
    Supports extra dictionary attributes and dataclass serialization.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Include extra attributes passed via `extra={...}`
        standard_attrs = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "id",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }

        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_record[key] = value

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, cls=DataclassJSONEncoder)


def configure_logging(
    log_dir: str, exam_id: str, student_id: str, date: str, level: int = logging.INFO
) -> None:
    """
    Configure the logging for the application.

    :param log_dir: Directory where the log file will be saved.
    :param exam_id: Identifier for the exam.
    :param student_id: Identifier for the student.
    :param date: Date string for the log file name.
    :param level: The logging level.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{exam_id}_{student_id}_{date}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logging
    for handler in root_logger.handlers.copy():
        root_logger.removeHandler(handler)

    # Keep logs for exactly 30 days
    file_handler = TimedRotatingFileHandler(
        filename=log_file_path,
        when="D",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )

    json_formatter = JsonFormatter()
    file_handler.setFormatter(json_formatter)

    root_logger.addHandler(file_handler)
