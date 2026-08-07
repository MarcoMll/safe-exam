"""Typed settings used by the proctoring agent. Owned by #33."""

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import yaml

from capture.capture_config import CaptureConfig


class ConfigError(ValueError):
    """Configuration file is missing, malformed, or invalid!"""


@dataclass(frozen=True)
class PhoneConfig:
    """Control whether and how confidently the agent detects phone usage."""

    # Enables phone detection.
    enabled: bool
    # Minimum detector confidence required to report a phone.
    confidence_threshold: float


@dataclass(frozen=True)
class GazeConfig:
    """Control gaze monitoring and when looking away becomes suspicious."""

    # Enables gaze monitoring.
    enabled: bool
    # Maximum allowed vertical head angle.
    head_pitch_threshold: float
    # Maximum allowed horizontal head angle.
    head_yaw_threshold: float
    # Maximum allowed vertical eye angle.
    eye_pitch_threshold: float
    # Maximum allowed horizontal eye angle.
    eye_yaw_threshold: float
    # Time a threshold must be exceeded before reporting a gaze violation.
    duration_threshold_seconds: float


@dataclass(frozen=True)
class MultiPersonConfig:
    """Control detection of additional people in the monitored area."""

    # Enables detection of multiple people.
    enabled: bool
    # Selects the frame region in which people are counted.
    roi_mode: str


@dataclass(frozen=True)
class DetectorConfig:
    """Group all detector-specific settings used by the agent."""

    # Settings for phone detection.
    phone: PhoneConfig
    # Settings for gaze monitoring.
    gaze: GazeConfig
    # Settings for detecting additional people.
    multi_person: MultiPersonConfig


@dataclass(frozen=True)
class FusionConfig:
    """Control how detector signals are combined into a flag decision."""

    # Contribution of a phone signal to the combined score.
    phone_weight: float
    # Contribution of a gaze signal to the combined score.
    gaze_weight: float
    # Contribution of an additional-person signal to the combined score.
    extra_person_weight: float
    # Extra score added when several signals occur together.
    multi_signal_bonus: float
    # Combined score at which an incident is flagged.
    flag_threshold: float
    # Minimum delay between consecutive flags.
    flag_cooldown_seconds: float


@dataclass(frozen=True)
class LoggingConfig:
    """Control the verbosity and destination of agent logs."""

    # Minimum severity level written to the logs.
    level: str
    # Directory in which log files are stored.
    log_dir: Path


@dataclass(frozen=True)
class Config:
    """Hold the complete runtime configuration for one proctoring session."""

    # Server endpoint to which the agent connects.
    server_url: str
    # Identifier of the monitored exam.
    exam_id: str
    # Identifier of the student taking the exam.
    student_id: str
    # Secret used to authenticate the agent with the server.
    auth_token: str = field(repr=False)

    # Number of frames analyzed per second.
    sampling_fps: float
    # Duration of recent video retained in memory.
    ring_buffer_seconds: float
    # Video duration saved before a flag is raised.
    clip_before_flag_seconds: float
    # Video duration saved after a flag is raised.
    clip_after_flag_seconds: float

    # Settings for all detectors.
    detectors: DetectorConfig
    # Settings for signal scoring and flag creation.
    fusion: FusionConfig
    # Settings for application logging.
    logging: LoggingConfig


def _load_yaml(path: str | Path) -> dict:
    """Read one YAML file and return its root mapping."""
    config_path = Path(path)

    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with config_path.open(encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping")

    return raw


def _field_path(parent: str, field_name: str) -> str:
    """Build a readable dotted path for validation errors."""
    return f"{parent}.{field_name}" if parent else field_name


def _expect_mapping(value, path: str) -> dict:
    """Return a mapping or raise a config error with its field path."""
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _require_exact_fields(raw: dict, expected: set[str], path: str = "") -> None:
    """Reject missing, unknown, and non-string mapping keys."""
    for key in raw:
        if not isinstance(key, str):
            location = path or "Config root"
            raise ConfigError(f"{location} keys must be strings")

    missing = sorted(expected - set(raw))
    if missing:
        raise ConfigError(f"Missing required field: {_field_path(path, missing[0])}")

    unknown = sorted(set(raw) - expected)
    if unknown:
        raise ConfigError(f"Unknown field: {_field_path(path, unknown[0])}")


def _expect_string(value, path: str) -> str:
    """Return a non-empty string value."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _expect_bool(value, path: str) -> bool:
    """Return a strict boolean value without accepting integers."""
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    return value


def _expect_number(value, path: str) -> float:
    """Return a numeric value as float without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{path} must be a number")
    return float(value)


def _expect_positive_number(value, path: str) -> float:
    number = _expect_number(value, path)
    if number <= 0:
        raise ConfigError(f"{path} must be greater than 0")
    return number


def _expect_non_negative_number(value, path: str) -> float:
    number = _expect_number(value, path)
    if number < 0:
        raise ConfigError(f"{path} must be greater than or equal to 0")
    return number


def _expect_unit_interval(value, path: str) -> float:
    number = _expect_number(value, path)
    if not 0 <= number <= 1:
        raise ConfigError(f"{path} must be between 0 and 1")
    return number


def _parse_config(raw: dict) -> Config:
    """Validate a raw YAML mapping and build the typed agent config."""
    raw = _expect_mapping(raw, "Config root")
    _require_exact_fields(
        raw,
        {
            "server_url",
            "exam_id",
            "student_id",
            "auth_token",
            "sampling_fps",
            "ring_buffer_seconds",
            "clip_before_flag_seconds",
            "clip_after_flag_seconds",
            "detectors",
            "fusion",
            "logging",
        },
    )

    detectors_raw = _expect_mapping(raw["detectors"], "detectors")
    _require_exact_fields(
        detectors_raw,
        {"phone", "gaze", "multi_person"},
        "detectors",
    )

    phone_raw = _expect_mapping(detectors_raw["phone"], "detectors.phone")
    _require_exact_fields(
        phone_raw,
        {"enabled", "confidence_threshold"},
        "detectors.phone",
    )

    gaze_raw = _expect_mapping(detectors_raw["gaze"], "detectors.gaze")
    _require_exact_fields(
        gaze_raw,
        {
            "enabled",
            "head_pitch_threshold",
            "head_yaw_threshold",
            "eye_pitch_threshold",
            "eye_yaw_threshold",
            "duration_threshold_seconds",
        },
        "detectors.gaze",
    )

    multi_person_raw = _expect_mapping(
        detectors_raw["multi_person"],
        "detectors.multi_person",
    )
    _require_exact_fields(
        multi_person_raw,
        {"enabled", "roi_mode"},
        "detectors.multi_person",
    )

    fusion_raw = _expect_mapping(raw["fusion"], "fusion")
    _require_exact_fields(
        fusion_raw,
        {
            "phone_weight",
            "gaze_weight",
            "extra_person_weight",
            "multi_signal_bonus",
            "flag_threshold",
            "flag_cooldown_seconds",
        },
        "fusion",
    )

    logging_raw = _expect_mapping(raw["logging"], "logging")
    _require_exact_fields(logging_raw, {"level", "log_dir"}, "logging")

    server_url = _expect_string(raw["server_url"], "server_url")
    parsed_url = urlparse(server_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ConfigError("server_url must be a valid HTTP or HTTPS URL")

    sampling_fps = _expect_positive_number(raw["sampling_fps"], "sampling_fps")
    ring_buffer_seconds = _expect_positive_number(
        raw["ring_buffer_seconds"],
        "ring_buffer_seconds",
    )
    clip_before_flag_seconds = _expect_non_negative_number(
        raw["clip_before_flag_seconds"],
        "clip_before_flag_seconds",
    )
    clip_after_flag_seconds = _expect_non_negative_number(
        raw["clip_after_flag_seconds"],
        "clip_after_flag_seconds",
    )
    if clip_before_flag_seconds + clip_after_flag_seconds > ring_buffer_seconds:
        raise ConfigError(
            "clip_before_flag_seconds + clip_after_flag_seconds "
            "must not exceed ring_buffer_seconds"
        )

    logging_level = _expect_string(logging_raw["level"], "logging.level").upper()
    allowed_logging_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if logging_level not in allowed_logging_levels:
        allowed = ", ".join(sorted(allowed_logging_levels))
        raise ConfigError(f"logging.level must be one of: {allowed}")

    return Config(
        server_url=server_url,
        exam_id=_expect_string(raw["exam_id"], "exam_id"),
        student_id=_expect_string(raw["student_id"], "student_id"),
        auth_token=_expect_string(raw["auth_token"], "auth_token"),
        sampling_fps=sampling_fps,
        ring_buffer_seconds=ring_buffer_seconds,
        clip_before_flag_seconds=clip_before_flag_seconds,
        clip_after_flag_seconds=clip_after_flag_seconds,
        detectors=DetectorConfig(
            phone=PhoneConfig(
                enabled=_expect_bool(
                    phone_raw["enabled"],
                    "detectors.phone.enabled",
                ),
                confidence_threshold=_expect_unit_interval(
                    phone_raw["confidence_threshold"],
                    "detectors.phone.confidence_threshold",
                ),
            ),
            gaze=GazeConfig(
                enabled=_expect_bool(
                    gaze_raw["enabled"],
                    "detectors.gaze.enabled",
                ),
                head_pitch_threshold=_expect_non_negative_number(
                    gaze_raw["head_pitch_threshold"],
                    "detectors.gaze.head_pitch_threshold",
                ),
                head_yaw_threshold=_expect_non_negative_number(
                    gaze_raw["head_yaw_threshold"],
                    "detectors.gaze.head_yaw_threshold",
                ),
                eye_pitch_threshold=_expect_non_negative_number(
                    gaze_raw["eye_pitch_threshold"],
                    "detectors.gaze.eye_pitch_threshold",
                ),
                eye_yaw_threshold=_expect_non_negative_number(
                    gaze_raw["eye_yaw_threshold"],
                    "detectors.gaze.eye_yaw_threshold",
                ),
                duration_threshold_seconds=_expect_non_negative_number(
                    gaze_raw["duration_threshold_seconds"],
                    "detectors.gaze.duration_threshold_seconds",
                ),
            ),
            multi_person=MultiPersonConfig(
                enabled=_expect_bool(
                    multi_person_raw["enabled"],
                    "detectors.multi_person.enabled",
                ),
                roi_mode=_expect_string(
                    multi_person_raw["roi_mode"],
                    "detectors.multi_person.roi_mode",
                ),
            ),
        ),
        fusion=FusionConfig(
            phone_weight=_expect_unit_interval(
                fusion_raw["phone_weight"],
                "fusion.phone_weight",
            ),
            gaze_weight=_expect_unit_interval(
                fusion_raw["gaze_weight"],
                "fusion.gaze_weight",
            ),
            extra_person_weight=_expect_unit_interval(
                fusion_raw["extra_person_weight"],
                "fusion.extra_person_weight",
            ),
            multi_signal_bonus=_expect_unit_interval(
                fusion_raw["multi_signal_bonus"],
                "fusion.multi_signal_bonus",
            ),
            flag_threshold=_expect_unit_interval(
                fusion_raw["flag_threshold"],
                "fusion.flag_threshold",
            ),
            flag_cooldown_seconds=_expect_non_negative_number(
                fusion_raw["flag_cooldown_seconds"],
                "fusion.flag_cooldown_seconds",
            ),
        ),
        logging=LoggingConfig(
            level=logging_level,
            log_dir=Path(_expect_string(logging_raw["log_dir"], "logging.log_dir")),
        ),
    )


def load_config(path: str | Path) -> Config:
    """Load, validate, and convert YAML into typed agent settings."""
    raw = _load_yaml(path)
    return _parse_config(raw)

def build_capture_config(
    config: Config,
    *,
    camera_index: int = 0,
    show_debug: bool = False,
) -> CaptureConfig:
    """Build webcam settings from the validated agent config."""
    return CaptureConfig(
        target_fps=config.sampling_fps,
        camera_index=camera_index,
        show_debug=show_debug,
    )
