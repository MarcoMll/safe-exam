"""Typed settings used by the proctoring agent. Owned by #33."""

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import yaml

from safe_exam.capture.capture_config import CaptureConfig


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
    # Target video bitrate for staged clips (ffmpeg -b:v), e.g. "500k".
    clip_bitrate: str
    # Directory where staged MP4s, sidecars, and the upload queue are stored.
    clip_dir: Path
    # How often metadata batches are POSTed to the server (#37).
    metadata_interval_seconds: float

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


_TOP_LEVEL_FIELDS = {
    "server_url",
    "exam_id",
    "student_id",
    "auth_token",
    "sampling_fps",
    "ring_buffer_seconds",
    "clip_before_flag_seconds",
    "clip_after_flag_seconds",
    "clip_bitrate",
    "clip_dir",
    "metadata_interval_seconds",
    "detectors",
    "fusion",
    "logging",
}

_DETECTOR_FIELDS = {"phone", "gaze", "multi_person"}
_PHONE_FIELDS = {"enabled", "confidence_threshold"}
_GAZE_FIELDS = {
    "enabled",
    "head_pitch_threshold",
    "head_yaw_threshold",
    "eye_pitch_threshold",
    "eye_yaw_threshold",
    "duration_threshold_seconds",
}
_MULTI_PERSON_FIELDS = {"enabled", "roi_mode"}
_FUSION_FIELDS = {
    "phone_weight",
    "gaze_weight",
    "extra_person_weight",
    "multi_signal_bonus",
    "flag_threshold",
    "flag_cooldown_seconds",
}
_LOGGING_FIELDS = {"level", "log_dir"}
_ALLOWED_LOGGING_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _parse_server_url(value) -> str:
    """Validate and return the server endpoint."""
    server_url = _expect_string(value, "server_url")
    parsed_url = urlparse(server_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ConfigError("server_url must be a valid HTTP or HTTPS URL")
    return server_url


def _parse_phone_config(raw) -> PhoneConfig:
    """Parse phone detector settings."""
    path = "detectors.phone"
    raw = _expect_mapping(raw, path)
    _require_exact_fields(raw, _PHONE_FIELDS, path)

    return PhoneConfig(
        enabled=_expect_bool(raw["enabled"], f"{path}.enabled"),
        confidence_threshold=_expect_unit_interval(
            raw["confidence_threshold"],
            f"{path}.confidence_threshold",
        ),
    )


def _parse_gaze_config(raw) -> GazeConfig:
    """Parse gaze detector settings."""
    path = "detectors.gaze"
    raw = _expect_mapping(raw, path)
    _require_exact_fields(raw, _GAZE_FIELDS, path)

    return GazeConfig(
        enabled=_expect_bool(raw["enabled"], f"{path}.enabled"),
        head_pitch_threshold=_expect_non_negative_number(
            raw["head_pitch_threshold"],
            f"{path}.head_pitch_threshold",
        ),
        head_yaw_threshold=_expect_non_negative_number(
            raw["head_yaw_threshold"],
            f"{path}.head_yaw_threshold",
        ),
        eye_pitch_threshold=_expect_non_negative_number(
            raw["eye_pitch_threshold"],
            f"{path}.eye_pitch_threshold",
        ),
        eye_yaw_threshold=_expect_non_negative_number(
            raw["eye_yaw_threshold"],
            f"{path}.eye_yaw_threshold",
        ),
        duration_threshold_seconds=_expect_non_negative_number(
            raw["duration_threshold_seconds"],
            f"{path}.duration_threshold_seconds",
        ),
    )


def _parse_multi_person_config(raw) -> MultiPersonConfig:
    """Parse additional-person detector settings."""
    path = "detectors.multi_person"
    raw = _expect_mapping(raw, path)
    _require_exact_fields(raw, _MULTI_PERSON_FIELDS, path)

    return MultiPersonConfig(
        enabled=_expect_bool(raw["enabled"], f"{path}.enabled"),
        roi_mode=_expect_string(raw["roi_mode"], f"{path}.roi_mode"),
    )


def _parse_detector_config(raw) -> DetectorConfig:
    """Parse all detector-specific settings."""
    path = "detectors"
    raw = _expect_mapping(raw, path)
    _require_exact_fields(raw, _DETECTOR_FIELDS, path)

    return DetectorConfig(
        phone=_parse_phone_config(raw["phone"]),
        gaze=_parse_gaze_config(raw["gaze"]),
        multi_person=_parse_multi_person_config(raw["multi_person"]),
    )


def _parse_fusion_config(raw) -> FusionConfig:
    """Parse signal-fusion settings."""
    path = "fusion"
    raw = _expect_mapping(raw, path)
    _require_exact_fields(raw, _FUSION_FIELDS, path)

    return FusionConfig(
        phone_weight=_expect_unit_interval(
            raw["phone_weight"],
            f"{path}.phone_weight",
        ),
        gaze_weight=_expect_unit_interval(
            raw["gaze_weight"],
            f"{path}.gaze_weight",
        ),
        extra_person_weight=_expect_unit_interval(
            raw["extra_person_weight"],
            f"{path}.extra_person_weight",
        ),
        multi_signal_bonus=_expect_unit_interval(
            raw["multi_signal_bonus"],
            f"{path}.multi_signal_bonus",
        ),
        flag_threshold=_expect_unit_interval(
            raw["flag_threshold"],
            f"{path}.flag_threshold",
        ),
        flag_cooldown_seconds=_expect_non_negative_number(
            raw["flag_cooldown_seconds"],
            f"{path}.flag_cooldown_seconds",
        ),
    )


def _parse_logging_config(raw) -> LoggingConfig:
    """Parse logging settings."""
    path = "logging"
    raw = _expect_mapping(raw, path)
    _require_exact_fields(raw, _LOGGING_FIELDS, path)

    level = _expect_string(raw["level"], f"{path}.level").upper()
    if level not in _ALLOWED_LOGGING_LEVELS:
        allowed = ", ".join(sorted(_ALLOWED_LOGGING_LEVELS))
        raise ConfigError(f"{path}.level must be one of: {allowed}")

    return LoggingConfig(
        level=level,
        log_dir=Path(_expect_string(raw["log_dir"], f"{path}.log_dir")),
    )


def _parse_clip_settings(raw: dict) -> tuple[float, float, float]:
    """Parse ring-buffer and clip-window durations."""
    ring_buffer_seconds = _expect_positive_number(
        raw["ring_buffer_seconds"],
        "ring_buffer_seconds",
    )
    clip_before = _expect_non_negative_number(
        raw["clip_before_flag_seconds"],
        "clip_before_flag_seconds",
    )
    clip_after = _expect_non_negative_number(
        raw["clip_after_flag_seconds"],
        "clip_after_flag_seconds",
    )

    if clip_before + clip_after > ring_buffer_seconds:
        raise ConfigError(
            "clip_before_flag_seconds + clip_after_flag_seconds "
            "must not exceed ring_buffer_seconds"
        )

    return ring_buffer_seconds, clip_before, clip_after


def _parse_config(raw: dict) -> Config:
    """Validate a raw YAML mapping and build the typed agent config."""
    raw = _expect_mapping(raw, "Config root")
    _require_exact_fields(raw, _TOP_LEVEL_FIELDS)

    ring_buffer_seconds, clip_before, clip_after = _parse_clip_settings(raw)

    return Config(
        server_url=_parse_server_url(raw["server_url"]),
        exam_id=_expect_string(raw["exam_id"], "exam_id"),
        student_id=_expect_string(raw["student_id"], "student_id"),
        auth_token=_expect_string(raw["auth_token"], "auth_token"),
        sampling_fps=_expect_positive_number(raw["sampling_fps"], "sampling_fps"),
        ring_buffer_seconds=ring_buffer_seconds,
        clip_before_flag_seconds=clip_before,
        clip_after_flag_seconds=clip_after,
        clip_bitrate=_expect_string(raw["clip_bitrate"], "clip_bitrate"),
        clip_dir=Path(_expect_string(raw["clip_dir"], "clip_dir")),
        metadata_interval_seconds=_expect_positive_number(
            raw["metadata_interval_seconds"],
            "metadata_interval_seconds",
        ),
        detectors=_parse_detector_config(raw["detectors"]),
        fusion=_parse_fusion_config(raw["fusion"]),
        logging=_parse_logging_config(raw["logging"]),
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
