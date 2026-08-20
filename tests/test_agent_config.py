"""Tests for the typed Phase 1 agent configuration."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from safe_exam.agent.config import (
    Config,
    ConfigError,
    DetectorConfig,
    FusionConfig,
    GazeConfig,
    LoggingConfig,
    MultiPersonConfig,
    PhoneConfig,
    _parse_config,
    build_capture_config,
    load_config,
)


def _build_config() -> Config:
    """Create one complete config shared by the tests below."""
    return Config(
        server_url="https://examguard.school.edu/",
        exam_id="EXAM_2026_FINAL",
        student_id="S12345",
        auth_token="secret-token",
        sampling_fps=5.0,
        ring_buffer_seconds=60.0,
        clip_before_flag_seconds=15.0,
        clip_after_flag_seconds=5.0,
        clip_bitrate="500k",
        clip_dir=Path("data/staged_clips"),
        metadata_interval_seconds=5.0,
        detectors=DetectorConfig(
            phone=PhoneConfig(
                enabled=True,
                confidence_threshold=0.50,
            ),
            gaze=GazeConfig(
                enabled=True,
                head_pitch_threshold=20.0,
                head_yaw_threshold=25.0,
                eye_pitch_threshold=15.0,
                eye_yaw_threshold=20.0,
                duration_threshold_seconds=4.0,
            ),
            multi_person=MultiPersonConfig(
                enabled=True,
                roi_mode="proximity",
            ),
        ),
        fusion=FusionConfig(
            phone_weight=0.5,
            gaze_weight=0.3,
            extra_person_weight=0.4,
            multi_signal_bonus=0.3,
            flag_threshold=0.5,
            flag_cooldown_seconds=30.0,
        ),
        logging=LoggingConfig(
            level="INFO",
            log_dir=Path("logs"),
        ),
    )


def _build_raw_config() -> dict:
    """Create the dictionary shape returned by yaml.safe_load()."""
    return {
        "server_url": "https://examguard.school.edu/",
        "exam_id": "EXAM_2026_FINAL",
        "student_id": "S12345",
        "auth_token": "secret-token",
        "sampling_fps": 5.0,
        "ring_buffer_seconds": 60.0,
        "clip_before_flag_seconds": 15.0,
        "clip_after_flag_seconds": 5.0,
        "clip_bitrate": "500k",
        "clip_dir": "data/staged_clips",
        "metadata_interval_seconds": 5.0,
        "detectors": {
            "phone": {
                "enabled": True,
                "confidence_threshold": 0.50,
            },
            "gaze": {
                "enabled": True,
                "head_pitch_threshold": 20.0,
                "head_yaw_threshold": 25.0,
                "eye_pitch_threshold": 15.0,
                "eye_yaw_threshold": 20.0,
                "duration_threshold_seconds": 4.0,
            },
            "multi_person": {
                "enabled": True,
                "roi_mode": "proximity",
            },
        },
        "fusion": {
            "phone_weight": 0.5,
            "gaze_weight": 0.3,
            "extra_person_weight": 0.4,
            "multi_signal_bonus": 0.3,
            "flag_threshold": 0.5,
            "flag_cooldown_seconds": 30.0,
        },
        "logging": {
            "level": "INFO",
            "log_dir": "logs",
        },
    }


def test_config_exposes_session_and_capture_settings():
    config = _build_config()

    assert config.server_url == "https://examguard.school.edu/"
    assert config.exam_id == "EXAM_2026_FINAL"
    assert config.student_id == "S12345"
    assert config.auth_token == "secret-token"
    assert config.sampling_fps == 5.0
    assert config.ring_buffer_seconds == 60.0
    assert config.clip_before_flag_seconds == 15.0
    assert config.clip_after_flag_seconds == 5.0
    assert config.clip_bitrate == "500k"
    assert config.clip_dir == Path("data/staged_clips")
    assert config.metadata_interval_seconds == 5.0


def test_config_exposes_nested_detector_settings():
    config = _build_config()

    assert config.detectors.phone.enabled is True
    assert config.detectors.phone.confidence_threshold == 0.50

    assert config.detectors.gaze.enabled is True
    assert config.detectors.gaze.head_pitch_threshold == 20.0
    assert config.detectors.gaze.head_yaw_threshold == 25.0
    assert config.detectors.gaze.eye_pitch_threshold == 15.0
    assert config.detectors.gaze.eye_yaw_threshold == 20.0
    assert config.detectors.gaze.duration_threshold_seconds == 4.0

    assert config.detectors.multi_person.enabled is True
    assert config.detectors.multi_person.roi_mode == "proximity"


def test_config_exposes_fusion_and_logging_settings():
    config = _build_config()

    assert config.fusion.phone_weight == 0.5
    assert config.fusion.gaze_weight == 0.3
    assert config.fusion.extra_person_weight == 0.4
    assert config.fusion.multi_signal_bonus == 0.3
    assert config.fusion.flag_threshold == 0.5
    assert config.fusion.flag_cooldown_seconds == 30.0

    assert config.logging.level == "INFO"
    assert config.logging.log_dir == Path("logs")
    assert isinstance(config.logging.log_dir, Path)


def test_config_is_immutable():
    config = _build_config()

    with pytest.raises(FrozenInstanceError):
        config.sampling_fps = 10.0


def test_nested_configs_are_immutable():
    config = _build_config()

    with pytest.raises(FrozenInstanceError):
        config.detectors.phone.confidence_threshold = 0.75


def test_auth_token_is_hidden_from_repr():
    config = _build_config()

    rendered = repr(config)

    assert "secret-token" not in rendered
    assert "auth_token" not in rendered


def test_load_config_parses_yaml_into_typed_config(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        yaml.safe_dump(_build_raw_config()),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config == _build_config()
    assert isinstance(config.detectors, DetectorConfig)
    assert isinstance(config.logging.log_dir, Path)


def test_example_config_matches_current_schema():
    project_root = Path(__file__).resolve().parents[1]

    config = load_config(project_root / "config" / "ex.config.yml")

    assert config.sampling_fps == 5.0
    assert config.metadata_interval_seconds == 5.0
    assert config.detectors.phone.confidence_threshold == 0.50
    assert config.auth_token == "replace-me"


@pytest.mark.parametrize("invalid_interval", [0, -1, "5", True])
def test_parse_config_rejects_invalid_metadata_interval(invalid_interval):
    raw = _build_raw_config()
    raw["metadata_interval_seconds"] = invalid_interval

    with pytest.raises(ConfigError, match="metadata_interval_seconds"):
        _parse_config(raw)


def test_parse_config_reports_missing_nested_field():
    raw = _build_raw_config()
    del raw["detectors"]["phone"]["enabled"]

    with pytest.raises(ConfigError, match=r"detectors\.phone\.enabled"):
        _parse_config(raw)


def test_parse_config_rejects_unknown_field():
    raw = _build_raw_config()
    raw["fusion"]["unexpected_weight"] = 0.2

    with pytest.raises(ConfigError, match=r"fusion\.unexpected_weight"):
        _parse_config(raw)


@pytest.mark.parametrize("invalid_fps", [0, -1, "5", True])
def test_parse_config_rejects_invalid_sampling_fps(invalid_fps):
    raw = _build_raw_config()
    raw["sampling_fps"] = invalid_fps

    with pytest.raises(ConfigError, match="sampling_fps"):
        _parse_config(raw)


def test_parse_config_rejects_confidence_outside_unit_interval():
    raw = _build_raw_config()
    raw["detectors"]["phone"]["confidence_threshold"] = 1.5

    with pytest.raises(ConfigError, match=r"detectors\.phone\.confidence_threshold"):
        _parse_config(raw)


def test_parse_config_rejects_clip_window_larger_than_buffer():
    raw = _build_raw_config()
    raw["ring_buffer_seconds"] = 10

    with pytest.raises(ConfigError, match="must not exceed ring_buffer_seconds"):
        _parse_config(raw)


def test_build_capture_config_uses_sampling_fps():
    config = _build_config()

    capture = build_capture_config(config, camera_index=1)

    assert capture.target_fps == 5.0
    assert capture.camera_index == 1
