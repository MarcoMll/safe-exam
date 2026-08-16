from pathlib import Path

import pytest

from safe_exam.agent.config import (
    Config,
    DetectorConfig,
    FusionConfig,
    GazeConfig,
    LoggingConfig,
    MultiPersonConfig,
    PhoneConfig,
)
from safe_exam.agent.fusion import SignalFuser
from safe_exam.processor.frame_result import FrameResult


@pytest.fixture
def fuser():
    full_config = Config(
        server_url="http://test.local",
        exam_id="test_exam",
        student_id="student123",
        auth_token="secret",
        sampling_fps=10.0,
        ring_buffer_seconds=30.0,
        clip_before_flag_seconds=5.0,
        clip_after_flag_seconds=5.0,
        clip_bitrate="500k",
        clip_dir=Path("data/staged_clips"),
        detectors=DetectorConfig(
            phone=PhoneConfig(enabled=True, confidence_threshold=0.8),
            gaze=GazeConfig(
                enabled=True,
                head_pitch_threshold=20.0,
                head_yaw_threshold=20.0,
                eye_pitch_threshold=10.0,
                eye_yaw_threshold=10.0,
                duration_threshold_seconds=2.0,
            ),
            multi_person=MultiPersonConfig(enabled=True, roi_mode="full"),
        ),
        fusion=FusionConfig(
            phone_weight=0.6,
            gaze_weight=0.5,
            extra_person_weight=0.7,
            multi_signal_bonus=0.2,
            flag_threshold=0.8,
            flag_cooldown_seconds=10.0,
        ),
        logging=LoggingConfig(level="INFO", log_dir=Path("logs")),
    )
    return SignalFuser.from_config(full_config)


def test_evaluate_gaze_violation_head_only(fuser: SignalFuser):
    """Test head pose triggering gaze violation independently."""
    # Head exceeds 20.0 threshold, eye is fine
    frame = FrameResult(face_detected=True, head_yaw=25.0, timestamp=1.0)
    assert not fuser._evaluate_gaze_violation(
        frame
    )  # Not yet a violation (duration < 2.0s)
    # Next frame at 4.0s (duration = 3.0s >= 2.0s threshold)
    frame2 = FrameResult(face_detected=True, head_yaw=25.0, timestamp=4.0)
    assert fuser._evaluate_gaze_violation(frame2)  # Now a violation


def test_evaluate_gaze_violation_eye_only(fuser: SignalFuser):
    """Test eye (iris) tracking triggering gaze violation independently."""
    # Eye exceeds 10.0 threshold, head is fine
    frame = FrameResult(face_detected=True, eye_pitch=15.0, timestamp=1.0)
    assert not fuser._evaluate_gaze_violation(
        frame
    )  # Not yet a violation (duration < 2.0s)
    frame2 = FrameResult(face_detected=True, eye_pitch=15.0, timestamp=4.0)
    assert fuser._evaluate_gaze_violation(frame2)  # Now a violation


def test_evaluate_gaze_violation_together(fuser: SignalFuser):
    """Test head pose and eye tracking triggering together."""
    # Both exceed threshold
    frame = FrameResult(
        face_detected=True, head_yaw=25.0, eye_pitch=15.0, timestamp=1.0
    )
    assert not fuser._evaluate_gaze_violation(
        frame
    )  # Not yet a violation (duration < 2.0s)
    frame2 = FrameResult(
        face_detected=True, head_yaw=25.0, eye_pitch=15.0, timestamp=4.0
    )
    assert fuser._evaluate_gaze_violation(frame2)  # Now a violation


def test_fuse_signals_score_calculation(fuser: SignalFuser):
    """Test score calculation for individual signals."""
    # Phone only
    frame = FrameResult(phone_confidence=0.85)
    score, reasons = fuser.fuse_signals(frame, gaze_violation=False)
    assert score == fuser.fusion_config.phone_weight
    assert "phone" in reasons

    # Gaze only
    frame = FrameResult()
    score, reasons = fuser.fuse_signals(frame, gaze_violation=True)
    assert score == fuser.fusion_config.gaze_weight
    assert "gaze_violation" in reasons


def test_fuse_signals_cap_at_one_and_bonus(fuser: SignalFuser):
    """Test score calculation capping at 1.0 and multi-signal bonus logic."""
    # Multi-person (0.7) + phone (0.6) + bonus (0.2) = 1.5 -> normalized to 1.0
    frame = FrameResult(phone_confidence=0.9, person_count=2)
    score, reasons = fuser.fuse_signals(frame, gaze_violation=False)

    assert score == 1.0
    assert "phone" in reasons
    assert "multiple_person" in reasons


def test_continuous_gaze_counter(fuser: SignalFuser):
    """
    Test a sequence of mock frames where gaze goes off-screen for 0.5s, 1.5s, and 3.0s,
    verifying that duration increments via frame timestamp delta and resets to 0.0 when
    gaze returns.
    """
    # Gaze off at 1.0s (duration 0.0)
    assert not fuser._evaluate_gaze_violation(
        FrameResult(face_detected=True, head_yaw=25.0, timestamp=1.0)
    )
    assert fuser.gaze_off_start_time == 1.0

    # Gaze off at 1.5s (duration 0.5)
    assert not fuser._evaluate_gaze_violation(
        FrameResult(face_detected=True, head_yaw=25.0, timestamp=1.5)
    )

    # Gaze off at 2.5s (duration 1.5)
    assert not fuser._evaluate_gaze_violation(
        FrameResult(face_detected=True, head_yaw=25.0, timestamp=2.5)
    )

    # Gaze off at 4.0s (duration 3.0 -> triggers violation)
    assert fuser._evaluate_gaze_violation(
        FrameResult(face_detected=True, head_yaw=25.0, timestamp=4.0)
    )

    # Gaze returns to center at 5.0s
    assert not fuser._evaluate_gaze_violation(
        FrameResult(face_detected=True, head_yaw=5.0, timestamp=5.0)
    )
    assert fuser.gaze_off_start_time is None


def test_cooldown_logic(fuser: SignalFuser):
    """
    Test that once a `FlagEvent` fires, subsequent frames within
    `config.flag_cooldown_seconds` do NOT emit duplicate flags,
    even if the score remains above threshold.
    """
    # Trigger a flag at timestamp 10.0
    frame = FrameResult(phone_confidence=0.9, person_count=2, timestamp=10.0)
    flag = fuser.process_frame(frame)
    assert flag is not None
    assert flag.score == 1.0

    # Send another frame with same high score at timestamp 15.0 (within 10.0 cooldown)
    frame2 = FrameResult(phone_confidence=0.9, person_count=2, timestamp=15.0)
    flag2 = fuser.process_frame(frame2)
    assert flag2 is None  # Cooldown active


def test_post_cooldown_event(fuser):
    """
    Verify a flag can fire again AFTER the cooldown period expires.
    """
    # Initial flag at 10.0
    frame = FrameResult(phone_confidence=0.9, person_count=2, timestamp=10.0)
    fuser.process_frame(frame)

    # Send frame after cooldown expires (timestamp 21.0 > 10.0 + 10.0)
    frame_post = FrameResult(phone_confidence=0.9, person_count=2, timestamp=21.0)
    flag_post = fuser.process_frame(frame_post)
    assert flag_post is not None
    assert flag_post.score == 1.0
