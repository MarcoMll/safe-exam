"""Tests for attention / off-center policy helpers."""

import pytest

from safe_exam.processor.attention_policy import (
    DEFAULT_ATTENTION_POLICY,
    AttentionPolicyConfig,
    is_attention_off_center,
)
from safe_exam.processor.frame_result import FrameResult


def _face(**kwargs) -> FrameResult:
    defaults = {
        "face_detected": True,
        "head_pitch": 0.0,
        "head_yaw": 0.0,
        "eye_pitch": 0.0,
        "eye_yaw": 0.0,
        "gaze_pitch": 0.0,
        "gaze_yaw": 0.0,
        "iris_offset_x": 0.0,
        "iris_offset_y": 0.0,
    }
    defaults.update(kwargs)
    return FrameResult(**defaults)


def test_no_face_never_off_center():
    result = FrameResult(face_detected=False, gaze_yaw=45.0)
    assert is_attention_off_center(result, DEFAULT_ATTENTION_POLICY) is False


def test_default_policy_yaw_only_triggers_on_large_gaze_yaw():
    on_center = _face(gaze_yaw=3.0, gaze_pitch=40.0)
    off_center = _face(gaze_yaw=12.0, gaze_pitch=0.0)

    assert is_attention_off_center(on_center, DEFAULT_ATTENTION_POLICY) is False
    assert is_attention_off_center(off_center, DEFAULT_ATTENTION_POLICY) is True


def test_default_policy_ignores_pitch():
    # pitch_threshold_deg=99 on default — looking down alone should not flag
    looking_down = _face(gaze_yaw=0.0, gaze_pitch=30.0)
    assert is_attention_off_center(looking_down, DEFAULT_ATTENTION_POLICY) is False


def test_both_mode_flags_on_pitch_or_yaw():
    policy = AttentionPolicyConfig(
        signal="gaze",
        mode="both",
        yaw_threshold_deg=5.0,
        pitch_threshold_deg=10.0,
    )
    assert is_attention_off_center(_face(gaze_yaw=8.0, gaze_pitch=0.0), policy) is True
    assert is_attention_off_center(_face(gaze_yaw=0.0, gaze_pitch=15.0), policy) is True
    assert is_attention_off_center(_face(gaze_yaw=2.0, gaze_pitch=5.0), policy) is False


def test_pitch_only_mode():
    policy = AttentionPolicyConfig(
        signal="head",
        mode="pitch_only",
        yaw_threshold_deg=1.0,
        pitch_threshold_deg=10.0,
    )
    assert (
        is_attention_off_center(_face(head_yaw=40.0, head_pitch=5.0), policy) is False
    )
    assert is_attention_off_center(_face(head_yaw=0.0, head_pitch=15.0), policy) is True


def test_head_and_eye_signals_use_their_angles():
    head_policy = AttentionPolicyConfig(
        signal="head",
        mode="yaw_only",
        yaw_threshold_deg=5.0,
    )
    eye_policy = AttentionPolicyConfig(
        signal="eye",
        mode="yaw_only",
        yaw_threshold_deg=5.0,
    )
    result = _face(head_yaw=20.0, eye_yaw=1.0, gaze_yaw=1.0)

    assert is_attention_off_center(result, head_policy) is True
    assert is_attention_off_center(result, eye_policy) is False


def test_iris_signal_requires_threshold():
    policy = AttentionPolicyConfig(signal="iris", iris_threshold=None)
    with pytest.raises(ValueError, match="iris_threshold"):
        is_attention_off_center(_face(), policy)


def test_iris_signal_uses_offsets():
    policy = AttentionPolicyConfig(signal="iris", iris_threshold=0.1)
    on_center = _face(iris_offset_x=0.05, iris_offset_y=0.05)
    off_center = _face(iris_offset_x=0.2, iris_offset_y=0.0)

    assert is_attention_off_center(on_center, policy) is False
    assert is_attention_off_center(off_center, policy) is True
