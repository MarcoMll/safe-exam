"""Tests for processor session stats aggregation."""

import time

from safe_exam.detectors.object.results import DetectedBox
from safe_exam.processor.attention_policy import (
    DEBUG_SIGNAL_POLICIES,
    DEFAULT_ATTENTION_POLICY,
    AttentionPolicyConfig,
)
from safe_exam.processor.frame_result import FrameResult
from safe_exam.processor.intrusion_policy import DEFAULT_INTRUSION_POLICY
from safe_exam.processor.session_stats import (
    ProcessorRunStats,
    build_summary,
    update_from_frame,
)


def _box(x1: float, y1: float, x2: float, y2: float) -> DetectedBox:
    return DetectedBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, class_id=0)


def test_update_counts_phone_and_face():
    stats = ProcessorRunStats()
    result = FrameResult(
        phone_detected=True,
        face_detected=True,
        gaze_yaw=0.0,
    )
    update_from_frame(
        stats,
        result,
        DEFAULT_ATTENTION_POLICY,
        DEFAULT_INTRUSION_POLICY,
        inference_time_ms=12.5,
    )

    assert stats.frame_count == 1
    assert stats.phone_detected_frames == 1
    assert stats.face_detected_frames == 1
    assert stats.attention_off_center_frames == 0
    assert stats.total_inference_time_ms == 12.5


def test_update_skips_attention_when_no_face():
    stats = ProcessorRunStats()
    result = FrameResult(face_detected=False, gaze_yaw=40.0, phone_detected=True)
    update_from_frame(
        stats,
        result,
        DEFAULT_ATTENTION_POLICY,
        DEFAULT_INTRUSION_POLICY,
    )

    assert stats.phone_detected_frames == 1
    assert stats.face_detected_frames == 0
    assert stats.attention_off_center_frames == 0


def test_update_counts_attention_off_center():
    stats = ProcessorRunStats()
    result = FrameResult(face_detected=True, gaze_yaw=20.0)
    update_from_frame(
        stats,
        result,
        DEFAULT_ATTENTION_POLICY,
        DEFAULT_INTRUSION_POLICY,
    )

    assert stats.attention_off_center_frames == 1


def test_update_counts_intrusion_when_lean_in():
    stats = ProcessorRunStats()
    primary = _box(200, 100, 600, 500)
    lean_in = _box(450, 80, 700, 480)
    result = FrameResult(
        person_count=2,
        person_boxes=[primary, lean_in],
        frame_width=1280,
        frame_height=720,
    )
    update_from_frame(
        stats,
        result,
        DEFAULT_ATTENTION_POLICY,
        DEFAULT_INTRUSION_POLICY,
    )

    assert stats.intrusion_suspected_frames == 1


def test_debug_policies_increment_signal_counters():
    stats = ProcessorRunStats()
    # head yaw large enough for DEBUG head policy (yaw 10°), gaze still on-center
    result = FrameResult(
        face_detected=True,
        head_yaw=25.0,
        head_pitch=0.0,
        eye_yaw=0.0,
        eye_pitch=0.0,
        gaze_yaw=0.0,
        gaze_pitch=0.0,
    )
    update_from_frame(
        stats,
        result,
        DEFAULT_ATTENTION_POLICY,
        DEFAULT_INTRUSION_POLICY,
        debug_policies=DEBUG_SIGNAL_POLICIES,
    )

    assert stats.head_off_center_frames == 1
    assert stats.eye_off_center_frames == 0
    assert stats.gaze_off_center_frames == 0
    assert stats.attention_off_center_frames == 0


def test_build_summary_averages_and_exposes_counters():
    stats = ProcessorRunStats(
        frame_count=4,
        total_inference_time_ms=40.0,
        phone_detected_frames=1,
        intrusion_suspected_frames=2,
        face_detected_frames=3,
        attention_off_center_frames=1,
    )
    started = time.perf_counter() - 2.0
    summary = build_summary(stats, started)

    assert summary["frame_count"] == 4
    assert summary["avg_inference_ms"] == 10.0
    assert summary["avg_fps"] > 0
    assert summary["phone_detected_frames"] == 1
    assert summary["intrusion_suspected_frames"] == 2
    assert summary["face_detected_frames"] == 3
    assert summary["attention_off_center_frames"] == 1


def test_build_summary_empty_session():
    stats = ProcessorRunStats()
    summary = build_summary(stats, time.perf_counter())

    assert summary["frame_count"] == 0
    assert summary["avg_inference_ms"] == 0.0


def test_active_policy_override_affects_attention_counter():
    stats = ProcessorRunStats()
    strict = AttentionPolicyConfig(
        signal="gaze",
        mode="yaw_only",
        yaw_threshold_deg=1.0,
    )
    result = FrameResult(face_detected=True, gaze_yaw=2.0)
    update_from_frame(stats, result, strict, DEFAULT_INTRUSION_POLICY)

    assert stats.attention_off_center_frames == 1
