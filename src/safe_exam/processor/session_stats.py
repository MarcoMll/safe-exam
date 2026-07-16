"""Session-level stats for the processor runner (not per-frame output)."""

import time
from dataclasses import dataclass

from safe_exam.processor.attention_policy import (
    AttentionPolicyConfig,
    is_attention_off_center,
)
from safe_exam.processor.frame_result import FrameResult

_DEBUG_COUNTER_ATTRS = {
    "head": "head_off_center_frames",
    "eye": "eye_off_center_frames",
    "gaze": "gaze_off_center_frames",
}


@dataclass
class ProcessorRunStats:
    frame_count: int = 0
    total_inference_time_ms: float = 0.0
    phone_detected_frames: int = 0
    extra_person_frames: int = 0
    face_detected_frames: int = 0
    head_off_center_frames: int = 0
    eye_off_center_frames: int = 0
    gaze_off_center_frames: int = 0
    attention_off_center_frames: int = 0


def update_from_frame(
    stats: ProcessorRunStats,
    result: FrameResult,
    active_policy: AttentionPolicyConfig,
    *,
    inference_time_ms: float = 0.0,
    debug_policies: dict[str, AttentionPolicyConfig] | None = None,
) -> None:
    """Update session counters from one processed frame."""
    stats.frame_count += 1
    stats.total_inference_time_ms += inference_time_ms

    if result.phone_detected:
        stats.phone_detected_frames += 1
    if result.extra_person_detected:
        stats.extra_person_frames += 1
    if not result.face_detected:
        return

    stats.face_detected_frames += 1

    if debug_policies:
        for name, policy in debug_policies.items():
            attr = _DEBUG_COUNTER_ATTRS.get(name)
            if attr and is_attention_off_center(result, policy):
                setattr(stats, attr, getattr(stats, attr) + 1)

    if is_attention_off_center(result, active_policy):
        stats.attention_off_center_frames += 1


def build_summary(
    stats: ProcessorRunStats, session_start: float
) -> dict[str, float | int]:
    elapsed = time.perf_counter() - session_start
    avg_inference_ms = (
        stats.total_inference_time_ms / stats.frame_count
        if stats.frame_count > 0
        else 0.0
    )
    avg_fps = stats.frame_count / elapsed if elapsed > 0 else 0.0

    return {
        "frame_count": stats.frame_count,
        "avg_inference_ms": avg_inference_ms,
        "avg_fps": avg_fps,
        "phone_detected_frames": stats.phone_detected_frames,
        "extra_person_frames": stats.extra_person_frames,
        "face_detected_frames": stats.face_detected_frames,
        "head_off_center_frames": stats.head_off_center_frames,
        "eye_off_center_frames": stats.eye_off_center_frames,
        "gaze_off_center_frames": stats.gaze_off_center_frames,
        "attention_off_center_frames": stats.attention_off_center_frames,
    }
