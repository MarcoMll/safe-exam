"""Session-level stats for the processor runner (not per-frame output)."""

import time
from dataclasses import dataclass

from safe_exam.detectors.face_gaze import FaceGazeConfig
from safe_exam.processor.frame_result import FrameResult


@dataclass
class ProcessorRunStats:
    frame_count: int = 0
    total_inference_time_ms: float = 0.0
    phone_detected_frames: int = 0
    extra_person_frames: int = 0
    face_detected_frames: int = 0
    head_off_center_frames: int = 0
    eye_off_center_frames: int = 0


def is_head_off_center(result: FrameResult, config: FaceGazeConfig) -> bool:
    """Return True when head pitch/yaw exceed configured thresholds."""
    return (
        abs(result.head_yaw) > config.yaw_threshold_deg
        or abs(result.head_pitch) > config.pitch_threshold_deg
    )


def is_eye_off_center(result: FrameResult, config: FaceGazeConfig) -> bool:
    """Return True when iris eye offset exceeds configured thresholds."""
    return (
        abs(result.eye_yaw) > config.eye_yaw_threshold_deg
        or abs(result.eye_pitch) > config.eye_pitch_threshold_deg
    )


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
    }
