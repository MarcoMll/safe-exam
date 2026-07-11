"""Session-level stats for the processor runner (not per-frame output)."""

import time
from dataclasses import dataclass


@dataclass
class ProcessorRunStats:
    frame_count: int = 0
    total_inference_time_ms: float = 0.0
    phone_detected_frames: int = 0
    extra_person_frames: int = 0


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
    }
