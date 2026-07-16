from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from safe_exam.processor.frame_result import FrameResult

AttentionSignal = Literal["head", "eye", "gaze", "iris"]
AttentionMode = Literal["both", "yaw_only", "pitch_only"]


@dataclass(frozen=True)
class AttentionPolicyConfig:
    """Runtime policy for interpreting raw attention signals."""

    signal: AttentionSignal = "gaze"
    mode: AttentionMode = "yaw_only"
    yaw_threshold_deg: float = 5.0
    pitch_threshold_deg: float = 99.0
    iris_threshold: float | None = None

    def label(self) -> str:
        return (
            f"signal={self.signal} mode={self.mode} "
            f"yaw={self.yaw_threshold_deg:.1f} "
            f"pitch={self.pitch_threshold_deg:.1f}"
        )


def _signal_angles(
    result: FrameResult,
    signal: AttentionSignal,
) -> tuple[float, float]:
    if signal == "head":
        return result.head_pitch, result.head_yaw
    if signal == "eye":
        return result.eye_pitch, result.eye_yaw
    if signal == "gaze":
        return result.gaze_pitch, result.gaze_yaw
    raise ValueError(f"Signal {signal!r} does not expose pitch/yaw angles.")


def _angles_off_center(
    pitch: float,
    yaw: float,
    policy: AttentionPolicyConfig,
) -> bool:
    if policy.mode == "yaw_only":
        return abs(yaw) > policy.yaw_threshold_deg
    if policy.mode == "pitch_only":
        return abs(pitch) > policy.pitch_threshold_deg
    return (
        abs(yaw) > policy.yaw_threshold_deg or abs(pitch) > policy.pitch_threshold_deg
    )


def is_attention_off_center(
    result: FrameResult,
    policy: AttentionPolicyConfig,
) -> bool:
    """Evaluate one frame against the active runtime attention policy."""
    if not result.face_detected:
        return False

    if policy.signal == "iris":
        if policy.iris_threshold is None:
            raise ValueError("iris_threshold is required when signal='iris'.")
        return (
            abs(result.iris_offset_x) > policy.iris_threshold
            or abs(result.iris_offset_y) > policy.iris_threshold
        )

    pitch, yaw = _signal_angles(result, policy.signal)
    return _angles_off_center(pitch, yaw, policy)


DEFAULT_ATTENTION_POLICY = AttentionPolicyConfig(
    signal="gaze",
    mode="yaw_only",
    yaw_threshold_deg=5.0,
    pitch_threshold_deg=99.0,
)

DEBUG_SIGNAL_POLICIES: dict[str, AttentionPolicyConfig] = {
    "head": AttentionPolicyConfig(
        signal="head",
        mode="both",
        yaw_threshold_deg=10.0,
        pitch_threshold_deg=10.0,
    ),
    "eye": AttentionPolicyConfig(
        signal="eye",
        mode="both",
        yaw_threshold_deg=3.0,
        pitch_threshold_deg=5.0,
    ),
    "gaze": AttentionPolicyConfig(
        signal="gaze",
        mode="yaw_only",
        yaw_threshold_deg=5.0,
        pitch_threshold_deg=99.0,
    ),
}
