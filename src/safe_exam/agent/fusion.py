"""Rule-based signal fusion and FlagEvent emission. Owned by #35."""

from __future__ import annotations

from dataclasses import dataclass

from safe_exam.agent.config import (
    Config,
    FusionConfig,
    GazeConfig,
    MultiPersonConfig,
    PhoneConfig,
)
from safe_exam.processor.attention_policy import (
    AttentionPolicyConfig,
    is_attention_off_center,
)
from safe_exam.processor.frame_result import FrameResult


@dataclass(frozen=True)
class FlagEvent:
    """Represents a flagged incident based on fused signals."""

    timestamp: float
    score: float
    reasons: list[str]


class SignalFuser:
    """Stateful fuser that evaluates signal logic and outputs flags."""

    def __init__(
        self,
        fusion_config: FusionConfig,
        gaze_config: GazeConfig,
        phone_config: PhoneConfig,
        multi_person_config: MultiPersonConfig,
        head_policy: AttentionPolicyConfig,
        eye_policy: AttentionPolicyConfig,
    ) -> None:
        self.fusion_config = fusion_config
        self.gaze_config = gaze_config
        self.phone_config = phone_config
        self.multi_person_config = multi_person_config
        # State tracking across frames
        self.gaze_off_start_time: float | None = None
        self.last_flag_time: float | None = None
        self.last_fused_score: float = 0.0
        # Reuse existing attention policy logic
        self.head_policy = head_policy
        self.eye_policy = eye_policy

    @classmethod
    def from_config(cls: type[SignalFuser], full_config: Config) -> SignalFuser:
        """Creates a `SignalFuser` instance from the global configuration."""
        head_policy = AttentionPolicyConfig(
            signal="head",
            mode="both",
            yaw_threshold_deg=full_config.detectors.gaze.head_yaw_threshold,
            pitch_threshold_deg=full_config.detectors.gaze.head_pitch_threshold,
        )

        eye_policy = AttentionPolicyConfig(
            signal="eye",
            mode="both",
            yaw_threshold_deg=full_config.detectors.gaze.eye_yaw_threshold,
            pitch_threshold_deg=full_config.detectors.gaze.eye_pitch_threshold,
        )

        return cls(
            fusion_config=full_config.fusion,
            gaze_config=full_config.detectors.gaze,
            phone_config=full_config.detectors.phone,
            multi_person_config=full_config.detectors.multi_person,
            head_policy=head_policy,
            eye_policy=eye_policy,
        )

    def _can_emit_flag(self, frame_timestamp: float) -> bool:
        """Determines if a new flag can be emitted based on the cooldown period."""
        if self.last_flag_time is None:
            return True
        time_since_last = frame_timestamp - self.last_flag_time
        return time_since_last >= self.fusion_config.flag_cooldown_seconds

    def _evaluate_gaze_violation(self, frame_result: FrameResult) -> bool:
        """
        Determine if gaze violation is currently active based on head and eye policies.
        Preserves state across frames to enforce duration threshold.
        """
        if not self.gaze_config.enabled:
            return False

        head_off = is_attention_off_center(frame_result, self.head_policy)
        eye_off = is_attention_off_center(frame_result, self.eye_policy)

        if head_off or eye_off:
            if self.gaze_off_start_time is None:
                self.gaze_off_start_time = frame_result.timestamp
            duration = frame_result.timestamp - self.gaze_off_start_time
            return duration >= self.gaze_config.duration_threshold_seconds
        else:
            self.gaze_off_start_time = None
            return False

    def fuse_signals(
        self, frame_result: FrameResult, gaze_violation: bool
    ) -> tuple[float, list[str]]:
        """
        Fuse the detection signals into a single confidence score and list of reasons.
        """
        score = 0.0
        reasons: list[str] = []

        if (
            self.phone_config.enabled
            and frame_result.phone_confidence >= self.phone_config.confidence_threshold
        ):
            score += self.fusion_config.phone_weight
            reasons.append("phone")

        if gaze_violation:
            score += self.fusion_config.gaze_weight
            reasons.append("gaze_violation")

        if self.multi_person_config.enabled and frame_result.person_count > 1:
            score += self.fusion_config.extra_person_weight
            reasons.append("multiple_person")

        if len(reasons) > 1:
            score += self.fusion_config.multi_signal_bonus

        return min(score, 1.0), reasons

    def process_frame(self, frame_result: FrameResult) -> FlagEvent | None:
        """
        Process a single frame result, update state and return a `FlagEvent` if
        thresholds are met and cooldown is respected.
        """
        gaze_violation = self._evaluate_gaze_violation(frame_result)

        score, reasons = self.fuse_signals(frame_result, gaze_violation)
        self.last_fused_score = score

        if score >= self.fusion_config.flag_threshold:
            if self._can_emit_flag(frame_result.timestamp):
                self.last_flag_time = frame_result.timestamp
                return FlagEvent(
                    timestamp=frame_result.timestamp, score=score, reasons=reasons
                )

        return None
