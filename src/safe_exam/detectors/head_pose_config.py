from dataclasses import dataclass


@dataclass
class HeadPoseConfig:
    """Configuration for MediaPipe head pose / gaze estimation."""

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    landmark_ids: tuple[int, ...] = (1, 33, 61, 199, 263, 291)
    nose_landmark_id: int = 1
    yaw_threshold_deg: float = 10.0
    pitch_threshold_deg: float = 10.0
    mirror_preview: bool = True
    draw_landmarks: bool = True
    line_scale: float = 10.0
    line_length: int = 3
