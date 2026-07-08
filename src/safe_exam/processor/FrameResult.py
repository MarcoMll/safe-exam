from dataclasses import dataclass


@dataclass
class FrameResult:
    phone_detected: bool = False
    phone_confidence: float = 0.0
    person_count: int = 0
    extra_person_detected: bool = False
    gaze_pitch: float = 0.0
    gaze_yaw: float = 0.0
    timestamp: float = 0.0
