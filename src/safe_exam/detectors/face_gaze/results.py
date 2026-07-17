from dataclasses import dataclass


@dataclass
class FaceGazeResult:
    """Detection output for one frame (no drawing attached)."""

    face_detected: bool = False
    face_count: int = 0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    eye_pitch: float = 0.0
    eye_yaw: float = 0.0
    gaze_pitch: float = 0.0
    gaze_yaw: float = 0.0
    iris_offset_x: float = 0.0
    iris_offset_y: float = 0.0
    head_direction: str = "No face detected"
    raw_angles: tuple[float | None, float | None, float | None] | None = None
    nose_2d: tuple[float, float] | None = None
    face_landmarks: object | None = None
