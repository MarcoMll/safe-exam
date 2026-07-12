from dataclasses import asdict, dataclass

from safe_exam.detectors.face_gaze import FaceGazeResult


@dataclass
class FrameResult:
    """Merged detection output for one frame (processor public contract)."""

    phone_detected: bool = False
    phone_confidence: float = 0.0
    person_count: int = 0
    extra_person_detected: bool = False

    face_detected: bool = False
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    eye_pitch: float = 0.0
    eye_yaw: float = 0.0
    gaze_pitch: float = 0.0
    gaze_yaw: float = 0.0
    iris_offset_x: float = 0.0
    iris_offset_y: float = 0.0
    head_direction: str = "No face detected"

    timestamp: float = 0.0


@dataclass
class ProcessFrameOutput:
    """Full processor output for one frame, including raw artifacts for debug."""

    result: FrameResult
    yolo_results: object
    face_gaze_result: FaceGazeResult
    inference_time_ms: float

    def as_dict(self) -> dict:
        return asdict(self.result)
