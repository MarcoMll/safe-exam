from dataclasses import asdict, dataclass, field

from safe_exam.detectors.face_gaze.results import FaceGazeResult
from safe_exam.detectors.object.results import DetectedBox


@dataclass
class FrameResult:
    """Merged detection output for one frame (processor public contract)."""

    phone_detected: bool = False
    phone_confidence: float = 0.0
    person_count: int = 0
    person_boxes: list[DetectedBox] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0

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
