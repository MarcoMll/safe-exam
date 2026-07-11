from dataclasses import asdict, dataclass

from safe_exam.detectors.head_pose_detector import HeadPoseResult


@dataclass
class FrameResult:
    """
    Merged detection output for one frame (processor public contract).
    """

    phone_detected: bool = False
    phone_confidence: float = 0.0
    person_count: int = 0
    extra_person_detected: bool = False
    gaze_pitch: float = 0.0
    gaze_yaw: float = 0.0
    timestamp: float = 0.0


@dataclass
class ProcessFrameOutput:
    """
    Full processor output for one frame, including raw artifacts for debug.
    """

    result: FrameResult
    yolo_results: object
    head_pose_result: HeadPoseResult
    inference_time_ms: float

    def as_dict(self) -> dict:
        """
        Convert the FrameResult to a dictionary.
        """
        return asdict(self.result)
