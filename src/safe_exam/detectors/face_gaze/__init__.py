from safe_exam.detectors.face_gaze.config import FaceGazeConfig
from safe_exam.detectors.face_gaze.overlay import (
    draw_face_gaze_on_frame,
    draw_face_gaze_overlay,
)
from safe_exam.detectors.face_gaze.results import FaceGazeResult

__all__ = [
    "FaceGazeConfig",
    "FaceGazeDetector",
    "FaceGazeResult",
    "draw_face_gaze_on_frame",
    "draw_face_gaze_overlay",
]


def __getattr__(name: str):
    if name == "FaceGazeDetector":
        from safe_exam.detectors.face_gaze.detector import FaceGazeDetector

        return FaceGazeDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
