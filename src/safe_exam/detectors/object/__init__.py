from safe_exam.detectors.object.config import ObjectDetectorConfig
from safe_exam.detectors.object.overlay import draw_object_overlay
from safe_exam.detectors.object.results import DetectedBox

__all__ = [
    "DetectedBox",
    "ObjectDetector",
    "ObjectDetectorConfig",
    "draw_object_overlay",
]


def __getattr__(name: str):
    if name == "ObjectDetector":
        from safe_exam.detectors.object.detector import ObjectDetector

        return ObjectDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
