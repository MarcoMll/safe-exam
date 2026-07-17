from dataclasses import dataclass


@dataclass(frozen=True)
class DetectedBox:
    """One YOLO bounding box (phone or person)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
