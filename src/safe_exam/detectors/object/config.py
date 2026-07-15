from dataclasses import dataclass


@dataclass
class ObjectDetectorConfig:
    """Configuration for the YOLO object detector."""

    model_name: str = "yolo26s.pt"
    phone_class_id: int = 67
    person_class_id: int = 0
    confidence_threshold: float = 0.50
