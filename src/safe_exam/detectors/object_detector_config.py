from dataclasses import dataclass


@dataclass
class ObjectDetectorConfig:
    """
    Configuration for the object detector.
    var model_name: The name of the model to use.
    var phone_class_id: The class id of the phone.
    var person_class_id: The class id of the person.
    var confidence_threshold: The confidence threshold for the detection.
    """

    model_name: str = "yolo26s.pt"
    phone_class_id: int = 67
    person_class_id: int = 0
    confidence_threshold: float = 0.25
