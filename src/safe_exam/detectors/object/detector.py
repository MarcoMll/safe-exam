import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from safe_exam.detectors.object.config import ObjectDetectorConfig
from safe_exam.utils.paths_initializer import get_paths


class ObjectDetector:
    """YOLO-based object detector for phones and persons."""

    def __init__(self, config: ObjectDetectorConfig | None = None):
        self.config = config or ObjectDetectorConfig()
        paths = get_paths()
        self.model = YOLO(paths.MODELS_DIR / self.config.model_name)

    def look_for_class(
        self, results: list[Results], target_class_index, threshold: float = 0
    ) -> tuple[bool, float]:
        max_confidence = 0.0

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if class_id == target_class_index:
                    max_confidence = max(max_confidence, confidence)

        return (max_confidence >= threshold), max_confidence

    def count_class(
        self, results: list[Results], target_class_index: int, threshold: float = 0
    ) -> int:
        count = 0

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id == target_class_index and confidence >= threshold:
                    count += 1

        return count

    def detect(self, frame: np.ndarray, classes=None):
        if classes is None:
            classes = [self.config.person_class_id, self.config.phone_class_id]

        return self.model.predict(source=frame, classes=classes, verbose=False)
