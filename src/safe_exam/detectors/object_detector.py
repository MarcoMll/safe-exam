import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from safe_exam.utils.paths_initializer import get_paths

class ObjectDetector:
    def __init__(self, model_name: str = "yolo26s.pt"):
        """
        Initialize a YOLO-based object detection model.
        :param model_name: name of the model
        """
        paths = get_paths()
        self.model = YOLO(paths.MODELS_DIR / model_name)

    def check_for_phone(self, results: list[Results]):
        """
        Performs phone detection on a given frame.
        :param results: list of prediction results as Result objects, obtained from a frame
        :return: a dictionary with the phone-detection status and confidence
        """
        max_confidence = 0.0

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                max_confidence = max(max_confidence, confidence)

        return {
            "phone_detected": max_confidence >= 0.25,
            "confidence": max_confidence,
        }

    def detect(self, frame: np.ndarray, classes=None):
        """
        Performs object detection on a given frame. If classes are not provided,
        the model will fallback to the default list of classes: 0, 67. For persons and cell-phones.
        :param frame: frame to perform object detection on
        :param classes: list of class indexes
        :return: list of detected objects
        """
        if classes is None:
            classes = [0, 67]

        return self.model.predict(
            source=frame,
            classes=classes
        )
