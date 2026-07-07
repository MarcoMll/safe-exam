import cv2

from safe_exam.capture.capture_loop import capture_frames
from safe_exam.capture.config import CaptureConfig
from safe_exam.detectors.object_detector import ObjectDetector


detector = ObjectDetector()

for frame in capture_frames(CaptureConfig()):
    result = detector.detect(frame)
    display = result.plot()

    cv2.imshow("safe-exam detection", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
