"""Optional standalone demo for testing face gaze detection only."""

import cv2

from safe_exam.capture.capture import capture_frames
from safe_exam.capture.capture_config import CaptureConfig
from safe_exam.detectors.face_gaze import FaceGazeDetector, draw_face_gaze_overlay


def main() -> None:
    """Run the face gaze detector with a live debug overlay."""
    detector = FaceGazeDetector()
    capture_config = CaptureConfig()

    try:
        for frame in capture_frames(capture_config):
            result = detector.detect(frame)
            display = draw_face_gaze_overlay(frame, result, detector.config)
            cv2.imshow("Face Gaze Demo", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
