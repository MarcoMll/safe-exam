"""Optional standalone demo for testing head pose detection only."""

import cv2

from safe_exam.capture.capture import capture_frames
from safe_exam.capture.capture_config import CaptureConfig
from safe_exam.detectors.head_pose_detector import HeadPoseDetector
from safe_exam.detectors.head_pose_overlay import draw_head_pose_overlay


def main() -> None:
    """
    Main function to test the head pose detector.
    """
    detector = HeadPoseDetector()
    capture_config = CaptureConfig()

    try:
        for frame in capture_frames(capture_config):
            result = detector.detect(frame)
            display = draw_head_pose_overlay(frame, result, detector.config)
            cv2.imshow("Head Pose Demo", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
