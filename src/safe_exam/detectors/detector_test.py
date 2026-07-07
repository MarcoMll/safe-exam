import cv2
import argparse

from safe_exam.capture.capture_loop import capture_frames
from safe_exam.capture.config import CaptureConfig
from safe_exam.detectors.object_detector import ObjectDetector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--target-fps", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    detector = ObjectDetector()
    config = CaptureConfig(camera_index=args.camera_index, target_fps=args.target_fps)

    for frame in capture_frames(config):
        results = detector.detect(frame)
        phone_result = detector.detect_phone(frame)

        display = results[0].plot()
        cv2.putText(
            display,
            (
                f"phone: {phone_result['phone_detected']} "
                f"conf: {phone_result['confidence']:.2f}"
            ),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("safe-exam detection", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
