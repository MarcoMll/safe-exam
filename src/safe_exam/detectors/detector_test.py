import cv2

from safe_exam.capture.capture_loop import capture_frames
from safe_exam.capture.config import CaptureConfig
from safe_exam.detectors.object_detector import ObjectDetector

def main():
    detector = ObjectDetector()
    config = CaptureConfig(camera_index=1, target_fps=12)

    for frame in capture_frames(config):
        results = detector.detect(frame)
        phone_results = detector.check_for_object(results, 67)
        persons_results = detector.count_class(results, 0)

        display = results[0].plot()
        cv2.putText(
            display,
            (
                f"phone_detected: {phone_results[0]}\n"
                f"confidence: {phone_results[1]:.2f}\n\n"
                f"persons_count: {persons_results}\n"
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
