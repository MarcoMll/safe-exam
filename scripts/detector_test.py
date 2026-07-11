import cv2

from safe_exam.capture.capture import capture_frames
from safe_exam.capture.capture_config import CaptureConfig
from safe_exam.detectors.object_detector import ObjectDetector


def main():
    """
    Main function to test the object detector.
    """
    detector = ObjectDetector()
    config = CaptureConfig(camera_index=1, target_fps=12)

    for frame in capture_frames(config):
        results = detector.detect(frame=frame, classes=[67, 0])

        phone_lookup_results = detector.look_for_class(
            results=results, target_class_index=67, threshold=0.25
        )
        person_lookup_results = detector.look_for_class(
            results=results, target_class_index=0, threshold=0.25
        )
        person_count_results = detector.count_class(
            results=results, target_class_index=0, threshold=0.25
        )

        display = results[0].plot()
        cv2.putText(
            display,
            (
                f"phone_detected: {phone_lookup_results[0]} \n"
                f"confidence: {phone_lookup_results[1]:.2f}\n"
                f"person_detected: {person_lookup_results[0]} \n"
                f"persons_count: {person_count_results}"
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
