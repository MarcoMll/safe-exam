import time

import cv2

from safe_exam.capture.config import CaptureConfig


def capture_frames(config: CaptureConfig):
    """Yield webcam frames (numpy arrays) at roughly config.target_fps."""
    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {config.camera_index}")

    interval = 1.0 / config.target_fps

    try:
        while True:
            start = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                break

            yield frame

            remaining = interval - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        cap.release()


def run_capture(config: CaptureConfig | None = None) -> None:
    """Show the live feed with a debug overlay. Press q to quit."""
    config = config or CaptureConfig()

    frame_count = 0
    t0 = time.perf_counter()
    fps = 0.0

    try:
        for frame in capture_frames(config):
            frame_count += 1
            now = time.perf_counter()
            if now - t0 >= 1.0:
                fps = frame_count / (now - t0)
                frame_count = 0
                t0 = now

            if not config.show_debug:
                continue

            h, w = frame.shape[:2]
            display = frame.copy()
            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display,
                f"Size: {w}x{h}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow(config.window_name, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
