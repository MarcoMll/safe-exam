import logging
import time
from dataclasses import dataclass

import cv2  # pylint: disable=no-member

from safe_exam.capture.capture_loop import capture_frames
from safe_exam.capture.config import CaptureConfig
from safe_exam.detectors.object_detector import ObjectDetector
from safe_exam.processor.frame_processor import process_frame
from safe_exam.utils.logging_utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@dataclass
class ProcessorRunStats:
    frame_count: int = 0
    total_inference_time_ms: float = 0.0
    phone_detected_frames: int = 0
    extra_person_frames: int = 0


def build_summary(
    stats: ProcessorRunStats, session_start: float
) -> dict[str, float | int]:
    elapsed = time.perf_counter() - session_start
    avg_inference_ms = (
        stats.total_inference_time_ms / stats.frame_count
        if stats.frame_count > 0
        else 0.0
    )
    avg_fps = stats.frame_count / elapsed if elapsed > 0 else 0.0

    return {
        "frame_count": stats.frame_count,
        "avg_inference_ms": avg_inference_ms,
        "avg_fps": avg_fps,
        "phone_detected_frames": stats.phone_detected_frames,
        "extra_person_frames": stats.extra_person_frames,
    }


def main():
    """
    Main function to run the frame processor.
    """
    detector = ObjectDetector()
    config = CaptureConfig(camera_index=0, target_fps=12)

    stats = ProcessorRunStats()
    summary_interval = 60
    session_start = time.perf_counter()

    logger.info(
        "Process started | camera_index=%s target_fps=%s debug=%s",
        config.camera_index,
        config.target_fps,
        True,
    )

    for frame in capture_frames(config):
        output, debug_frame, inference_time_ms = process_frame(
            frame, detector, debug=True
        )

        stats.frame_count += 1
        stats.total_inference_time_ms += inference_time_ms

        if output["phone_detected"]:
            stats.phone_detected_frames += 1

        if output["extra_person_detected"]:
            stats.extra_person_frames += 1

        if stats.frame_count % summary_interval == 0:
            summary = build_summary(stats, session_start)

            logger.info(
                "frames=%s avg_inference_ms=%.1f avg_fps=%.2f "
                "phone_frames=%s extra_person_frames=%s",
                summary["frame_count"],
                summary["avg_inference_ms"],
                summary["avg_fps"],
                summary["phone_detected_frames"],
                summary["extra_person_frames"],
            )

        if debug_frame is not None:
            cv2.imshow("frame processor", debug_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

    summary = build_summary(stats, session_start)

    logger.info(
        "processor stopped | total_frames=%s avg_inference_ms=%.1f "
        "avg_fps=%.2f phone_frames=%s extra_person_frames=%s",
        summary["frame_count"],
        summary["avg_inference_ms"],
        summary["avg_fps"],
        summary["phone_detected_frames"],
        summary["extra_person_frames"],
    )


if __name__ == "__main__":
    main()
