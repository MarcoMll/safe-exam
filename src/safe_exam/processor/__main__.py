import argparse
import logging
import time

import cv2  # pylint: disable=no-member

from safe_exam.capture.capture import capture_frames
from safe_exam.capture.capture_config import CaptureConfig
from safe_exam.detectors.head_pose_config import HeadPoseConfig
from safe_exam.detectors.head_pose_detector import HeadPoseDetector
from safe_exam.detectors.object_detector import ObjectDetector
from safe_exam.processor.debug_overlay import draw_composite_overlay
from safe_exam.processor.frame_processor import process_frame
from safe_exam.processor.session_stats import ProcessorRunStats, build_summary
from safe_exam.utils.logging_utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def main(
    debug: bool = True,
    camera_index: int = 0,
    target_fps: float = 12.0,
) -> None:
    """Run the integrated capture + detection pipeline."""
    object_detector = ObjectDetector()
    head_pose_detector = HeadPoseDetector(
        HeadPoseConfig(draw_landmarks=debug, mirror_preview=False)
    )
    capture_config = CaptureConfig(
        camera_index=camera_index,
        target_fps=target_fps,
    )

    stats = ProcessorRunStats()
    summary_interval = 60
    session_start = time.perf_counter()

    logger.info(
        "Process started | camera_index=%s target_fps=%s debug=%s",
        capture_config.camera_index,
        capture_config.target_fps,
        debug,
    )

    try:
        for frame in capture_frames(capture_config):
            output = process_frame(frame, object_detector, head_pose_detector)

            stats.frame_count += 1
            stats.total_inference_time_ms += output.inference_time_ms

            result_dict = output.as_dict()
            if result_dict["phone_detected"]:
                stats.phone_detected_frames += 1
            if result_dict["extra_person_detected"]:
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

            if debug:
                debug_frame = draw_composite_overlay(
                    yolo_results=output.yolo_results,
                    result=output.result,
                    inference_time_ms=output.inference_time_ms,
                    head_pose_result=output.head_pose_result,
                    head_pose_config=head_pose_detector.config,
                )
                cv2.imshow("frame processor", debug_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        head_pose_detector.close()
        if debug:
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
    parser = argparse.ArgumentParser(description="Run the safe-exam frame processor")
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable the composite debug overlay window",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=12.0,
        help="Target capture frame rate (default: 12)",
    )
    args = parser.parse_args()
    main(
        debug=not args.no_debug,
        camera_index=args.camera_index,
        target_fps=args.target_fps,
    )
