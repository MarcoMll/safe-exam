import argparse
import logging
import time

import cv2  # pylint: disable=no-member

from safe_exam.capture.capture import capture_frames
from safe_exam.capture.capture_config import CaptureConfig
from safe_exam.detectors.face_gaze import FaceGazeConfig, FaceGazeDetector
from safe_exam.detectors.object import ObjectDetector
from safe_exam.processor.debug_overlay import draw_composite_overlay
from safe_exam.processor.frame_processor import process_frame
from safe_exam.processor.session_stats import (
    ProcessorRunStats,
    build_summary,
    is_eye_off_center,
    is_head_off_center,
)
from safe_exam.utils.logging_utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def main(
    debug: bool = True,
    camera_index: int = 0,
    target_fps: float = 12.0,
) -> None:
    """Run the integrated capture + detection pipeline."""
    face_gaze_config = FaceGazeConfig(
        draw_landmarks=debug,
        mirror_preview=False,
        refine_landmarks=True,
    )
    object_detector = ObjectDetector()
    face_gaze_detector = FaceGazeDetector(config=face_gaze_config)
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
            output = process_frame(frame, object_detector, face_gaze_detector)

            stats.frame_count += 1
            stats.total_inference_time_ms += output.inference_time_ms

            result = output.result
            if result.phone_detected:
                stats.phone_detected_frames += 1
            if result.extra_person_detected:
                stats.extra_person_frames += 1
            if result.face_detected:
                stats.face_detected_frames += 1
                if is_head_off_center(result, face_gaze_config):
                    stats.head_off_center_frames += 1
                if is_eye_off_center(result, face_gaze_config):
                    stats.eye_off_center_frames += 1

            if stats.frame_count % summary_interval == 0:
                summary = build_summary(stats, session_start)
                logger.info(
                    "frames=%s avg_inference_ms=%.1f avg_fps=%.2f "
                    "phone_frames=%s extra_person_frames=%s "
                    "face_frames=%s head_off_center=%s eye_off_center=%s",
                    summary["frame_count"],
                    summary["avg_inference_ms"],
                    summary["avg_fps"],
                    summary["phone_detected_frames"],
                    summary["extra_person_frames"],
                    summary["face_detected_frames"],
                    summary["head_off_center_frames"],
                    summary["eye_off_center_frames"],
                )

            if debug:
                debug_frame = draw_composite_overlay(
                    yolo_results=output.yolo_results,
                    result=output.result,
                    inference_time_ms=output.inference_time_ms,
                    face_gaze_result=output.face_gaze_result,
                    face_gaze_config=face_gaze_detector.config,
                )
                cv2.imshow("frame processor", debug_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        face_gaze_detector.close()
        if debug:
            cv2.destroyAllWindows()

    summary = build_summary(stats, session_start)
    logger.info(
        "processor stopped | total_frames=%s avg_inference_ms=%.1f "
        "avg_fps=%.2f phone_frames=%s extra_person_frames=%s "
        "face_frames=%s head_off_center=%s eye_off_center=%s",
        summary["frame_count"],
        summary["avg_inference_ms"],
        summary["avg_fps"],
        summary["phone_detected_frames"],
        summary["extra_person_frames"],
        summary["face_detected_frames"],
        summary["head_off_center_frames"],
        summary["eye_off_center_frames"],
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
