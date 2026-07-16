"""Live capture loop for the unified processor pipeline."""

import logging
import time

import cv2  # pylint: disable=no-member

from safe_exam.capture.capture import capture_frames
from safe_exam.capture.capture_config import CaptureConfig
from safe_exam.detectors.face_gaze import FaceGazeConfig, FaceGazeDetector
from safe_exam.detectors.object import ObjectDetector
from safe_exam.processor.attention_policy import (
    DEBUG_SIGNAL_POLICIES,
    DEFAULT_ATTENTION_POLICY,
    AttentionPolicyConfig,
)
from safe_exam.processor.debug_overlay import draw_composite_overlay
from safe_exam.processor.frame_processor import process_frame
from safe_exam.processor.session_stats import (
    ProcessorRunStats,
    build_summary,
    update_from_frame,
)

logger = logging.getLogger(__name__)

SUMMARY_INTERVAL = 60


def _log_summary(summary: dict[str, float | int], *, stopped: bool = False) -> None:
    if stopped:
        logger.info(
            "processor stopped | total_frames=%s avg_inference_ms=%.1f "
            "avg_fps=%.2f phone_frames=%s extra_person_frames=%s "
            "face_frames=%s head_off_center=%s eye_off_center=%s "
            "gaze_off_center=%s attention_off_center=%s",
            summary["frame_count"],
            summary["avg_inference_ms"],
            summary["avg_fps"],
            summary["phone_detected_frames"],
            summary["extra_person_frames"],
            summary["face_detected_frames"],
            summary["head_off_center_frames"],
            summary["eye_off_center_frames"],
            summary["gaze_off_center_frames"],
            summary["attention_off_center_frames"],
        )
        return

    logger.info(
        "frames=%s avg_inference_ms=%.1f avg_fps=%.2f "
        "phone_frames=%s extra_person_frames=%s face_frames=%s "
        "head_off_center=%s eye_off_center=%s gaze_off_center=%s "
        "attention_off_center=%s",
        summary["frame_count"],
        summary["avg_inference_ms"],
        summary["avg_fps"],
        summary["phone_detected_frames"],
        summary["extra_person_frames"],
        summary["face_detected_frames"],
        summary["head_off_center_frames"],
        summary["eye_off_center_frames"],
        summary["gaze_off_center_frames"],
        summary["attention_off_center_frames"],
    )


def run_processor(
    *,
    debug: bool = True,
    camera_index: int = 0,
    target_fps: float = 12.0,
    attention_policy: AttentionPolicyConfig | None = None,
) -> None:
    """Run the integrated capture + detection pipeline."""
    active_policy = attention_policy or DEFAULT_ATTENTION_POLICY
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
    session_start = time.perf_counter()

    logger.info(
        "Process started | camera_index=%s target_fps=%s debug=%s policy=%s",
        capture_config.camera_index,
        capture_config.target_fps,
        debug,
        active_policy.label(),
    )

    try:
        for frame in capture_frames(capture_config):
            output = process_frame(frame, object_detector, face_gaze_detector)
            update_from_frame(
                stats,
                output.result,
                active_policy,
                inference_time_ms=output.inference_time_ms,
                debug_policies=DEBUG_SIGNAL_POLICIES,
            )

            if stats.frame_count % SUMMARY_INTERVAL == 0:
                _log_summary(build_summary(stats, session_start))

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

    _log_summary(build_summary(stats, session_start), stopped=True)
