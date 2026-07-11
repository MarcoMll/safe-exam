"""Composite debug overlay for the unified processor pipeline."""

import cv2

from safe_exam.detectors.head_pose_config import HeadPoseConfig
from safe_exam.detectors.head_pose_detector import HeadPoseResult
from safe_exam.detectors.head_pose_overlay import draw_head_pose_on_frame
from safe_exam.detectors.object_detector_overlay import draw_object_overlay
from safe_exam.processor.frame_result import FrameResult


def draw_summary_text(
    frame,
    result: FrameResult,
    inference_time_ms: float,
    head_pose_result: HeadPoseResult | None = None,
):
    """Draw unified processor summary text on a frame."""
    cv2.putText(
        frame,
        f"Phone: {result.phone_detected}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Phone conf: {result.phone_confidence:.2f}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Persons: {result.person_count}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Extra person: {result.extra_person_detected}",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Gaze pitch: {result.gaze_pitch:.2f}",
        (10, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Gaze yaw: {result.gaze_yaw:.2f}",
        (10, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    if head_pose_result:
        cv2.putText(
            frame,
            f"Direction: {head_pose_result.direction}",
            (10, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    cv2.putText(
        frame,
        f"Inference: {inference_time_ms:.1f} ms",
        (10, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    return frame


def draw_composite_overlay(
    yolo_results,
    result: FrameResult,
    inference_time_ms: float,
    head_pose_result: HeadPoseResult | None = None,
    head_pose_config: HeadPoseConfig | None = None,
):
    """Combine object detection and head pose overlays with summary text."""
    debug_frame = draw_object_overlay(yolo_results)

    if head_pose_result and head_pose_config:
        debug_frame = draw_head_pose_on_frame(
            debug_frame, head_pose_result, head_pose_config
        )

    return draw_summary_text(debug_frame, result, inference_time_ms, head_pose_result)
