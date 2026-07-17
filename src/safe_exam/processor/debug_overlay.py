"""Composite debug overlay for the unified processor pipeline."""

import cv2

from safe_exam.detectors.face_gaze import (
    FaceGazeConfig,
    FaceGazeResult,
    draw_face_gaze_on_frame,
)
from safe_exam.detectors.object import draw_object_overlay
from safe_exam.processor.frame_result import FrameResult
from safe_exam.processor.intrusion_policy import (
    IntrusionPolicyConfig,
    is_intrusion_suspected_for_frame,
)


def _put_text(frame, text, x, y, color=(0, 255, 0)):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def draw_summary_text(
    frame,
    result: FrameResult,
    inference_time_ms: float,
    intrusion_policy: IntrusionPolicyConfig,
    face_gaze_result: FaceGazeResult | None = None,
):
    """Draw unified processor summary text on a frame."""
    left_x = 10
    right_x = 320
    y = 25
    line_h = 22

    _put_text(frame, f"Phone: {result.phone_detected}", left_x, y)
    y += line_h
    _put_text(frame, f"Phone conf: {result.phone_confidence:.2f}", left_x, y)
    y += line_h
    _put_text(frame, f"Persons: {result.person_count}", left_x, y)
    y += line_h
    intrusion_suspected = is_intrusion_suspected_for_frame(result, intrusion_policy)
    _put_text(frame, f"Intrusion: {intrusion_suspected}", left_x, y)

    y = 25
    _put_text(
        frame,
        f"Head: p={result.head_pitch:.1f} y={result.head_yaw:.1f}",
        right_x,
        y,
        (0, 255, 255),
    )
    y += line_h
    if face_gaze_result:
        _put_text(
            frame,
            f"  dir: {face_gaze_result.head_direction}",
            right_x,
            y,
            (0, 255, 255),
        )
        y += line_h
    _put_text(
        frame,
        f"Eye: p={result.eye_pitch:.1f} y={result.eye_yaw:.1f}",
        right_x,
        y,
        (255, 255, 0),
    )
    y += line_h
    _put_text(
        frame,
        f"  iris: x={result.iris_offset_x:.2f} y={result.iris_offset_y:.2f}",
        right_x,
        y,
        (255, 255, 0),
    )
    y += line_h
    _put_text(
        frame,
        f"Gaze: p={result.gaze_pitch:.1f} y={result.gaze_yaw:.1f}",
        right_x,
        y,
        (255, 200, 100),
    )
    y += line_h
    _put_text(
        frame,
        f"Inference: {inference_time_ms:.1f} ms",
        right_x,
        y,
        (255, 255, 0),
    )
    return frame


def draw_composite_overlay(
    yolo_results,
    result: FrameResult,
    inference_time_ms: float,
    intrusion_policy: IntrusionPolicyConfig,
    face_gaze_result: FaceGazeResult | None = None,
    face_gaze_config: FaceGazeConfig | None = None,
):
    """Combine object detection and face gaze overlays with summary text."""
    debug_frame = draw_object_overlay(yolo_results)

    if face_gaze_result and face_gaze_config:
        debug_frame = draw_face_gaze_on_frame(
            debug_frame, face_gaze_result, face_gaze_config
        )

    return draw_summary_text(
        debug_frame,
        result,
        inference_time_ms,
        intrusion_policy,
        face_gaze_result,
    )
