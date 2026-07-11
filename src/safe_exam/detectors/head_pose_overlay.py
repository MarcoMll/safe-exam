"""Drawing helpers for head pose / gaze detection results."""

import cv2
import mediapipe as mp
import numpy as np

from safe_exam.detectors.head_pose_config import HeadPoseConfig
from safe_exam.detectors.head_pose_detector import HeadPoseResult


def draw_nose_overlay(
    frame: np.ndarray,
    angles: tuple[float, float, float],
    nose_2d: tuple[float, float],
    config: HeadPoseConfig,
) -> np.ndarray:
    """Draw the nose direction line on a frame."""
    if not config.draw_landmarks:
        return frame

    pitch, yaw, _ = angles
    nose_start = (int(nose_2d[0]), int(nose_2d[1]))
    nose_end = (
        int(nose_2d[0] + yaw * config.line_scale),
        int(nose_2d[1] - pitch * config.line_scale),
    )
    cv2.line(frame, nose_start, nose_end, (0, 255, 0), int(config.line_length))
    return frame


def draw_face_mesh(
    frame: np.ndarray, face_landmarks, config: HeadPoseConfig
) -> np.ndarray:
    """Draw face mesh landmarks on a frame."""
    if not config.draw_landmarks:
        return frame

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=1, circle_radius=1
        ),
    )
    return frame


def draw_direction_text(frame: np.ndarray, direction_text: str) -> np.ndarray:
    """Draw direction label text on a frame."""
    cv2.putText(
        frame,
        direction_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    return frame


def draw_head_pose_on_frame(
    frame: np.ndarray,
    result: HeadPoseResult,
    config: HeadPoseConfig,
) -> np.ndarray:
    """Draw head pose visuals onto an existing frame (no copy/mirror)."""
    if not result.face_detected or result.face_landmarks is None:
        return frame

    frame = draw_face_mesh(frame, result.face_landmarks, config)

    if result.raw_angles and result.nose_2d:
        frame = draw_nose_overlay(frame, result.raw_angles, result.nose_2d, config)

    return frame


def draw_head_pose_overlay(
    frame: np.ndarray, result: HeadPoseResult, config: HeadPoseConfig
) -> np.ndarray:
    """Build a standalone debug frame for head-pose-only demos."""
    display_frame = frame.copy()

    if config.mirror_preview:
        display_frame = cv2.flip(display_frame, 1)

    if not result.face_detected or result.face_landmarks is None:
        cv2.putText(
            display_frame,
            result.direction,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        return display_frame

    display_frame = draw_head_pose_on_frame(display_frame, result, config)
    return draw_direction_text(display_frame, result.direction)
