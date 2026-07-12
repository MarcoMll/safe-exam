"""Drawing helpers for face gaze detection (head pose + iris)."""

import cv2
import mediapipe as mp
import numpy as np

from safe_exam.detectors.face_gaze.config import FaceGazeConfig
from safe_exam.detectors.face_gaze.detector import FaceGazeResult
from safe_exam.detectors.face_gaze.iris_estimation import landmark_to_xy


def draw_nose_overlay(
    frame: np.ndarray,
    angles: tuple[float, float, float],
    nose_2d: tuple[float, float],
    config: FaceGazeConfig,
) -> np.ndarray:
    """Draw the head direction line from the nose (head pose, not eye gaze)."""
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


def draw_iris_markers(
    frame: np.ndarray,
    result: FaceGazeResult,
    config: FaceGazeConfig,
) -> np.ndarray:
    """Draw iris center markers and offset line when refined landmarks exist."""
    if (
        not config.draw_landmarks
        or not result.face_detected
        or result.face_landmarks is None
    ):
        return frame

    img_h, img_w = frame.shape[:2]
    landmarks = result.face_landmarks.landmark

    for iris_id in (config.left_iris_landmark_id, config.right_iris_landmark_id):
        if iris_id >= len(landmarks):
            continue
        x, y = landmark_to_xy(landmarks[iris_id], img_w, img_h)
        cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 0), -1)

    if result.nose_2d and (result.iris_offset_x or result.iris_offset_y):
        cx, cy = int(result.nose_2d[0]), int(result.nose_2d[1])
        offset_len = 25
        end_x = int(cx + result.iris_offset_x * offset_len * 2)
        end_y = int(cy + result.iris_offset_y * offset_len * 2)
        cv2.line(frame, (cx, cy), (end_x, end_y), (255, 255, 0), 2)

    return frame


def draw_face_mesh(
    frame: np.ndarray, face_landmarks, config: FaceGazeConfig
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
    """Draw head direction label text on a frame."""
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


def draw_face_gaze_on_frame(
    frame: np.ndarray,
    result: FaceGazeResult,
    config: FaceGazeConfig,
) -> np.ndarray:
    """Draw face gaze visuals onto an existing frame (no copy/mirror)."""
    if not result.face_detected or result.face_landmarks is None:
        return frame

    frame = draw_face_mesh(frame, result.face_landmarks, config)

    if result.raw_angles and result.nose_2d:
        frame = draw_nose_overlay(frame, result.raw_angles, result.nose_2d, config)

    frame = draw_iris_markers(frame, result, config)
    return frame


def draw_face_gaze_overlay(
    frame: np.ndarray, result: FaceGazeResult, config: FaceGazeConfig
) -> np.ndarray:
    """Build a standalone debug frame for face-gaze-only demos."""
    display_frame = frame.copy()

    if config.mirror_preview:
        display_frame = cv2.flip(display_frame, 1)

    if not result.face_detected or result.face_landmarks is None:
        cv2.putText(
            display_frame,
            result.head_direction,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        return display_frame

    display_frame = draw_face_gaze_on_frame(display_frame, result, config)
    return draw_direction_text(display_frame, result.head_direction)
