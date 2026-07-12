"""Iris-based eye gaze estimation and primary face selection."""

from __future__ import annotations

from safe_exam.detectors.face_gaze.config import FaceGazeConfig


def landmark_to_xy(landmark, img_w: int, img_h: int) -> tuple[float, float]:
    """Convert a normalized MediaPipe landmark to pixel coordinates."""
    return landmark.x * img_w, landmark.y * img_h


def compute_face_bbox_area(face_landmarks, img_w: int, img_h: int) -> float:
    """Compute bounding box area for a face from all landmarks."""
    xs = [lm.x * img_w for lm in face_landmarks.landmark]
    ys = [lm.y * img_h for lm in face_landmarks.landmark]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return width * height


def select_primary_face(landmarks_list, img_w: int, img_h: int):
    """Select the face with the largest bounding box."""
    return max(
        landmarks_list,
        key=lambda face: compute_face_bbox_area(face, img_w, img_h),
    )


def _eye_normalized_offset(
    face_landmarks,
    iris_id: int,
    corner_ids: tuple[int, ...],
    img_w: int,
    img_h: int,
) -> tuple[float, float] | None:
    """Compute normalized iris offset within one eye region (-0.5 to 0.5)."""
    landmarks = face_landmarks.landmark
    if iris_id >= len(landmarks):
        return None

    iris_x, iris_y = landmark_to_xy(landmarks[iris_id], img_w, img_h)

    xs = [landmarks[idx].x * img_w for idx in corner_ids if idx < len(landmarks)]
    ys = [landmarks[idx].y * img_h for idx in corner_ids if idx < len(landmarks)]
    if not xs or not ys:
        return None

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return None

    norm_x = ((iris_x - min_x) / width) - 0.5
    norm_y = ((iris_y - min_y) / height) - 0.5
    return norm_x, norm_y


def estimate_iris_offset(
    face_landmarks,
    img_w: int,
    img_h: int,
    config: FaceGazeConfig,
) -> tuple[float, float, float, float]:
    """
    Estimate eye gaze offset from iris position within the eye sockets.

    Returns (eye_pitch, eye_yaw, iris_offset_x, iris_offset_y).
    """
    if not config.refine_landmarks:
        return 0.0, 0.0, 0.0, 0.0

    left = _eye_normalized_offset(
        face_landmarks,
        config.left_iris_landmark_id,
        config.left_eye_corner_ids,
        img_w,
        img_h,
    )
    right = _eye_normalized_offset(
        face_landmarks,
        config.right_iris_landmark_id,
        config.right_eye_corner_ids,
        img_w,
        img_h,
    )

    offsets = [o for o in (left, right) if o is not None]
    if not offsets:
        return 0.0, 0.0, 0.0, 0.0

    iris_offset_x = sum(o[0] for o in offsets) / len(offsets)
    iris_offset_y = sum(o[1] for o in offsets) / len(offsets)

    scale = config.eye_offset_scale
    eye_yaw = iris_offset_x * scale * 2
    eye_pitch = -iris_offset_y * scale * 2

    return eye_pitch, eye_yaw, iris_offset_x, iris_offset_y
