"""MediaPipe head pose detector (inference only).

For drawing, use ``safe_exam.detectors.head_pose_overlay``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from safe_exam.detectors.head_pose_config import HeadPoseConfig


@dataclass
class HeadPoseResult:
    """Detection output for one frame (no drawing attached)."""

    face_detected: bool = False
    gaze_pitch: float = 0.0
    gaze_yaw: float = 0.0
    direction: str = "No face detected"
    raw_angles: tuple[float | None, float | None, float | None] | None = None
    nose_2d: tuple[float, float] | None = None
    face_landmarks: object | None = None


def create_face_mesh(config: HeadPoseConfig) -> mp.solutions.face_mesh.FaceMesh:
    """Create the MediaPipe face mesh model."""
    return mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to RGB for MediaPipe."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def run_face_mesh(frame_rgb: np.ndarray, face_mesh: mp.solutions.face_mesh.FaceMesh):
    """Run face mesh on a preprocessed RGB frame."""
    frame_rgb.flags.writeable = False
    results = face_mesh.process(frame_rgb)
    frame_rgb.flags.writeable = True
    return results


def extract_pose_points(
    face_landmarks,
    img_w: int,
    img_h: int,
    config: HeadPoseConfig,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int, float]],
    tuple[float, float] | None,
]:
    """Extract 2D/3D landmark points used for head pose estimation."""
    face_2d = []
    face_3d = []
    nose_2d = None

    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in config.landmark_ids:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append((x, y))
            face_3d.append((x, y, lm.z))
            if idx == config.nose_landmark_id:
                nose_2d = (lm.x * img_w, lm.y * img_h)

    return face_2d, face_3d, nose_2d


def build_camera_matrix(img_w: int, img_h: int) -> np.ndarray:
    """Build a simple camera matrix for solvePnP."""
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    return np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype="double",
    )


def estimate_head_pose(
    face_2d: list[tuple[int, int]],
    face_3d: list[tuple[int, int, float]],
    camera_matrix: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    """Estimate head pose angles using solvePnP."""
    face_2d_array = np.array(face_2d, dtype=np.float64)
    face_3d_array = np.array(face_3d, dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rotation_vector, _ = cv2.solvePnP(
        face_3d_array, face_2d_array, camera_matrix, dist_coeffs
    )
    if not ok:
        return (None, None, None)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    return angles


def angles_to_degrees(
    angles: tuple[float | None, float | None, float | None],
) -> tuple[float, float]:
    """Convert raw solvePnP angles to pitch/yaw degrees used by the detector."""
    pitch = (angles[0] or 0.0) * 360
    yaw = (angles[1] or 0.0) * 360
    return pitch, yaw


def classify_direction(
    angles: tuple[float, float, float], config: HeadPoseConfig
) -> str:
    """Classify head direction from pitch/yaw angles."""
    pitch = angles[0] * 360
    yaw = angles[1] * 360

    if yaw > config.yaw_threshold_deg:
        return "Looking Right"
    if yaw < -config.yaw_threshold_deg:
        return "Looking Left"
    if pitch > config.pitch_threshold_deg:
        return "Looking Up"
    if pitch < -config.pitch_threshold_deg:
        return "Looking Down"
    return "Forward"


class HeadPoseDetector:
    """MediaPipe-based head pose detector."""

    def __init__(self, config: HeadPoseConfig | None = None):
        self.config = config or HeadPoseConfig()
        self.face_mesh = create_face_mesh(self.config)

    def detect(self, frame: np.ndarray) -> HeadPoseResult:
        """Run head pose detection on one frame without drawing."""
        frame_rgb = preprocess_frame(frame)
        results = run_face_mesh(frame_rgb, self.face_mesh)

        if not results.multi_face_landmarks:
            return HeadPoseResult()

        face_landmarks = results.multi_face_landmarks[0]
        img_h, img_w = frame.shape[:2]
        face_2d, face_3d, nose_2d = extract_pose_points(
            face_landmarks, img_w, img_h, self.config
        )

        if len(face_2d) != len(self.config.landmark_ids) or nose_2d is None:
            return HeadPoseResult(
                direction="Insufficient landmarks",
                face_landmarks=face_landmarks,
            )

        camera_matrix = build_camera_matrix(img_w, img_h)
        angles = estimate_head_pose(face_2d, face_3d, camera_matrix)

        if angles[0] is None:
            return HeadPoseResult(
                direction="Pose estimation failed",
                face_landmarks=face_landmarks,
            )

        gaze_pitch, gaze_yaw = angles_to_degrees(angles)
        direction = classify_direction(angles, self.config)

        return HeadPoseResult(
            face_detected=True,
            gaze_pitch=gaze_pitch,
            gaze_yaw=gaze_yaw,
            direction=direction,
            raw_angles=angles,
            nose_2d=nose_2d,
            face_landmarks=face_landmarks,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.face_mesh.close()
