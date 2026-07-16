from dataclasses import dataclass


@dataclass
class FaceGazeConfig:
    """Configuration for MediaPipe face gaze (head pose + iris)."""

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_num_faces: int = 1
    refine_landmarks: bool = True
    select_largest_face: bool = True

    landmark_ids: tuple[int, ...] = (1, 33, 61, 199, 263, 291)
    nose_landmark_id: int = 1

    left_iris_landmark_id: int = 468
    right_iris_landmark_id: int = 473
    left_eye_corner_ids: tuple[int, ...] = (33, 133, 159, 145)
    right_eye_corner_ids: tuple[int, ...] = (263, 362, 386, 374)

    yaw_threshold_deg: float = 10.0
    pitch_threshold_deg: float = 10.0
    eye_yaw_threshold_deg: float = 3.0
    eye_pitch_threshold_deg: float = 5.0
    eye_offset_scale: float = 30.0

    mirror_preview: bool = True
    draw_landmarks: bool = True
    line_scale: float = 10.0
    line_length: int = 3
