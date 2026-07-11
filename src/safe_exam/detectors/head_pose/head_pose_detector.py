import mediapipe as mp
import cv2
import numpy as np
from safe_exam.detectors.head_pose.head_pose_config import HeadPoseConfig

def create_face_mesh(config : HeadPoseConfig) -> mp.solutions.face_mesh.FaceMesh:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence = config.min_detection_confidence, min_tracking_confidence = config.min_tracking_confidence)
    return face_mesh

def preprocess_frame(frame: np.ndarray, config: HeadPoseConfig) -> np.ndarray:
    """Preprocess the frame for head pose detection."""
    # if config.mirror_preview:
    #     frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

def run_face_mesh(frame_rgb: np.ndarray, face_mesh: mp.solutions.face_mesh.FaceMesh):
    """Run the face mesh model on the frame."""
    frame_rgb.flags.writeable = False
    results = face_mesh.process(frame_rgb)
    frame_rgb.flags.writeable = True
    return results

def extract_pose_points(face_landmarks, img_w: int, img_h: int, config: HeadPoseConfig) -> tuple[list[tuple[int, int]], list[tuple[int, int, float]], tuple[float, float] | None, tuple[float, float, float] | None]:
    """Extract the 2D and 3D points for head pose estimation."""
    face_2d = []
    face_3d = []
    nose_2d = None
    nose_3d = None

    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in config.landmark_ids:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append((x, y))
            face_3d.append((x, y, lm.z))  # Scale z to make it more comparable to x and y
            if idx == config.nose_landmark_id:
                nose_2d = (lm.x * img_w, lm.y * img_h)
                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

    return face_2d, face_3d, nose_2d, nose_3d

def build_camera_matrix(img_w: int, img_h: int) -> np.ndarray:
    """Build the camera matrix for head pose estimation."""
    focal_length = img_w # this is a rought estimate, it would be better to get the accual camera's focal length
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                               [0, focal_length, center[1]],
                               [0, 0, 1]], dtype="double")
    return camera_matrix

def estimate_head_pose(face_2d: list[tuple[int, int]], face_3d: list[tuple[int, int, float]], camera_matrix: np.ndarray) -> tuple[float, float, float]:
    """Estimate the head pose using solvePnP.
    
    """
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion (not realistic but fine for prototype)
    ok, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, camera_matrix, dist_coeffs)
    if ok:
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Get Euler angles from rotation matrix
        angles, __, __, __, __, __ = cv2.RQDecomp3x3(rotation_matrix)

        return angles
    else:
        return (None, None, None)  # Return None if pose estimation fails

def classify_direction(angles: tuple[float, float, float], config: HeadPoseConfig) -> str:
    """Classify the head pose based on the angles and thresholds."""
    #pitch, yaw, __ = angles
    text = ""
    pitch = angles[0] * 360
    yaw = angles[1] * 360
    # Classify based on thresholds
    if yaw > config.yaw_threshold_deg:
        text = "Looking Right"
        return text
    elif yaw < -config.yaw_threshold_deg:
        text = "Looking Left"
        return text
    elif pitch > config.pitch_threshold_deg:
        text = "Looking Up"
        return text
    elif pitch < -config.pitch_threshold_deg:
        text = "Looking Down"
        return text
    else:
        text = "Forward"
        return text
    
def draw_nose_overlay(frame: np.ndarray, angles: tuple[float, float, float], nose_2d: tuple[int, int], config: HeadPoseConfig) -> np.ndarray:
    """Draw the head pose overlay on the frame."""
    pitch, yaw, __ = angles

    if config.draw_landmarks:
        nose_start = (int(nose_2d[0]), int(nose_2d[1]))
        nose_end = (int(nose_2d[0] + yaw * config.line_scale), int(nose_2d[1] - pitch * config.line_scale))
        cv2.line(frame, nose_start, nose_end, (0, 255, 0), int(config.line_length))
    return frame

def draw_face_mesh(frame: np.ndarray, face_landmarks, config: HeadPoseConfig) -> np.ndarray:
    """Draw the face mesh landmarks on the frame."""
    if config.draw_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        )
    return frame

def draw_direction_text(frame: np.ndarray, direction_text: str) -> np.ndarray:
    """Draw the head pose direction text on the frame."""
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

def process_head_pose_frame(frame: np.ndarray, face_mesh: mp.solutions.face_mesh.FaceMesh, config: HeadPoseConfig):
    '''Runs the whole head-pose dectection pipeline'''
    display_frame = frame.copy()

    if config.mirror_preview:
        display_frame = cv2.flip(display_frame, 1)

    frame_rgb = preprocess_frame(display_frame, config)
    results = run_face_mesh(frame_rgb, face_mesh)

    if not results.multi_face_landmarks:
        cv2.putText(display_frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,)
        return display_frame, "No face detected", (None, None, None)

    face_landmarks = results.multi_face_landmarks[0]
    img_h, img_w = display_frame.shape[:2]
    face_2d, face_3d, nose_2d, nose_3d = extract_pose_points(face_landmarks, img_w, img_h, config)

    display_frame = draw_face_mesh(display_frame, face_landmarks, config)

    if len(face_2d) != len(config.landmark_ids) or nose_2d is None:
        cv2.putText(display_frame, "Insufficient landmarks", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,)
        return display_frame, "Insufficient landmarks", (None, None, None)
    
    camera_matrix = build_camera_matrix(img_w, img_h)
    angles = estimate_head_pose(face_2d, face_3d, camera_matrix)

    if angles[0] is None:
        cv2.putText(display_frame, "Pose estimation failed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,)
        return display_frame, "Pose estimation failed", angles
    
    direction = classify_direction(angles, config)
    display_frame = draw_nose_overlay(display_frame, angles, nose_2d, config)
    display_frame = draw_direction_text(display_frame, direction)

    return display_frame, direction, angles