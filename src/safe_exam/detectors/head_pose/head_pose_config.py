from dataclasses import dataclass

@dataclass
class HeadPoseConfig:
    '''
    Configuration for the head pose detector.
    var min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for the detection to be considered successful.
    var min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the tracking to be considered successful.
    var landmark_ids: The landmark ids to be used for head pose estimation.
    var mirror_preview: Whether to mirror the preview image. (mainly for debugging)
    var draw_landmarks: Whether to draw the landmarks on the image.
    var line_scale: The scale of the lines drawn on the image.
    '''
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    landmark_ids: tuple[int, ...] = (1, 33, 61, 199, 263, 291) # these are the landmark ids
    nose_landmark_id: int = 1 # this is the landmark id for the nose tip
    yaw_threshold_deg: float = 10.0
    pitch_threshold_deg: float = 10.0
    mirror_preview: bool = True
    draw_landmarks: bool = True
    line_scale: float = 10.0
    line_length: int = 3
