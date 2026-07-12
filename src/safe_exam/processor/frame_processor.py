import time

from safe_exam.detectors.face_gaze import FaceGazeDetector
from safe_exam.detectors.object import ObjectDetector
from safe_exam.processor.frame_result import FrameResult, ProcessFrameOutput


def process_frame(
    frame,
    object_detector: ObjectDetector,
    face_gaze_detector: FaceGazeDetector,
) -> ProcessFrameOutput:
    """Run all detectors on one frame and return merged results."""
    result = FrameResult(timestamp=time.time())
    config = object_detector.config

    start = time.perf_counter()
    yolo_results = object_detector.detect(
        frame=frame,
        classes=[config.person_class_id, config.phone_class_id],
    )
    yolo_inference_time_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    face_gaze_result = face_gaze_detector.detect(frame=frame)
    face_gaze_inference_time_ms = (time.perf_counter() - start) * 1000

    inference_time_ms = yolo_inference_time_ms + face_gaze_inference_time_ms

    phone_detected, phone_confidence = object_detector.look_for_class(
        results=yolo_results,
        target_class_index=config.phone_class_id,
        threshold=config.confidence_threshold,
    )

    person_count = object_detector.count_class(
        results=yolo_results,
        target_class_index=config.person_class_id,
        threshold=config.confidence_threshold,
    )

    result.phone_detected = phone_detected
    result.phone_confidence = phone_confidence
    result.person_count = person_count
    result.extra_person_detected = person_count > 1

    result.face_detected = face_gaze_result.face_detected
    result.head_pitch = face_gaze_result.head_pitch
    result.head_yaw = face_gaze_result.head_yaw
    result.eye_pitch = face_gaze_result.eye_pitch
    result.eye_yaw = face_gaze_result.eye_yaw
    result.gaze_pitch = face_gaze_result.gaze_pitch
    result.gaze_yaw = face_gaze_result.gaze_yaw
    result.iris_offset_x = face_gaze_result.iris_offset_x
    result.iris_offset_y = face_gaze_result.iris_offset_y
    result.head_direction = face_gaze_result.head_direction

    return ProcessFrameOutput(
        result=result,
        yolo_results=yolo_results,
        face_gaze_result=face_gaze_result,
        inference_time_ms=inference_time_ms,
    )
