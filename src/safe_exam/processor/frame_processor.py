import time

from safe_exam.detectors.head_pose_detector import HeadPoseDetector
from safe_exam.detectors.object_detector import ObjectDetector
from safe_exam.processor.frame_result import FrameResult, ProcessFrameOutput


def process_frame(
    frame,
    object_detector: ObjectDetector,
    head_pose_detector: HeadPoseDetector,
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
    head_pose_result = head_pose_detector.detect(frame=frame)
    head_pose_inference_time_ms = (time.perf_counter() - start) * 1000

    inference_time_ms = yolo_inference_time_ms + head_pose_inference_time_ms

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
    result.gaze_pitch = head_pose_result.gaze_pitch
    result.gaze_yaw = head_pose_result.gaze_yaw

    return ProcessFrameOutput(
        result=result,
        yolo_results=yolo_results,
        head_pose_result=head_pose_result,
        inference_time_ms=inference_time_ms,
    )
