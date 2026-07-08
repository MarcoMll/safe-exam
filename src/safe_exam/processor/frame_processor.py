import time
from dataclasses import asdict

import cv2

from safe_exam.processor.FrameResult import FrameResult


def draw_debug_overlay(
    yolo_results,
    result: FrameResult,
    inference_time_ms: float,
):
    """
    Draws a debug overlay on the frame.
    :param yolo_results: the YOLO results
    :param result: the frame result
    :param inference_time_ms: the inference time in milliseconds
    :return: the debug frame
    """

    debug_frame = yolo_results[0].plot()

    cv2.putText(
        debug_frame,
        f"Phone: {result.phone_detected}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        debug_frame,
        f"Phone conf: {result.phone_confidence:.2f}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        debug_frame,
        f"Persons: {result.person_count}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        debug_frame,
        f"Extra person: {result.extra_person_detected}",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        debug_frame,
        f"Gaze pitch: {result.gaze_pitch:.2f}",
        (10, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        debug_frame,
        f"Gaze yaw: {result.gaze_yaw:.2f}",
        (10, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        debug_frame,
        f"Inference: {inference_time_ms:.1f} ms",
        (10, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    return debug_frame


def process_frame(
    frame,
    object_detector,
    debug: bool = False,
):
    """
    Processes a single frame and returns the output, debug frame, and inference time.
    :param frame: the frame to process
    :param object_detector: the object detector to use
    :param debug: whether to return the debug frame
    :return: a tuple containing the output, debug frame, and inference time
    """
    result = FrameResult(timestamp=time.time())

    start = time.perf_counter()
    yolo_results = object_detector.detect(frame=frame, classes=[0, 67])
    inference_time_ms = (time.perf_counter() - start) * 1000

    phone_detected, phone_confidence = object_detector.look_for_class(
        results=yolo_results,
        target_class_index=67,
        threshold=0.25,
    )

    person_count = object_detector.count_class(
        results=yolo_results,
        target_class_index=0,
        threshold=0.25,
    )

    result.phone_detected = phone_detected
    result.phone_confidence = phone_confidence
    result.person_count = person_count
    result.extra_person_detected = person_count > 1

    # Placeholder for gaze detection
    result.gaze_pitch = 0.0
    result.gaze_yaw = 0.0

    debug_frame = None
    if debug:
        debug_frame = draw_debug_overlay(
            yolo_results=yolo_results,
            result=result,
            inference_time_ms=inference_time_ms,
        )

    return asdict(result), debug_frame, inference_time_ms
