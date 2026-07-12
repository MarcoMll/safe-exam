"""Drawing helpers for YOLO object detection results."""

import numpy as np


def draw_object_overlay(yolo_results) -> np.ndarray:
    """Draw YOLO bounding boxes on a frame."""
    return yolo_results[0].plot()
