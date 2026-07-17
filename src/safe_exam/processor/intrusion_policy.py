from __future__ import annotations

from dataclasses import dataclass

from safe_exam.detectors.object.results import DetectedBox
from safe_exam.processor.frame_result import FrameResult


@dataclass(frozen=True)
class IntrusionPolicyConfig:
    """Runtime policy for interpreting multi-person detections."""

    roi_center_fraction: float = 0.6
    min_secondary_area_pct: float = 0.05
    primary_overlap_iou: float = 0.1
    min_rules_to_match: int = 2

    def label(self) -> str:
        return (
            f"roi={self.roi_center_fraction:.2f} "
            f"area={self.min_secondary_area_pct:.2f} "
            f"iou={self.primary_overlap_iou:.2f} "
            f"rules={self.min_rules_to_match}"
        )


@dataclass(frozen=True)
class IntrusionFrameFeatures:
    """Per-frame spatial features for calibration CSV logging."""

    person_count: int
    primary_area_pct: float
    max_secondary_area_pct: float
    max_secondary_iou: float
    any_secondary_in_roi: bool
    intrusion_suspected: bool


def box_area(box: DetectedBox) -> float:
    """Return the pixel area of one bounding box."""
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def box_center(box: DetectedBox) -> tuple[float, float]:
    """Return the center point of one bounding box."""
    return (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2


def intersection_over_union(box1: DetectedBox, box2: DetectedBox) -> float:
    """Return IoU between two boxes."""
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)
    intersection_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union_area = box_area(box1) + box_area(box2) - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def is_in_center_roi(
    box: DetectedBox,
    frame_width: int,
    frame_height: int,
    roi_center_fraction: float,
) -> bool:
    """True when the box center lies inside the central ROI."""
    if frame_width <= 0 or frame_height <= 0:
        return False

    margin_x = frame_width * (1.0 - roi_center_fraction) / 2.0
    margin_y = frame_height * (1.0 - roi_center_fraction) / 2.0
    center_x, center_y = box_center(box)

    return (
        margin_x <= center_x <= frame_width - margin_x
        and margin_y <= center_y <= frame_height - margin_y
    )


def box_area_fraction(box: DetectedBox, frame_area: float) -> float:
    """Return box area as a fraction of the full frame area."""
    if frame_area <= 0:
        return 0.0
    return box_area(box) / frame_area


def select_primary_person(boxes: list[DetectedBox]) -> DetectedBox | None:
    """Pick the primary student as the largest detected person box."""
    if not boxes:
        return None
    return max(boxes, key=box_area)


def _secondary_passes_policy(
    secondary: DetectedBox,
    primary: DetectedBox,
    frame_width: int,
    frame_height: int,
    policy: IntrusionPolicyConfig,
) -> bool:
    """True when a non-primary person looks spatially intrusive."""
    frame_area = frame_width * frame_height
    rules_matched = 0

    if is_in_center_roi(
        secondary,
        frame_width,
        frame_height,
        policy.roi_center_fraction,
    ):
        rules_matched += 1

    if box_area_fraction(secondary, frame_area) >= policy.min_secondary_area_pct:
        rules_matched += 1

    if intersection_over_union(secondary, primary) >= policy.primary_overlap_iou:
        rules_matched += 1

    return rules_matched >= policy.min_rules_to_match


def is_spatial_intrusion_suspected(
    boxes: list[DetectedBox],
    frame_width: int,
    frame_height: int,
    policy: IntrusionPolicyConfig,
) -> bool:
    """Evaluate one frame against the active spatial intrusion policy."""
    primary = select_primary_person(boxes)
    if primary is None:
        return False

    for box in boxes:
        if box is primary:
            continue
        if _secondary_passes_policy(
            secondary=box,
            primary=primary,
            frame_width=frame_width,
            frame_height=frame_height,
            policy=policy,
        ):
            return True

    return False


def is_intrusion_suspected_for_frame(
    result: FrameResult,
    policy: IntrusionPolicyConfig,
) -> bool:
    """Evaluate a merged frame result against the intrusion policy."""
    return is_spatial_intrusion_suspected(
        result.person_boxes,
        result.frame_width,
        result.frame_height,
        policy,
    )


def intrusion_features_for_frame(
    result: FrameResult,
    policy: IntrusionPolicyConfig,
) -> IntrusionFrameFeatures:
    """Extract spatial features used by calibration tooling."""
    boxes = result.person_boxes
    frame_area = result.frame_width * result.frame_height
    primary = select_primary_person(boxes)

    primary_area_pct = (
        box_area_fraction(primary, frame_area) if primary is not None else 0.0
    )
    max_secondary_area_pct = 0.0
    max_secondary_iou = 0.0
    any_secondary_in_roi = False

    if primary is not None:
        for box in boxes:
            if box is primary:
                continue
            max_secondary_area_pct = max(
                max_secondary_area_pct,
                box_area_fraction(box, frame_area),
            )
            max_secondary_iou = max(
                max_secondary_iou,
                intersection_over_union(box, primary),
            )
            if is_in_center_roi(
                box,
                result.frame_width,
                result.frame_height,
                policy.roi_center_fraction,
            ):
                any_secondary_in_roi = True

    return IntrusionFrameFeatures(
        person_count=len(boxes),
        primary_area_pct=primary_area_pct,
        max_secondary_area_pct=max_secondary_area_pct,
        max_secondary_iou=max_secondary_iou,
        any_secondary_in_roi=any_secondary_in_roi,
        intrusion_suspected=is_intrusion_suspected_for_frame(result, policy),
    )


DEFAULT_INTRUSION_POLICY = IntrusionPolicyConfig()
