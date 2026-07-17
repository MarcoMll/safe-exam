"""Tests for spatial intrusion policy helpers."""

from safe_exam.detectors.object.results import DetectedBox
from safe_exam.processor.frame_result import FrameResult
from safe_exam.processor.intrusion_policy import (
    DEFAULT_INTRUSION_POLICY,
    IntrusionPolicyConfig,
    intersection_over_union,
    intrusion_features_for_frame,
    is_intrusion_suspected_for_frame,
    is_spatial_intrusion_suspected,
    select_primary_person,
)

FRAME_W = 1280
FRAME_H = 720


def _box(x1: float, y1: float, x2: float, y2: float) -> DetectedBox:
    return DetectedBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, class_id=0)


def _frame_result(*boxes: DetectedBox) -> FrameResult:
    return FrameResult(
        person_count=len(boxes),
        person_boxes=list(boxes),
        frame_width=FRAME_W,
        frame_height=FRAME_H,
    )


def test_select_primary_person_picks_largest_box():
    small = _box(0, 0, 100, 100)
    large = _box(0, 0, 200, 200)
    assert select_primary_person([small, large]) is large


def test_select_primary_person_empty_returns_none():
    assert select_primary_person([]) is None


def test_intersection_over_union_zero_when_no_overlap():
    a = _box(0, 0, 10, 10)
    b = _box(20, 20, 30, 30)
    assert intersection_over_union(a, b) == 0.0


def test_background_person_does_not_trigger_intrusion():
    primary = _box(200, 100, 600, 500)
    background = _box(10, 10, 80, 120)
    result = _frame_result(primary, background)

    assert is_intrusion_suspected_for_frame(result, DEFAULT_INTRUSION_POLICY) is False


def test_lean_in_overlap_triggers_intrusion():
    primary = _box(200, 100, 600, 500)
    lean_in = _box(450, 80, 700, 480)
    result = _frame_result(primary, lean_in)

    assert is_intrusion_suspected_for_frame(result, DEFAULT_INTRUSION_POLICY) is True


def test_solo_frame_never_triggers_intrusion():
    primary = _box(200, 100, 600, 500)
    result = _frame_result(primary)

    assert (
        is_spatial_intrusion_suspected(
            result.person_boxes,
            FRAME_W,
            FRAME_H,
            DEFAULT_INTRUSION_POLICY,
        )
        is False
    )


def test_intrusion_features_for_background_frame():
    primary = _box(200, 100, 600, 500)
    background = _box(10, 10, 80, 120)
    result = _frame_result(primary, background)
    features = intrusion_features_for_frame(result, DEFAULT_INTRUSION_POLICY)

    assert features.person_count == 2
    assert features.max_secondary_iou == 0.0
    assert features.intrusion_suspected is False


def test_strict_policy_requires_more_rules():
    primary = _box(200, 100, 600, 500)
    background = _box(10, 10, 80, 120)
    strict = IntrusionPolicyConfig(min_rules_to_match=3)

    assert (
        is_spatial_intrusion_suspected(
            [primary, background],
            FRAME_W,
            FRAME_H,
            strict,
        )
        is False
    )
