"""Tests for camera open/reuse in capture_frames."""

from __future__ import annotations

import numpy as np

from safe_exam.capture.capture import capture_frames
from safe_exam.capture.capture_config import CaptureConfig


class _FakeCap:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)
        self.released = False
        self.reads = 0

    def read(self):
        self.reads += 1
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True


def test_capture_frames_reuses_injected_cap_without_opening(
    monkeypatch,
) -> None:
    opened: list[int] = []
    monkeypatch.setattr(
        "safe_exam.capture.capture.open_camera",
        lambda index: opened.append(index) or _FakeCap([]),
    )

    frames = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8),
    ]
    fake_cap = _FakeCap(frames)

    yielded = list(capture_frames(CaptureConfig(target_fps=100.0), cap=fake_cap))

    assert len(yielded) == 2
    assert opened == []
    assert fake_cap.released is False


def test_capture_frames_opens_and_releases_when_cap_omitted(
    monkeypatch,
) -> None:
    fake_cap = _FakeCap([np.zeros((4, 4, 3), dtype=np.uint8)])
    opened: list[int] = []

    def fake_open(index: int):
        opened.append(index)
        return fake_cap

    monkeypatch.setattr("safe_exam.capture.capture.open_camera", fake_open)

    yielded = list(capture_frames(CaptureConfig(camera_index=2, target_fps=100.0)))

    assert opened == [2]
    assert len(yielded) == 1
    assert fake_cap.released is True
