"""Ring buffer for pre/post-flag clip extraction. Owned by #34."""

from collections import deque
from math import ceil
from threading import Lock

import numpy as np

FrameEntry = tuple[float, np.ndarray]
# entry example:
# (
#     1786363501.42, -> timestamp
#     numpy_array, -> frame
# )


class RingBuffer:
    """Keep a bounded sequence of recent webcam frames."""

    def __init__(
        self,
        *,
        ring_buffer_seconds: float,
        sampling_fps: float,
        clip_before_flag_seconds: float,
        clip_after_flag_seconds: float,
    ) -> None:
        self._ring_buffer_seconds = ring_buffer_seconds
        self._clip_before_flag_seconds = clip_before_flag_seconds
        self._clip_after_flag_seconds = clip_after_flag_seconds

        self._lock = Lock()
        self._max_frames = ceil(ring_buffer_seconds * sampling_fps)
        self._frames: deque[FrameEntry] = deque(maxlen=self._max_frames)

    def add_frame(self, timestamp: float, frame: np.ndarray) -> None:
        """Add one timestamped frame to the buffer."""
        with self._lock:
            self._frames.append((timestamp, frame))

    def snapshot(self) -> tuple[FrameEntry, ...]:
        """Return the stored entries in chronological insertion order."""
        with self._lock:
            return tuple(self._frames)

    def extract_clip(self, flag_timestamp: float) -> list[FrameEntry]:
        """Return frames inside the configured window around a flag."""
        window_start = flag_timestamp - self._clip_before_flag_seconds
        window_end = flag_timestamp + self._clip_after_flag_seconds

        return [
            (timestamp, frame)
            for timestamp, frame in self.snapshot()
            if window_start <= timestamp <= window_end
        ]

    def __len__(self) -> int:
        """Return the current number of stored frames."""
        with self._lock:
            return len(self._frames)
