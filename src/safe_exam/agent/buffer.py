"""Ring buffer for pre/post-flag clip extraction. Owned by #34."""

from collections import deque
from math import ceil

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

        self._max_frames = ceil(ring_buffer_seconds * sampling_fps)
        self._frames: deque[FrameEntry] = deque(maxlen=self._max_frames)

    def add_frame(self, timestamp: float, frame: np.ndarray) -> None:
        """Add one timestamped frame to the buffer."""
        self._frames.append((timestamp, frame))

    def snapshot(self) -> tuple[FrameEntry, ...]:
        """Return the stored entries in chronological insertion order."""
        return tuple(self._frames)

    # created this one as a helper method for testing
    def __len__(self) -> int:
        """Return the current number of stored frames."""
        return len(self._frames)
