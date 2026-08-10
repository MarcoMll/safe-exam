import numpy as np

from safe_exam.agent.buffer import RingBuffer


def _make_buffer(
    *, ring_buffer_seconds: float = 2, sampling_fps: float = 2
) -> RingBuffer:
    return RingBuffer(
        ring_buffer_seconds=ring_buffer_seconds,
        sampling_fps=sampling_fps,
        clip_before_flag_seconds=1,
        clip_after_flag_seconds=1,
    )


def _frame(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


def test_buffer_starts_empty():
    ring_buffer = _make_buffer()

    assert len(ring_buffer) == 0
    assert ring_buffer.snapshot() == ()


def test_buffer_grows_until_capacity_without_losing_frames():
    ring_buffer = _make_buffer()
    frames = [_frame(value) for value in range(4)]

    for timestamp, frame in enumerate(frames):
        ring_buffer.add_frame(float(timestamp), frame)
        assert len(ring_buffer) == timestamp + 1

    entries = ring_buffer.snapshot()
    assert [timestamp for timestamp, _ in entries] == [0.0, 1.0, 2.0, 3.0]
    assert all(
        stored_frame is frame
        for (_, stored_frame), frame in zip(entries, frames, strict=True)
    )


def test_buffer_frame_count_is_bounded():
    ring_buffer = _make_buffer()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    for timestamp in range(10):
        ring_buffer.add_frame(float(timestamp), frame)

    assert len(ring_buffer) == 4


def test_buffer_evicts_oldest_frame_and_preserves_order():
    ring_buffer = _make_buffer()

    for timestamp in range(5):
        ring_buffer.add_frame(float(timestamp), _frame(timestamp))

    entries = ring_buffer.snapshot()
    assert [timestamp for timestamp, _ in entries] == [1.0, 2.0, 3.0, 4.0]
    assert [int(frame[0, 0, 0]) for _, frame in entries] == [1, 2, 3, 4]


def test_buffer_capacity_is_rounded_up_for_fractional_frame_count():
    ring_buffer = _make_buffer(ring_buffer_seconds=2.1, sampling_fps=2)

    for timestamp in range(6):
        ring_buffer.add_frame(float(timestamp), _frame(timestamp))

    assert len(ring_buffer) == 5
    assert [timestamp for timestamp, _ in ring_buffer.snapshot()] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]


def test_buffer_keeps_timestamp_paired_with_its_frame():
    ring_buffer = _make_buffer()
    expected = [(10.5, _frame(10)), (11.5, _frame(20))]

    for timestamp, frame in expected:
        ring_buffer.add_frame(timestamp, frame)

    entries = ring_buffer.snapshot()
    assert [timestamp for timestamp, _ in entries] == [10.5, 11.5]
    np.testing.assert_array_equal(entries[0][1], expected[0][1])
    np.testing.assert_array_equal(entries[1][1], expected[1][1])


def test_buffer_stores_frame_by_reference():
    ring_buffer = _make_buffer()
    frame = _frame(1)

    ring_buffer.add_frame(1.0, frame)
    frame.fill(9)

    stored_frame = ring_buffer.snapshot()[0][1]
    assert stored_frame is frame
    assert np.all(stored_frame == 9)
