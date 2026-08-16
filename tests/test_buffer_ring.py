from concurrent.futures import ThreadPoolExecutor

import numpy as np

from safe_exam.agent.buffer import RingBuffer


def _make_buffer(
    *,
    ring_buffer_seconds: float = 2,
    sampling_fps: float = 2,
    clip_before_flag_seconds: float = 1,
    clip_after_flag_seconds: float = 1,
) -> RingBuffer:
    return RingBuffer(
        ring_buffer_seconds=ring_buffer_seconds,
        sampling_fps=sampling_fps,
        clip_before_flag_seconds=clip_before_flag_seconds,
        clip_after_flag_seconds=clip_after_flag_seconds,
    )


def _frame(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


def test_buffer_starts_empty():
    ring_buffer = _make_buffer()

    assert len(ring_buffer) == 0
    assert ring_buffer.snapshot() == ()


def test_extract_clip_returns_empty_list_for_empty_buffer():
    ring_buffer = _make_buffer()

    clip = ring_buffer.extract_clip(flag_timestamp=100.0)

    assert clip == []


def test_extract_clip_returns_frames_inside_flag_window():
    ring_buffer = _make_buffer(
        ring_buffer_seconds=10,
        sampling_fps=1,
        clip_before_flag_seconds=2,
        clip_after_flag_seconds=1,
    )

    for timestamp in range(96, 103):
        ring_buffer.add_frame(float(timestamp), _frame(timestamp % 256))

    clip = ring_buffer.extract_clip(flag_timestamp=100.0)

    assert [timestamp for timestamp, _ in clip] == [
        98.0,
        99.0,
        100.0,
        101.0,
    ]


def test_extract_clip_returns_available_part_of_window():
    ring_buffer = _make_buffer(
        ring_buffer_seconds=10,
        sampling_fps=1,
        clip_before_flag_seconds=3,
        clip_after_flag_seconds=2,
    )

    ring_buffer.add_frame(100.0, _frame(1))
    ring_buffer.add_frame(101.0, _frame(2))

    clip = ring_buffer.extract_clip(flag_timestamp=100.0)

    assert [timestamp for timestamp, _ in clip] == [100.0, 101.0]


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


def test_buffer_accepts_frames_from_multiple_threads():
    ring_buffer = _make_buffer(
        ring_buffer_seconds=100,
        sampling_fps=10,
    )
    frame = _frame(0)

    def add_batch(start: int) -> None:
        for timestamp in range(start, start + 100):
            ring_buffer.add_frame(float(timestamp), frame)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(add_batch, [0, 100, 200, 300]))

    entries = ring_buffer.snapshot()
    timestamps = {timestamp for timestamp, _ in entries}

    assert len(entries) == 400
    assert timestamps == {float(timestamp) for timestamp in range(400)}
