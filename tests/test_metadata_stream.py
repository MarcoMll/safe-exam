"""Tests for lightweight metadata stream buffering and upload."""

import json
from pathlib import Path

import pytest

from safe_exam.agent.network import (
    METADATA_SCHEMA_VERSION,
    LocalBatchStore,
    MetadataStreamThread,
)
from safe_exam.detectors.face_gaze.results import FaceGazeResult
from safe_exam.processor.frame_result import FrameResult, ProcessFrameOutput


def _output(result: FrameResult) -> ProcessFrameOutput:
    return ProcessFrameOutput(
        result=result,
        yolo_results=None,
        face_gaze_result=FaceGazeResult(),
        inference_time_ms=0.0,
    )


def test_metadata_stream_builds_ingest_endpoint_from_server_url(tmp_path: Path):
    stream = MetadataStreamThread(
        server_url="https://examguard.school.edu/",
        session_id="session-1",
        auth_token="secret",
        batch_store_dir=tmp_path,
    )

    assert stream.endpoint_url == "https://examguard.school.edu/metadata/ingest"
    assert stream.interval_seconds == 5.0


def test_metadata_stream_rejects_non_positive_interval(tmp_path: Path):
    with pytest.raises(ValueError, match="interval_seconds"):
        MetadataStreamThread(
            server_url="https://examguard.school.edu",
            session_id="session-1",
            auth_token="secret",
            interval_seconds=0,
            batch_store_dir=tmp_path,
        )


def test_record_frame_buffers_signal_when_recording(tmp_path: Path):
    stream = MetadataStreamThread(
        server_url="https://examguard.school.edu",
        session_id="session-1",
        auth_token="secret",
        batch_store_dir=tmp_path,
    )
    frame_result = FrameResult(
        timestamp=1720000000.0,
        phone_detected=True,
        phone_confidence=0.91,
        head_pitch=-5.2,
        head_yaw=3.1,
        eye_pitch=-2.1,
        eye_yaw=1.3,
        gaze_pitch=-4.2,
        gaze_yaw=2.5,
        person_count=2,
    )

    stream.recording = True
    stream.record_frame(
        _output(frame_result),
        gaze_off_seconds=1.25,
        fused_score=0.75,
    )

    signals = stream._drain_signals()

    assert len(signals) == 1
    assert signals[0]["timestamp"] == 1720000000.0
    assert signals[0]["phone_detected"] is True
    assert signals[0]["phone_confidence"] == 0.91
    assert signals[0]["head_pitch"] == -5.2
    assert signals[0]["person_count"] == 2
    assert signals[0]["extra_person_detected"] is True
    assert signals[0]["gaze_off_seconds"] == 1.25
    assert signals[0]["fused_score"] == 0.75
    assert stream._drain_signals() == ()


def test_record_frame_ignores_frames_when_not_recording(tmp_path: Path):
    stream = MetadataStreamThread(
        server_url="https://examguard.school.edu",
        session_id="session-1",
        auth_token="secret",
        batch_store_dir=tmp_path,
    )

    stream.record_frame(
        _output(FrameResult(timestamp=1720000000.0)),
        gaze_off_seconds=0.0,
        fused_score=0.0,
    )

    assert stream._drain_signals() == ()


def test_flush_posts_batch_and_clears_buffer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stream = MetadataStreamThread(
        server_url="https://examguard.school.edu",
        session_id="session-1",
        auth_token="secret",
        batch_store_dir=tmp_path,
    )
    posted: list[dict] = []

    def fake_post(**kwargs) -> int:
        posted.append(kwargs)
        return 200

    monkeypatch.setattr("safe_exam.agent.network._post_metadata_batch", fake_post)

    stream.recording = True
    stream.record_frame(
        _output(FrameResult(timestamp=1720000000.0)),
        gaze_off_seconds=0.0,
        fused_score=0.1,
    )
    stream._flush()

    assert len(posted) == 1
    assert posted[0]["auth_token"] == "secret"
    assert posted[0]["endpoint_url"] == "https://examguard.school.edu/metadata/ingest"
    assert posted[0]["batch"]["session_id"] == "session-1"
    assert posted[0]["batch"]["schema_version"] == METADATA_SCHEMA_VERSION
    assert len(posted[0]["batch"]["signals"]) == 1
    assert stream._drain_signals() == ()
    assert stream.batch_store.pending() == []


def test_flush_restores_signals_when_server_rejects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stream = MetadataStreamThread(
        server_url="https://examguard.school.edu",
        session_id="session-1",
        auth_token="secret",
        batch_store_dir=tmp_path,
    )
    monkeypatch.setattr(
        "safe_exam.agent.network._post_metadata_batch",
        lambda **kwargs: 500,
    )

    stream.recording = True
    stream.record_frame(
        _output(FrameResult(timestamp=1720000000.0)),
        gaze_off_seconds=0.0,
        fused_score=0.1,
    )
    stream._flush()

    pending_paths = stream.batch_store.pending()
    assert len(pending_paths) == 1
    batch = stream.batch_store.load(pending_paths[0])
    assert batch["session_id"] == "session-1"
    assert batch["signals"][0]["timestamp"] == 1720000000.0


def test_flush_restores_signals_when_post_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stream = MetadataStreamThread(
        server_url="https://examguard.school.edu",
        session_id="session-1",
        auth_token="secret",
        batch_store_dir=tmp_path,
    )

    def fake_post(**kwargs) -> int:
        raise ConnectionError("offline")

    monkeypatch.setattr("safe_exam.agent.network._post_metadata_batch", fake_post)

    stream.recording = True
    stream.record_frame(
        _output(FrameResult(timestamp=1720000000.0)),
        gaze_off_seconds=0.0,
        fused_score=0.1,
    )
    stream._flush()

    pending_paths = stream.batch_store.pending()
    assert len(pending_paths) == 1
    saved_batch = stream.batch_store.load(pending_paths[0])
    assert saved_batch["signals"][0]["timestamp"] == 1720000000.0


def test_stop_flushes_remaining_signals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stream = MetadataStreamThread(
        server_url="https://examguard.school.edu",
        session_id="session-1",
        auth_token="secret",
        interval_seconds=0.05,
        batch_store_dir=tmp_path,
    )
    posted: list[dict] = []

    def fake_post(**kwargs) -> int:
        posted.append(kwargs["batch"])
        return 200

    monkeypatch.setattr("safe_exam.agent.network._post_metadata_batch", fake_post)

    stream.start()
    try:
        stream.record_frame(
            _output(FrameResult(timestamp=1720000000.0)),
            gaze_off_seconds=0.0,
            fused_score=0.1,
        )
    finally:
        stream.stop()

    assert stream._thread is None
    assert any(len(batch["signals"]) == 1 for batch in posted)
    assert stream._drain_signals() == ()


def test_repeated_start_does_not_create_another_thread(tmp_path: Path) -> None:
    stream = MetadataStreamThread(
        server_url="http://localhost:8000",
        session_id="test-session",
        auth_token="test-token",
        interval_seconds=1.0,
        batch_store_dir=tmp_path,
    )

    stream.start()
    first_thread = stream._thread

    stream.start()

    assert stream._thread is first_thread

    stream.stop()


def test_repeated_stop_does_not_fail(tmp_path: Path) -> None:
    stream = MetadataStreamThread(
        server_url="http://localhost:8000",
        session_id="test-session",
        auth_token="test-token",
        interval_seconds=1.0,
        batch_store_dir=tmp_path,
    )

    stream.start()

    stream.stop()
    stream.stop()

    assert stream.recording is False
    assert stream._thread is None


def test_can_start_again_after_stop(tmp_path: Path) -> None:
    stream = MetadataStreamThread(
        server_url="http://localhost:8000",
        session_id="test-session",
        auth_token="test-token",
        interval_seconds=1.0,
        batch_store_dir=tmp_path,
    )

    stream.start()
    first_thread = stream._thread

    stream.stop()

    stream.start()
    second_thread = stream._thread

    assert stream.recording is True
    assert second_thread is not None
    assert second_thread.is_alive()
    assert second_thread is not first_thread

    stream.stop()


def test_flush_retries_persisted_batches_before_new_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stream = MetadataStreamThread(
        server_url="https://examguard.school.edu",
        session_id="session-1",
        auth_token="secret",
        batch_store_dir=tmp_path,
    )
    old_batch = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "batch_id": "old-batch",
        "session_id": "session-1",
        "created_at": 1000.0,
        "signals": [{"timestamp": 999.0}],
    }
    stream.batch_store.save(old_batch)

    posted_ids: list[str] = []

    def fake_post(**kwargs) -> int:
        posted_ids.append(kwargs["batch"]["batch_id"])
        return 200

    monkeypatch.setattr("safe_exam.agent.network._post_metadata_batch", fake_post)

    stream.recording = True
    stream.record_frame(
        _output(FrameResult(timestamp=1720000000.0)),
        gaze_off_seconds=0.0,
        fused_score=0.1,
    )

    stream._flush()

    assert posted_ids[0] == "old-batch"
    assert len(posted_ids) == 2
    assert stream.batch_store.pending() == []


def test_local_batch_store_round_trip(tmp_path: Path) -> None:
    batch_store = LocalBatchStore(tmp_path)
    batch = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "batch_id": "batch-123",
        "session_id": "session-1",
        "created_at": 1720000000.0,
        "signals": [],
    }

    path_to_batch = batch_store.save(batch)

    assert isinstance(path_to_batch, Path)
    assert path_to_batch.exists()
    assert batch_store.pending() == [path_to_batch]
    assert batch_store.load(path_to_batch) == batch

    raw = json.loads(path_to_batch.read_text(encoding="utf-8"))
    assert raw == batch

    batch_store.delete(path_to_batch)
    assert batch_store.pending() == []
