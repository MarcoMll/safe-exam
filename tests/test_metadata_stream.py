"""Tests for lightweight metadata stream buffering."""

import pytest

from safe_exam.agent.network import MetadataStreamThread
from safe_exam.detectors.face_gaze.results import FaceGazeResult
from safe_exam.processor.frame_result import FrameResult, ProcessFrameOutput


def _output(result: FrameResult) -> ProcessFrameOutput:
    return ProcessFrameOutput(
        result=result,
        yolo_results=None,
        face_gaze_result=FaceGazeResult(),
        inference_time_ms=0.0,
    )


def test_metadata_stream_builds_ingest_endpoint_from_server_url():
    stream = MetadataStreamThread(
        server_url="https://examguard.school.edu/",
        session_id="session-1",
        auth_token="secret",
    )

    assert stream.endpoint_url == "https://examguard.school.edu/metadata/ingest"
    assert stream.interval_seconds == 5.0


def test_metadata_stream_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="interval_seconds"):
        MetadataStreamThread(
            server_url="https://examguard.school.edu",
            session_id="session-1",
            auth_token="secret",
            interval_seconds=0,
        )


def test_record_frame_buffers_signal_when_recording():
    stream = MetadataStreamThread(
        server_url="https://examguard.school.edu",
        session_id="session-1",
        auth_token="secret",
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

    stream.start()
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


def test_record_frame_ignores_frames_when_not_recording():
    stream = MetadataStreamThread(
        server_url="https://examguard.school.edu",
        session_id="session-1",
        auth_token="secret",
    )

    stream.record_frame(
        _output(FrameResult(timestamp=1720000000.0)),
        gaze_off_seconds=0.0,
        fused_score=0.0,
    )

    assert stream._drain_signals() == ()
