"""Tests for lightweight metadata stream buffering."""
import time
from unittest.mock import MagicMock

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

    stream.start_recording()
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

    stream.stop_recording()


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

def test_start_recording_starts_background_thread():
    stream = MetadataStreamThread(
        server_url="http://localhost:8000",
        session_id="test-session",
        auth_token="test-token",
        interval_seconds=1.0,
    )

    stream.start_recording()

    assert stream.recording is True
    assert stream._thread is not None
    assert stream._thread.is_alive()

    stream.stop_recording()

def test_interval_creates_pending_packet():
    stream = MetadataStreamThread(
        server_url="https://idinahui.pidor/spencer-dolbaeb",
        session_id="test-session",
        auth_token="test-token",
        interval_seconds=0.1,
    )

    mock_output = MagicMock(spec=ProcessFrameOutput)
    mock_output.as_dict.return_value = {"frame_id": 1}
    mock_output.result = MagicMock(spec=FrameResult)
    mock_output.result.person_count = 1

    stream.start_recording()
    stream.record_frame(
        output=mock_output,
        gaze_off_seconds=0.5,
        fused_score=0.8,
    )

    time.sleep(1)

    packets = stream.get_pending_packets()
    stream.stop_recording()

    assert len(packets) == 1

    packet = packets[0]

    assert packet["session_id"] == "test-session"
    assert isinstance(packet["timestamp"], float)
    assert len(packet["signals"]) == 1

    signal = packet["signals"][0]

    assert signal["frame_id"] == 1
    assert signal["extra_person_detected"] is False
    assert signal["gaze_off_seconds"] == 0.5
    assert signal["fused_score"] == 0.8

def test_stop_recording_loses_partial_packet():
    stream = MetadataStreamThread(
        server_url="https://idinahui.pidor/spencer-dolbaeb",
        session_id="test-session",
        auth_token="test-token",
        interval_seconds=10,
    )

    mock_output = MagicMock(spec=ProcessFrameOutput)
    mock_output.as_dict.return_value = {"frame_id": 1}
    mock_output.result = MagicMock(spec=FrameResult)
    mock_output.result.person_count = 1

    stream.start_recording()
    stream.record_frame(
        output=mock_output,
        gaze_off_seconds=0.5,
        fused_score=0.8,
    )

    time.sleep(1)

    stream.stop_recording()

    packets = stream.get_pending_packets()

    assert len(packets) == 1
    assert len(packets[0]["signals"]) == 1

    assert stream._thread is not None
    assert not stream._thread.is_alive()

def test_repeated_start_does_not_create_another_thread():
    stream = MetadataStreamThread(
        server_url="http://localhost:8000",
        session_id="test-session",
        auth_token="test-token",
        interval_seconds=1.0,
    )

    stream.start_recording()
    first_thread = stream._thread

    stream.start_recording()

    assert stream._thread is first_thread

    stream.stop_recording()

def test_repeated_stop_does_not_fail():
    stream = MetadataStreamThread(
        server_url="http://localhost:8000",
        session_id="test-session",
        auth_token="test-token",
        interval_seconds=1.0,
    )

    stream.start_recording()

    stream.stop_recording()
    stream.stop_recording()

    assert stream.recording is False
    assert stream._thread is not None
    assert not stream._thread.is_alive()

def test_can_start_again_after_stop():
    stream = MetadataStreamThread(
        server_url="http://localhost:8000",
        session_id="test-session",
        auth_token="test-token",
        interval_seconds=1.0,
    )

    stream.start_recording()
    first_thread = stream._thread

    stream.stop_recording()

    stream.start_recording()
    second_thread = stream._thread

    assert stream.recording is True
    assert second_thread is not None
    assert second_thread.is_alive()
    assert second_thread is not first_thread

    stream.stop_recording()
