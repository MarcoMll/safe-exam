"""Tests for session lifecycle (#39)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import requests
from test_agent_config import _build_config

from safe_exam.agent.buffer import RingBuffer
from safe_exam.agent.fusion import FlagEvent
from safe_exam.agent.session import (
    ChecklistError,
    ExamSession,
    SessionError,
    _PendingFlag,
    check_server,
    end_session,
    run_checklist,
    start_session,
)
from safe_exam.processor.frame_result import FrameResult


def test_check_server_raises_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*_args, **_kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("safe_exam.agent.session.requests.get", fake_get)

    with pytest.raises(ChecklistError, match="not reachable"):
        check_server("http://127.0.0.1:8000")


def test_start_session_returns_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_config()

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict[str, str]:
            return {"session_id": "session-abc"}

    monkeypatch.setattr(
        "safe_exam.agent.session.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    assert start_session(config) == "session-abc"


def test_start_session_raises_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_config()

    class FakeResponse:
        status_code = 503

    monkeypatch.setattr(
        "safe_exam.agent.session.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(SessionError, match="Session start failed"):
        start_session(config)


def test_run_checklist_stops_on_first_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_config()
    calls: list[str] = []

    def fake_check_server(_url: str) -> None:
        calls.append("server")
        raise ChecklistError("server down")

    def fake_check_auth(_url: str, _token: str) -> None:
        calls.append("auth")

    monkeypatch.setattr("safe_exam.agent.session.check_server", fake_check_server)
    monkeypatch.setattr("safe_exam.agent.session.check_auth", fake_check_auth)

    with pytest.raises(ChecklistError, match="server down"):
        run_checklist(config)

    assert calls == ["server"]


def test_run_checklist_returns_open_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_config()

    class FakeCap:
        def __init__(self) -> None:
            self.released = False

        def read(self):
            return True, object()

        def release(self) -> None:
            self.released = True

    fake_cap = FakeCap()
    monkeypatch.setattr("safe_exam.agent.session.check_server", lambda *_a: None)
    monkeypatch.setattr("safe_exam.agent.session.check_auth", lambda *_a: None)
    monkeypatch.setattr("safe_exam.agent.session.check_disk_space", lambda *_a: None)
    monkeypatch.setattr(
        "safe_exam.agent.session.open_camera",
        lambda _index: fake_cap,
    )

    cap = run_checklist(config)

    assert cap is fake_cap
    assert fake_cap.released is False


def test_end_session_logs_and_continues_on_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config()

    def fake_post(*_args, **_kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("safe_exam.agent.session.requests.post", fake_post)

    end_session(
        config,
        "session-abc",
        duration_seconds=10.0,
        flag_count=1,
        clips_uploaded=1,
        clips_pending=0,
    )


def test_exam_session_shutdown_calls_end_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config()
    session = ExamSession(config)
    session.session_id = "session-abc"
    session.started_at = 1000.0
    session.flag_count = 2

    class FakeUploader:
        clips_uploaded = 2

        def stop(self, **_kwargs) -> None:
            return None

    session.clip_uploader = FakeUploader()  # type: ignore[assignment]

    posted: list[dict] = []

    def fake_end_session(
        cfg,
        session_id,
        *,
        duration_seconds,
        flag_count,
        clips_uploaded,
        clips_pending,
    ) -> None:
        posted.append(
            {
                "session_id": session_id,
                "duration_seconds": duration_seconds,
                "flag_count": flag_count,
                "clips_uploaded": clips_uploaded,
                "clips_pending": clips_pending,
            }
        )

    monkeypatch.setattr("safe_exam.agent.session.end_session", fake_end_session)
    monkeypatch.setattr(
        "safe_exam.agent.session.load_pending_uploads",
        lambda _path: [],
    )

    session.shutdown()

    assert len(posted) == 1
    assert posted[0]["session_id"] == "session-abc"
    assert posted[0]["flag_count"] == 2
    assert posted[0]["clips_uploaded"] == 2
    assert posted[0]["clips_pending"] == 0


def test_exam_session_shutdown_releases_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config()
    session = ExamSession(config)

    class FakeCap:
        def __init__(self) -> None:
            self.released = False

        def release(self) -> None:
            self.released = True

    fake_cap = FakeCap()
    session._cap = fake_cap

    monkeypatch.setattr("safe_exam.agent.session.end_session", lambda *a, **k: None)
    monkeypatch.setattr(
        "safe_exam.agent.session.load_pending_uploads",
        lambda _path: [],
    )

    session.shutdown()

    assert fake_cap.released is True
    assert session._cap is None


def test_flush_ready_pending_waits_for_clip_after_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config()
    session = ExamSession(config)
    session.ring_buffer = RingBuffer(
        ring_buffer_seconds=config.ring_buffer_seconds,
        sampling_fps=config.sampling_fps,
        clip_before_flag_seconds=config.clip_before_flag_seconds,
        clip_after_flag_seconds=config.clip_after_flag_seconds,
    )

    flag_ts = 100.0
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    session.ring_buffer.add_frame(flag_ts, frame)
    session.ring_buffer.add_frame(flag_ts + 5.0, frame)

    staged: list[float] = []

    def fake_stage_clip(frames, *, flag, config, gaze_off_seconds, frame_result):
        staged.append(flag.timestamp)
        return Path("clip.mp4"), Path("clip.json")

    monkeypatch.setattr("safe_exam.agent.session.stage_clip", fake_stage_clip)

    pending = _PendingFlag(
        flag=FlagEvent(timestamp=flag_ts, score=0.9, reasons=["phone"]),
        gaze_off_seconds=0.0,
        frame_result=FrameResult(timestamp=flag_ts),
        ready_at=flag_ts + config.clip_after_flag_seconds,
    )
    session._pending_flags.append(pending)

    session._flush_ready_pending(flag_ts + 4.9)
    assert session._pending_flags == [pending]
    assert staged == []

    session._flush_ready_pending(flag_ts + 5.0)
    assert session._pending_flags == []
    assert staged == [flag_ts]


def test_shutdown_flushes_unfinished_pending_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config()
    session = ExamSession(config)
    session.ring_buffer = RingBuffer(
        ring_buffer_seconds=config.ring_buffer_seconds,
        sampling_fps=config.sampling_fps,
        clip_before_flag_seconds=config.clip_before_flag_seconds,
        clip_after_flag_seconds=config.clip_after_flag_seconds,
    )

    flag_ts = 200.0
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    session.ring_buffer.add_frame(flag_ts, frame)

    staged: list[float] = []

    def fake_stage_clip(frames, *, flag, config, gaze_off_seconds, frame_result):
        staged.append(flag.timestamp)
        return Path("clip.mp4"), Path("clip.json")

    monkeypatch.setattr("safe_exam.agent.session.stage_clip", fake_stage_clip)
    monkeypatch.setattr("safe_exam.agent.session.end_session", lambda *a, **k: None)
    monkeypatch.setattr(
        "safe_exam.agent.session.load_pending_uploads",
        lambda _path: [],
    )

    session._pending_flags.append(
        _PendingFlag(
            flag=FlagEvent(timestamp=flag_ts, score=0.8, reasons=["gaze_violation"]),
            gaze_off_seconds=1.0,
            frame_result=FrameResult(timestamp=flag_ts),
            ready_at=flag_ts + config.clip_after_flag_seconds,
        )
    )

    session.shutdown()

    assert session._pending_flags == []
    assert staged == [flag_ts]


def test_exam_session_run_requires_start() -> None:
    session = ExamSession(_build_config())

    with pytest.raises(RuntimeError, match="start\\(\\) must be called"):
        session.run()


def test_main_returns_1_on_config_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import argparse

    from safe_exam.agent import __main__ as agent_main

    missing = tmp_path / "missing.yml"
    monkeypatch.setattr(
        agent_main,
        "_parse_args",
        lambda: argparse.Namespace(config=missing, camera=0),
    )

    assert agent_main.main() == 1
