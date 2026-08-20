"""Tests for session lifecycle (#39)."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests
from test_agent_config import _build_config

from safe_exam.agent.session import (
    ChecklistError,
    ExamSession,
    SessionError,
    check_server,
    end_session,
    run_checklist,
    start_session,
)


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

        def stop(self) -> None:
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
