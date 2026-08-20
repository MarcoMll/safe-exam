"""Tests for background clip upload (#38)."""

import time
from pathlib import Path

import pytest

from safe_exam.agent.network import (
    ClipUploadThread,
    _add_to_upload_queue,
    _remove_from_upload_queue,
    load_pending_uploads,
)


def test_clip_upload_builds_endpoint_from_server_url(tmp_path: Path) -> None:
    uploader = ClipUploadThread(
        server_url="https://examguard.school.edu/",
        auth_token="secret",
        clip_dir=tmp_path,
    )

    assert uploader.endpoint_url == "https://examguard.school.edu/clip/upload"
    assert uploader.poll_interval_seconds == 1.0
    assert uploader.max_retries == 5


def test_clip_upload_rejects_non_positive_poll_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="poll_interval_seconds"):
        ClipUploadThread(
            server_url="https://examguard.school.edu",
            auth_token="secret",
            clip_dir=tmp_path,
            poll_interval_seconds=0,
        )


def test_clip_upload_start_and_stop_do_not_block(tmp_path: Path) -> None:
    uploader = ClipUploadThread(
        server_url="https://examguard.school.edu",
        auth_token="secret",
        clip_dir=tmp_path,
        poll_interval_seconds=0.05,
    )

    uploader.start()
    try:
        assert uploader.running is True
        assert uploader._thread is not None
        assert uploader._thread.is_alive()
    finally:
        uploader.stop()

    assert uploader.running is False
    assert uploader._thread is None


def test_remove_from_upload_queue_drops_one_entry(tmp_path: Path) -> None:
    queue_path = tmp_path / "upload_queue.jsonl"
    first_clip = tmp_path / "first.mp4"
    first_sidecar = tmp_path / "first.json"
    second_clip = tmp_path / "second.mp4"
    second_sidecar = tmp_path / "second.json"

    _add_to_upload_queue(queue_path, clip_path=first_clip, sidecar_path=first_sidecar)
    _add_to_upload_queue(queue_path, clip_path=second_clip, sidecar_path=second_sidecar)

    _remove_from_upload_queue(
        queue_path,
        clip_path=first_clip,
        sidecar_path=first_sidecar,
    )

    assert load_pending_uploads(queue_path) == [
        (second_clip.resolve(), second_sidecar.resolve()),
    ]


def test_upload_one_success_deletes_files_and_removes_queue_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "upload_queue.jsonl"
    clip_path = tmp_path / "clip.mp4"
    sidecar_path = tmp_path / "clip.json"
    clip_path.write_bytes(b"fake mp4")
    sidecar_path.write_text('{"exam_id": "X"}', encoding="utf-8")
    _add_to_upload_queue(queue_path, clip_path=clip_path, sidecar_path=sidecar_path)

    uploader = ClipUploadThread(
        server_url="https://examguard.school.edu",
        auth_token="secret",
        clip_dir=tmp_path,
    )

    def fake_post_clip(**kwargs) -> int:
        assert kwargs["endpoint_url"] == "https://examguard.school.edu/clip/upload"
        assert kwargs["auth_token"] == "secret"
        assert kwargs["clip_path"] == clip_path
        assert kwargs["sidecar_path"] == sidecar_path
        return 200

    monkeypatch.setattr("safe_exam.agent.network._post_clip", fake_post_clip)

    assert uploader._upload_one(clip_path, sidecar_path) is True
    assert not clip_path.exists()
    assert not sidecar_path.exists()
    assert load_pending_uploads(queue_path) == []


def test_upload_one_keeps_files_when_server_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "upload_queue.jsonl"
    clip_path = tmp_path / "clip.mp4"
    sidecar_path = tmp_path / "clip.json"
    clip_path.write_bytes(b"fake mp4")
    sidecar_path.write_text('{"exam_id": "X"}', encoding="utf-8")
    _add_to_upload_queue(queue_path, clip_path=clip_path, sidecar_path=sidecar_path)

    uploader = ClipUploadThread(
        server_url="https://examguard.school.edu",
        auth_token="secret",
        clip_dir=tmp_path,
    )
    monkeypatch.setattr(
        "safe_exam.agent.network._post_clip",
        lambda **kwargs: 500,
    )

    assert uploader._upload_one(clip_path, sidecar_path) is False
    assert clip_path.exists()
    assert sidecar_path.exists()
    assert load_pending_uploads(queue_path) == [
        (clip_path.resolve(), sidecar_path.resolve()),
    ]


def test_process_pending_uploads_only_first_queue_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "upload_queue.jsonl"
    first_clip = tmp_path / "first.mp4"
    first_sidecar = tmp_path / "first.json"
    second_clip = tmp_path / "second.mp4"
    second_sidecar = tmp_path / "second.json"

    for clip, sidecar in (
        (first_clip, first_sidecar),
        (second_clip, second_sidecar),
    ):
        clip.write_bytes(b"x")
        sidecar.write_text("{}", encoding="utf-8")

    _add_to_upload_queue(
        queue_path,
        clip_path=first_clip,
        sidecar_path=first_sidecar,
    )
    _add_to_upload_queue(
        queue_path,
        clip_path=second_clip,
        sidecar_path=second_sidecar,
    )

    uploaded: list[str] = []

    def fake_post_clip(**kwargs) -> int:
        uploaded.append(kwargs["clip_path"].name)
        return 200

    monkeypatch.setattr("safe_exam.agent.network._post_clip", fake_post_clip)

    uploader = ClipUploadThread(
        server_url="https://examguard.school.edu",
        auth_token="secret",
        clip_dir=tmp_path,
    )
    uploader._process_pending()

    assert uploaded == ["first.mp4"]
    assert not first_clip.exists()
    assert second_clip.exists()
    assert load_pending_uploads(queue_path) == [
        (second_clip.resolve(), second_sidecar.resolve()),
    ]


def test_upload_one_retries_with_backoff_then_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "upload_queue.jsonl"
    clip_path = tmp_path / "clip.mp4"
    sidecar_path = tmp_path / "clip.json"
    clip_path.write_bytes(b"fake mp4")
    sidecar_path.write_text('{"exam_id": "X"}', encoding="utf-8")
    _add_to_upload_queue(queue_path, clip_path=clip_path, sidecar_path=sidecar_path)

    uploader = ClipUploadThread(
        server_url="https://examguard.school.edu",
        auth_token="secret",
        clip_dir=tmp_path,
        max_retries=5,
    )
    responses = iter([500, 500, 200])

    def fake_post_clip(**kwargs) -> int:
        return next(responses)

    monkeypatch.setattr("safe_exam.agent.network._post_clip", fake_post_clip)
    monkeypatch.setattr(uploader, "_backoff_seconds", lambda attempt: 0.0)

    assert uploader._upload_one(clip_path, sidecar_path) is True
    assert not clip_path.exists()
    assert load_pending_uploads(queue_path) == []


def test_upload_one_stops_retrying_until_restart_after_max_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "upload_queue.jsonl"
    clip_path = tmp_path / "clip.mp4"
    sidecar_path = tmp_path / "clip.json"
    clip_path.write_bytes(b"fake mp4")
    sidecar_path.write_text('{"exam_id": "X"}', encoding="utf-8")
    _add_to_upload_queue(queue_path, clip_path=clip_path, sidecar_path=sidecar_path)

    uploader = ClipUploadThread(
        server_url="https://examguard.school.edu",
        auth_token="secret",
        clip_dir=tmp_path,
        max_retries=3,
    )
    attempts = {"count": 0}

    def fake_post_clip(**kwargs) -> int:
        attempts["count"] += 1
        return 500

    monkeypatch.setattr("safe_exam.agent.network._post_clip", fake_post_clip)
    monkeypatch.setattr(uploader, "_backoff_seconds", lambda attempt: 0.0)

    assert uploader._upload_one(clip_path, sidecar_path) is False
    assert attempts["count"] == 3
    assert clip_path.exists()
    assert load_pending_uploads(queue_path) == [
        (clip_path.resolve(), sidecar_path.resolve()),
    ]

    attempts["count"] = 0
    assert uploader._upload_one(clip_path, sidecar_path) is False
    assert attempts["count"] == 0


def test_upload_one_removes_stale_queue_entry_when_files_missing(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "upload_queue.jsonl"
    clip_path = tmp_path / "missing.mp4"
    sidecar_path = tmp_path / "missing.json"
    _add_to_upload_queue(queue_path, clip_path=clip_path, sidecar_path=sidecar_path)

    uploader = ClipUploadThread(
        server_url="https://examguard.school.edu",
        auth_token="secret",
        clip_dir=tmp_path,
    )

    assert uploader._upload_one(clip_path, sidecar_path) is False
    assert load_pending_uploads(queue_path) == []


def test_background_thread_uploads_pending_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "upload_queue.jsonl"
    clip_path = tmp_path / "clip.mp4"
    sidecar_path = tmp_path / "clip.json"
    clip_path.write_bytes(b"fake mp4")
    sidecar_path.write_text('{"exam_id": "X"}', encoding="utf-8")
    _add_to_upload_queue(queue_path, clip_path=clip_path, sidecar_path=sidecar_path)

    uploader = ClipUploadThread(
        server_url="https://examguard.school.edu",
        auth_token="secret",
        clip_dir=tmp_path,
        poll_interval_seconds=0.05,
    )
    monkeypatch.setattr(
        "safe_exam.agent.network._post_clip",
        lambda **kwargs: 200,
    )

    uploader.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if not load_pending_uploads(queue_path):
                break
            time.sleep(0.05)
    finally:
        uploader.stop()

    assert not clip_path.exists()
    assert not sidecar_path.exists()
    assert load_pending_uploads(queue_path) == []


def test_stop_drains_queue_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "upload_queue.jsonl"
    clip_path = tmp_path / "clip.mp4"
    sidecar_path = tmp_path / "clip.json"
    clip_path.write_bytes(b"fake mp4")
    sidecar_path.write_text('{"exam_id": "X"}', encoding="utf-8")
    _add_to_upload_queue(queue_path, clip_path=clip_path, sidecar_path=sidecar_path)

    uploaded: list[str] = []

    def fake_post_clip(**kwargs) -> int:
        time.sleep(0.05)
        uploaded.append(kwargs["clip_path"].name)
        return 200

    monkeypatch.setattr("safe_exam.agent.network._post_clip", fake_post_clip)

    uploader = ClipUploadThread(
        server_url="https://examguard.school.edu",
        auth_token="secret",
        clip_dir=tmp_path,
        poll_interval_seconds=0.05,
    )
    # Do not start the background thread — stop() itself must drain.
    uploader.stop(drain_timeout_seconds=5.0)

    assert uploaded == ["clip.mp4"]
    assert not clip_path.exists()
    assert load_pending_uploads(queue_path) == []
    assert uploader.running is False
