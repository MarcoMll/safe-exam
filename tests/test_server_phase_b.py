"""Tests for dev server Phase B endpoints."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from server import storage
from server.main import app

AUTH_HEADERS = {"Authorization": "Bearer replace-me"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Use isolated storage paths for each test."""
    monkeypatch.setattr(storage, "sessions_root", lambda: tmp_path / "sessions")
    monkeypatch.setattr(storage, "metadata_root", lambda: tmp_path / "metadata")
    monkeypatch.setattr(storage, "clips_root", lambda: tmp_path / "clips")
    return TestClient(app)


def test_session_start_returns_uuid(client: TestClient) -> None:
    response = client.post(
        "/session/start",
        headers=AUTH_HEADERS,
        json={"exam_id": "EXAM_2026_FINAL", "student_id": "S12345"},
    )

    assert response.status_code == 200
    session_id = response.json()["session_id"]
    assert session_id

    record_path = storage.sessions_root() / f"{session_id}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "open"
    assert record["exam_id"] == "EXAM_2026_FINAL"


def test_session_end_closes_existing_session(client: TestClient) -> None:
    start = client.post(
        "/session/start",
        headers=AUTH_HEADERS,
        json={"exam_id": "EXAM_2026_FINAL", "student_id": "S12345"},
    )
    session_id = start.json()["session_id"]

    end = client.post(
        "/session/end",
        headers=AUTH_HEADERS,
        json={
            "session_id": session_id,
            "duration_seconds": 42.0,
            "flag_count": 1,
            "clips_uploaded": 1,
            "clips_pending": 0,
        },
    )

    assert end.status_code == 200
    assert end.json() == {"status": "closed"}

    record = json.loads(
        (storage.sessions_root() / f"{session_id}.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "closed"
    assert record["flag_count"] == 1


def test_session_end_unknown_id_returns_404(client: TestClient) -> None:
    response = client.post(
        "/session/end",
        headers=AUTH_HEADERS,
        json={
            "session_id": "00000000-0000-0000-0000-000000000000",
            "duration_seconds": 1.0,
            "flag_count": 0,
            "clips_uploaded": 0,
            "clips_pending": 0,
        },
    )

    assert response.status_code == 404


def test_metadata_ingest_appends_jsonl(client: TestClient) -> None:
    start = client.post(
        "/session/start",
        headers=AUTH_HEADERS,
        json={"exam_id": "EXAM_2026_FINAL", "student_id": "S12345"},
    )
    session_id = start.json()["session_id"]

    response = client.post(
        "/metadata/ingest",
        headers=AUTH_HEADERS,
        json={
            "session_id": session_id,
            "timestamp": 1720000000.0,
            "signals": [
                {"timestamp": 1719999995.2, "phone_detected": False, "fused_score": 0.0}
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"received": 1}

    metadata_path = storage.metadata_root() / f"{session_id}.jsonl"
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    batch = json.loads(lines[0])
    assert batch["session_id"] == session_id
    assert len(batch["signals"]) == 1


def test_metadata_ingest_unknown_session_returns_404(client: TestClient) -> None:
    response = client.post(
        "/metadata/ingest",
        headers=AUTH_HEADERS,
        json={
            "session_id": "00000000-0000-0000-0000-000000000000",
            "timestamp": 1.0,
            "signals": [{"timestamp": 1.0}],
        },
    )

    assert response.status_code == 404


def test_health_requires_no_auth(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_auth_check_accepts_known_token(client: TestClient) -> None:
    response = client.get("/auth/check", headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_auth_check_rejects_missing_token(client: TestClient) -> None:
    response = client.get("/auth/check")
    assert response.status_code == 401


def test_clip_upload_stores_files(client: TestClient) -> None:
    response = client.post(
        "/clip/upload",
        headers=AUTH_HEADERS,
        files={
            "clip": ("clip.mp4", b"fake-mp4-bytes", "video/mp4"),
            "sidecar": (
                "clip.json",
                b'{"exam_id":"EXAM_2026_FINAL","student_id":"S12345","timestamp":1720000000.0}',
                "application/json",
            ),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "stored"}

    clip_path = storage.clips_root() / "EXAM_2026_FINAL" / "S12345" / "1720000000.mp4"
    sidecar_path = (
        storage.clips_root() / "EXAM_2026_FINAL" / "S12345" / "1720000000.json"
    )
    assert clip_path.is_file()
    assert sidecar_path.is_file()
    assert clip_path.read_bytes() == b"fake-mp4-bytes"
