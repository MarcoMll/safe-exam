"""Session start/end routes."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from server import storage
from server.auth import require_bearer
from server.schemas.session import (
    SessionEndRequest,
    SessionEndResponse,
    SessionStartRequest,
    SessionStartResponse,
)

router = APIRouter(tags=["session"])


def load_session(session_id: str) -> dict:
    """Load a session record from the filesystem."""
    session_path = storage.sessions_root() / f"{session_id}.json"
    if not session_path.is_file():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    try:
        record = json.loads(session_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail="Session record is corrupted",
        ) from exc

    if not isinstance(record, dict):
        raise HTTPException(status_code=500, detail="Session record is invalid")

    return record


@router.post("/session/start", response_model=SessionStartResponse)
def session_start(
    body: SessionStartRequest,
    _token: str = Depends(require_bearer),
) -> SessionStartResponse:
    """Register a new exam session and return its session ID."""
    session_id = str(uuid.uuid4())
    started_at = time.time()

    record = {
        "session_id": session_id,
        "exam_id": body.exam_id,
        "student_id": body.student_id,
        "started_at": started_at,
        "started_at_iso": datetime.fromtimestamp(
            started_at, tz=timezone.utc
        ).isoformat(),
        "status": "open",
    }

    storage.sessions_root().mkdir(parents=True, exist_ok=True)
    session_path = storage.sessions_root() / f"{session_id}.json"
    session_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    return SessionStartResponse(session_id=session_id)


@router.post("/session/end", response_model=SessionEndResponse)
def session_end(
    body: SessionEndRequest,
    _token: str = Depends(require_bearer),
) -> SessionEndResponse:
    """End an exam session and record the results."""
    record = load_session(body.session_id)

    record["status"] = "closed"
    record["ended_at"] = time.time()
    record["duration_seconds"] = body.duration_seconds
    record["flag_count"] = body.flag_count
    record["clips_uploaded"] = body.clips_uploaded
    record["clips_pending"] = body.clips_pending

    session_path = storage.sessions_root() / f"{body.session_id}.json"
    session_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    return SessionEndResponse(status="closed")
