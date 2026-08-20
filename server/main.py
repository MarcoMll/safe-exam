"""ExamGuard dev server - implements docs/api-contract.md (Phase A + B)."""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

app = FastAPI(title="ExamGuard Dev Server")

STORAGE_ROOT = Path(__file__).resolve().parent / "storage" / "clips"
SESSIONS_ROOT = Path(__file__).resolve().parent / "storage" / "sessions"
METADATA_ROOT = Path(__file__).resolve().parent / "storage" / "metadata"
_DEFAULT_TOKENS = {"replace-me", "dev", "demo"}


def _allowed_tokens() -> set[str]:
    """Get the set of allowed tokens from environment or defaults."""
    extra = os.environ.get("EXAMGUARD_DEV_TOKEN", "").strip()
    tokens = set(_DEFAULT_TOKENS)
    if extra:
        tokens.add(extra)
    return tokens


def require_bearer(
    authorization: str | None = Header(default=None),
) -> str:
    """Reject the request unless Authorization is Bearer + a known token."""
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )

    token = authorization.removeprefix("Bearer ").strip()
    if token not in _allowed_tokens():
        raise HTTPException(status_code=401, detail="Unknown token")

    return token


def _load_session(session_id: str) -> dict:
    """Load a session record from the filesystem."""
    session_path = SESSIONS_ROOT / f"{session_id}.json"
    if not session_path.is_file():
        raise HTTPException(status_code=404, detail="Session not found")

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


class SessionStartRequest(BaseModel):
    """Request to start a new session."""

    exam_id: str = Field(..., min_length=1)
    student_id: str = Field(..., min_length=1)


class SessionStartResponse(BaseModel):
    """Response to a session start request."""

    session_id: str


class SessionEndRequest(BaseModel):
    """Request to end an exam session."""

    session_id: str = Field(..., min_length=1)
    duration_seconds: float = Field(..., ge=0)
    flag_count: int = Field(..., ge=0)
    clips_uploaded: int = Field(..., ge=0)
    clips_pending: int = Field(..., ge=0)


class SessionEndResponse(BaseModel):
    status: str


class MetadataSignal(BaseModel):
    timestamp: float
    model_config = {"extra": "allow"}


class MetadataIngestRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    timestamp: float
    signals: list[MetadataSignal]


class MetadataIngestResponse(BaseModel):
    received: int


@app.post("/session/start")
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

    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    session_path = SESSIONS_ROOT / f"{session_id}.json"
    session_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    return SessionStartResponse(session_id=session_id)


@app.post("/session/end")
def session_end(
    body: SessionEndRequest,
    _token: str = Depends(require_bearer),
) -> SessionEndResponse:
    """End an exam session and record the results."""
    record = _load_session(body.session_id)

    record["status"] = "closed"
    record["ended_at"] = time.time()
    record["duration_seconds"] = body.duration_seconds
    record["flag_count"] = body.flag_count
    record["clips_uploaded"] = body.clips_uploaded
    record["clips_pending"] = body.clips_pending

    session_path = SESSIONS_ROOT / f"{body.session_id}.json"
    session_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    return SessionEndResponse(status="closed")


@app.post("/metadata/ingest")
def metadata_ingest(
    body: MetadataIngestRequest,
    _token: str = Depends(require_bearer),
) -> MetadataIngestResponse:
    """Ingest metadata signals for a session."""
    _load_session(body.session_id)

    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    metadata_path = METADATA_ROOT / f"{body.session_id}.jsonl"

    batch = {
        "session_id": body.session_id,
        "timestamp": body.timestamp,
        "signals": [signal.model_dump() for signal in body.signals],
    }

    line = json.dumps(batch) + "\n"
    with metadata_path.open("a", encoding="utf-8") as metadata_file:
        metadata_file.write(line)

    return MetadataIngestResponse(received=len(body.signals))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/check")
def auth_check(_token: str = Depends(require_bearer)) -> dict[str, str]:
    """Light auth probe for the pre-exam checklist."""
    return {"status": "ok"}


@app.post("/clip/upload")
async def clip_upload(
    clip: UploadFile = File(...),
    sidecar: UploadFile = File(...),
    _token: str = Depends(require_bearer),
) -> dict[str, str]:
    """Upload a clip and its sidecar JSON metadata."""
    sidecar_bytes = await sidecar.read()
    try:
        meta = json.loads(sidecar_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="sidecar is not valid JSON",
        ) from exc

    exam_id = str(meta.get("exam_id") or "unknown_exam")
    student_id = str(meta.get("student_id") or "unknown_student")
    timestamp = int(float(meta.get("timestamp") or 0))

    dest_dir = STORAGE_ROOT / exam_id / student_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    clip_path = dest_dir / f"{timestamp}.mp4"
    sidecar_path = dest_dir / f"{timestamp}.json"

    clip_path.write_bytes(await clip.read())
    sidecar_path.write_bytes(sidecar_bytes)

    return {"status": "stored"}
