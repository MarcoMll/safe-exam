"""ExamGuard dev server - implements docs/api-contract.md (Phase A)."""

import json
import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile

app = FastAPI(title="ExamGuard Dev Server")

STORAGE_ROOT = Path(__file__).resolve().parent / "storage" / "clips"
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
