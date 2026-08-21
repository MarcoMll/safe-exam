"""Clip upload route."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from server import storage
from server.auth import require_bearer

router = APIRouter(tags=["clip"])


@router.post("/clip/upload")
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

    dest_dir = storage.clips_root() / exam_id / student_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    clip_path = dest_dir / f"{timestamp}.mp4"
    sidecar_path = dest_dir / f"{timestamp}.json"

    clip_path.write_bytes(await clip.read())
    sidecar_path.write_bytes(sidecar_bytes)

    return {"status": "stored"}
