"""Metadata ingest route."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends

from server import storage
from server.auth import require_bearer
from server.routes.session import load_session
from server.schemas.session import MetadataIngestRequest, MetadataIngestResponse

router = APIRouter(tags=["metadata"])


@router.post("/metadata/ingest", response_model=MetadataIngestResponse)
def metadata_ingest(
    body: MetadataIngestRequest,
    _token: str = Depends(require_bearer),
) -> MetadataIngestResponse:
    """Ingest metadata signals for a session."""
    load_session(body.session_id)

    storage.metadata_root().mkdir(parents=True, exist_ok=True)
    metadata_path = storage.metadata_root() / f"{body.session_id}.jsonl"

    batch = {
        "session_id": body.session_id,
        "timestamp": body.timestamp,
        "signals": [signal.model_dump() for signal in body.signals],
    }

    line = json.dumps(batch) + "\n"
    with metadata_path.open("a", encoding="utf-8") as metadata_file:
        metadata_file.write(line)

    return MetadataIngestResponse(received=len(body.signals))
