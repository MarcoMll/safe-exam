"""Session and metadata ingest schemas (Phase B contract)."""

from pydantic import BaseModel, Field


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
    """Response to a session end request."""

    status: str


class MetadataSignal(BaseModel):
    """Signal for a metadata batch ingestion."""

    timestamp: float
    model_config = {"extra": "allow"}


class MetadataIngestRequest(BaseModel):
    """Request to ingest a metadata batch."""

    session_id: str = Field(..., min_length=1)
    timestamp: float
    signals: list[MetadataSignal]


class MetadataIngestResponse(BaseModel):
    """Response to a metadata batch ingestion request."""

    received: int
