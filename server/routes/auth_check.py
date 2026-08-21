"""Auth probe for the pre-exam checklist."""

from fastapi import APIRouter, Depends

from server.auth import require_bearer

router = APIRouter(tags=["auth"])


@router.get("/auth/check")
def auth_check(_token: str = Depends(require_bearer)) -> dict[str, str]:
    """Light auth probe for the pre-exam checklist."""
    return {"status": "ok"}
