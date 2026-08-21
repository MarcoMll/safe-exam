"""Bearer token validation for agent requests."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException

_DEFAULT_TOKENS = {"replace-me", "dev", "demo"}


def allowed_tokens() -> set[str]:
    """Return the set of accepted Bearer tokens for local/dev use."""
    tokens = set(_DEFAULT_TOKENS)
    extra = os.environ.get("EXAMGUARD_DEV_TOKEN", "").strip()

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
    if token not in allowed_tokens():
        raise HTTPException(status_code=401, detail="Unknown token")

    return token
