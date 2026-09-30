"""Request-scoped context (correlation IDs) and a single source of "now"."""

from __future__ import annotations

import re
import secrets
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{8,64}$")


def new_request_id() -> str:
    return secrets.token_hex(12)


def accept_request_id(candidate: str | None) -> str:
    """Accept a caller-supplied correlation ID only if it is short and log-safe."""
    if candidate and _REQUEST_ID_RE.fullmatch(candidate):
        return candidate
    return new_request_id()


def current_request_id() -> str | None:
    return request_id_var.get()


def utcnow() -> datetime:
    return datetime.now(UTC)
