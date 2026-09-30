"""The slice of the LLM gateway contract the intelligence area consumes.

The request/result types and errors are re-exported from :mod:`docassist.llm.base` (the
single source of truth); :class:`Gateway` narrows ``container.llm`` to the two methods
this area calls, so tests can inject any object that provides them.
"""

from __future__ import annotations

import uuid
from typing import Protocol

from docassist.core.enums import Classification
from docassist.llm.base import (
    ChatMessage,
    LLMError,
    LLMOutputInvalid,
    LLMPolicyDenied,
    LLMRefused,
    LLMRequest,
    LLMResult,
    LLMTask,
    LLMUnavailable,
)


class Gateway(Protocol):
    """What the intelligence area needs from ``container.llm``."""

    async def complete(
        self, request: LLMRequest, *, org_id: uuid.UUID, user_id: uuid.UUID | None
    ) -> LLMResult: ...  # pragma: no cover - protocol

    def route(self, classification: Classification) -> str | None: ...  # pragma: no cover


__all__ = [
    "ChatMessage",
    "Gateway",
    "LLMError",
    "LLMOutputInvalid",
    "LLMPolicyDenied",
    "LLMRefused",
    "LLMRequest",
    "LLMResult",
    "LLMTask",
    "LLMUnavailable",
]
