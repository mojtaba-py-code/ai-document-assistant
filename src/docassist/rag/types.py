"""Result types of the assistant (answers, agent runs, conversations)."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

AnswerStatus = Literal["answered", "insufficient_context", "refused"]
ConfidenceLabel = Literal["low", "medium", "high"]


def confidence_label(value: float) -> ConfidenceLabel:
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "medium"
    return "low"


def _jsonable(value: Any) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class Citation:
    n: int
    source_id: str
    document_id: uuid.UUID
    document_title: str
    version_number: int
    page_start: int | None
    page_end: int | None
    section: str | None
    quote: str
    chunk_id: uuid.UUID | None


@dataclass(frozen=True, slots=True)
class Evidence:
    source_id: str
    document_id: uuid.UUID
    document_title: str
    page_start: int | None
    section: str | None
    excerpt: str


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    status: AnswerStatus
    answer: str
    confidence: float
    confidence_label: ConfidenceLabel
    citations: list[Citation]
    evidence: list[Evidence]
    warnings: list[str]
    model: str | None
    provider: str | None
    usage: Usage
    conversation_id: uuid.UUID | None
    message_id: uuid.UUID | None
    latency_ms: int
    prompt_version: str = ""
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = _jsonable(asdict(self))
        return data


@dataclass(frozen=True, slots=True)
class AgentCitation:
    n: int
    document_id: uuid.UUID
    document_title: str
    quote: str


@dataclass(frozen=True, slots=True)
class AgentStep:
    tool: str
    ok: bool
    detail: str
    duration_ms: int


@dataclass(frozen=True, slots=True)
class AgentResult:
    status: AnswerStatus
    answer: str
    citations: list[AgentCitation]
    steps: list[AgentStep]
    warnings: list[str]
    model: str | None
    provider: str | None
    usage: Usage
    latency_ms: int
    iterations: int = 0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = _jsonable(asdict(self))
        return data


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    id: uuid.UUID
    role: str
    content: str
    status: str | None
    confidence: float | None
    citations: list[dict[str, Any]]
    warnings: list[str]
    model: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ConversationDetail:
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[ConversationMessage] = field(default_factory=list)
