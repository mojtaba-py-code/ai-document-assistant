"""The LLM contract every other area codes against.

Providers (Anthropic, an OpenAI-compatible local model server, the offline extractive
provider) implement :class:`LLMProvider`; callers never talk to a provider directly but go
through :class:`docassist.llm.gateway.LLMGateway`, which adds data-governance routing,
pseudonymisation, budgets, rate limits, a circuit breaker, schema validation and usage
accounting.

Message content
---------------
``ChatMessage.content`` is either plain text or a list of content blocks in the
Anthropic block shape, which is the canonical internal representation:

* ``{"type": "text", "text": ...}``
* ``{"type": "tool_use", "id": ..., "name": ..., "input": {...}}`` (assistant turns)
* ``{"type": "tool_result", "tool_use_id": ..., "content": str, "is_error": bool}``
  (user turns)

Providers that speak another wire format translate from/to this shape. An assistant turn
whose content is a *list* is treated as provider-originated (``LLMResult.raw_content``
echoed back in an agent loop) and is forwarded verbatim.

This module has no dependency on the database, storage or HTTP layers (import-linter
contract "The LLM layer never touches the database or storage directly").
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from docassist.core.enums import Classification
from docassist.core.errors import PermissionDenied, ServiceUnavailable


class LLMTask(StrEnum):
    ANSWER = "answer"
    SUMMARIZE = "summarize"
    EXTRACT = "extract"
    CLASSIFY = "classify"
    COMPARE = "compare"
    RERANK = "rerank"
    AGENT = "agent"


FAST_TASKS: frozenset[LLMTask] = frozenset({LLMTask.CLASSIFY, LLMTask.RERANK, LLMTask.EXTRACT})
"""Tasks served by the fast model tier; every other task uses the main model."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool offered to the model. ``input_schema`` must be a strict JSON schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMRequest:
    task: LLMTask
    system: str
    messages: list[ChatMessage]
    output_schema: dict[str, Any] | None = None
    max_output_tokens: int | None = None
    data_classification: Classification = Classification.INTERNAL
    tools: list[ToolSpec] | None = None
    pseudonymize: bool = True


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Normalised provider response.

    ``text``, ``data`` and ``tool_calls`` are in the *application* domain (pseudonyms
    restored). ``raw_content`` is the assistant turn in the *provider* domain (exactly what
    the provider produced, pseudonyms intact) and exists only to be echoed back unchanged in
    the next turn of the same agent loop - never display it.
    """

    text: str
    data: dict[str, Any] | None
    tool_calls: list[ToolCall]
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    latency_ms: int
    pseudonymized: int
    raw_content: list[dict[str, Any]] = field(default_factory=list)
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class LLMError(ServiceUnavailable):
    """The model call failed in a way that retrying the same request will not fix."""

    status_code = 502
    code = "llm_error"
    title = "AI service error"
    default_message = "The AI service could not complete the request."

    # Token usage already consumed when the failure happened (set by providers when known).
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None

    def with_usage(self, input_tokens: int, output_tokens: int, model: str | None) -> LLMError:
        self.input_tokens = max(0, input_tokens)
        self.output_tokens = max(0, output_tokens)
        self.model = model
        return self


class LLMUnavailable(LLMError):
    """Transient: timeout, connection failure, rate limit, 5xx or an open circuit."""

    status_code = 503
    code = "llm_unavailable"
    title = "AI service unavailable"
    default_message = "The AI service is temporarily unavailable. Please retry later."


class LLMPolicyDenied(PermissionDenied):
    """The data-governance policy forbids sending this content to any configured model."""

    code = "llm_policy_denied"
    default_message = "The data-governance policy does not allow AI processing of this content."


class LLMOutputInvalid(LLMError):
    """The model output was truncated or did not match the requested schema."""

    code = "llm_output_invalid"
    default_message = "The AI service returned an invalid response."


class LLMRefused(LLMError):
    """The model declined the request (``stop_reason == "refusal"``)."""

    status_code = 422
    code = "llm_refused"
    title = "Request declined"
    default_message = "The AI model declined to process this request."


# --------------------------------------------------------------------------- #
# Collaborator protocols
# --------------------------------------------------------------------------- #
@runtime_checkable
class LLMProvider(Protocol):
    name: str
    is_external: bool
    supports_tools: bool

    async def complete(self, request: LLMRequest, *, model: str) -> LLMResult: ...

    async def aclose(self) -> None: ...


class UsageRecorder(Protocol):
    async def record(
        self,
        *,
        org_id: uuid.UUID,
        user_id: uuid.UUID | None,
        task: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: int,
        status: str,
    ) -> None: ...


class BudgetGuard(Protocol):
    async def check(self, org_id: uuid.UUID) -> None:
        """Raise :class:`docassist.core.errors.QuotaExceeded` when the org is over budget."""
        ...


class OrgLlmPolicy(Protocol):
    async def external_ceiling(self, org_id: uuid.UUID) -> Classification | None:
        """The organisation's own (lower or equal) external-classification ceiling, if any."""
        ...


class NullUsageRecorder:
    """Discards usage records (tests, CLI tools without a database)."""

    async def record(self, **_kwargs: Any) -> None:
        return None


class UnlimitedBudget:
    async def check(self, org_id: uuid.UUID) -> None:
        return None
