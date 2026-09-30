"""Claude via the official ``anthropic`` SDK (1.x, built on ``httpx2``).

Request rules for the current model family (a violation is an HTTP 400 on Claude Opus 5.5):

* no ``thinking`` parameter - thinking is adaptive by default and cannot be disabled;
  depth is controlled with ``output_config.effort`` (models that accept it only - Claude
  Haiku 4.5 does not);
* no ``temperature`` / ``top_p`` / ``top_k``;
* no assistant prefill (the last message must be a user turn - enforced by the gateway);
* never a forced ``tool_choice`` - tools are sent with ``"strict": True`` and
  ``tool_choice={"type": "auto"}``;
* structured output via ``output_config.format`` (``json_schema``); the schema is passed
  through :func:`anthropic.transform_schema`, which keeps the constraints the API enforces
  and moves the rest (``maxLength``, ``maxItems``...) into descriptions - the gateway still
  validates the full schema locally;
* on models with server-side refusal fallbacks the call goes through
  ``client.beta.messages.create(..., betas=[...], fallbacks="default")``.

``stop_reason`` is checked *before* any content is read: ``refusal`` raises
:class:`LLMRefused`; ``max_tokens`` with a schema or tools raises
:class:`LLMOutputInvalid` (a truncated JSON document / tool call is never used).

The system prompt carries a ``cache_control`` breakpoint, so a stable system prompt that
reaches the model's minimum cacheable length is read from the prompt cache on repeated
calls (shorter prefixes are simply not cached).
"""

from __future__ import annotations

import json
import time
from typing import Any

import anthropic
import httpx2

from docassist.core.logging import get_logger
from docassist.llm.base import (
    LLMError,
    LLMOutputInvalid,
    LLMRefused,
    LLMRequest,
    LLMResult,
    LLMUnavailable,
    ToolCall,
)

log = get_logger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = frozenset(
    {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
)
"""Models on which ``fallbacks="default"`` is sent (when enabled in settings)."""

EFFORT_MODEL_PREFIXES = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-fable",
    "claude-mythos",
)
"""Models that accept ``output_config.effort`` (Claude Haiku 4.5 rejects it)."""

_MODEL_INTERNAL_BLOCKS = frozenset({"thinking", "redacted_thinking", "tool_use"})


def supports_effort(model: str) -> bool:
    return model.startswith(EFFORT_MODEL_PREFIXES)


def uses_server_fallbacks(model: str, enabled: bool) -> bool:
    return enabled and model in FALLBACK_MODELS


def echoable_content(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assistant content in the shape to echo back on the next agent turn.

    ``fallback`` blocks are audit markers and are dropped. After a mid-output fallback the
    declined model's thinking/tool_use blocks that precede the final marker must not be
    echoed; text blocks and everything after the boundary are kept unchanged (thinking
    blocks are never modified - they are bound to the exact conversation prefix).
    """
    boundary = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=-1)
    out: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        kind = block.get("type")
        if kind == "fallback":
            continue
        if index < boundary and kind in _MODEL_INTERNAL_BLOCKS:
            continue
        out.append(block)
    return out


class AnthropicProvider:
    name = "anthropic"
    is_external = True
    supports_tools = True

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        effort: str = "medium",
        enable_server_fallbacks: bool = True,
        default_max_output_tokens: int = 8_000,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("an Anthropic API key is required")
        self._effort = effort
        self._fallbacks = enable_server_fallbacks
        self._default_max_tokens = default_max_output_tokens
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout_seconds,
            "max_retries": max_retries,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if http_client is not None:
            kwargs["http_client"] = http_client
        self._client = anthropic.AsyncAnthropic(**kwargs)

    # ------------------------------------------------------------------ #
    def build_params(self, request: LLMRequest, *, model: str) -> dict[str, Any]:
        """The exact keyword arguments sent to ``messages.create`` (public for tests)."""
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_output_tokens or self._default_max_tokens,
            "system": [
                {"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        output_config: dict[str, Any] = {}
        if supports_effort(model):
            output_config["effort"] = self._effort
        if request.output_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": anthropic.transform_schema(request.output_schema),
            }
        if output_config:
            params["output_config"] = output_config
        if request.tools:
            params["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": anthropic.transform_schema(tool.input_schema),
                    "strict": True,
                }
                for tool in request.tools
            ]
            params["tool_choice"] = {"type": "auto"}
        return params

    async def complete(self, request: LLMRequest, *, model: str) -> LLMResult:
        params = self.build_params(request, model=model)
        started = time.perf_counter()
        try:
            if uses_server_fallbacks(model, self._fallbacks):
                response: Any = await self._client.beta.messages.create(
                    **params, betas=[FALLBACK_BETA], fallbacks="default"
                )
            else:
                response = await self._client.messages.create(**params)
        except anthropic.APIConnectionError as exc:  # includes APITimeoutError
            raise LLMUnavailable(
                internal_detail=f"anthropic connection: {type(exc).__name__}"
            ) from exc
        except anthropic.RateLimitError as exc:
            raise LLMUnavailable(internal_detail="anthropic rate limited (429)") from exc
        except anthropic.APIStatusError as exc:
            detail = f"anthropic status {exc.status_code} ({type(exc).__name__})"
            if exc.status_code >= 500:
                raise LLMUnavailable(internal_detail=detail) from exc
            raise LLMError(internal_detail=detail) from exc
        except anthropic.APIResponseValidationError as exc:
            raise LLMError(internal_detail="anthropic response failed validation") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        return self._parse(response, request, latency_ms)

    def _parse(self, response: Any, request: LLMRequest, latency_ms: int) -> LLMResult:
        usage = response.usage
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        served_model = str(response.model)
        stop_reason = str(response.stop_reason or "")

        # stop_reason is inspected BEFORE any content is read.
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            raise LLMRefused(internal_detail=f"refusal category={category}").with_usage(
                input_tokens, output_tokens, served_model
            )
        if stop_reason == "max_tokens" and (request.output_schema is not None or request.tools):
            raise LLMOutputInvalid(internal_detail="output truncated at max_tokens").with_usage(
                input_tokens, output_tokens, served_model
            )
        if stop_reason in {"model_context_window_exceeded", "pause_turn"}:
            raise LLMOutputInvalid(
                internal_detail=f"unusable stop_reason {stop_reason}"
            ).with_usage(input_tokens, output_tokens, served_model)

        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        raw: list[dict[str, Any]] = []
        for block in response.content:
            dumped = block.model_dump(mode="json", by_alias=True, exclude_none=True)
            raw.append(dumped)
            kind = getattr(block, "type", None)
            if kind == "text":
                texts.append(str(block.text))
            elif kind == "tool_use":
                arguments = block.input
                if not isinstance(arguments, dict):
                    raise LLMOutputInvalid(internal_detail="tool_use input is not an object")
                tool_calls.append(
                    ToolCall(id=str(block.id), name=str(block.name), arguments=arguments)
                )
            # thinking / redacted_thinking / fallback blocks carry no answer text
        text = "".join(texts)
        data: dict[str, Any] | None = None
        if request.output_schema is not None:
            data = _parse_json_object(text)
        return LLMResult(
            text=text,
            data=data,
            tool_calls=tool_calls,
            model=served_model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            latency_ms=latency_ms,
            pseudonymized=0,
            raw_content=echoable_content(raw),
            cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        )

    async def aclose(self) -> None:
        await self._client.close()


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None
