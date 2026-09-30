"""AnthropicProvider: exact request shapes, response parsing and error mapping.

The SDK (1.x) is built on httpx2, so it is exercised through
``anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))`` - the real SDK
code builds and sends the request; the handler captures the wire body.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2
import pytest

from docassist.llm.anthropic_provider import (
    FALLBACK_BETA,
    AnthropicProvider,
    echoable_content,
    supports_effort,
)
from docassist.llm.base import (
    ChatMessage,
    LLMError,
    LLMOutputInvalid,
    LLMRefused,
    LLMRequest,
    LLMTask,
    LLMUnavailable,
    ToolSpec,
)

API_KEY = "sk-ant-api03-" + "x" * 40
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string", "maxLength": 50}},
}
TOOL = ToolSpec(
    name="search_documents",
    description="Search",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {"query": {"type": "string", "maxLength": 500}},
    },
)


def _message(
    content: list[dict[str, Any]],
    *,
    stop: str = "end_turn",
    model: str = "claude-opus-5-5",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 50},
        **extra,
    }


class Capture:
    def __init__(
        self, status: int = 200, body: dict[str, Any] | None = None, exc: Exception | None = None
    ) -> None:
        self.status = status
        self.body = body or _message([{"type": "text", "text": '{"answer": "ok"}'}])
        self.exc = exc
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.exc is not None:
            raise self.exc
        return httpx2.Response(self.status, json=self.body, headers={"request-id": "req_1"})

    @property
    def last_body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


def provider(capture: Capture, **kwargs: Any) -> AnthropicProvider:
    http = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(capture))
    return AnthropicProvider(api_key=API_KEY, max_retries=0, http_client=http, **kwargs)


def request(**kwargs: Any) -> LLMRequest:
    base: dict[str, Any] = {
        "task": LLMTask.ANSWER,
        "system": "You are a careful assistant.",
        "messages": [ChatMessage(role="user", content="What are the payment terms?")],
    }
    base.update(kwargs)
    return LLMRequest(**base)


async def test_opus_request_shape_uses_effort_fallbacks_cache_and_json_schema() -> None:
    capture = Capture()
    p = provider(capture, effort="high")
    out = await p.complete(
        request(output_schema=SCHEMA, max_output_tokens=900), model="claude-opus-5-5"
    )
    sent = capture.requests[-1]
    body = capture.last_body
    assert sent.url.path == "/v1/messages"
    assert sent.url.params.get("beta") == "true"
    assert sent.headers["anthropic-beta"] == FALLBACK_BETA
    assert body["fallbacks"] == "default"
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 900
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["type"] == "json_schema"
    schema = body["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["answer"]
    answer = schema["properties"]["answer"]
    assert "maxLength" not in answer  # moved into the description by transform_schema
    assert "maxLength: 50" in answer["description"]
    assert body["system"] == [
        {
            "type": "text",
            "text": "You are a careful assistant.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    for forbidden in ("thinking", "temperature", "top_p", "top_k", "tool_choice", "tools"):
        assert forbidden not in body
    assert body["messages"][-1]["role"] == "user"
    assert out.data == {"answer": "ok"}
    assert out.input_tokens == 120 and out.output_tokens == 30 and out.cache_read_input_tokens == 50
    assert out.provider == "anthropic" and out.model == "claude-opus-5-5"


async def test_fallbacks_can_be_disabled() -> None:
    capture = Capture()
    await provider(capture, enable_server_fallbacks=False).complete(
        request(), model="claude-opus-5-5"
    )
    assert "fallbacks" not in capture.last_body
    assert "anthropic-beta" not in capture.requests[-1].headers
    assert capture.requests[-1].url.params.get("beta") is None


async def test_haiku_gets_no_effort_no_thinking_and_no_fallbacks() -> None:
    capture = Capture(body=_message([{"type": "text", "text": "fine"}], model="claude-haiku-4-5"))
    out = await provider(capture).complete(request(task=LLMTask.CLASSIFY), model="claude-haiku-4-5")
    body = capture.last_body
    assert body["model"] == "claude-haiku-4-5"
    assert "output_config" not in body
    assert "thinking" not in body and "fallbacks" not in body
    assert "anthropic-beta" not in capture.requests[-1].headers
    assert out.text == "fine" and out.data is None


def test_effort_support_table() -> None:
    assert supports_effort("claude-opus-5-5")
    assert supports_effort("claude-sonnet-5-5")
    assert not supports_effort("claude-haiku-4-5")


async def test_tools_are_strict_with_auto_choice_and_tool_use_is_parsed() -> None:
    capture = Capture(
        body=_message(
            [
                {"type": "thinking", "thinking": "", "signature": "sig-1"},
                {"type": "text", "text": "Let me search."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search_documents",
                    "input": {"query": "a/b é"},
                },
            ],
            stop="tool_use",
        )
    )
    out = await provider(capture).complete(
        request(task=LLMTask.AGENT, tools=[TOOL]), model="claude-opus-5-5"
    )
    body = capture.last_body
    assert body["tool_choice"] == {"type": "auto"}
    assert body["tools"][0]["strict"] is True
    assert body["tools"][0]["input_schema"]["additionalProperties"] is False
    assert out.text == "Let me search."  # thinking text is never part of the answer
    assert out.tool_calls[0].name == "search_documents"
    assert out.tool_calls[0].arguments == {"query": "a/b é"}  # parsed JSON, not string-matched
    # raw content keeps the thinking block (with its signature) for echoing back unchanged
    assert out.raw_content[0] == {"type": "thinking", "thinking": "", "signature": "sig-1"}
    assert out.stop_reason == "tool_use"


async def test_refusal_is_checked_before_content() -> None:
    body = _message(
        [{"type": "text", "text": "partial"}],
        stop="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": "x"},
    )
    with pytest.raises(LLMRefused) as info:
        await provider(Capture(body=body)).complete(request(), model="claude-opus-5-5")
    assert info.value.input_tokens == 120
    assert "cyber" in (info.value.internal_detail or "")


async def test_max_tokens_with_schema_is_invalid_output() -> None:
    body = _message([{"type": "text", "text": '{"answer": "trunc'}], stop="max_tokens")
    with pytest.raises(LLMOutputInvalid):
        await provider(Capture(body=body)).complete(
            request(output_schema=SCHEMA), model="claude-opus-5-5"
        )


async def test_max_tokens_without_schema_returns_text() -> None:
    body = _message([{"type": "text", "text": "long answer"}], stop="max_tokens")
    out = await provider(Capture(body=body)).complete(request(), model="claude-opus-5-5")
    assert out.text == "long answer" and out.stop_reason == "max_tokens"


async def test_unparseable_json_yields_no_data_for_the_gateway_to_repair() -> None:
    body = _message([{"type": "text", "text": "not json"}])
    out = await provider(Capture(body=body)).complete(
        request(output_schema=SCHEMA), model="claude-opus-5-5"
    )
    assert out.data is None


def test_echoable_content_drops_fallback_markers_and_declined_internal_blocks() -> None:
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "declined"},
        {"type": "text", "text": "partial "},
        {"type": "tool_use", "id": "t0", "name": "x", "input": {}},
        {
            "type": "fallback",
            "from": {"model": "claude-opus-5-5"},
            "to": {"model": "claude-opus-5"},
        },
        {"type": "thinking", "thinking": "", "signature": "kept"},
        {"type": "text", "text": "rest"},
    ]
    assert echoable_content(blocks) == [
        {"type": "text", "text": "partial "},
        {"type": "thinking", "thinking": "", "signature": "kept"},
        {"type": "text", "text": "rest"},
    ]


async def test_fallback_served_response_reports_the_serving_model() -> None:
    body = _message(
        [
            {
                "type": "fallback",
                "from": {"model": "claude-opus-5-5"},
                "to": {"model": "claude-opus-5"},
            },
            {"type": "text", "text": '{"answer": "ok"}'},
        ],
        model="claude-opus-5",
    )
    out = await provider(Capture(body=body)).complete(
        request(output_schema=SCHEMA), model="claude-opus-5-5"
    )
    assert out.model == "claude-opus-5"
    assert all(block["type"] != "fallback" for block in out.raw_content)


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (429, LLMUnavailable),
        (500, LLMUnavailable),
        (529, LLMUnavailable),
        (400, LLMError),
        (401, LLMError),
        (403, LLMError),
        (404, LLMError),
    ],
)
async def test_http_errors_are_mapped(status: int, error: type[Exception]) -> None:
    capture = Capture(
        status=status, body={"type": "error", "error": {"type": "x", "message": "boom"}}
    )
    with pytest.raises(error) as info:
        await provider(capture).complete(request(), model="claude-opus-5-5")
    if status < 500 and status != 429:
        assert not isinstance(info.value, LLMUnavailable)
    detail = str(info.value.internal_detail) + info.value.public_message
    assert API_KEY not in detail and "sk-ant" not in detail


@pytest.mark.parametrize("exc", [httpx2.ConnectError("refused"), httpx2.ReadTimeout("slow")])
async def test_transport_failures_are_unavailable(exc: Exception) -> None:
    with pytest.raises(LLMUnavailable):
        await provider(Capture(exc=exc)).complete(request(), model="claude-opus-5-5")


def test_api_key_is_required() -> None:
    with pytest.raises(ValueError, match="API key"):
        AnthropicProvider(api_key="")
