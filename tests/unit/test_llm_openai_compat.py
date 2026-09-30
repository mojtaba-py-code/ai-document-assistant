"""OpenAI-compatible local model server provider: wire format, parsing, retries, limits."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

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
from docassist.llm.openai_compat import (
    MAX_RESPONSE_BYTES,
    OpenAICompatibleChatProvider,
    to_openai_messages,
)
from docassist.security.ssrf import EgressDenied, EgressPolicy

BASE = "http://llm.internal:8000/v1"
POLICY = EgressPolicy.build([], ["llm.internal"])
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}
TOOL = ToolSpec(
    "lookup",
    "Look up",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["q"],
        "properties": {"q": {"type": "string"}},
    },
)


def completion(message: dict[str, Any], finish: str = "stop") -> dict[str, Any]:
    return {
        "model": "local-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 7},
    }


class Server:
    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def make(server: Server, **kwargs: Any) -> OpenAICompatibleChatProvider:
    return OpenAICompatibleChatProvider(
        base_url=BASE,
        api_key="local-key",
        egress=POLICY,
        is_external=False,
        max_retries=kwargs.pop("max_retries", 1),
        client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
        **kwargs,
    )


def req(**kwargs: Any) -> LLMRequest:
    base: dict[str, Any] = {
        "task": LLMTask.ANSWER,
        "system": "sys",
        "messages": [ChatMessage("user", "q")],
    }
    base.update(kwargs)
    return LLMRequest(**base)


async def test_structured_output_request_and_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    server = Server(
        httpx.Response(200, json=completion({"role": "assistant", "content": '{"answer": "yes"}'}))
    )
    out = await make(server).complete(
        req(output_schema=SCHEMA, max_output_tokens=300), model="local-model"
    )
    body = server.body()
    assert server.requests[0].url.path == "/v1/chat/completions"
    assert server.requests[0].headers["authorization"] == "Bearer local-key"
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": SCHEMA, "strict": True},
    }
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["max_tokens"] == 300 and body["stream"] is False
    assert "tools" not in body
    assert out.data == {"answer": "yes"} and out.input_tokens == 50 and out.output_tokens == 7
    assert out.provider == "openai_compatible" and out.stop_reason == "end_turn"


async def test_tool_calls_round_trip() -> None:
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q": "x"}'},
            }
        ],
    }
    server = Server(httpx.Response(200, json=completion(message, "tool_calls")))
    out = await make(server).complete(req(task=LLMTask.AGENT, tools=[TOOL]), model="m")
    body = server.body()
    assert body["tool_choice"] == "auto"
    assert body["tools"][0]["function"]["strict"] is True
    assert out.tool_calls[0].arguments == {"q": "x"}
    assert out.raw_content == [
        {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "x"}}
    ]
    assert out.stop_reason == "tool_use"


def test_canonical_blocks_are_translated() -> None:
    messages = [
        ChatMessage("user", "task"),
        ChatMessage(
            "assistant",
            [
                {"type": "text", "text": "looking"},
                {"type": "tool_use", "id": "c1", "name": "lookup", "input": {"q": 1}},
            ],
        ),
        ChatMessage(
            "user", [{"type": "tool_result", "tool_use_id": "c1", "content": "result text"}]
        ),
    ]
    out = to_openai_messages("sys", messages)
    assert out[2] == {
        "role": "assistant",
        "content": "looking",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q": 1}'},
            }
        ],
    }
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "result text"}


@pytest.mark.parametrize(
    ("message", "finish", "error"),
    [
        ({"role": "assistant", "content": "{", "refusal": None}, "length", LLMOutputInvalid),
        ({"role": "assistant", "content": None, "refusal": "no"}, "stop", LLMRefused),
        ({"role": "assistant", "content": ""}, "content_filter", LLMRefused),
    ],
)
async def test_truncation_and_refusal(
    message: dict[str, Any], finish: str, error: type[Exception]
) -> None:
    server = Server(httpx.Response(200, json=completion(message, finish)))
    with pytest.raises(error):
        await make(server).complete(req(output_schema=SCHEMA), model="m")


async def test_transient_errors_are_retried_then_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("docassist.llm.openai_compat.asyncio.sleep", _no_sleep)
    server = Server(httpx.Response(503), httpx.ConnectError("down"))
    with pytest.raises(LLMUnavailable):
        await make(server).complete(req(), model="m")
    assert len(server.requests) == 2


async def test_transient_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("docassist.llm.openai_compat.asyncio.sleep", _no_sleep)
    server = Server(
        httpx.Response(429),
        httpx.Response(200, json=completion({"role": "assistant", "content": "ok"})),
    )
    out = await make(server).complete(req(), model="m")
    assert out.text == "ok"


async def test_client_errors_are_not_retried() -> None:
    server = Server(httpx.Response(400, json={"error": "bad"}))
    with pytest.raises(LLMError) as info:
        await make(server).complete(req(), model="m")
    assert not isinstance(info.value, LLMUnavailable)
    assert len(server.requests) == 1


async def test_oversized_response_is_rejected() -> None:
    server = Server(httpx.Response(200, content=b"x" * (MAX_RESPONSE_BYTES + 1)))
    with pytest.raises(LLMOutputInvalid):
        await make(server).complete(req(), model="m")


async def test_malformed_payloads() -> None:
    for payload in (b"not json", b"[]", json.dumps({"choices": []}).encode()):
        server = Server(httpx.Response(200, content=payload))
        with pytest.raises(LLMOutputInvalid):
            await make(server).complete(req(), model="m")


def test_endpoint_must_be_on_the_egress_allowlist() -> None:
    with pytest.raises(EgressDenied):
        OpenAICompatibleChatProvider(
            base_url="http://evil.example/v1", api_key=None, egress=POLICY, is_external=False
        )


async def _no_sleep(_seconds: float) -> None:
    return None
