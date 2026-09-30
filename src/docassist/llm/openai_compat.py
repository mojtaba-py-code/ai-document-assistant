"""Chat completions over the OpenAI-compatible ``POST /chat/completions`` protocol.

Used for on-premises / local model servers (vLLM, Ollama ``/v1``, LM Studio, TGI). All
traffic goes through the SSRF-guarded client (:func:`docassist.security.ssrf.guarded_client`):
the base URL must be on the egress allowlist (a private server must be on
``outbound.private_network_allowlist``), redirects are refused and the response body is
read with a hard size cap.

Structured output uses ``response_format={"type": "json_schema", ...}``; tools are optional
(``tools`` + ``tool_choice="auto"``). The canonical Anthropic-shaped content blocks used
internally (see :mod:`docassist.llm.base`) are translated to and from the OpenAI wire shape.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

import httpx

from docassist.core.logging import get_logger
from docassist.llm.base import (
    ChatMessage,
    LLMError,
    LLMOutputInvalid,
    LLMRefused,
    LLMRequest,
    LLMResult,
    LLMUnavailable,
    ToolCall,
)
from docassist.security.ssrf import EgressDenied, EgressPolicy, guarded_client, validate_url

log = get_logger(__name__)

_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def to_openai_messages(system: str, messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Translate canonical messages (text or Anthropic-shaped blocks) to the OpenAI shape."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        if isinstance(message.content, str):
            out.append({"role": message.role, "content": message.content})
            continue
        if message.role == "assistant":
            text = "".join(b.get("text", "") for b in message.content if b.get("type") == "text")
            calls = [
                {
                    "id": str(b["id"]),
                    "type": "function",
                    "function": {
                        "name": str(b["name"]),
                        "arguments": json.dumps(b.get("input", {})),
                    },
                }
                for b in message.content
                if b.get("type") == "tool_use"
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
            continue
        texts: list[str] = []
        for block in message.content:
            kind = block.get("type")
            if kind == "tool_result":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id", "")),
                        "content": _tool_result_text(block.get("content")),
                    }
                )
            elif kind == "text":
                texts.append(str(block.get("text", "")))
        if texts:
            out.append({"role": "user", "content": "\n\n".join(texts)})
    return out


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return ""


class OpenAICompatibleChatProvider:
    supports_tools = True

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        egress: EgressPolicy,
        is_external: bool,
        name: str = "openai_compatible",
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        default_max_output_tokens: int = 8_000,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        validate_url(base_url, egress)  # fail fast on a non-allowlisted endpoint
        self.name = name
        self.is_external = is_external
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout_seconds
        self._retries = max_retries
        self._default_max_tokens = default_max_output_tokens
        self._client = client or guarded_client(egress)

    def build_body(self, request: LLMRequest, *, model: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": to_openai_messages(request.system, request.messages),
            "max_tokens": request.max_output_tokens or self._default_max_tokens,
            "stream": False,
        }
        if request.output_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": request.output_schema,
                    "strict": True,
                },
            }
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                        "strict": True,
                    },
                }
                for tool in request.tools
            ]
            body["tool_choice"] = "auto"
        return body

    async def complete(self, request: LLMRequest, *, model: str) -> LLMResult:
        body = self.build_body(request, model=model)
        started = time.perf_counter()
        payload = await self._post(body)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return self._parse(payload, request, model, latency_ms)

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        last: str = "no attempt"
        for attempt in range(self._retries + 1):
            try:
                status, raw = await self._send(body)
            except EgressDenied as exc:
                raise LLMError(internal_detail=f"egress denied: {exc}") from exc
            except httpx.HTTPError as exc:
                last = f"transport {type(exc).__name__}"
            else:
                if status == 200:
                    try:
                        payload = json.loads(raw)
                    except ValueError as exc:
                        raise LLMOutputInvalid(internal_detail="response is not JSON") from exc
                    if not isinstance(payload, dict):
                        raise LLMOutputInvalid(internal_detail="response is not a JSON object")
                    return payload
                if status not in _RETRYABLE:
                    raise LLMError(internal_detail=f"model server returned {status}")
                last = f"status {status}"
            if attempt < self._retries:
                await asyncio.sleep(min(10.0, 0.5 * 2**attempt) * random.uniform(0.8, 1.2))
        raise LLMUnavailable(internal_detail=f"model server unavailable ({last})")

    async def _send(self, body: dict[str, Any]) -> tuple[int, bytes]:
        async with self._client.stream(
            "POST", self._url, json=body, headers=self._headers, timeout=self._timeout
        ) as response:
            chunks = bytearray()
            async for chunk in response.aiter_bytes():
                chunks += chunk
                if len(chunks) > MAX_RESPONSE_BYTES:
                    raise LLMOutputInvalid(internal_detail="model server response too large")
            return response.status_code, bytes(chunks)

    def _parse(
        self, payload: dict[str, Any], request: LLMRequest, model: str, latency_ms: int
    ) -> LLMResult:
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        try:
            choice = payload["choices"][0]
            message = choice.get("message") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMOutputInvalid(internal_detail="response has no choices") from exc
        finish = str(choice.get("finish_reason") or "")
        served = str(payload.get("model") or model)
        if finish == "content_filter" or message.get("refusal"):
            raise LLMRefused(internal_detail="model server refusal").with_usage(
                input_tokens, output_tokens, served
            )
        if finish == "length" and (request.output_schema is not None or request.tools):
            raise LLMOutputInvalid(internal_detail="output truncated").with_usage(
                input_tokens, output_tokens, served
            )
        text = message.get("content") or ""
        if not isinstance(text, str):
            raise LLMOutputInvalid(internal_detail="message content is not text")
        tool_calls: list[ToolCall] = []
        raw: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        for call in message.get("tool_calls") or []:
            try:
                function = call["function"]
                arguments = json.loads(function.get("arguments") or "{}")
                call_id, name = str(call["id"]), str(function["name"])
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMOutputInvalid(internal_detail="malformed tool call") from exc
            if not isinstance(arguments, dict):
                raise LLMOutputInvalid(internal_detail="tool arguments are not an object")
            tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
            raw.append({"type": "tool_use", "id": call_id, "name": name, "input": arguments})
        data: dict[str, Any] | None = None
        if request.output_schema is not None:
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
            data = parsed if isinstance(parsed, dict) else None
        stop = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}.get(
            finish, finish
        )
        return LLMResult(
            text=text,
            data=data,
            tool_calls=tool_calls,
            model=served,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop,
            latency_ms=latency_ms,
            pseudonymized=0,
            raw_content=raw,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
