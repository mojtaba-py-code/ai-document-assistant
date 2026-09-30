"""Embeddings over the widely implemented ``POST /embeddings`` protocol.

Works with OpenAI-compatible servers (vLLM, Ollama ``/v1``, text-embeddings-inference,
Azure deployments behind a gateway, Voyage-style APIs). All traffic goes through the SSRF
guarded client: the base URL must be on the egress allowlist, redirects are refused, the
response is size-capped and every vector is validated before use.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Sequence
from typing import Any

import httpx

from docassist.embeddings.base import (
    EmbeddingError,
    EmbeddingKind,
    EmbeddingValidationError,
    validate_vectors,
)
from docassist.observability import metrics
from docassist.security.ssrf import EgressDenied, EgressPolicy, guarded_client, validate_url

_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatibleEmbedder:
    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimensions: int,
        api_key: str | None,
        egress: EgressPolicy,
        is_external: bool,
        batch_size: int = 64,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        send_dimensions: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        validate_url(base_url, egress)  # fail fast at startup on a non-allowlisted endpoint
        self.model = model
        self.dimensions = dimensions
        self.is_external = is_external
        self._url = base_url.rstrip("/") + "/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._batch = batch_size
        self._retries = max_retries
        self._send_dimensions = send_dimensions
        self._client = client or guarded_client(egress)
        self._timeout = timeout_seconds

    async def _post(self, texts: list[str]) -> list[list[float]]:
        body: dict[str, Any] = {"model": self.model, "input": texts, "encoding_format": "float"}
        if self._send_dimensions:
            body["dimensions"] = self.dimensions
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                response = await self._client.post(
                    self._url, json=body, headers=self._headers, timeout=self._timeout
                )
            except EgressDenied:
                raise
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    return self._parse(response, len(texts))
                if response.status_code not in _RETRYABLE:
                    raise EmbeddingError(f"embedding provider returned {response.status_code}")
                last_error = EmbeddingError(f"embedding provider returned {response.status_code}")
                retry_after = response.headers.get("retry-after", "")
                if retry_after.isdigit():
                    await asyncio.sleep(min(30, int(retry_after)))
                    continue
            if attempt < self._retries:
                await asyncio.sleep(min(20.0, 0.5 * 2**attempt) * random.uniform(0.8, 1.2))
        raise EmbeddingError("embedding provider unavailable") from last_error

    def _parse(self, response: httpx.Response, count: int) -> list[list[float]]:
        try:
            payload = response.json()
            items = sorted(payload["data"], key=lambda item: int(item["index"]))
            vectors = [[float(v) for v in item["embedding"]] for item in items]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingValidationError("malformed embedding response") from exc
        return validate_vectors(vectors, count, self.dimensions)

    async def embed(self, texts: Sequence[str], *, kind: EmbeddingKind) -> list[list[float]]:
        out: list[list[float]] = []
        started = time.perf_counter()
        for start in range(0, len(texts), self._batch):
            out.extend(await self._post(list(texts[start : start + self._batch])))
        metrics.EMBEDDING_LATENCY.labels(provider=self.name).observe(time.perf_counter() - started)
        return out

    async def aclose(self) -> None:
        await self._client.aclose()
