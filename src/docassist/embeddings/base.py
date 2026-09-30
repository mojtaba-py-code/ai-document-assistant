"""Embedding provider abstraction.

Data-governance note: an embedding API receives the *full text* of every chunk, exactly like
an LLM does. ``is_external`` therefore feeds the same classification policy as the LLM
gateway - RESTRICTED text is never sent to an external embedding service.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

EmbeddingKind = Literal["document", "query"]


class EmbeddingError(Exception):
    """Transient provider failure (retryable)."""


class EmbeddingValidationError(EmbeddingError):
    """Provider returned something unusable (wrong count/dimension/non-finite values)."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimensions: int
    is_external: bool

    async def embed(self, texts: Sequence[str], *, kind: EmbeddingKind) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


def validate_vectors(
    vectors: list[list[float]], expected_count: int, dimensions: int
) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise EmbeddingValidationError(f"expected {expected_count} vectors, got {len(vectors)}")
    for vector in vectors:
        if len(vector) != dimensions:
            raise EmbeddingValidationError(f"expected dimension {dimensions}, got {len(vector)}")
        if not all(math.isfinite(v) for v in vector):
            raise EmbeddingValidationError("non-finite value in embedding")
    return vectors


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]
