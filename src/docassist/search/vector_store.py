"""Vector store abstraction.

Contract for every implementation:

* ``query`` returns ``(chunk_id, cosine similarity clamped to 0..1)`` pairs, best first, and
  applies authorisation **inside** the retrieval: pgvector embeds the SQL policy in the
  similarity query; Qdrant pushes an equivalent payload filter into the ANN search and then
  re-verifies every hit against PostgreSQL (the authoritative source), dropping hits whose
  payload is stale.
* ``upsert_version`` / ``sync_document`` / ``delete_document`` keep an external index in step
  with PostgreSQL (the source of truth for embeddings and ACLs). They are no-ops for pgvector,
  whose embeddings are written by the ingestion pipeline in the same transaction as the
  chunks. They return the number of points written/removed.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession

from docassist.authz.principal import Principal
from docassist.search.types import SearchFilters


class VectorStoreError(Exception):
    """A vector store operation failed (transient: retry / degrade)."""


class VectorStoreConfigError(VectorStoreError):
    """The vector store is misconfigured (e.g. dimension mismatch) - retrying cannot help."""


@runtime_checkable
class VectorStore(Protocol):
    name: str

    async def query(
        self,
        session: AsyncSession,
        principal: Principal,
        vector: Sequence[float],
        *,
        filters: SearchFilters,
        limit: int,
        now: datetime,
    ) -> list[tuple[uuid.UUID, float]]: ...

    async def upsert_version(self, org_id: uuid.UUID, version_id: uuid.UUID) -> int: ...

    async def delete_document(self, org_id: uuid.UUID, document_id: uuid.UUID) -> int: ...

    async def sync_document(self, org_id: uuid.UUID, document_id: uuid.UUID) -> int: ...

    async def health(self) -> bool: ...

    async def aclose(self) -> None: ...


def check_vector(vector: Sequence[float], dimensions: int) -> list[float]:
    """Validate a query vector before it reaches a store (length and finiteness)."""
    values = [float(v) for v in vector]
    if len(values) != dimensions:
        raise VectorStoreConfigError(
            f"query vector has {len(values)} dimensions, the store expects {dimensions}"
        )
    if not all(math.isfinite(v) for v in values):
        raise VectorStoreError("query vector contains non-finite values")
    return values
