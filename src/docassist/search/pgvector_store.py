"""pgvector implementation of :class:`~docassist.search.vector_store.VectorStore`.

The similarity query joins ``chunk_embeddings -> document_chunks -> documents ->
document_versions`` and carries the complete authorisation scope
(:func:`docassist.search.sql.chunk_scope`) in its ``WHERE`` clause, ordered by
``embedding <=> :query`` (cosine distance) so the HNSW index can serve it.

Recall under restrictive filters: before pgvector 0.8 an HNSW scan yields at most
``hnsw.ef_search`` candidates *before* the WHERE clause is applied, so a tenant owning a small
share of a large shared table (or a user who may read few documents) can get fewer results
than exist. The store therefore raises ``hnsw.ef_search`` to at least the requested limit
(transaction-local ``set_config``) and, whenever the approximate query returns fewer rows than
requested, re-runs it as an **exact** query (``ORDER BY distance + 0`` is a sort key the HNSW
index cannot provide, so every row satisfying the authorisation predicates is ranked). Both
queries embed the same authorisation.

Embeddings live in PostgreSQL and are written by the ingestion pipeline in the chunk
transaction, so the synchronisation methods are no-ops.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import Select, and_, literal, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.authz.principal import Principal
from docassist.core.errors import ServiceUnavailable
from docassist.core.logging import get_logger
from docassist.db.models import ChunkEmbedding, DocumentChunk
from docassist.db.session import Database
from docassist.search.hybrid import clamp01
from docassist.search.sql import scoped, vector_param
from docassist.search.types import SearchFilters
from docassist.search.vector_store import check_vector

log = get_logger(__name__)

MAX_EF_SEARCH = 1_000
_SET_EF_SEARCH = text("SELECT set_config('hnsw.ef_search', :value, true)")


class PgVectorStore:
    name = "pgvector"

    def __init__(self, *, db: Database, model: str, dimensions: int, ef_search: int = 100) -> None:
        self._db = db
        self._model = model
        self._dimensions = dimensions
        self._ef_search = max(1, min(ef_search, MAX_EF_SEARCH))
        self._ef_supported: bool | None = None

    async def _configure(self, session: AsyncSession, limit: int) -> None:
        """``SET LOCAL hnsw.ef_search`` (tolerated when the server does not know the setting)."""
        if self._ef_supported is False:
            return
        value = {"value": str(max(self._ef_search, min(limit, MAX_EF_SEARCH)))}
        if self._ef_supported:
            await session.execute(_SET_EF_SEARCH, value)
            return
        try:
            async with session.begin_nested():  # first use: probe inside a savepoint
                await session.execute(_SET_EF_SEARCH, value)
        except DBAPIError:
            self._ef_supported = False
            log.warning("hnsw_ef_search_unavailable")
        else:
            self._ef_supported = True

    def statements(
        self,
        principal: Principal,
        vector: list[float],
        *,
        filters: SearchFilters,
        now: datetime,
        limit: int,
    ) -> tuple[Select[Any], Select[Any]]:
        """The ``(approximate, exact)`` similarity queries - identical authorisation scope."""
        distance = ChunkEmbedding.embedding.cosine_distance(vector_param(vector))
        stmt = scoped(
            select(ChunkEmbedding.chunk_id, distance.label("distance"))
            .select_from(ChunkEmbedding)
            .join(
                DocumentChunk,
                and_(
                    DocumentChunk.id == ChunkEmbedding.chunk_id,
                    DocumentChunk.organization_id == ChunkEmbedding.organization_id,
                ),
            ),
            principal,
            now,
            filters,
        ).where(
            ChunkEmbedding.organization_id == principal.org_id,
            ChunkEmbedding.model == self._model,
        )
        approximate = stmt.order_by(distance).limit(limit)
        # "+ 0" makes the sort key an expression the HNSW index cannot provide.
        exact = stmt.order_by(distance + literal(0.0), ChunkEmbedding.chunk_id).limit(limit)
        return approximate, exact

    async def query(
        self,
        session: AsyncSession,
        principal: Principal,
        vector: Sequence[float],
        *,
        filters: SearchFilters,
        limit: int,
        now: datetime,
    ) -> list[tuple[uuid.UUID, float]]:
        if principal.org_id is None or limit <= 0:
            return []
        query_vector = check_vector(vector, self._dimensions)
        await self._configure(session, limit)
        approximate, exact = self.statements(
            principal, query_vector, filters=filters, now=now, limit=limit
        )
        rows = (await session.execute(approximate)).all()
        if len(rows) < limit:
            rows = (await session.execute(exact)).all()
        ranked = sorted(rows, key=lambda row: (float(row.distance), str(row.chunk_id)))
        return [(row.chunk_id, clamp01(1.0 - float(row.distance))) for row in ranked]

    async def upsert_version(self, org_id: uuid.UUID, version_id: uuid.UUID) -> int:
        return 0

    async def delete_document(self, org_id: uuid.UUID, document_id: uuid.UUID) -> int:
        return 0

    async def sync_document(self, org_id: uuid.UUID, document_id: uuid.UUID) -> int:
        return 0

    async def health(self) -> bool:
        try:
            await self._db.ping()
        except ServiceUnavailable:
            return False
        return True

    async def aclose(self) -> None:
        return None
