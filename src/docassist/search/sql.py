"""Authorisation-scoped SQL building blocks shared by every retrieval query.

:func:`chunk_scope` is the single definition of "which chunks may this principal retrieve":

* the document policy :func:`docassist.authz.policy.readable_clause` (organisation, status
  READY, classification ceiling, ``allowed_roles``, owner / department / grant rules);
* the chunk belongs to an **indexed** version of that document (a quarantined or failed
  version never contributes text) and, unless ``include_old_versions`` is set, to the
  document's *current* version;
* the caller's :class:`~docassist.search.types.SearchFilters`, ANDed on top.

Every keyword, vector and hydration query embeds these predicates in its ``WHERE`` clause, so
an unauthorised row is never read into the application ("never retrieve-then-filter"). RLS
adds tenant isolation underneath as a second, independent layer.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import ColumnElement, Select, and_, bindparam, cast, false, select
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.types import String

from docassist.authz.policy import readable_clause
from docassist.authz.principal import Principal
from docassist.core.enums import VersionStatus
from docassist.core.text import clean_line_text
from docassist.db.models import ChunkEmbedding, Document, DocumentChunk, DocumentVersion
from docassist.search.types import RetrievedChunk, SearchFilters

MAX_TITLE_CHARS = 300
MAX_SECTION_CHARS = 300


def chunk_scope(
    principal: Principal, now: datetime, filters: SearchFilters
) -> list[ColumnElement[bool]]:
    """Predicates over ``DocumentChunk`` / ``Document`` / ``DocumentVersion`` (see module doc)."""
    if principal.org_id is None:
        return [false()]
    conditions: list[ColumnElement[bool]] = [
        DocumentChunk.organization_id == principal.org_id,
        readable_clause(principal, now),
        DocumentVersion.status == VersionStatus.INDEXED.value,
    ]
    if not filters.include_old_versions:
        conditions.append(DocumentChunk.version_id == Document.current_version_id)
    if filters.doc_types:
        conditions.append(Document.doc_type.in_(list(filters.doc_types)))
    if filters.department_ids:
        conditions.append(Document.department_id.in_(list(filters.department_ids)))
    if filters.classifications:
        conditions.append(Document.classification.in_(list(filters.classifications)))
    if filters.document_ids:
        conditions.append(Document.id.in_(list(filters.document_ids)))
    if filters.tags:
        conditions.append(Document.tags.overlap(cast(list(filters.tags), ARRAY(String(64)))))
    if filters.created_after is not None:
        conditions.append(Document.created_at >= filters.created_after)
    if filters.created_before is not None:
        conditions.append(Document.created_at < filters.created_before)
    return conditions


def join_documents[*Ts](stmt: Select[*Ts]) -> Select[*Ts]:
    """Join ``Document`` and ``DocumentVersion`` onto a statement that already has chunks.

    Joins use the composite ``(organization_id, id)`` keys so they can never pair rows of
    different tenants, mirroring the composite foreign keys of the schema.
    """
    return stmt.join(
        Document,
        and_(
            Document.id == DocumentChunk.document_id,
            Document.organization_id == DocumentChunk.organization_id,
        ),
    ).join(
        DocumentVersion,
        and_(
            DocumentVersion.id == DocumentChunk.version_id,
            DocumentVersion.organization_id == DocumentChunk.organization_id,
            DocumentVersion.document_id == DocumentChunk.document_id,
        ),
    )


def scoped[*Ts](
    stmt: Select[*Ts], principal: Principal, now: datetime, filters: SearchFilters
) -> Select[*Ts]:
    """``stmt`` (selecting from ``DocumentChunk``) joined and restricted to readable chunks."""
    return join_documents(stmt).where(*chunk_scope(principal, now, filters))


def vector_param(values: Sequence[float]) -> ColumnElement[Any]:
    """A query vector as ``CAST(:text AS vector)``.

    The value travels as text and PostgreSQL casts it, which works whether or not the
    asyncpg connection registered pgvector's binary codec (with the codec registered, the
    ORM ``Vector`` type's text bind value is rejected by the driver).
    """
    literal_text = "[" + ",".join(repr(float(v)) for v in values) + "]"
    return cast(bindparam(None, literal_text, type_=String()), Vector())


def to_floats(value: Any) -> list[float]:
    """pgvector values arrive as numpy arrays, ``Vector`` objects or lists - normalise them."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "to_list"):
        value = value.to_list()
    return [float(v) for v in value]


async def readable_chunk_ids(
    session: AsyncSession,
    principal: Principal,
    now: datetime,
    filters: SearchFilters,
    chunk_ids: Iterable[uuid.UUID],
) -> set[uuid.UUID]:
    """The subset of ``chunk_ids`` the principal may retrieve *right now* (authoritative)."""
    ids = list(dict.fromkeys(chunk_ids))
    if not ids:
        return set()
    stmt = scoped(
        select(DocumentChunk.id).where(DocumentChunk.id.in_(ids)), principal, now, filters
    )
    return set((await session.execute(stmt)).scalars().all())


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """A hydrated, authorised chunk row (display fields sanitised)."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    version_number: int
    is_current: bool
    document_title: str
    classification: str
    doc_type: str
    department_id: uuid.UUID | None
    page_start: int | None
    page_end: int | None
    section: str | None
    heading_path: tuple[str, ...]
    content: str
    content_sha256: str
    chunk_index: int
    injection_score: float
    injection_flags: tuple[str, ...]
    pii_types: tuple[str, ...]

    def retrieved(
        self, *, score: float, keyword_score: float | None, semantic_score: float | None
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            version_id=self.version_id,
            version_number=self.version_number,
            is_current=self.is_current,
            document_title=self.document_title,
            classification=self.classification,
            doc_type=self.doc_type,
            department_id=self.department_id,
            page_start=self.page_start,
            page_end=self.page_end,
            section=self.section,
            heading_path=self.heading_path,
            content=self.content,
            score=score,
            keyword_score=keyword_score,
            semantic_score=semantic_score,
            injection_score=self.injection_score,
            injection_flags=self.injection_flags,
            pii_types=self.pii_types,
            content_sha256=self.content_sha256,
            chunk_index=self.chunk_index,
        )


async def load_chunks(
    session: AsyncSession,
    principal: Principal,
    now: datetime,
    filters: SearchFilters,
    chunk_ids: Sequence[uuid.UUID],
) -> dict[uuid.UUID, ChunkRecord]:
    """Hydrate chunks by id - re-applying the authorisation scope (defence in depth)."""
    ids = list(dict.fromkeys(chunk_ids))
    if not ids:
        return {}
    stmt = scoped(
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.version_id,
            DocumentVersion.version_number,
            Document.current_version_id,
            Document.title,
            Document.classification,
            Document.doc_type,
            Document.department_id,
            DocumentChunk.page_start,
            DocumentChunk.page_end,
            DocumentChunk.section,
            DocumentChunk.heading_path,
            DocumentChunk.content,
            DocumentChunk.content_sha256,
            DocumentChunk.chunk_index,
            DocumentChunk.injection_score,
            DocumentChunk.injection_flags,
            DocumentChunk.pii_types,
        ).where(DocumentChunk.id.in_(ids)),
        principal,
        now,
        filters,
    )
    records: dict[uuid.UUID, ChunkRecord] = {}
    for row in (await session.execute(stmt)).all():
        section = clean_line_text(row.section, MAX_SECTION_CHARS) if row.section else None
        records[row.id] = ChunkRecord(
            chunk_id=row.id,
            document_id=row.document_id,
            version_id=row.version_id,
            version_number=int(row.version_number),
            is_current=row.version_id == row.current_version_id,
            document_title=clean_line_text(row.title, MAX_TITLE_CHARS),
            classification=row.classification,
            doc_type=row.doc_type,
            department_id=row.department_id,
            page_start=row.page_start,
            page_end=row.page_end,
            section=section or None,
            heading_path=tuple(
                h for h in (clean_line_text(p, MAX_SECTION_CHARS) for p in row.heading_path) if h
            ),
            content=row.content,
            content_sha256=row.content_sha256,
            chunk_index=int(row.chunk_index),
            injection_score=float(row.injection_score or 0.0),
            injection_flags=tuple(row.injection_flags or ()),
            pii_types=tuple(row.pii_types or ()),
        )
    return records


async def load_embeddings(
    session: AsyncSession,
    principal: Principal,
    chunk_ids: Sequence[uuid.UUID],
    *,
    model: str,
) -> dict[uuid.UUID, list[float]]:
    """Stored embeddings (of ``model``) for already-authorised chunk ids, for MMR."""
    ids = list(dict.fromkeys(chunk_ids))
    if not ids or principal.org_id is None:
        return {}
    stmt = select(ChunkEmbedding.chunk_id, ChunkEmbedding.embedding).where(
        ChunkEmbedding.organization_id == principal.org_id,
        ChunkEmbedding.model == model,
        ChunkEmbedding.chunk_id.in_(ids),
    )
    return {row.chunk_id: to_floats(row.embedding) for row in (await session.execute(stmt)).all()}
