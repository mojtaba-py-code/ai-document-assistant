"""Authorised loading of documents, versions, chunks and extracted fields.

Every query here embeds :func:`docassist.authz.policy.readable_clause` in its SQL - chunks
and fields are joined to their document and filtered in the database, never fetched and
then filtered in Python. A document the caller cannot see raises :class:`NotFound`; one
the caller can list but not read raises :class:`PermissionDenied` (or :class:`Conflict`
while it is not ready), mirroring the platform-wide existence-hiding rule.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.authz.policy import listable_clause, readable_clause
from docassist.authz.principal import Principal
from docassist.core.enums import Classification, DocumentStatus, VersionStatus
from docassist.core.errors import Conflict, NotFound, PermissionDenied
from docassist.core.text import clean_line_text
from docassist.db.models import Document, DocumentChunk, DocumentVersion, ExtractedField
from docassist.intelligence.schemas import FieldValue
from docassist.intelligence.textops import format_decimal

MAX_CHUNKS = 5_000


@dataclass(frozen=True, slots=True)
class DocView:
    id: uuid.UUID
    org_id: uuid.UUID
    title: str
    doc_type: str
    classification: Classification
    status: str
    current_version_id: uuid.UUID | None
    version_count: int
    tags: tuple[str, ...]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class VersionView:
    id: uuid.UUID
    document_id: uuid.UUID
    number: int
    status: str
    page_count: int | None
    chunk_count: int | None
    language: str | None
    processed_at: datetime | None


@dataclass(frozen=True, slots=True)
class ChunkView:
    id: uuid.UUID
    index: int
    content: str
    page_start: int | None
    page_end: int | None
    section: str | None
    heading_path: tuple[str, ...] = ()
    injection_score: float = 0.0
    pii_types: tuple[str, ...] = ()
    char_start: int | None = None
    char_end: int | None = None


@dataclass(frozen=True, slots=True)
class FieldRow:
    id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    field: str
    value_text: str | None
    value_date: date | None
    value_number: Decimal | None
    currency: str | None
    confidence: float
    method: str
    chunk_id: uuid.UUID | None
    page: int | None
    evidence: str | None

    def display(self) -> str | None:
        """Human-readable value (date ISO, number with currency, or the text)."""
        if self.value_date is not None:
            return self.value_date.isoformat()
        if self.value_number is not None:
            number = format_decimal(self.value_number)
            return f"{number} {self.currency}" if self.currency else number
        if self.value_text:
            return clean_line_text(self.value_text, 1_000)
        return self.currency

    def to_value(self) -> FieldValue:
        return FieldValue(
            field=self.field,
            value=self.display(),
            value_text=clean_line_text(self.value_text, 1_000) if self.value_text else None,
            value_date=self.value_date,
            value_number=format_decimal(self.value_number)
            if self.value_number is not None
            else None,
            currency=self.currency,
            confidence=round(float(self.confidence), 4),
            method="llm" if self.method == "llm" else "rules",
            page=self.page,
            chunk_id=self.chunk_id,
            evidence=clean_line_text(self.evidence, 500) if self.evidence else None,
        )


def _doc_view(doc: Document) -> DocView:
    return DocView(
        id=doc.id,
        org_id=doc.organization_id,
        title=clean_line_text(doc.title, 300) or "Untitled",
        doc_type=doc.doc_type,
        classification=Classification(doc.classification),
        status=doc.status,
        current_version_id=doc.current_version_id,
        version_count=doc.version_count,
        tags=tuple(doc.tags or ()),
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


async def load_readable_document(
    session: AsyncSession, principal: Principal, document_id: uuid.UUID, now: datetime
) -> DocView:
    """The document if the principal may read its content, else the appropriate error."""
    doc = (
        await session.execute(
            select(Document).where(Document.id == document_id, readable_clause(principal, now))
        )
    ).scalar_one_or_none()
    if doc is not None:
        return _doc_view(doc)
    status = (
        await session.execute(
            select(Document.status).where(
                Document.id == document_id, listable_clause(principal, now)
            )
        )
    ).scalar_one_or_none()
    if status is None:
        raise NotFound(internal_detail="document not visible")
    if status != DocumentStatus.READY.value:
        raise Conflict(
            "The document is not ready for analysis.", internal_detail=f"status={status}"
        )
    raise PermissionDenied(
        "You don't have access to this document's content.",
        internal_detail="listable, not readable",
    )


async def load_version(
    session: AsyncSession, doc: DocView, version_number: int | None
) -> VersionView:
    """An indexed version of an already-authorised document (default: the current one)."""
    stmt = select(DocumentVersion).where(
        DocumentVersion.document_id == doc.id, DocumentVersion.organization_id == doc.org_id
    )
    if version_number is None:
        if doc.current_version_id is None:
            raise Conflict("The document has no indexed version yet.")
        stmt = stmt.where(DocumentVersion.id == doc.current_version_id)
    else:
        stmt = stmt.where(DocumentVersion.version_number == version_number)
    version = (await session.execute(stmt)).scalar_one_or_none()
    if version is None:
        raise NotFound("The requested version does not exist.")
    if version.status != VersionStatus.INDEXED.value:
        raise Conflict(
            "The requested version has not been indexed.",
            internal_detail=f"status={version.status}",
        )
    return VersionView(
        id=version.id,
        document_id=version.document_id,
        number=version.version_number,
        status=version.status,
        page_count=version.page_count,
        chunk_count=version.chunk_count,
        language=version.language,
        processed_at=version.processed_at,
    )


async def load_chunks(
    session: AsyncSession,
    principal: Principal,
    now: datetime,
    doc: DocView,
    version: VersionView,
    *,
    limit: int = MAX_CHUNKS,
) -> list[ChunkView]:
    """Chunks of one version in reading order - the readable clause is part of the query."""
    rows = (
        await session.execute(
            select(DocumentChunk)
            .join(
                Document,
                and_(
                    Document.id == DocumentChunk.document_id,
                    Document.organization_id == DocumentChunk.organization_id,
                ),
            )
            .where(
                DocumentChunk.document_id == doc.id,
                DocumentChunk.version_id == version.id,
                readable_clause(principal, now),
            )
            .order_by(DocumentChunk.chunk_index)
            .limit(limit)
        )
    ).scalars()
    return [
        ChunkView(
            id=row.id,
            index=row.chunk_index,
            content=row.content,
            page_start=row.page_start,
            page_end=row.page_end,
            section=clean_line_text(row.section, 300) if row.section else None,
            heading_path=tuple(clean_line_text(h, 300) for h in (row.heading_path or ())),
            injection_score=float(row.injection_score or 0.0),
            pii_types=tuple(row.pii_types or ()),
            char_start=row.char_start,
            char_end=row.char_end,
        )
        for row in rows
    ]


async def load_fields(
    session: AsyncSession,
    principal: Principal,
    now: datetime,
    doc: DocView,
    version_id: uuid.UUID,
    *,
    method: str | None = None,
) -> list[FieldRow]:
    """Extracted fields of one version (readable clause embedded), best first per field."""
    stmt = (
        select(ExtractedField)
        .join(
            Document,
            and_(
                Document.id == ExtractedField.document_id,
                Document.organization_id == ExtractedField.organization_id,
            ),
        )
        .where(
            ExtractedField.document_id == doc.id,
            ExtractedField.version_id == version_id,
            readable_clause(principal, now),
        )
        .order_by(ExtractedField.field, ExtractedField.confidence.desc(), ExtractedField.id)
        .limit(2_000)
    )
    if method is not None:
        stmt = stmt.where(ExtractedField.method == method)
    return [field_row(row) for row in (await session.execute(stmt)).scalars()]


def field_row(row: ExtractedField) -> FieldRow:
    return FieldRow(
        id=row.id,
        document_id=row.document_id,
        version_id=row.version_id,
        field=row.field,
        value_text=row.value_text,
        value_date=row.value_date,
        value_number=row.value_number,
        currency=row.currency,
        confidence=float(row.confidence),
        method=row.method,
        chunk_id=row.chunk_id,
        page=row.page,
        evidence=row.evidence,
    )


def best_per_field(rows: list[FieldRow]) -> dict[str, FieldRow]:
    """Highest-confidence row per field; LLM rows win ties (they are evidence-verified)."""
    best: dict[str, FieldRow] = {}
    for row in rows:
        current = best.get(row.field)
        if current is None or (row.confidence, row.method == "llm") > (
            current.confidence,
            current.method == "llm",
        ):
            best[row.field] = row
    return best
