"""Deterministic deadline lookup over extracted fields (no model involved).

Answers "which contracts expire in the next 90 days?" from ``extracted_fields`` rows of the
*current* version of documents the principal can read - the authorisation predicate
(:func:`docassist.authz.policy.readable_clause`) is part of the SQL, so unreadable documents
are never fetched. Rows from several extraction methods for the same (document, field,
date) are merged, keeping the most confident one.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.authz.policy import readable_clause
from docassist.authz.principal import Principal
from docassist.core.enums import Classification
from docassist.db.models import Document, DocumentVersion, ExtractedField

DEADLINE_FIELDS: tuple[str, ...] = (
    "expiration_date",
    "expiry_date",
    "termination_date",
    "renewal_date",
    "due_date",
    "payment_due_date",
    "notice_deadline",
)
FIELD_LABELS = {
    "expiration_date": "expires",
    "expiry_date": "expires",
    "termination_date": "terminates",
    "renewal_date": "renews",
    "due_date": "is due",
    "payment_due_date": "payment is due",
    "notice_deadline": "notice deadline",
}
MAX_ROWS = 200


@dataclass(frozen=True, slots=True)
class DeadlineRow:
    document_id: uuid.UUID
    document_title: str
    doc_type: str
    classification: str
    version_number: int
    field: str
    due: date
    days_left: int
    evidence: str | None
    page: int | None
    chunk_id: uuid.UUID | None
    confidence: float

    @property
    def label(self) -> str:
        return FIELD_LABELS.get(self.field, self.field.replace("_", " "))


async def find_deadlines(
    session: AsyncSession,
    principal: Principal,
    *,
    start: date,
    end: date,
    today: date,
    now: datetime,
    doc_types: Sequence[str] = (),
    max_classification: Classification | None = None,
    limit: int = 50,
) -> list[DeadlineRow]:
    limit = max(1, min(limit, MAX_ROWS))
    field = ExtractedField
    stmt = (
        select(
            field,
            Document.title,
            Document.doc_type,
            Document.classification,
            DocumentVersion.version_number,
        )
        .join(
            Document,
            and_(
                Document.organization_id == field.organization_id,
                Document.id == field.document_id,
            ),
        )
        .join(
            DocumentVersion,
            and_(
                DocumentVersion.organization_id == field.organization_id,
                DocumentVersion.id == field.version_id,
            ),
        )
        .where(
            readable_clause(principal, now),
            field.version_id == Document.current_version_id,
            field.field.in_(DEADLINE_FIELDS),
            field.value_date.is_not(None),
            field.value_date >= start,
            field.value_date <= end,
        )
        .order_by(field.value_date, Document.title, field.id)
        .limit(limit * 4)
    )
    if doc_types:
        stmt = stmt.where(Document.doc_type.in_(list(doc_types)))
    if max_classification is not None:
        allowed = [c.value for c in Classification.at_most(max_classification)]
        stmt = stmt.where(Document.classification.in_(allowed))
    rows = (await session.execute(stmt)).all()
    best: dict[tuple[uuid.UUID, str, date], DeadlineRow] = {}
    for extracted, title, doc_type, classification, version_number in rows:
        due = extracted.value_date
        if due is None:
            continue
        candidate = DeadlineRow(
            document_id=extracted.document_id,
            document_title=title,
            doc_type=doc_type,
            classification=classification,
            version_number=version_number,
            field=extracted.field,
            due=due,
            days_left=(due - today).days,
            evidence=extracted.evidence,
            page=extracted.page,
            chunk_id=extracted.chunk_id,
            confidence=float(extracted.confidence),
        )
        key = (candidate.document_id, candidate.field, due)
        current = best.get(key)
        if current is None or candidate.confidence > current.confidence:
            best[key] = candidate
    ordered = sorted(best.values(), key=lambda r: (r.due, r.document_title, str(r.document_id)))
    return ordered[:limit]
