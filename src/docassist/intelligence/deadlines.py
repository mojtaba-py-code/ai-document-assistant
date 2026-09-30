"""Upcoming deadlines from extracted date fields, over readable documents only.

Window math
    ``today`` is the current date *in the caller's time zone* (UTC by default). A deadline
    is included when ``today - overdue_days <= date <= today + within_days`` (both ends
    inclusive) and ``days_left = date - today`` (negative when overdue).

Only fields of each document's *current* version count, the readable clause is part of
the SQL, and when both the rules extractor and the LLM extractor produced the same
(document, field, date) only the higher-confidence row is returned (LLM wins ties).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta, tzinfo
from typing import TYPE_CHECKING

from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.authz.policy import readable_clause
from docassist.authz.principal import Principal
from docassist.core.context import utcnow
from docassist.core.errors import ValidationFailed
from docassist.core.text import clean_line_text
from docassist.db.models import Document, ExtractedField
from docassist.intelligence.schemas import DeadlineItem, DeadlineList
from docassist.intelligence.textops import parse_timezone

if TYPE_CHECKING:
    from docassist.api.container import Container

DEADLINE_FIELDS: tuple[str, ...] = (
    "expiration_date",
    "expiry_date",
    "termination_date",
    "renewal_date",
    "due_date",
    "payment_due_date",
    "notice_deadline",
)
"""Fields that represent something that must happen (or ends) on a date."""

DATE_FIELDS: tuple[str, ...] = (*DEADLINE_FIELDS, "effective_date", "issue_date", "invoice_date")
"""Every date field a caller may ask for explicitly."""

MAX_WITHIN_DAYS = 3_650
DEFAULT_LIMIT = 200
MAX_LIMIT = 10_000


def resolve_fields(fields: Sequence[str] | None) -> tuple[str, ...]:
    """Requested fields (validated against :data:`DATE_FIELDS`), default the deadline set."""
    if not fields:
        return DEADLINE_FIELDS
    unknown = sorted(set(fields) - set(DATE_FIELDS))
    if unknown:
        raise ValidationFailed("Unknown date field.", extra={"allowed_fields": list(DATE_FIELDS)})
    return tuple(dict.fromkeys(fields))


def local_today(now: datetime, tz: tzinfo) -> date:
    """The calendar date at instant ``now`` in time zone ``tz``."""
    aware = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return aware.astimezone(tz).date()


def window(today: date, within_days: int, overdue_days: int = 0) -> tuple[date, date]:
    if not 0 <= within_days <= MAX_WITHIN_DAYS or not 0 <= overdue_days <= MAX_WITHIN_DAYS:
        raise ValidationFailed(f"Day ranges must be between 0 and {MAX_WITHIN_DAYS}.")
    return today - timedelta(days=overdue_days), today + timedelta(days=within_days)


async def query_deadlines(
    session: AsyncSession,
    principal: Principal,
    *,
    now: datetime,
    start: date,
    end: date,
    today: date,
    fields: Sequence[str],
    doc_type: str | None = None,
    document_ids: Sequence[uuid.UUID] = (),
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[DeadlineItem], bool]:
    """Deadline rows in ``[start, end]`` ordered by date; ``(items, truncated)``."""
    ef = ExtractedField
    rank = (
        func.row_number()
        .over(
            partition_by=(ef.document_id, ef.field, ef.value_date),
            order_by=(
                ef.confidence.desc(),
                case((ef.method == "llm", 0), else_=1),
                ef.id,
            ),
        )
        .label("rank")
    )
    conditions = [
        readable_clause(principal, now),
        ef.version_id == Document.current_version_id,
        ef.field.in_(list(fields)),
        ef.value_date.is_not(None),
        ef.value_date >= start,
        ef.value_date <= end,
    ]
    if doc_type:
        conditions.append(Document.doc_type == doc_type)
    if document_ids:
        conditions.append(ef.document_id.in_(list(document_ids)))
    ranked = (
        select(
            ef.id.label("field_id"),
            ef.document_id,
            ef.version_id,
            ef.field,
            ef.value_date,
            ef.value_text,
            ef.evidence,
            ef.page,
            ef.chunk_id,
            ef.confidence,
            ef.method,
            Document.title,
            Document.doc_type,
            Document.classification,
            rank,
        )
        .join(
            Document,
            and_(Document.id == ef.document_id, Document.organization_id == ef.organization_id),
        )
        .where(*conditions)
        .subquery()
    )
    rows = (
        await session.execute(
            select(ranked)
            .where(ranked.c.rank == 1)
            .order_by(ranked.c.value_date, ranked.c.title, ranked.c.field, ranked.c.field_id)
            .limit(limit + 1)
        )
    ).all()
    items = [
        DeadlineItem(
            document_id=row.document_id,
            document_title=clean_line_text(row.title, 300),
            doc_type=row.doc_type,
            classification=row.classification,
            version_id=row.version_id,
            field=row.field,
            date=row.value_date,
            days_left=(row.value_date - today).days,
            value_text=clean_line_text(row.value_text, 300) if row.value_text else None,
            evidence=clean_line_text(row.evidence, 500) if row.evidence else None,
            page=row.page,
            chunk_id=row.chunk_id,
            confidence=round(float(row.confidence), 4),
            method="llm" if row.method == "llm" else "rules",
        )
        for row in rows[:limit]
    ]
    return items, len(rows) > limit


class DeadlineService:
    """Deadline queries; also the contract used by the RAG deterministic deadline path."""

    def __init__(self, container: Container) -> None:
        self._c = container

    async def window(
        self,
        principal: Principal,
        *,
        within_days: int,
        doc_type: str | None = None,
        fields: Sequence[str] | None = None,
        tz: str | None = None,
        overdue_days: int = 0,
        limit: int = DEFAULT_LIMIT,
        now: datetime | None = None,
    ) -> DeadlineList:
        principal.require_org()
        zone = parse_timezone(tz)
        instant = now or utcnow()
        today = local_today(instant, zone)
        start, end = window(today, within_days, overdue_days)
        wanted = resolve_fields(fields)
        limit = max(1, min(limit, MAX_LIMIT))
        async with self._c.db.session(principal.db_context) as session:
            items, truncated = await query_deadlines(
                session,
                principal,
                now=instant,
                start=start,
                end=end,
                today=today,
                fields=wanted,
                doc_type=doc_type,
                limit=limit,
            )
        return DeadlineList(
            items=items,
            today=today,
            window_start=start,
            window_end=end,
            timezone=(tz or "UTC").strip() or "UTC",
            truncated=truncated,
        )

    async def upcoming(
        self,
        principal: Principal,
        *,
        within_days: int,
        doc_type: str | None = None,
        fields: Sequence[str] | None = None,
        tz: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[DeadlineItem]:
        """Deadlines from today (caller's zone) to ``today + within_days``, soonest first."""
        result = await self.window(
            principal, within_days=within_days, doc_type=doc_type, fields=fields, tz=tz, limit=limit
        )
        return result.items
