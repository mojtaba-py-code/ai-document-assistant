"""``IntelligenceService`` - the facade other areas and the HTTP layer use.

Sub-services are exposed as attributes as well (``summarizer``, ``comparer``,
``deadlines``, ``extractor``, ``reports``, ``exports``) - the RAG deterministic deadline
path may call either :meth:`IntelligenceService.upcoming_deadlines` or
``container.intelligence.deadlines.upcoming``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from docassist.authz.principal import Principal
from docassist.core.context import utcnow
from docassist.intelligence.access import load_fields, load_readable_document, load_version
from docassist.intelligence.compare import Comparer
from docassist.intelligence.deadlines import DEFAULT_LIMIT, DeadlineService
from docassist.intelligence.exports import ExportService
from docassist.intelligence.extraction import Extractor
from docassist.intelligence.reports import ReportBuilder
from docassist.intelligence.schemas import (
    CompareRequest,
    ComparisonResult,
    DeadlineItem,
    DeadlineList,
    DocumentFields,
    ExtractionKind,
    ExtractionResult,
    ReportResponse,
    SummaryResult,
    SummaryStyle,
)
from docassist.intelligence.summarize import Summarizer

if TYPE_CHECKING:
    from docassist.api.container import Container


class IntelligenceService:
    def __init__(self, container: Container) -> None:
        self._c = container
        self.summarizer = Summarizer(container)
        self.comparer = Comparer(container)
        self.deadlines = DeadlineService(container)
        self.extractor = Extractor(container)
        self.reports = ReportBuilder(container, self.summarizer)
        self.exports = ExportService(container, self.reports)

    async def summarize(
        self, principal: Principal, document_id: uuid.UUID, *, style: SummaryStyle = "brief"
    ) -> SummaryResult:
        return await self.summarizer.summarize(principal, document_id, style=style)

    async def compare(self, principal: Principal, request: CompareRequest) -> ComparisonResult:
        return await self.comparer.compare(principal, request)

    async def upcoming_deadlines(
        self, principal: Principal, within_days: int, doc_type: str | None = None
    ) -> list[DeadlineItem]:
        """Contract used by the RAG deterministic path: readable documents only, UTC."""
        return await self.deadlines.upcoming(principal, within_days=within_days, doc_type=doc_type)

    async def deadline_window(
        self,
        principal: Principal,
        *,
        within_days: int,
        doc_type: str | None = None,
        fields: Sequence[str] | None = None,
        tz: str | None = None,
        overdue_days: int = 0,
        limit: int = DEFAULT_LIMIT,
    ) -> DeadlineList:
        return await self.deadlines.window(
            principal,
            within_days=within_days,
            doc_type=doc_type,
            fields=fields,
            tz=tz,
            overdue_days=overdue_days,
            limit=limit,
        )

    async def extract(
        self,
        principal: Principal,
        document_id: uuid.UUID,
        *,
        kind: ExtractionKind | None = None,
        persist: bool = True,
    ) -> ExtractionResult:
        return await self.extractor.extract(principal, document_id, kind=kind, persist=persist)

    async def document_fields(
        self, principal: Principal, document_id: uuid.UUID, *, version_number: int | None = None
    ) -> DocumentFields:
        """Stored fields (rules and LLM) of the current or a given version of a readable
        document, ordered by field name then confidence."""
        principal.require_org()
        now = utcnow()
        async with self._c.db.session(principal.db_context) as session:
            doc = await load_readable_document(session, principal, document_id, now)
            version = await load_version(session, doc, version_number)
            rows = await load_fields(session, principal, now, doc, version.id)
        return DocumentFields(
            document_id=doc.id,
            version_id=version.id,
            version_number=version.number,
            fields=[row.to_value() for row in rows],
        )

    async def report(
        self,
        principal: Principal,
        document_id: uuid.UUID,
        *,
        summary_mode: Literal["extractive", "llm"] = "extractive",
    ) -> ReportResponse:
        return await self.reports.report(principal, document_id, summary_mode=summary_mode)
