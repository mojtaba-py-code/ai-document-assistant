"""Document intelligence endpoints: summaries, comparisons, deadlines, extraction, reports.

RBAC gates: ``intelligence:use`` for AI features and the deadline/report views;
``document:read`` for structured extraction and stored fields (persisting extracted
values additionally requires ``intelligence:use``, checked by the service). Document-level
access (the readable clause) is enforced inside every service query: invisible documents
are 404, listable-but-unreadable ones 403.
"""

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, Query, Response

from docassist.api.container import Container
from docassist.api.deps import get_container, require
from docassist.authz.permissions import Permission
from docassist.authz.principal import Principal
from docassist.core.enums import DocumentType
from docassist.intelligence.schemas import (
    CompareRequest,
    ComparisonResult,
    DeadlineList,
    DocumentFields,
    ExtractionResult,
    ExtractRequest,
    ReportResponse,
    SummarizeRequest,
    SummaryResult,
)

router = APIRouter(prefix="/api/v1/intelligence", tags=["intelligence"])

_use = require(Permission.INTELLIGENCE_USE)
_read = require(Permission.DOCUMENT_READ)


@router.post("/summarize", response_model=SummaryResult)
async def summarize(
    payload: SummarizeRequest,
    principal: Principal = Depends(_use),
    container: Container = Depends(get_container),
) -> SummaryResult:
    return await container.intelligence.summarize(
        principal, payload.document_id, style=payload.style
    )


@router.post("/compare", response_model=ComparisonResult)
async def compare(
    payload: CompareRequest,
    principal: Principal = Depends(_use),
    container: Container = Depends(get_container),
) -> ComparisonResult:
    return await container.intelligence.compare(principal, payload)


@router.get("/deadlines", response_model=DeadlineList)
async def deadlines(
    *,
    within_days: int = Query(90, ge=0, le=3_650),
    overdue_days: int = Query(0, ge=0, le=3_650),
    doc_type: DocumentType | None = Query(None),
    fields: list[str] | None = Query(None, max_length=20),
    tz: str = Query("UTC", min_length=1, max_length=64),
    limit: int = Query(200, ge=1, le=1_000),
    principal: Principal = Depends(_use),
    container: Container = Depends(get_container),
) -> DeadlineList:
    return await container.intelligence.deadline_window(
        principal,
        within_days=within_days,
        overdue_days=overdue_days,
        doc_type=doc_type.value if doc_type else None,
        fields=fields,
        tz=tz,
        limit=limit,
    )


@router.post("/extract", response_model=ExtractionResult)
async def extract(
    payload: ExtractRequest,
    principal: Principal = Depends(_read),
    container: Container = Depends(get_container),
) -> ExtractionResult:
    return await container.intelligence.extract(
        principal, payload.document_id, kind=payload.kind, persist=payload.persist
    )


@router.get("/documents/{document_id}/fields", response_model=DocumentFields)
async def document_fields(
    document_id: uuid.UUID,
    version: int | None = Query(None, ge=1, le=1_000_000),
    principal: Principal = Depends(_read),
    container: Container = Depends(get_container),
) -> DocumentFields:
    return await container.intelligence.document_fields(
        principal, document_id, version_number=version
    )


@router.get("/documents/{document_id}/report", response_model=ReportResponse)
async def document_report(
    document_id: uuid.UUID,
    summary: Literal["extractive", "llm"] = Query("extractive"),
    format: Literal["json", "markdown"] = Query("json"),  # noqa: A002 - public query name
    principal: Principal = Depends(_use),
    container: Container = Depends(get_container),
) -> ReportResponse | Response:
    result = await container.intelligence.report(principal, document_id, summary_mode=summary)
    if format == "markdown":
        return Response(
            content=result.markdown,
            media_type="text/markdown; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )
    return result
