"""Search endpoint: ``POST /api/v1/search``.

Requires ``search:use``. Every result is a chunk the caller may read *now* (authorisation is
part of every retrieval query); documents the caller cannot read never influence the
response - not even through titles, counts or errors. Responses are ``Cache-Control:
no-store`` because snippets contain document text.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

from docassist.api.container import Container
from docassist.api.deps import get_container, require
from docassist.authz.permissions import Permission
from docassist.authz.principal import Principal
from docassist.core.enums import Classification, DocumentType
from docassist.search.types import MAX_FILTER_VALUES, MAX_TAG_CHARS, SearchFilters

router = APIRouter(prefix="/api/v1/search", tags=["search"])

MAX_QUERY_BODY_CHARS = 10_000
_search_user = require(Permission.SEARCH_USE)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchFiltersIn(_Strict):
    doc_types: list[DocumentType] = Field(default_factory=list, max_length=MAX_FILTER_VALUES)
    department_ids: list[uuid.UUID] = Field(default_factory=list, max_length=MAX_FILTER_VALUES)
    classifications: list[Classification] = Field(
        default_factory=list, max_length=MAX_FILTER_VALUES
    )
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=MAX_FILTER_VALUES)
    tags: list[Annotated[str, StringConstraints(min_length=1, max_length=MAX_TAG_CHARS)]] = Field(
        default_factory=list, max_length=MAX_FILTER_VALUES
    )
    created_after: AwareDatetime | None = None
    created_before: AwareDatetime | None = None
    include_old_versions: bool = False

    def to_filters(self) -> SearchFilters:
        return SearchFilters(
            doc_types=tuple(t.value for t in self.doc_types),
            department_ids=tuple(self.department_ids),
            classifications=tuple(c.value for c in self.classifications),
            document_ids=tuple(self.document_ids),
            tags=tuple(self.tags),
            created_after=self.created_after,
            created_before=self.created_before,
            include_old_versions=self.include_old_versions,
        )


class SearchRequest(_Strict):
    query: str = Field(min_length=1, max_length=MAX_QUERY_BODY_CHARS)
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    filters: SearchFiltersIn | None = None
    limit: int = Field(default=10, ge=1, le=50)


class SearchHit(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    version_number: int
    is_current: bool
    classification: str
    doc_type: str
    page_start: int | None
    page_end: int | None
    section: str | None
    snippet: str
    score: float
    keyword_score: float | None
    semantic_score: float | None
    flagged: bool


class SearchResponseOut(BaseModel):
    results: list[SearchHit]
    mode_used: str
    degraded: bool
    took_ms: float


@router.post("", response_model=SearchResponseOut)
async def search(
    payload: SearchRequest,
    response: Response,
    principal: Principal = Depends(_search_user),
    container: Container = Depends(get_container),
) -> SearchResponseOut:
    filters = payload.filters.to_filters() if payload.filters else SearchFilters()
    result = await container.search.search(
        principal, query=payload.query, mode=payload.mode, filters=filters, limit=payload.limit
    )
    response.headers["Cache-Control"] = "no-store"
    return SearchResponseOut(
        results=[
            SearchHit(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                document_title=r.document_title,
                version_number=r.version_number,
                is_current=r.is_current,
                classification=r.classification,
                doc_type=r.doc_type,
                page_start=r.page_start,
                page_end=r.page_end,
                section=r.section,
                snippet=r.snippet,
                score=r.score,
                keyword_score=r.keyword_score,
                semantic_score=r.semantic_score,
                flagged=r.flagged,
            )
            for r in result.results
        ],
        mode_used=result.mode_used,
        degraded=result.degraded,
        took_ms=result.took_ms,
    )
