"""Shared helpers for the LLM / RAG / assistant tests.

* :class:`ScriptedProvider` - an in-memory LLM provider that returns scripted results and
  records every request it receives (what an external provider would "see").
* :func:`make_gateway` - an :class:`LLMGateway` over given providers with in-memory usage,
  budget and an in-process rate limiter.
* :func:`seed_document` - inserts a ready document + current version + chunks (+ extracted
  fields) directly through the ORM, under the organisation's RLS context (the ingestion
  pipeline is not needed to test answering).
* :class:`KeywordRetriever` - a small, deterministic stand-in for ``SearchService.retrieve``
  that applies the real authorisation predicate (``readable_clause``) in SQL and ranks
  chunks by query-term overlap.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import and_, select

from docassist.authz.policy import readable_clause
from docassist.authz.principal import Principal
from docassist.cache.ratelimit import RateLimiter
from docassist.core.context import utcnow
from docassist.core.errors import QuotaExceeded
from docassist.db.models import (
    ChunkEmbedding,
    Document,
    DocumentChunk,
    DocumentVersion,
    ExtractedField,
)
from docassist.db.session import DbContext
from docassist.ingestion.injection import scan_for_injection
from docassist.llm.base import LLMRequest, LLMResult, ToolCall
from docassist.llm.gateway import LOCAL, MAIN, LLMGateway
from docassist.llm.local_extractive import terms
from docassist.search.types import RetrievalResult, RetrievedChunk, SearchFilters

Responder = Callable[[LLMRequest, str], LLMResult | Exception]


def result(
    *,
    text: str = "",
    data: dict[str, Any] | None = None,
    tool_calls: Sequence[ToolCall] = (),
    model: str = "fake-model",
    provider: str = "fake",
    input_tokens: int = 100,
    output_tokens: int = 20,
    raw_content: list[dict[str, Any]] | None = None,
) -> LLMResult:
    if data is not None and not text:
        text = json.dumps(data)
    raw = raw_content
    if raw is None:
        raw = ([{"type": "text", "text": text}] if text else []) + [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
            for c in tool_calls
        ]
    return LLMResult(
        text=text,
        data=data,
        tool_calls=list(tool_calls),
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason="tool_use" if tool_calls else "end_turn",
        latency_ms=1,
        pseudonymized=0,
        raw_content=raw,
    )


class ScriptedProvider:
    """Returns the scripted responses in order (or calls ``responder`` for each request)."""

    def __init__(
        self,
        responses: Sequence[LLMResult | Exception] = (),
        *,
        name: str = "fake",
        is_external: bool = True,
        supports_tools: bool = True,
        responder: Responder | None = None,
    ) -> None:
        self.name = name
        self.is_external = is_external
        self.supports_tools = supports_tools
        self._responses = list(responses)
        self._responder = responder
        self.requests: list[LLMRequest] = []
        self.models: list[str] = []
        self.closed = False

    async def complete(self, request: LLMRequest, *, model: str) -> LLMResult:
        self.requests.append(request)
        self.models.append(model)
        if self._responder is not None:
            outcome = self._responder(request, model)
        elif self._responses:
            outcome = self._responses.pop(0)
        else:
            raise AssertionError("ScriptedProvider ran out of responses")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        self.closed = True

    def seen_text(self) -> str:
        """Everything this provider received (system + messages), as one string."""
        parts: list[str] = []
        for request in self.requests:
            parts.append(request.system)
            for message in request.messages:
                parts.append(
                    message.content
                    if isinstance(message.content, str)
                    else json.dumps(message.content)
                )
        return "\n".join(parts)


@dataclass
class MemoryUsage:
    rows: list[dict[str, Any]] = field(default_factory=list)

    async def record(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


@dataclass
class MemoryBudget:
    exceeded: bool = False
    checks: int = 0

    async def check(self, org_id: uuid.UUID) -> None:
        self.checks += 1
        if self.exceeded:
            raise QuotaExceeded(internal_detail="test budget exceeded")


@dataclass
class FixedOrgPolicy:
    ceiling: Any = None

    async def external_ceiling(self, org_id: uuid.UUID) -> Any:
        return self.ceiling


def make_gateway(
    settings: Any,
    main: Any,
    local: Any = None,
    *,
    usage: MemoryUsage | None = None,
    budget: MemoryBudget | None = None,
    limiter: RateLimiter | None = None,
    org_policy: Any = None,
) -> LLMGateway:
    providers = {MAIN: main}
    if local is not None:
        providers[LOCAL] = local
    return LLMGateway(
        settings,
        providers,
        usage=usage or MemoryUsage(),
        budget=budget or MemoryBudget(),
        limiter=limiter or RateLimiter(None, "test", enabled=True),
        org_policy=org_policy,
    )


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChunkSpec:
    content: str
    page: int | None = 1
    section: str | None = None
    injection_score: float | None = None  # None: computed by the real scanner


@dataclass(frozen=True)
class FieldSpec:
    field: str
    value_date: date | None = None
    value_text: str | None = None
    evidence: str | None = None
    page: int | None = 1
    confidence: float = 0.9
    chunk_index: int | None = 0


@dataclass(frozen=True)
class SeededDocument:
    id: uuid.UUID
    version_id: uuid.UUID
    chunk_ids: list[uuid.UUID]
    title: str


async def seed_document(
    container: Any,
    *,
    org_id: uuid.UUID,
    owner_id: uuid.UUID,
    title: str,
    chunks: Sequence[ChunkSpec | str],
    classification: str = "INTERNAL",
    doc_type: str = "other",
    department_id: uuid.UUID | None = None,
    allowed_roles: Sequence[str] = (),
    fields: Sequence[FieldSpec] = (),
    status: str = "ready",
    tags: Sequence[str] = (),
    embed: bool = False,
) -> SeededDocument:
    specs = [c if isinstance(c, ChunkSpec) else ChunkSpec(c) for c in chunks]
    async with container.db.transaction(DbContext(org_id=org_id)) as session:
        document = Document(
            organization_id=org_id,
            department_id=department_id,
            owner_id=owner_id,
            title=title,
            classification=classification,
            doc_type=doc_type,
            doc_type_source="user",
            allowed_roles=list(allowed_roles),
            tags=list(tags),
            status=status,
            version_count=1,
        )
        session.add(document)
        await session.flush()
        body = "\n".join(s.content for s in specs).encode()
        version = DocumentVersion(
            organization_id=org_id,
            document_id=document.id,
            version_number=1,
            storage_key=f"test/{uuid.uuid4().hex}",
            original_filename="seed.txt",
            extension="txt",
            detected_mime="text/plain",
            size_bytes=len(body),
            sha256=hashlib.sha256(body + uuid.uuid4().bytes).hexdigest(),
            status="indexed",
            page_count=max((s.page or 1) for s in specs) if specs else 0,
            chunk_count=len(specs),
            created_by=owner_id,
        )
        session.add(version)
        await session.flush()
        chunk_ids: list[uuid.UUID] = []
        for index, spec in enumerate(specs):
            report = scan_for_injection(spec.content)
            score = report.score if spec.injection_score is None else spec.injection_score
            chunk = DocumentChunk(
                organization_id=org_id,
                document_id=document.id,
                version_id=version.id,
                chunk_index=index,
                content=spec.content,
                content_sha256=hashlib.sha256(spec.content.encode()).hexdigest(),
                page_start=spec.page,
                page_end=spec.page,
                section=spec.section,
                heading_path=[spec.section] if spec.section else [],
                token_count=max(1, len(spec.content) // 4),
                injection_score=score,
                injection_flags=list(report.flags),
            )
            session.add(chunk)
            await session.flush()
            chunk_ids.append(chunk.id)
            if embed:
                vector = (await container.embeddings.embed([spec.content], kind="document"))[0]
                session.add(
                    ChunkEmbedding(
                        chunk_id=chunk.id,
                        organization_id=org_id,
                        document_id=document.id,
                        version_id=version.id,
                        model=container.embeddings.model,
                        embedding=vector,
                    )
                )
        for spec in fields:
            session.add(
                ExtractedField(
                    organization_id=org_id,
                    document_id=document.id,
                    version_id=version.id,
                    field=spec.field,
                    value_text=spec.value_text,
                    value_date=spec.value_date,
                    confidence=spec.confidence,
                    method="rules",
                    chunk_id=chunk_ids[spec.chunk_index]
                    if spec.chunk_index is not None and chunk_ids
                    else None,
                    page=spec.page,
                    evidence=spec.evidence,
                )
            )
        document.current_version_id = version.id
        await session.flush()
        return SeededDocument(
            id=document.id, version_id=version.id, chunk_ids=chunk_ids, title=title
        )


# --------------------------------------------------------------------------- #
# Retrieval stand-in
# --------------------------------------------------------------------------- #
class KeywordRetriever:
    """Deterministic ``retrieve`` over seeded chunks with the real authorisation SQL."""

    def __init__(self, container: Any, *, exclude_threshold: float | None = None) -> None:
        self._container = container
        self._exclude = (
            exclude_threshold
            if exclude_threshold is not None
            else container.settings.retrieval.injection_exclude_threshold
        )
        self.calls: list[tuple[str, SearchFilters | None]] = []

    async def retrieve(
        self,
        principal: Principal,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        self.calls.append((query, filters))
        filters = filters or SearchFilters()
        now: datetime = utcnow()
        stmt = (
            select(DocumentChunk, Document, DocumentVersion.version_number)
            .join(
                Document,
                and_(
                    Document.organization_id == DocumentChunk.organization_id,
                    Document.id == DocumentChunk.document_id,
                ),
            )
            .join(
                DocumentVersion,
                and_(
                    DocumentVersion.organization_id == DocumentChunk.organization_id,
                    DocumentVersion.id == DocumentChunk.version_id,
                ),
            )
            .where(readable_clause(principal, now))
        )
        if not filters.include_old_versions:
            stmt = stmt.where(DocumentChunk.version_id == Document.current_version_id)
        if filters.doc_types:
            stmt = stmt.where(Document.doc_type.in_(filters.doc_types))
        if filters.document_ids:
            stmt = stmt.where(Document.id.in_(filters.document_ids))
        if filters.classifications:
            stmt = stmt.where(Document.classification.in_(filters.classifications))
        async with self._container.db.session(principal.db_context) as session:
            rows = (await session.execute(stmt)).all()
        wanted = set(terms(query))
        scored: list[tuple[float, RetrievedChunk]] = []
        excluded = 0
        for chunk, document, version_number in rows:
            words = set(terms(chunk.content + " " + document.title + " " + (chunk.section or "")))
            overlap = len(wanted & words) / len(wanted) if wanted else 0.0
            if overlap <= 0:
                continue
            if chunk.injection_score >= self._exclude:
                excluded += 1
                continue
            scored.append(
                (
                    overlap,
                    RetrievedChunk(
                        chunk_id=chunk.id,
                        document_id=document.id,
                        version_id=chunk.version_id,
                        version_number=version_number,
                        is_current=chunk.version_id == document.current_version_id,
                        document_title=document.title,
                        classification=document.classification,
                        doc_type=document.doc_type,
                        department_id=document.department_id,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        section=chunk.section,
                        heading_path=tuple(chunk.heading_path or ()),
                        content=chunk.content,
                        score=round(overlap, 4),
                        keyword_score=round(overlap, 4),
                        semantic_score=None,
                        injection_score=chunk.injection_score,
                        injection_flags=tuple(chunk.injection_flags or ()),
                        pii_types=tuple(chunk.pii_types or ()),
                        content_sha256=chunk.content_sha256,
                        chunk_index=chunk.chunk_index,
                    ),
                )
            )
        scored.sort(key=lambda item: (-item[0], str(item[1].chunk_id)))
        chunks = [chunk for _, chunk in scored[: top_k or 8]]
        return RetrievalResult(chunks, excluded_injection=excluded)


def chunk(
    content: str,
    *,
    title: str = "Doc",
    classification: str = "INTERNAL",
    score: float = 0.9,
    injection_score: float = 0.0,
    page: int | None = 1,
    section: str | None = None,
    document_id: uuid.UUID | None = None,
    is_current: bool = True,
) -> RetrievedChunk:
    """An in-memory RetrievedChunk for unit tests."""
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        version_id=uuid.uuid4(),
        version_number=1,
        is_current=is_current,
        document_title=title,
        classification=classification,
        doc_type="other",
        department_id=None,
        page_start=page,
        page_end=page,
        section=section,
        heading_path=(),
        content=content,
        score=score,
        keyword_score=score,
        semantic_score=None,
        injection_score=injection_score,
        injection_flags=(),
        pii_types=(),
    )


async def seed_golden(
    container: Any, factory: Any, dataset: Any, today: date
) -> dict[str, Principal]:
    """Load a golden dataset into fresh organisations; returns ``user_key -> Principal``."""
    from docassist.rag.evaluation import render_text

    orgs = {key: await factory.org() for key in sorted({u.org_key for u in dataset.users})}
    departments: dict[tuple[str, str], uuid.UUID] = {}

    async def department(org_key: str, name: str) -> uuid.UUID:
        if (org_key, name) not in departments:
            departments[(org_key, name)] = await factory.department(orgs[org_key], name=name)
        return departments[(org_key, name)]

    users: dict[str, Any] = {}
    for spec in dataset.users:
        users[spec.key] = await factory.user(
            orgs[spec.org_key],
            spec.role,
            departments=[await department(spec.org_key, d) for d in spec.departments],
            managed=[await department(spec.org_key, d) for d in spec.managed],
            clearance=spec.clearance,
        )
    for doc in dataset.documents:
        numbers = [page.number for page in doc.pages]
        await seed_document(
            container,
            org_id=orgs[doc.org_key],
            owner_id=users[doc.owner_key].id,
            title=doc.title,
            classification=doc.classification,
            doc_type=doc.doc_type,
            department_id=await department(doc.org_key, doc.department_key)
            if doc.department_key
            else None,
            chunks=[
                ChunkSpec(render_text(p.text, today), page=p.number, section=p.section)
                for p in doc.pages
            ],
            fields=[
                FieldSpec(
                    f.field,
                    value_date=today + timedelta(days=f.days_from_today),
                    evidence=render_text(f.evidence, today),
                    page=f.page,
                    chunk_index=numbers.index(f.page),
                )
                for f in doc.fields
            ],
        )
    return {key: await factory.principal(user) for key, user in users.items()}
