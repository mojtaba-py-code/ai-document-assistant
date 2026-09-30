"""Shared helpers for the intelligence/export tests.

* :class:`FakeGateway` - a scripted stand-in for ``container.llm`` (``complete`` + ``route``)
  that records every request; install it with :func:`installed_gateway`, which restores the
  previous gateway afterwards.
* Seeding helpers insert documents, versions, chunks and extracted fields directly through
  the ORM (the ingestion pipeline is not needed for these tests).
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy import select, update

from docassist.core.enums import Classification
from docassist.core.text import estimate_tokens
from docassist.db.models import Document, DocumentChunk, DocumentVersion, ExtractedField, Job
from docassist.db.session import DbContext
from docassist.intelligence.llm_contract import LLMRequest, LLMResult
from docassist.jobs.queue import ClaimedJob, JobError
from docassist.jobs.registry import JobContext

# --------------------------------------------------------------------------- #
# Fake gateway
# --------------------------------------------------------------------------- #
Responder = Callable[[LLMRequest], dict[str, Any] | Exception]

_SOURCE_RE = re.compile(r'<source id="(C\d+)" nonce="([0-9a-f]+)"([^>]*)>\n(.*?)\n</source>', re.S)
_HUNK_RE = re.compile(r'<hunk id="(H\d+)"')
_FIELD_RE = re.compile(r'<field-change id="(F\d+)"')


def prompt_text(request: LLMRequest) -> str:
    parts = []
    for message in request.messages:
        content = message.content
        parts.append(content if isinstance(content, str) else json.dumps(content))
    return "\n".join(parts)


def sources_in(request: LLMRequest) -> dict[str, str]:
    """``{"C1": content, ...}`` of the spotlighted sources in a request (content escaped)."""
    return {m[1]: m[4] for m in _SOURCE_RE.finditer(prompt_text(request))}


def source_attrs(request: LLMRequest) -> dict[str, str]:
    return {m[1]: m[3] for m in _SOURCE_RE.finditer(prompt_text(request))}


def hunk_ids_in(request: LLMRequest) -> list[str]:
    text = prompt_text(request)
    return _FIELD_RE.findall(text) + _HUNK_RE.findall(text)


@dataclass
class FakeGateway:
    responder: Responder | None = None
    denied: set[Classification] = field(default_factory=set)
    requests: list[LLMRequest] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = "fake-model"
    provider: str = "fake"

    def route(self, classification: Classification) -> str | None:
        return None if classification in self.denied else self.provider

    async def complete(self, request: LLMRequest, *, org_id: Any, user_id: Any) -> LLMResult:
        self.requests.append(request)
        self.calls.append({"org_id": org_id, "user_id": user_id, "task": request.task})
        outcome: dict[str, Any] | Exception = self.responder(request) if self.responder else {}
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResult(
            text=json.dumps(outcome),
            data=outcome,
            tool_calls=[],
            model=self.model,
            provider=self.provider,
            input_tokens=estimate_tokens(request.system + prompt_text(request)),
            output_tokens=estimate_tokens(json.dumps(outcome)),
            stop_reason="end_turn",
            latency_ms=1,
            pseudonymized=0,
            raw_content=[],
        )


@contextmanager
def installed_gateway(container: Any, gateway: Any) -> Iterator[Any]:
    """Temporarily replace ``container.llm`` (``None`` simulates an unwired LLM area)."""
    had = "llm" in container.__dict__
    previous = container.__dict__.get("llm")
    container.llm = gateway
    try:
        yield gateway
    finally:
        if had:
            container.llm = previous
        else:
            container.__dict__.pop("llm", None)


@contextmanager
def installed_storage(container: Any) -> Iterator[Any]:
    """Make sure ``container.storage`` exists (the documents area wires it in production)."""
    from docassist.documents.storage import LocalEncryptedStorage

    if "storage" in container.__dict__:
        yield container.storage
        return
    container.storage = LocalEncryptedStorage(container.settings.storage.root, container.ring)
    try:
        yield container.storage
    finally:
        container.__dict__.pop("storage", None)


@contextmanager
def replaced_attribute(container: Any, name: str, value: Any) -> Iterator[Any]:
    had = name in container.__dict__
    previous = container.__dict__.get(name)
    setattr(container, name, value)
    try:
        yield value
    finally:
        if had:
            setattr(container, name, previous)
        else:
            container.__dict__.pop(name, None)


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
@dataclass
class ChunkSpec:
    content: str
    page: int | None = 1
    section: str | None = None
    injection_score: float = 0.0
    pii_types: tuple[str, ...] = ()
    char_start: int | None = None
    char_end: int | None = None


@dataclass
class FieldSpec:
    field: str
    value_text: str | None = None
    value_date: date | None = None
    value_number: Decimal | None = None
    currency: str | None = None
    confidence: float = 0.8
    method: str = "rules"
    chunk: int | None = 0
    evidence: str | None = None
    page: int | None = 1


@dataclass
class SeededDoc:
    id: uuid.UUID
    org_id: uuid.UUID
    version_ids: list[uuid.UUID]
    chunk_ids: list[list[uuid.UUID]]

    @property
    def current_version_id(self) -> uuid.UUID:
        return self.version_ids[-1]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def seed_document(
    container: Any,
    org_id: uuid.UUID,
    owner_id: uuid.UUID,
    *,
    title: str = "Master Services Agreement",
    classification: str = "INTERNAL",
    doc_type: str = "contract",
    department_id: uuid.UUID | None = None,
    status: str = "ready",
    versions: list[list[ChunkSpec]] | None = None,
    fields: dict[int, list[FieldSpec]] | None = None,
    allowed_roles: list[str] | None = None,
    current_index: int | None = None,
    tags: list[str] | None = None,
) -> SeededDoc:
    """Insert a document with indexed versions, chunks and fields (RLS-scoped to ``org_id``)."""
    versions = versions or [list(CONTRACT)]
    fields = fields or {}
    ctx = DbContext(org_id=org_id)
    async with container.db.transaction(ctx) as session:
        doc = Document(
            organization_id=org_id,
            department_id=department_id,
            owner_id=owner_id,
            title=title,
            classification=classification,
            doc_type=doc_type,
            status=status,
            version_count=len(versions),
            allowed_roles=allowed_roles or [],
            tags=tags or [],
        )
        session.add(doc)
        await session.flush()
        version_ids: list[uuid.UUID] = []
        chunk_ids: list[list[uuid.UUID]] = []
        for index, chunks in enumerate(versions):
            version = DocumentVersion(
                organization_id=org_id,
                document_id=doc.id,
                version_number=index + 1,
                storage_key=f"{org_id.hex}/aa/{uuid.uuid4().hex}",
                original_filename="doc.txt",
                extension="txt",
                detected_mime="text/plain",
                size_bytes=100,
                sha256=_sha(f"{doc.id}-{index}"),
                status="indexed",
                page_count=max((c.page or 1) for c in chunks) if chunks else 0,
                chunk_count=len(chunks),
                created_by=owner_id,
            )
            session.add(version)
            await session.flush()
            version_ids.append(version.id)
            rows = [
                DocumentChunk(
                    organization_id=org_id,
                    document_id=doc.id,
                    version_id=version.id,
                    chunk_index=position,
                    content=spec.content,
                    content_sha256=_sha(spec.content),
                    page_start=spec.page,
                    page_end=spec.page,
                    section=spec.section,
                    token_count=estimate_tokens(spec.content),
                    char_start=spec.char_start,
                    char_end=spec.char_end,
                    injection_score=spec.injection_score,
                    pii_types=list(spec.pii_types),
                )
                for position, spec in enumerate(chunks)
            ]
            session.add_all(rows)
            await session.flush()
            ids = [row.id for row in rows]
            chunk_ids.append(ids)
            for spec in fields.get(index, []):
                session.add(
                    ExtractedField(
                        organization_id=org_id,
                        document_id=doc.id,
                        version_id=version.id,
                        field=spec.field,
                        value_text=spec.value_text,
                        value_date=spec.value_date,
                        value_number=spec.value_number,
                        currency=spec.currency,
                        confidence=spec.confidence,
                        method=spec.method,
                        chunk_id=ids[spec.chunk] if spec.chunk is not None and ids else None,
                        page=spec.page,
                        evidence=spec.evidence,
                    )
                )
        current = version_ids[current_index if current_index is not None else -1]
        doc.current_version_id = current
        await session.flush()
        return SeededDoc(doc.id, org_id, version_ids, chunk_ids)


async def llm_field_rows(
    container: Any, org_id: uuid.UUID, version_id: uuid.UUID
) -> list[ExtractedField]:
    async with container.db.session(DbContext(org_id=org_id)) as session:
        rows = (
            await session.execute(
                select(ExtractedField)
                .where(ExtractedField.version_id == version_id)
                .order_by(ExtractedField.field, ExtractedField.id)
            )
        ).scalars()
        return list(rows)


async def set_document(container: Any, org_id: uuid.UUID, doc_id: uuid.UUID, **values: Any) -> None:
    async with container.db.transaction(DbContext(org_id=org_id)) as session:
        await session.execute(update(Document).where(Document.id == doc_id).values(**values))


# --------------------------------------------------------------------------- #
# Export jobs
# --------------------------------------------------------------------------- #
async def export_job(container: Any, export_id: uuid.UUID) -> Job | None:
    async with container.worker_db.session(DbContext.anonymous()) as session:
        return (
            await session.execute(select(Job).where(Job.idempotency_key == f"export:{export_id}"))
        ).scalar_one_or_none()


async def run_export_job(
    container: Any, export_id: uuid.UUID, *, attempts: int = 1, max_attempts: int = 5
) -> dict[str, Any] | JobError | None:
    """Execute the queued ``export_generate`` job of one export through its handler."""
    from docassist.intelligence.jobs import export_generate

    job = await export_job(container, export_id)
    assert job is not None, "export job was not enqueued"
    claimed = ClaimedJob(
        id=job.id,
        kind=job.kind,
        organization_id=job.organization_id,
        payload=dict(job.payload),
        attempts=attempts,
        max_attempts=max_attempts,
        request_id=job.request_id,
    )
    ctx = JobContext(container=container, worker_db=container.worker_db)
    try:
        result = await export_generate(ctx, claimed)
    except JobError as exc:
        status = "dead"
        outcome: dict[str, Any] | JobError | None = exc
    else:
        status = "succeeded"
        outcome = result
    async with container.worker_db.transaction(DbContext.anonymous()) as session:
        await session.execute(update(Job).where(Job.id == job.id).values(status=status))
    return outcome


# --------------------------------------------------------------------------- #
# Tenants and sample content
# --------------------------------------------------------------------------- #
CONTRACT = [
    ChunkSpec(
        "MASTER SERVICES AGREEMENT\nThis Master Services Agreement is made between Alpha Ltd "
        "and Beta Inc. It is effective from 1 January 2026 and expires on 31 December 2027.",
        page=1,
        section="Parties",
    ),
    ChunkSpec(
        "Payment terms: net 30 days. The total contract value is USD 120,000.00. Late payments "
        "incur a fee of 1.5% per month.",
        page=2,
        section="Payment",
    ),
    ChunkSpec(
        "This Agreement shall renew automatically for successive one-year terms unless either "
        "party gives ninety (90) days written notice. This Agreement is governed by the laws "
        "of England.",
        page=3,
        section="Term",
    ),
]


class Tenant:
    """One organisation with the usual cast of users."""

    def __init__(self) -> None:
        self.org: uuid.UUID
        self.dept: uuid.UUID
        self.other_dept: uuid.UUID
        self.admin: Any
        self.manager: Any
        self.employee: Any
        self.outsider: Any
        self.auditor: Any


async def tenant(factory: Any) -> Tenant:
    t = Tenant()
    t.org = await factory.org()
    t.dept = await factory.department(t.org)
    t.other_dept = await factory.department(t.org)
    t.admin = await factory.user(t.org, "organization_admin")
    t.manager = await factory.user(t.org, "department_manager", managed=[t.dept])
    t.employee = await factory.user(t.org, "employee", departments=[t.dept])
    t.outsider = await factory.user(t.org, "employee", departments=[t.other_dept])
    t.auditor = await factory.user(t.org, "auditor")
    return t
