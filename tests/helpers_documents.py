"""Shared helpers for the documents-area tests."""

from __future__ import annotations

import asyncio
import hashlib
import io
import struct
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, update

from docassist.core.context import utcnow
from docassist.core.enums import DocumentStatus, VersionStatus
from docassist.db.models import (
    AuditEvent,
    ChunkEmbedding,
    Document,
    DocumentChunk,
    DocumentVersion,
    ExtractedField,
    Job,
)
from docassist.db.session import DbContext

TEXT_BODY = b"Quarterly report\nRevenue grew in every region.\n"


@dataclass
class Tenant:
    """One organisation with two departments and a user of every tenant role."""

    org: uuid.UUID
    finance: uuid.UUID
    hr: uuid.UUID
    admin: Any
    manager: Any  # department manager of finance (RESTRICTED clearance)
    alice: Any  # employee in finance (CONFIDENTIAL clearance)
    carol: Any  # second employee in finance
    bob: Any  # employee in hr
    auditor: Any


async def make_tenant(factory: Any) -> Tenant:
    org = await factory.org()
    finance = await factory.department(org, name="Finance")
    hr = await factory.department(org, name="HR")
    return Tenant(
        org=org,
        finance=finance,
        hr=hr,
        admin=await factory.user(org, "organization_admin"),
        manager=await factory.user(org, "department_manager", managed=[finance]),
        alice=await factory.user(org, "employee", departments=[finance]),
        carol=await factory.user(org, "employee", departments=[finance]),
        bob=await factory.user(org, "employee", departments=[hr]),
        auditor=await factory.user(org, "auditor"),
    )


def unique_text(label: str = "doc") -> bytes:
    """Distinct content per call (duplicate detection would otherwise kick in)."""
    return f"{label} {uuid.uuid4()}\n".encode() + TEXT_BODY


class BytesSource:
    """In-memory :class:`~docassist.documents.scanning.ScanSource`."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    @property
    def size(self) -> int:
        return len(self.data)

    def open(self) -> io.BytesIO:
        return io.BytesIO(self.data)


async def stream_of(data: bytes, chunk: int = 64 * 1024) -> AsyncIterator[bytes]:
    for start in range(0, len(data), chunk):
        yield data[start : start + chunk]


async def upload_file(
    client: Any,
    headers: dict[str, str],
    content: bytes,
    filename: str,
    *,
    classification: str = "INTERNAL",
    content_type: str = "application/octet-stream",
    **fields: Any,
) -> Any:
    data = {"classification": classification}
    data.update({k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in fields.items()})
    return await client.post(
        "/api/v1/documents",
        headers=headers,
        data=data,
        files={"file": (filename, content, content_type)},
    )


async def service_upload(
    target: Any, principal: Any, content: bytes, filename: str, **kwargs: Any
) -> Any:
    """Upload through a ``DocumentService`` (or a container's ``documents`` service)."""
    service = getattr(target, "documents", target)
    kwargs.setdefault("classification", "INTERNAL")
    kwargs.setdefault("declared_mime", None)
    return await service.upload(principal, stream=stream_of(content), filename=filename, **kwargs)


async def make_ready(
    container: Any,
    org_id: uuid.UUID,
    document_id: uuid.UUID,
    *,
    chunks: list[tuple[str, int]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> uuid.UUID:
    """Simulate the ingestion worker: index the latest version and insert chunks,
    embeddings and one extracted field. Returns the version id."""
    chunks = chunks if chunks is not None else [("Revenue grew in every region.", 1)]
    dimensions = container.settings.embedding.dimensions
    async with container.db.transaction(DbContext.system_for_org(org_id)) as session:
        version = (
            await session.execute(
                select(DocumentVersion)
                .where(DocumentVersion.document_id == document_id)
                .order_by(DocumentVersion.version_number.desc())
                .limit(1)
            )
        ).scalar_one()
        version.status = VersionStatus.INDEXED.value
        version.chunk_count = len(chunks)
        version.page_count = max((page for _text, page in chunks), default=1)
        version.doc_metadata = metadata or {"title": "Report", "author": "Finance"}
        version.processed_at = utcnow()
        for index, (text, page) in enumerate(chunks):
            chunk_id = uuid.uuid4()
            session.add(
                DocumentChunk(
                    id=chunk_id,
                    organization_id=org_id,
                    document_id=document_id,
                    version_id=version.id,
                    chunk_index=index,
                    content=text,
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    page_start=page,
                    page_end=page,
                    section="Summary",
                    heading_path=["Summary"],
                    block_types=["paragraph"],
                    token_count=max(1, len(text) // 4),
                )
            )
            await session.flush()
            session.add(
                ChunkEmbedding(
                    chunk_id=chunk_id,
                    organization_id=org_id,
                    document_id=document_id,
                    version_id=version.id,
                    model="test",
                    embedding=[0.01] * dimensions,
                )
            )
        session.add(
            ExtractedField(
                organization_id=org_id,
                document_id=document_id,
                version_id=version.id,
                field="due_date",
                value_text="2026-10-14",
                confidence=0.9,
                method="rules",
            )
        )
        await session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(status=DocumentStatus.READY.value, current_version_id=version.id)
        )
        return version.id


async def row_counts(container: Any, org_id: uuid.UUID, document_id: uuid.UUID) -> dict[str, int]:
    async with container.db.session(DbContext.system_for_org(org_id)) as session:
        counts = {}
        for name, model in (
            ("chunks", DocumentChunk),
            ("embeddings", ChunkEmbedding),
            ("fields", ExtractedField),
        ):
            rows = (
                await session.execute(select(model).where(model.document_id == document_id))
            ).all()
            counts[name] = len(rows)
        return counts


async def audit_actions(container: Any, org_id: uuid.UUID) -> list[tuple[str, str, dict[str, Any]]]:
    async with container.db.session(DbContext.system_for_org(org_id)) as session:
        rows = (
            await session.execute(
                select(AuditEvent.action, AuditEvent.outcome, AuditEvent.details)
                .where(AuditEvent.organization_id == org_id)
                .order_by(AuditEvent.id)
            )
        ).all()
    return [(r[0], r[1], r[2]) for r in rows]


async def jobs_for(container: Any, org_id: uuid.UUID) -> list[Job]:
    async with container.db.session(DbContext.system_for_org(org_id)) as session:
        return list(
            (await session.execute(select(Job).where(Job.organization_id == org_id)))
            .scalars()
            .all()
        )


async def version_row(container: Any, org_id: uuid.UUID, version_id: uuid.UUID) -> DocumentVersion:
    async with container.db.session(DbContext.system_for_org(org_id)) as session:
        return (
            await session.execute(select(DocumentVersion).where(DocumentVersion.id == version_id))
        ).scalar_one()


@dataclass
class FakeClamd:
    """A minimal clamd speaking the INSTREAM protocol on an ephemeral local port."""

    reply: bytes = b"stream: OK\x00"
    delay: float = 0.0
    received: bytearray = field(default_factory=bytearray)
    commands: list[bytes] = field(default_factory=list)
    chunk_sizes: list[int] = field(default_factory=list)
    port: int = 0

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            command = await reader.readuntil(b"\x00")
            self.commands.append(command)
            if command == b"zINSTREAM\x00":
                while True:
                    (size,) = struct.unpack("!I", await reader.readexactly(4))
                    self.chunk_sizes.append(size)
                    if size == 0:
                        break
                    self.received += await reader.readexactly(size)
            elif command == b"zPING\x00":
                writer.write(b"PONG\x00")
                await writer.drain()
                return
            if self.delay:
                await asyncio.sleep(self.delay)
            writer.write(self.reply)
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    @asynccontextmanager
    async def running(self) -> AsyncIterator[FakeClamd]:
        server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = server.sockets[0].getsockname()[1]
        try:
            yield self
        finally:
            server.close()
            await server.wait_closed()
