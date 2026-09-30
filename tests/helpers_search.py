"""Seeding helpers for search tests.

The ingestion pipeline is built in parallel, so tests write documents, versions, chunks and
embeddings directly through the ORM (as the app role, so RLS applies) and embed chunk text with
the same offline :class:`HashingEmbedder` configuration the test container uses.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import delete, insert, update

from docassist.core.text import estimate_tokens
from docassist.db.models import (
    ChunkEmbedding,
    Document,
    DocumentChunk,
    DocumentGrant,
    DocumentVersion,
)
from docassist.db.session import DbContext
from docassist.embeddings.hashing import HashingEmbedder
from docassist.search.sql import vector_param

EMBEDDER = HashingEmbedder(dimensions=1024, model="hashing-v1")


@dataclass(frozen=True)
class ChunkSpec:
    content: str
    section: str | None = None
    heading_path: tuple[str, ...] = ()
    page: int | None = 1
    injection_score: float = 0.0
    injection_flags: tuple[str, ...] = ()
    pii_types: tuple[str, ...] = ()


@dataclass
class SeededVersion:
    id: uuid.UUID
    number: int
    chunk_ids: list[uuid.UUID] = field(default_factory=list)


@dataclass
class SeededDoc:
    id: uuid.UUID
    org_id: uuid.UUID
    owner_id: uuid.UUID
    title: str
    versions: list[SeededVersion]

    @property
    def version(self) -> SeededVersion:
        return self.versions[-1]

    @property
    def chunk_ids(self) -> list[uuid.UUID]:
        return self.versions[-1].chunk_ids

    @property
    def all_chunk_ids(self) -> list[uuid.UUID]:
        return [c for v in self.versions for c in v.chunk_ids]


def _spec(item: str | ChunkSpec) -> ChunkSpec:
    return item if isinstance(item, ChunkSpec) else ChunkSpec(content=item)


async def _insert_version(
    session: Any,
    *,
    org_id: uuid.UUID,
    document_id: uuid.UUID,
    number: int,
    created_by: uuid.UUID,
    chunks: list[ChunkSpec],
    embed: bool,
    status: str,
    embedder: HashingEmbedder,
) -> SeededVersion:
    version = DocumentVersion(
        organization_id=org_id,
        document_id=document_id,
        version_number=number,
        storage_key=f"test/{uuid.uuid4().hex}",
        original_filename=f"v{number}.txt",
        extension="txt",
        detected_mime="text/plain",
        size_bytes=sum(len(c.content) for c in chunks),
        sha256=hashlib.sha256(uuid.uuid4().bytes).hexdigest(),
        status=status,
        chunk_count=len(chunks),
        semantic_indexed=embed,
        created_by=created_by,
    )
    session.add(version)
    await session.flush()
    seeded = SeededVersion(version.id, number)
    rows = []
    for index, spec in enumerate(chunks):
        chunk = DocumentChunk(
            organization_id=org_id,
            document_id=document_id,
            version_id=version.id,
            chunk_index=index,
            content=spec.content,
            content_sha256=hashlib.sha256(spec.content.encode()).hexdigest(),
            page_start=spec.page,
            page_end=spec.page,
            section=spec.section,
            heading_path=list(spec.heading_path),
            token_count=estimate_tokens(spec.content),
            injection_score=spec.injection_score,
            injection_flags=list(spec.injection_flags),
            pii_types=list(spec.pii_types),
        )
        session.add(chunk)
        rows.append((chunk, spec))
    await session.flush()
    for chunk, spec in rows:
        seeded.chunk_ids.append(chunk.id)
        if embed:
            await session.execute(
                insert(ChunkEmbedding).values(
                    chunk_id=chunk.id,
                    organization_id=org_id,
                    document_id=document_id,
                    version_id=version.id,
                    model=embedder.model,
                    embedding=vector_param(embedder.embed_one(spec.content)),
                )
            )
    return seeded


async def seed_document(
    container: Any,
    *,
    org_id: uuid.UUID,
    owner_id: uuid.UUID,
    chunks: list[str | ChunkSpec] | tuple[str | ChunkSpec, ...] = ("placeholder text",),
    title: str = "Test document",
    classification: str = "INTERNAL",
    department_id: uuid.UUID | None = None,
    allowed_roles: list[str] | tuple[str, ...] = (),
    tags: list[str] | tuple[str, ...] = (),
    doc_type: str = "other",
    status: str = "ready",
    version_status: str = "indexed",
    embed: bool = True,
    created_at: datetime | None = None,
    embedder: HashingEmbedder = EMBEDDER,
) -> SeededDoc:
    async with container.db.transaction(DbContext(org_id=org_id)) as session:
        doc = Document(
            organization_id=org_id,
            department_id=department_id,
            owner_id=owner_id,
            title=title,
            classification=classification,
            doc_type=doc_type,
            status=status,
            allowed_roles=list(allowed_roles),
            tags=list(tags),
            version_count=1,
        )
        if created_at is not None:
            doc.created_at = created_at
        session.add(doc)
        await session.flush()
        version = await _insert_version(
            session,
            org_id=org_id,
            document_id=doc.id,
            number=1,
            created_by=owner_id,
            chunks=[_spec(c) for c in chunks],
            embed=embed,
            status=version_status,
            embedder=embedder,
        )
        doc.current_version_id = version.id
        await session.flush()
        return SeededDoc(doc.id, org_id, owner_id, title, [version])


async def add_version(
    container: Any,
    doc: SeededDoc,
    chunks: list[str | ChunkSpec] | tuple[str | ChunkSpec, ...],
    *,
    make_current: bool = True,
    embed: bool = True,
    status: str = "indexed",
    embedder: HashingEmbedder = EMBEDDER,
) -> SeededVersion:
    async with container.db.transaction(DbContext(org_id=doc.org_id)) as session:
        number = doc.versions[-1].number + 1
        version = await _insert_version(
            session,
            org_id=doc.org_id,
            document_id=doc.id,
            number=number,
            created_by=doc.owner_id,
            chunks=[_spec(c) for c in chunks],
            embed=embed,
            status=status,
            embedder=embedder,
        )
        values: dict[str, Any] = {"version_count": number}
        if make_current:
            values["current_version_id"] = version.id
        await session.execute(update(Document).where(Document.id == doc.id).values(**values))
    doc.versions.append(version)
    return version


async def add_grant(
    container: Any,
    doc: SeededDoc,
    *,
    user_id: uuid.UUID | None = None,
    department_id: uuid.UUID | None = None,
    role: str | None = None,
    permission: str = "read",
    expires_at: datetime | None = None,
) -> uuid.UUID:
    grantee_type = "user" if user_id else "department" if department_id else "role"
    async with container.db.transaction(DbContext(org_id=doc.org_id)) as session:
        grant = DocumentGrant(
            organization_id=doc.org_id,
            document_id=doc.id,
            grantee_type=grantee_type,
            grantee_user_id=user_id,
            grantee_department_id=department_id,
            grantee_role=role,
            permission=permission,
            granted_by=doc.owner_id,
            expires_at=expires_at,
        )
        session.add(grant)
        await session.flush()
        return grant.id  # type: ignore[no-any-return]


async def update_document(container: Any, doc: SeededDoc, **values: Any) -> None:
    async with container.db.transaction(DbContext(org_id=doc.org_id)) as session:
        await session.execute(update(Document).where(Document.id == doc.id).values(**values))


async def update_grant(container: Any, doc: SeededDoc, grant_id: uuid.UUID, **values: Any) -> None:
    async with container.db.transaction(DbContext(org_id=doc.org_id)) as session:
        await session.execute(
            update(DocumentGrant).where(DocumentGrant.id == grant_id).values(**values)
        )


async def revoke_grant(container: Any, doc: SeededDoc, grant_id: uuid.UUID) -> None:
    async with container.db.transaction(DbContext(org_id=doc.org_id)) as session:
        await session.execute(delete(DocumentGrant).where(DocumentGrant.id == grant_id))


async def soft_delete(container: Any, doc: SeededDoc) -> None:
    """What the document service does on delete: status ``deleted`` + chunks removed."""
    async with container.db.transaction(DbContext(org_id=doc.org_id)) as session:
        await session.execute(
            update(Document).where(Document.id == doc.id).values(status="deleted")
        )
        await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))


async def update_version(
    container: Any, doc: SeededDoc, version_id: uuid.UUID, **values: Any
) -> None:
    async with container.db.transaction(DbContext(org_id=doc.org_id)) as session:
        await session.execute(
            update(DocumentVersion).where(DocumentVersion.id == version_id).values(**values)
        )


class FailingEmbedder:
    """Embedding provider that always fails (degraded-mode tests)."""

    name = "failing"
    model = "hashing-v1"
    dimensions = 1024
    is_external = True

    def __init__(self, exc: Exception | None = None) -> None:
        self.calls = 0
        self._exc = exc or RuntimeError("provider down")

    async def embed(self, texts: Any, *, kind: str) -> list[list[float]]:
        self.calls += 1
        raise self._exc

    async def aclose(self) -> None:
        return None


class CountingEmbedder(HashingEmbedder):
    """The offline embedder, counting calls (cache tests)."""

    def __init__(self) -> None:
        super().__init__(dimensions=1024, model="hashing-v1")
        self.calls = 0

    async def embed(self, texts: Any, *, kind: Any) -> list[list[float]]:
        self.calls += 1
        return await super().embed(texts, kind=kind)


class FailingVectorStore:
    name = "failing"

    async def query(self, *args: Any, **kwargs: Any) -> list[tuple[uuid.UUID, float]]:
        raise ConnectionError("vector store down")

    async def upsert_version(self, org_id: uuid.UUID, version_id: uuid.UUID) -> int:
        raise ConnectionError("vector store down")

    async def delete_document(self, org_id: uuid.UUID, document_id: uuid.UUID) -> int:
        raise ConnectionError("vector store down")

    async def sync_document(self, org_id: uuid.UUID, document_id: uuid.UUID) -> int:
        raise ConnectionError("vector store down")

    async def health(self) -> bool:
        return False

    async def aclose(self) -> None:
        return None
