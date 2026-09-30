"""Qdrant implementation of :class:`~docassist.search.vector_store.VectorStore` (optional extra).

This module imports ``qdrant_client`` at import time; :mod:`docassist.search.wiring` imports
it only when ``retrieval.backend == "qdrant"``, so pgvector deployments never need the extra.

Authorisation, twice:

1. **In the ANN query** - every point carries its document's ACL facts as payload
   (organisation, status, classification, owner, department, ``allowed_roles``, grant
   principals ``user:<id>`` / ``dept:<id>`` / ``role:<role>``, version status, current-version
   flag, doc type, tags, creation time). :func:`policy_filter` compiles the document policy
   into a Qdrant ``Filter`` (organisation as the tenant key), so the similarity search itself
   only walks points the principal could read.
2. **In PostgreSQL** - payloads can be stale (a revoked grant, a reclassified or deleted
   document not yet re-synchronised) and grant *expiry* is time-dependent, so every hit is
   re-verified with the authoritative SQL scope (:func:`docassist.search.sql.readable_chunk_ids`)
   in the caller's RLS-scoped session. Hits that fail are dropped; stale payloads can only
   ever *hide* results, never reveal them.

PostgreSQL is the source of truth for vectors too: synchronisation reads
``chunk_embeddings`` (the model in use) and writes points whose id is the chunk id.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.authz.principal import Principal
from docassist.core.config import Settings
from docassist.core.context import utcnow
from docassist.core.enums import (
    Classification,
    DocumentStatus,
    GranteeType,
    GrantPermission,
    VersionStatus,
)
from docassist.core.logging import get_logger
from docassist.db.models import (
    ChunkEmbedding,
    Document,
    DocumentChunk,
    DocumentGrant,
    DocumentVersion,
)
from docassist.db.session import Database, DbContext
from docassist.search.hybrid import clamp01
from docassist.search.sql import readable_chunk_ids, to_floats
from docassist.search.types import SearchFilters
from docassist.search.vector_store import VectorStoreConfigError, check_vector

log = get_logger(__name__)

_KEYWORD = models.PayloadSchemaType.KEYWORD
PAYLOAD_INDEXES: tuple[tuple[str, Any], ...] = (
    (
        "organization_id",
        models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
    ),
    ("document_id", _KEYWORD),
    ("version_id", _KEYWORD),
    ("status", _KEYWORD),
    ("version_status", _KEYWORD),
    ("classification", _KEYWORD),
    ("owner_id", _KEYWORD),
    ("department_id", _KEYWORD),
    ("allowed_roles", _KEYWORD),
    ("grant_principals", _KEYWORD),
    ("doc_type", _KEYWORD),
    ("tags", _KEYWORD),
    ("is_current", models.PayloadSchemaType.BOOL),
    ("created_at", models.PayloadSchemaType.FLOAT),
)
_LOCAL_INDEX_WARNING = "Payload indexes have no effect in the local Qdrant"
_READ_PERMISSIONS = frozenset({GrantPermission.READ.value, GrantPermission.MANAGE.value})


# --------------------------------------------------------------------------- #
# Policy -> Qdrant filter
# --------------------------------------------------------------------------- #
def _match(key: str, value: str | bool) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _any(key: str, values: Iterable[str]) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchAny(any=list(values)))


def principal_grant_keys(principal: Principal) -> list[str]:
    keys = [f"user:{principal.user_id}", f"role:{principal.role.value}"]
    keys.extend(f"dept:{d}" for d in sorted(principal.department_ids, key=str))
    return keys


def policy_filter(principal: Principal, filters: SearchFilters) -> models.Filter | None:
    """Compile the document read policy + search filters; ``None`` means "nothing visible"."""
    if principal.org_id is None:
        return None
    allowed = [c.value for c in Classification.at_most(principal.clearance)]
    if filters.classifications:
        allowed = [c for c in allowed if c in filters.classifications]
    if not allowed:
        return None
    access: list[Any] = [
        _any("classification", [Classification.PUBLIC.value, Classification.INTERNAL.value]),
        _match("owner_id", str(principal.user_id)),
        _any("grant_principals", principal_grant_keys(principal)),
    ]
    if principal.department_ids:
        access.append(
            models.Filter(
                must=[
                    _match("classification", Classification.CONFIDENTIAL.value),
                    _any("department_id", sorted(str(d) for d in principal.department_ids)),
                ]
            )
        )
    must: list[Any] = [
        _match("organization_id", str(principal.org_id)),
        _match("status", DocumentStatus.READY.value),
        _match("version_status", VersionStatus.INDEXED.value),
        _any("classification", allowed),
        models.Filter(
            should=[
                models.IsEmptyCondition(is_empty=models.PayloadField(key="allowed_roles")),
                _any("allowed_roles", [principal.role.value]),
            ]
        ),
        models.Filter(should=access),
    ]
    if not filters.include_old_versions:
        must.append(_match("is_current", True))
    if filters.doc_types:
        must.append(_any("doc_type", filters.doc_types))
    if filters.department_ids:
        must.append(_any("department_id", [str(d) for d in filters.department_ids]))
    if filters.document_ids:
        must.append(_any("document_id", [str(d) for d in filters.document_ids]))
    if filters.tags:
        must.append(_any("tags", filters.tags))
    if filters.created_after is not None or filters.created_before is not None:
        must.append(
            models.FieldCondition(
                key="created_at",
                range=models.Range(
                    gte=filters.created_after.timestamp() if filters.created_after else None,
                    lt=filters.created_before.timestamp() if filters.created_before else None,
                ),
            )
        )
    return models.Filter(must=must)


def _grant_key(grant: DocumentGrant) -> str | None:
    if grant.grantee_type == GranteeType.USER.value and grant.grantee_user_id:
        return f"user:{grant.grantee_user_id}"
    if grant.grantee_type == GranteeType.DEPARTMENT.value and grant.grantee_department_id:
        return f"dept:{grant.grantee_department_id}"
    if grant.grantee_type == GranteeType.ROLE.value and grant.grantee_role:
        return f"role:{grant.grantee_role}"
    return None


def document_payload(
    doc: Document, grants: Iterable[DocumentGrant], now: datetime
) -> dict[str, Any]:
    """Document-level ACL/filter payload (grants already expired at ``now`` are left out)."""
    principals = {
        key
        for g in grants
        if g.document_id == doc.id
        and g.permission in _READ_PERMISSIONS
        and (g.expires_at is None or g.expires_at > now)
        and (key := _grant_key(g)) is not None
    }
    return {
        "organization_id": str(doc.organization_id),
        "document_id": str(doc.id),
        "department_id": str(doc.department_id) if doc.department_id else None,
        "owner_id": str(doc.owner_id),
        "classification": doc.classification,
        "allowed_roles": sorted(doc.allowed_roles or []),
        "grant_principals": sorted(principals),
        "status": doc.status,
        "doc_type": doc.doc_type,
        "tags": sorted(doc.tags or []),
        "created_at": doc.created_at.timestamp(),
    }


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _DocumentState:
    org_id: uuid.UUID
    document_id: uuid.UUID
    status: str
    current_version_id: uuid.UUID | None
    payload: dict[str, Any]
    versions: dict[uuid.UUID, str]  # version id -> version status
    chunks: dict[uuid.UUID, uuid.UUID]  # chunk id (with an embedding of our model) -> version


class QdrantVectorStore:
    name = "qdrant"

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        db: Database,
        collection: str,
        dimensions: int,
        model: str,
        overfetch: int = 3,
        max_fetch: int = 1_000,
        batch_size: int = 128,
    ) -> None:
        self._client = client
        self._db = db
        self._collection = collection
        self._dimensions = dimensions
        self._model = model
        self._overfetch = max(1, overfetch)
        self._max_fetch = max(1, max_fetch)
        self._batch = max(1, batch_size)
        self._ready = False

    @classmethod
    def from_settings(
        cls, settings: Settings, *, db: Database, dimensions: int, model: str
    ) -> QdrantVectorStore:
        cfg = settings.retrieval
        if not cfg.qdrant_url:
            raise ValueError("retrieval.qdrant_url is required when retrieval.backend is 'qdrant'")
        client = AsyncQdrantClient(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key.get_secret_value() if cfg.qdrant_api_key else None,
            timeout=max(1, int(settings.outbound.timeout_seconds)),
            # No network round-trip at construction time; ensure_collection() verifies the
            # collection (and fails loudly on a mismatch) on first use instead.
            check_compatibility=False,
        )
        return cls(
            client, db=db, collection=cfg.qdrant_collection, dimensions=dimensions, model=model
        )

    # ------------------------------------------------------------------ #
    async def ensure_collection(self) -> None:
        """Create the collection and payload indexes once; verify the vector size."""
        if self._ready:
            return
        if not await self._client.collection_exists(self._collection):
            try:
                await self._client.create_collection(
                    self._collection,
                    vectors_config=models.VectorParams(
                        size=self._dimensions, distance=models.Distance.COSINE
                    ),
                    # Multitenancy: per-tenant HNSW graphs keyed by the is_tenant payload index.
                    hnsw_config=models.HnswConfigDiff(payload_m=16, m=0),
                )
            except Exception:
                if not await self._client.collection_exists(self._collection):
                    raise  # not a create race with another replica
        info = await self._client.get_collection(self._collection)
        size = getattr(info.config.params.vectors, "size", None)
        if size != self._dimensions:
            raise VectorStoreConfigError(
                f"collection {self._collection!r} has vector size {size}, expected "
                f"{self._dimensions}"
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_LOCAL_INDEX_WARNING)
            for field_name, schema in PAYLOAD_INDEXES:
                await self._client.create_payload_index(
                    self._collection, field_name, field_schema=schema, wait=True
                )
        self._ready = True

    async def query(
        self,
        session: AsyncSession,
        principal: Principal,
        vector: Sequence[float],
        *,
        filters: SearchFilters,
        limit: int,
        now: datetime,
    ) -> list[tuple[uuid.UUID, float]]:
        if principal.org_id is None or limit <= 0:
            return []
        query_vector = check_vector(vector, self._dimensions)
        query_filter = policy_filter(principal, filters)
        if query_filter is None:
            return []
        await self.ensure_collection()
        fetch = min(self._max_fetch, max(limit, limit * self._overfetch))
        response = await self._client.query_points(
            self._collection,
            query=query_vector,
            query_filter=query_filter,
            limit=fetch,
            with_payload=False,
            with_vectors=False,
        )
        hits = [(uuid.UUID(str(point.id)), clamp01(point.score)) for point in response.points]
        if not hits:
            return []
        verified = await readable_chunk_ids(session, principal, now, filters, [h[0] for h in hits])
        kept = [hit for hit in hits if hit[0] in verified]
        if len(kept) < len(hits):
            log.info("qdrant_stale_hits_dropped", dropped=len(hits) - len(kept))
        return kept[:limit]

    # ------------------------------------------------------------------ #
    # Synchronisation (PostgreSQL -> Qdrant)
    # ------------------------------------------------------------------ #
    def _filter(self, org_id: uuid.UUID, **equals: str) -> models.Filter:
        conditions: list[Any] = [_match("organization_id", str(org_id))]
        conditions.extend(_match(key, value) for key, value in equals.items())
        return models.Filter(must=conditions)

    async def _load(self, session: AsyncSession, document_id: uuid.UUID) -> _DocumentState | None:
        doc = await session.get(Document, document_id)
        if doc is None:
            return None
        grants = (
            (
                await session.execute(
                    select(DocumentGrant).where(DocumentGrant.document_id == doc.id)
                )
            )
            .scalars()
            .all()
        )
        versions = (
            await session.execute(
                select(DocumentVersion.id, DocumentVersion.status).where(
                    DocumentVersion.document_id == doc.id
                )
            )
        ).all()
        chunks = (
            await session.execute(
                select(DocumentChunk.id, DocumentChunk.version_id)
                .join(
                    ChunkEmbedding,
                    and_(
                        ChunkEmbedding.chunk_id == DocumentChunk.id,
                        ChunkEmbedding.organization_id == DocumentChunk.organization_id,
                    ),
                )
                .where(DocumentChunk.document_id == doc.id, ChunkEmbedding.model == self._model)
            )
        ).all()
        return _DocumentState(
            org_id=doc.organization_id,
            document_id=doc.id,
            status=doc.status,
            current_version_id=doc.current_version_id,
            payload=document_payload(doc, grants, utcnow()),
            versions={row.id: row.status for row in versions},
            chunks={row.id: row.version_id for row in chunks},
        )

    async def _upsert(self, state: _DocumentState, chunk_ids: Sequence[uuid.UUID]) -> int:
        written = 0
        for start in range(0, len(chunk_ids), self._batch):
            batch = list(chunk_ids[start : start + self._batch])
            async with self._db.session(DbContext.system_for_org(state.org_id)) as session:
                rows = (
                    await session.execute(
                        select(ChunkEmbedding.chunk_id, ChunkEmbedding.embedding).where(
                            ChunkEmbedding.chunk_id.in_(batch),
                            ChunkEmbedding.model == self._model,
                        )
                    )
                ).all()
            points = []
            for row in rows:
                version_id = state.chunks[row.chunk_id]
                payload = {
                    **state.payload,
                    "version_id": str(version_id),
                    "version_status": state.versions.get(version_id, ""),
                    "is_current": version_id == state.current_version_id,
                }
                vector = to_floats(row.embedding)
                if len(vector) != self._dimensions:
                    raise VectorStoreConfigError(
                        f"stored embedding has {len(vector)} dimensions, expected "
                        f"{self._dimensions}"
                    )
                points.append(
                    models.PointStruct(id=str(row.chunk_id), vector=vector, payload=payload)
                )
            if points:
                await self._client.upsert(self._collection, points=points, wait=True)
                written += len(points)
        return written

    async def _point_ids(self, org_id: uuid.UUID, document_id: uuid.UUID) -> set[uuid.UUID]:
        ids: set[uuid.UUID] = set()
        offset: Any = None
        while True:
            points, offset = await self._client.scroll(
                self._collection,
                scroll_filter=self._filter(org_id, document_id=str(document_id)),
                limit=512,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.update(uuid.UUID(str(p.id)) for p in points)
            if offset is None:
                return ids

    async def _reconcile(self, state: _DocumentState, fresh: set[uuid.UUID]) -> int:
        """Make the document's points match PostgreSQL; ``fresh`` ids were just upserted."""
        doc_filter = self._filter(state.org_id, document_id=str(state.document_id))
        await self._client.set_payload(
            self._collection,
            payload=state.payload,
            points=models.FilterSelector(filter=doc_filter),
            wait=True,
        )
        existing = await self._point_ids(state.org_id, state.document_id)
        stale = sorted(existing - set(state.chunks), key=str)
        if stale:
            await self._client.delete(
                self._collection,
                points_selector=models.PointIdsList(points=[str(i) for i in stale]),
                wait=True,
            )
        indexed_versions = {state.chunks[c] for c in existing | fresh if c in state.chunks}
        for version_id in sorted(indexed_versions, key=str):
            await self._client.set_payload(
                self._collection,
                payload={
                    "is_current": version_id == state.current_version_id,
                    "version_status": state.versions.get(version_id, ""),
                },
                points=models.FilterSelector(
                    filter=self._filter(
                        state.org_id,
                        document_id=str(state.document_id),
                        version_id=str(version_id),
                    )
                ),
                wait=True,
            )
        missing = sorted(set(state.chunks) - existing - fresh, key=str)
        written = await self._upsert(state, missing)
        return written + len(stale)

    async def upsert_version(self, org_id: uuid.UUID, version_id: uuid.UUID) -> int:
        await self.ensure_collection()
        async with self._db.session(DbContext.system_for_org(org_id)) as session:
            version = await session.get(DocumentVersion, version_id)
            state = await self._load(session, version.document_id) if version else None
        if version is None:
            # The version (and its document) is gone: remove anything still indexed for it.
            return await self._delete(self._filter(org_id, version_id=str(version_id)))
        if state is None or state.status == DocumentStatus.DELETED.value:
            return await self.delete_document(org_id, version.document_id)
        chunk_ids = sorted((c for c, v in state.chunks.items() if v == version_id), key=str)
        written = await self._upsert(state, chunk_ids)
        return written + await self._reconcile(state, set(chunk_ids))

    async def sync_document(self, org_id: uuid.UUID, document_id: uuid.UUID) -> int:
        await self.ensure_collection()
        async with self._db.session(DbContext.system_for_org(org_id)) as session:
            state = await self._load(session, document_id)
        if state is None or state.status == DocumentStatus.DELETED.value:
            return await self.delete_document(org_id, document_id)
        return await self._reconcile(state, set())

    async def _delete(self, selector_filter: models.Filter) -> int:
        count = await self._client.count(self._collection, count_filter=selector_filter, exact=True)
        if count.count:
            await self._client.delete(
                self._collection,
                points_selector=models.FilterSelector(filter=selector_filter),
                wait=True,
            )
        return int(count.count)

    async def delete_document(self, org_id: uuid.UUID, document_id: uuid.UUID) -> int:
        await self.ensure_collection()
        return await self._delete(self._filter(org_id, document_id=str(document_id)))

    async def health(self) -> bool:
        try:
            await self._client.get_collections()
        except Exception:  # noqa: BLE001 - any failure means "not healthy"
            return False
        return True

    async def aclose(self) -> None:
        await self._client.close()
