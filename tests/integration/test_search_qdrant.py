"""Qdrant vector store (in-memory client) against the real PostgreSQL schema.

* parity: for random organisations the Qdrant store returns exactly the chunks pgvector and
  the Python policy allow (current versions and all versions);
* stale payloads: reclassification, revoked/expired grants and deletion are enforced at query
  time by the PostgreSQL re-verification, before any re-sync happens;
* synchronisation: version switches, reconciliation of stale/missing points, deletions, the
  job handlers and configuration errors.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import delete, select

from docassist.authz.policy import DocumentFacts, can_read
from docassist.core.context import utcnow
from docassist.core.enums import Classification, GranteeType, GrantPermission, Role
from docassist.db.models import Document, DocumentChunk, DocumentGrant
from docassist.db.session import DbContext
from docassist.jobs.queue import ClaimedJob, PermanentJobError
from docassist.jobs.registry import JobContext
from docassist.search.jobs import vector_delete_document, vector_sync_document, vector_sync_version
from docassist.search.qdrant_store import QdrantVectorStore, policy_filter
from docassist.search.service import SearchService
from docassist.search.types import SearchFilters
from docassist.search.vector_store import VectorStoreConfigError
from tests.helpers_search import (
    EMBEDDER,
    add_grant,
    add_version,
    revoke_grant,
    seed_document,
    soft_delete,
    update_document,
    update_grant,
)

pytestmark = [pytest.mark.db]

QUERY = EMBEDDER.embed_one("service level agreement")
ALL_VERSIONS = SearchFilters(include_old_versions=True)
CURRENT = SearchFilters()


@pytest_asyncio.fixture
async def qdrant(container: Any) -> AsyncIterator[tuple[QdrantVectorStore, AsyncQdrantClient]]:
    client = AsyncQdrantClient(location=":memory:")
    store = QdrantVectorStore(
        client,
        db=container.db,
        collection=f"chunks_{uuid.uuid4().hex[:8]}",
        dimensions=1024,
        model="hashing-v1",
    )
    yield store, client
    await store.aclose()


async def _query(
    container: Any, store: Any, principal: Any, filters: SearchFilters = CURRENT
) -> set[uuid.UUID]:
    async with container.db.session(principal.db_context) as session:
        hits = await store.query(
            session, principal, QUERY, filters=filters, limit=500, now=utcnow()
        )
    return {chunk_id for chunk_id, _ in hits}


async def _raw(
    client: AsyncQdrantClient, store: QdrantVectorStore, principal: Any
) -> set[uuid.UUID]:
    """What the Qdrant payload filter alone would return (no PostgreSQL re-verification)."""
    flt = policy_filter(principal, SearchFilters())
    assert flt is not None
    response = await client.query_points(
        store._collection, query=QUERY, query_filter=flt, limit=500
    )
    return {uuid.UUID(str(p.id)) for p in response.points}


@pytest.mark.parametrize("seed", [5, 77])
async def test_parity_with_pgvector_and_python_policy(
    container, factory, qdrant, seed: int
) -> None:
    store, _ = qdrant
    rng = random.Random(seed)
    org, other_org = await factory.org(), await factory.org()
    depts = [await factory.department(org) for _ in range(3)]
    roles = [Role.ORGANIZATION_ADMIN, Role.DEPARTMENT_MANAGER, Role.EMPLOYEE]
    users = [
        await factory.user(
            org,
            rng.choice(roles).value,
            departments=rng.sample(depts, rng.randint(0, 2)),
            clearance=rng.choice(list(Classification)).value,
        )
        for _ in range(6)
    ]
    outsider = await factory.user(other_org, "organization_admin")
    now = utcnow()
    docs = []
    for i in range(20):
        doc = await seed_document(
            container,
            org_id=org,
            owner_id=rng.choice(users).id,
            classification=rng.choice(list(Classification)).value,
            department_id=rng.choice([None, *depts]),
            status=rng.choice(["ready", "ready", "ready", "processing", "failed"]),
            allowed_roles=rng.choice([[], [], ["employee"], ["department_manager"]]),
            chunks=[f"service level agreement {i} part {j}" for j in range(rng.randint(1, 3))],
        )
        if rng.random() < 0.3:
            await add_version(container, doc, [f"service level agreement {i} revised"])
        for _ in range(rng.choice([0, 1, 2])):
            kind = rng.choice(list(GranteeType))
            await add_grant(
                container,
                doc,
                user_id=rng.choice(users).id if kind is GranteeType.USER else None,
                department_id=rng.choice(depts) if kind is GranteeType.DEPARTMENT else None,
                role=rng.choice(roles).value if kind is GranteeType.ROLE else None,
                permission=rng.choice(list(GrantPermission)).value,
                expires_at=rng.choice([None, now - timedelta(hours=1), now + timedelta(days=3)]),
            )
        await store.sync_document(org, doc.id)
        docs.append(doc)
    async with container.db.session(DbContext(org_id=org)) as session:
        rows = (await session.execute(select(Document))).scalars().all()
        grants = (await session.execute(select(DocumentGrant))).scalars().all()
    facts = {row.id: DocumentFacts.from_row(row, grants) for row in rows}

    for user in [*users, outsider]:
        principal = await factory.principal(user)
        readable = [
            d for d in docs if principal.org_id == org and can_read(principal, facts[d.id], now)
        ]
        expected_current = {c for d in readable for c in d.chunk_ids}
        expected_all = {c for d in readable for c in d.all_chunk_ids}
        assert await _query(container, store, principal) == expected_current
        assert await _query(container, container.vector_store, principal) == expected_current
        assert await _query(container, store, principal, ALL_VERSIONS) == expected_all
        assert (
            await _query(container, container.vector_store, principal, ALL_VERSIONS) == expected_all
        )


async def test_stale_payloads_are_dropped_by_postgres_reverification(
    container, factory, qdrant
) -> None:
    store, client = qdrant
    org = await factory.org()
    dept = await factory.department(org)
    owner = await factory.user(org, "department_manager", departments=[dept])
    member = await factory.user(org, "employee", departments=[dept], clearance="RESTRICTED")
    doc = await seed_document(
        container,
        org_id=org,
        owner_id=owner.id,
        classification="CONFIDENTIAL",
        department_id=dept,
        chunks=["service level agreement for the data centre"],
    )
    await store.sync_document(org, doc.id)
    principal = await factory.principal(member)
    assert await _query(container, store, principal) == set(doc.chunk_ids)

    # reclassified in PostgreSQL, Qdrant not yet re-synchronised
    await update_document(container, doc, classification="RESTRICTED")
    assert await _raw(client, store, principal) == set(doc.chunk_ids)  # stale payload matches
    assert await _query(container, store, principal) == set()  # ...but is re-verified away
    await store.sync_document(org, doc.id)
    assert await _raw(client, store, principal) == set()

    # an explicit grant makes it readable again; revoking it takes effect before any re-sync
    grant = await add_grant(container, doc, user_id=member.id)
    await store.sync_document(org, doc.id)
    assert await _query(container, store, principal) == set(doc.chunk_ids)
    await revoke_grant(container, doc, grant)
    assert await _raw(client, store, principal) == set(doc.chunk_ids)
    assert await _query(container, store, principal) == set()

    # grant expiry is time dependent: it is enforced at query time even with a fresh payload
    grant = await add_grant(
        container, doc, user_id=member.id, expires_at=utcnow() + timedelta(hours=1)
    )
    await store.sync_document(org, doc.id)
    assert await _query(container, store, principal) == set(doc.chunk_ids)
    await update_grant(container, doc, grant, expires_at=utcnow() - timedelta(seconds=1))
    assert await _raw(client, store, principal) == set(doc.chunk_ids)
    assert await _query(container, store, principal) == set()

    # deletion: chunks disappear from PostgreSQL immediately; points are removed by the job
    await update_grant(container, doc, grant, expires_at=None)
    owner_principal = await factory.principal(owner)
    assert await _query(container, store, owner_principal) == set(doc.chunk_ids)
    await soft_delete(container, doc)
    assert await _query(container, store, owner_principal) == set()
    assert await store.delete_document(org, doc.id) == 1
    assert await _raw(client, store, owner_principal) == set()


async def test_version_switch_updates_current_flags(container, factory, qdrant) -> None:
    store, client = qdrant
    org = await factory.org()
    owner = await factory.user(org, "employee")
    doc = await seed_document(
        container, org_id=org, owner_id=owner.id, chunks=["service level agreement v1"]
    )
    assert await store.upsert_version(org, doc.version.id) >= 1
    v2 = await add_version(
        container, doc, ["service level agreement v2 a", "service level agreement v2 b"]
    )
    written = await store.upsert_version(org, v2.id)
    assert written == 2
    principal = await factory.principal(owner)
    assert await _query(container, store, principal) == set(v2.chunk_ids)
    assert await _query(container, store, principal, ALL_VERSIONS) == set(doc.all_chunk_ids)
    points, _ = await client.scroll(store._collection, limit=10, with_payload=True)
    flags = {uuid.UUID(str(p.id)): (p.payload or {})["is_current"] for p in points}
    assert flags == {
        **dict.fromkeys(doc.versions[0].chunk_ids, False),
        **dict.fromkeys(v2.chunk_ids, True),
    }


async def test_reconcile_removes_stale_and_adds_missing_points(container, factory, qdrant) -> None:
    store, client = qdrant
    org = await factory.org()
    owner = await factory.user(org, "employee")
    doc = await seed_document(
        container,
        org_id=org,
        owner_id=owner.id,
        chunks=["service level agreement one", "service level agreement two"],
    )
    assert await store.sync_document(org, doc.id) == 2  # both points were missing
    ghost = uuid.uuid4()
    await client.upsert(
        store._collection,
        points=[
            models.PointStruct(
                id=str(ghost),
                vector=QUERY,
                payload={"organization_id": str(org), "document_id": str(doc.id)},
            )
        ],
    )
    async with container.db.transaction(DbContext(org_id=org)) as session:
        await session.execute(delete(DocumentChunk).where(DocumentChunk.id == doc.chunk_ids[1]))
    assert await store.sync_document(org, doc.id) == 2  # ghost + deleted chunk removed
    points, _ = await client.scroll(store._collection, limit=10)
    assert {uuid.UUID(str(p.id)) for p in points} == {doc.chunk_ids[0]}
    assert await store.sync_document(org, doc.id) == 0  # idempotent


async def test_upsert_of_a_vanished_version_removes_its_points(container, factory, qdrant) -> None:
    store, client = qdrant
    org = await factory.org()
    await store.ensure_collection()
    missing_version = uuid.uuid4()
    await client.upsert(
        store._collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=QUERY,
                payload={"organization_id": str(org), "version_id": str(missing_version)},
            )
        ],
    )
    assert await store.upsert_version(org, missing_version) == 1
    assert (await client.count(store._collection)).count == 0


async def test_sync_of_deleted_document_removes_points(container, factory, qdrant) -> None:
    store, client = qdrant
    org = await factory.org()
    owner = await factory.user(org, "employee")
    doc = await seed_document(
        container, org_id=org, owner_id=owner.id, chunks=["service level agreement"]
    )
    await store.sync_document(org, doc.id)
    await soft_delete(container, doc)
    assert await store.sync_document(org, doc.id) == 1
    assert await store.upsert_version(org, doc.version.id) == 0
    assert (await client.count(store._collection)).count == 0


async def test_dimension_mismatch_is_a_configuration_error(container, factory) -> None:
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        "wrong", vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE)
    )
    store = QdrantVectorStore(
        client, db=container.db, collection="wrong", dimensions=1024, model="hashing-v1"
    )
    with pytest.raises(VectorStoreConfigError):
        await store.ensure_collection()
    job = ClaimedJob(
        id=uuid.uuid4(),
        kind="vector_sync_document",
        organization_id=uuid.uuid4(),
        payload={"document_id": str(uuid.uuid4())},
        attempts=1,
        max_attempts=5,
        request_id=None,
    )
    ctx = JobContext(container=SimpleNamespace(vector_store=store), worker_db=container.worker_db)  # type: ignore[arg-type]
    with pytest.raises(PermanentJobError):
        await vector_sync_document(ctx, job)
    assert await store.health() is True
    await store.aclose()


async def test_job_handlers_drive_the_store(container, factory, qdrant) -> None:
    store, client = qdrant
    org = await factory.org()
    owner = await factory.user(org, "employee")
    doc = await seed_document(
        container, org_id=org, owner_id=owner.id, chunks=["service level agreement a", "b text"]
    )
    ctx = JobContext(container=SimpleNamespace(vector_store=store), worker_db=container.worker_db)  # type: ignore[arg-type]

    def job(kind: str, payload: dict[str, str]) -> ClaimedJob:
        return ClaimedJob(
            id=uuid.uuid4(),
            kind=kind,
            organization_id=org,
            payload=payload,
            attempts=1,
            max_attempts=5,
            request_id=None,
        )

    result = await vector_sync_version(
        ctx, job("vector_sync_version", {"version_id": str(doc.version.id)})
    )
    assert result == {"backend": "qdrant", "points": 2}
    result = await vector_delete_document(
        ctx, job("vector_delete_document", {"document_id": str(doc.id)})
    )
    assert result == {"backend": "qdrant", "points": 2}
    assert (await client.count(store._collection)).count == 0


async def test_search_service_on_qdrant(container, factory, qdrant) -> None:
    store, _ = qdrant
    org = await factory.org()
    finance, sales = await factory.department(org), await factory.department(org)
    owner = await factory.user(org, "employee", departments=[finance])
    visible = await seed_document(
        container,
        org_id=org,
        owner_id=owner.id,
        title="SLA",
        classification="INTERNAL",
        chunks=["The service level agreement guarantees 99.9 percent uptime."],
    )
    hidden = await seed_document(
        container,
        org_id=org,
        owner_id=owner.id,
        title="Finance SLA",
        classification="CONFIDENTIAL",
        department_id=finance,
        chunks=["Finance service level agreement penalties."],
    )
    for doc in (visible, hidden):
        await store.sync_document(org, doc.id)
    service = SearchService(container, vector_store=store)
    reader = await factory.principal(await factory.user(org, "employee", departments=[sales]))
    for mode in ("semantic", "hybrid"):
        response = await service.search(reader, query="service level agreement", mode=mode)
        assert not response.degraded
        assert {r.document_id for r in response.results} == {visible.id}
        assert all(r.document_title != "Finance SLA" for r in response.results)
    retrieved = await service.retrieve(reader, "service level agreement uptime")
    assert {c.document_id for c in retrieved} == {visible.id}
