"""Vector-sync job handlers: payload validation and error classification (fake store)."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from docassist.jobs.queue import ClaimedJob, JobError, PermanentJobError
from docassist.jobs.registry import HANDLER_MODULES, HANDLERS, JobContext
from docassist.search.jobs import vector_delete_document, vector_sync_document, vector_sync_version
from docassist.search.vector_store import VectorStoreConfigError


class RecordingStore:
    name = "fake"

    def __init__(self, exc: Exception | None = None) -> None:
        self.calls: list[tuple[str, uuid.UUID, uuid.UUID]] = []
        self.exc = exc

    async def _op(self, name: str, org: uuid.UUID, target: uuid.UUID) -> int:
        self.calls.append((name, org, target))
        if self.exc:
            raise self.exc
        return 3

    async def upsert_version(self, org_id: uuid.UUID, version_id: uuid.UUID) -> int:
        return await self._op("upsert_version", org_id, version_id)

    async def sync_document(self, org_id: uuid.UUID, document_id: uuid.UUID) -> int:
        return await self._op("sync_document", org_id, document_id)

    async def delete_document(self, org_id: uuid.UUID, document_id: uuid.UUID) -> int:
        return await self._op("delete_document", org_id, document_id)


def _ctx(store: Any) -> JobContext:
    return JobContext(container=SimpleNamespace(vector_store=store), worker_db=None)  # type: ignore[arg-type]


def _job(kind: str, payload: dict[str, Any], org: uuid.UUID | None) -> ClaimedJob:
    return ClaimedJob(
        id=uuid.uuid4(),
        kind=kind,
        organization_id=org,
        payload=payload,
        attempts=1,
        max_attempts=5,
        request_id=None,
    )


def test_handlers_are_registered() -> None:
    assert "docassist.search.jobs" in HANDLER_MODULES
    assert HANDLERS["vector_sync_version"] is vector_sync_version
    assert HANDLERS["vector_sync_document"] is vector_sync_document
    assert HANDLERS["vector_delete_document"] is vector_delete_document


@pytest.mark.parametrize(
    ("handler", "kind", "key", "operation"),
    [
        (vector_sync_version, "vector_sync_version", "version_id", "upsert_version"),
        (vector_sync_document, "vector_sync_document", "document_id", "sync_document"),
        (vector_delete_document, "vector_delete_document", "document_id", "delete_document"),
    ],
)
async def test_handlers_call_the_store_with_the_job_organisation(
    handler: Any, kind: str, key: str, operation: str
) -> None:
    store, org, target = RecordingStore(), uuid.uuid4(), uuid.uuid4()
    result = await handler(_ctx(store), _job(kind, {key: str(target)}, org))
    assert result == {"backend": "fake", "points": 3}
    assert store.calls == [(operation, org, target)]


@pytest.mark.parametrize(
    ("payload", "org"),
    [
        ({}, uuid.uuid4()),
        ({"version_id": 123}, uuid.uuid4()),
        ({"version_id": "not-a-uuid"}, uuid.uuid4()),
        ({"version_id": str(uuid.uuid4())}, None),
    ],
)
async def test_malformed_jobs_are_permanent_failures(payload: dict, org: uuid.UUID | None) -> None:
    store = RecordingStore()
    with pytest.raises(PermanentJobError) as info:
        await vector_sync_version(_ctx(store), _job("vector_sync_version", payload, org))
    assert info.value.code == "invalid_payload"
    assert store.calls == []


async def test_configuration_errors_are_permanent_and_outages_are_retried() -> None:
    job = _job("vector_sync_document", {"document_id": str(uuid.uuid4())}, uuid.uuid4())
    with pytest.raises(PermanentJobError) as permanent:
        await vector_sync_document(_ctx(RecordingStore(VectorStoreConfigError("dims"))), job)
    assert permanent.value.code == "vector_store_config"
    with pytest.raises(JobError) as transient:
        await vector_sync_document(_ctx(RecordingStore(ConnectionError("down"))), job)
    assert not isinstance(transient.value, PermanentJobError)
    assert transient.value.code == "vector_store_unavailable"
    with pytest.raises(PermanentJobError):  # a JobError raised by the store passes through
        await vector_sync_document(_ctx(RecordingStore(PermanentJobError("gone"))), job)


async def test_pgvector_store_sync_is_a_no_op() -> None:
    from docassist.search.pgvector_store import PgVectorStore

    store = PgVectorStore(db=None, model="m", dimensions=4)  # type: ignore[arg-type]
    org, target = uuid.uuid4(), uuid.uuid4()
    assert await store.upsert_version(org, target) == 0
    assert await store.sync_document(org, target) == 0
    assert await store.delete_document(org, target) == 0
    await store.aclose()
    job = _job("vector_sync_version", {"version_id": str(target)}, org)
    assert await vector_sync_version(_ctx(store), job) == {"backend": "pgvector", "points": 0}
