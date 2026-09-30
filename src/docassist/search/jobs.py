"""Background job handlers for the search area: keep an external vector index in sync.

Job kinds (enqueued by the document service / ingestion pipeline only when
``retrieval.backend == "qdrant"``; payloads carry identifiers only):

* ``vector_sync_version``   ``{"version_id": ...}``  - index a newly indexed version and refresh
  the document's ACL payload / current-version flags;
* ``vector_sync_document``  ``{"document_id": ...}`` - reconcile a document after permission or
  metadata changes (payload refresh, stale points removed, missing points added);
* ``vector_delete_document`` ``{"document_id": ...}`` - remove every point of a document.

The organisation always comes from the job row (set server-side at enqueue time), never from
the payload. Malformed payloads and configuration errors are permanent failures
(dead-letter); vector-store outages are transient (retried with backoff).
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from docassist.core.logging import get_logger
from docassist.jobs.queue import ClaimedJob, JobError, PermanentJobError
from docassist.jobs.registry import JobContext, job_handler
from docassist.search.vector_store import VectorStore, VectorStoreConfigError

log = get_logger(__name__)


def _org_id(job: ClaimedJob) -> uuid.UUID:
    if job.organization_id is None:
        raise PermanentJobError("vector job without organisation", code="invalid_payload")
    return job.organization_id


def _uuid_field(job: ClaimedJob, key: str) -> uuid.UUID:
    raw = job.payload.get(key)
    if not isinstance(raw, str):
        raise PermanentJobError(f"payload field {key!r} missing", code="invalid_payload")
    try:
        return uuid.UUID(raw)
    except ValueError as exc:
        raise PermanentJobError(
            f"payload field {key!r} is not a UUID", code="invalid_payload"
        ) from exc


async def _run(
    ctx: JobContext,
    job: ClaimedJob,
    key: str,
    operation: Callable[[VectorStore, uuid.UUID, uuid.UUID], Awaitable[int]],
) -> dict[str, Any]:
    org_id = _org_id(job)
    target = _uuid_field(job, key)
    store = ctx.container.vector_store
    try:
        count = await operation(store, org_id, target)
    except VectorStoreConfigError as exc:
        raise PermanentJobError("vector store misconfigured", code="vector_store_config") from exc
    except JobError:
        raise
    except Exception as exc:
        log.warning("vector_job_failed", kind=job.kind, error_type=type(exc).__name__)
        raise JobError("vector store unavailable", code="vector_store_unavailable") from exc
    return {"backend": store.name, "points": count}


@job_handler("vector_sync_version")
async def vector_sync_version(ctx: JobContext, job: ClaimedJob) -> dict[str, Any] | None:
    return await _run(ctx, job, "version_id", lambda s, org, vid: s.upsert_version(org, vid))


@job_handler("vector_sync_document")
async def vector_sync_document(ctx: JobContext, job: ClaimedJob) -> dict[str, Any] | None:
    return await _run(ctx, job, "document_id", lambda s, org, did: s.sync_document(org, did))


@job_handler("vector_delete_document")
async def vector_delete_document(ctx: JobContext, job: ClaimedJob) -> dict[str, Any] | None:
    return await _run(ctx, job, "document_id", lambda s, org, did: s.delete_document(org, did))
