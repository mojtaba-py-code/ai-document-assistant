"""Postgres job queue semantics: outbox idempotency, SKIP LOCKED, leases, fencing, dead letters."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import select, text

from docassist.core.enums import JobStatus
from docassist.db.models import Job
from docassist.db.session import DbContext
from docassist.jobs.queue import JobError, JobQueue, PermanentJobError, enqueue, requeue_dead

pytestmark = [pytest.mark.db]


def queue(worker: str = "w1", lease: int = 60) -> JobQueue:
    return JobQueue(worker, lease_seconds=lease, backoff_base=1.0, backoff_max=5.0)


async def _enqueue(container, org, kind: str, key: str | None = None, **kw) -> uuid.UUID | None:
    async with container.db.transaction(DbContext(org_id=org)) as session:
        return await enqueue(
            session, kind=kind, organization_id=org, payload={"x": 1}, idempotency_key=key, **kw
        )


async def _claim(container, q: JobQueue, kind: str):
    async with container.worker_db.transaction(DbContext.anonymous()) as session:
        return await q.claim(session, [kind])


async def test_idempotent_enqueue(container, factory) -> None:
    org = await factory.org()
    key = f"k-{uuid.uuid4()}"
    first = await _enqueue(container, org, "t.idem", key)
    second = await _enqueue(container, org, "t.idem", key)
    assert first is not None and second is None


async def test_enqueue_rolls_back_with_business_transaction(container, factory) -> None:
    org = await factory.org()
    kind = f"t.rollback.{uuid.uuid4().hex[:6]}"
    with pytest.raises(RuntimeError):
        async with container.db.transaction(DbContext(org_id=org)) as session:
            await enqueue(session, kind=kind, organization_id=org, payload={})
            raise RuntimeError("business failure")
    assert await _claim(container, queue(), kind) is None


async def test_concurrent_claims_never_share_a_job(container, factory) -> None:
    org = await factory.org()
    kind = f"t.conc.{uuid.uuid4().hex[:6]}"
    for _ in range(6):
        await _enqueue(container, org, kind)
    claimed = await asyncio.gather(*[_claim(container, queue(f"w{i}"), kind) for i in range(10)])
    ids = [c.id for c in claimed if c is not None]
    assert len(ids) == 6 and len(set(ids)) == 6


async def test_complete_requires_ownership(container, factory) -> None:
    org = await factory.org()
    kind = f"t.fence.{uuid.uuid4().hex[:6]}"
    await _enqueue(container, org, kind)
    q1 = queue("w1")
    job = await _claim(container, q1, kind)
    assert job is not None
    async with container.worker_db.transaction(DbContext.anonymous()) as session:
        assert not await queue("intruder").complete(session, job.id)
        assert await q1.complete(session, job.id, {"ok": True})


async def test_lease_expiry_and_fencing(container, factory) -> None:
    org = await factory.org()
    kind = f"t.lease.{uuid.uuid4().hex[:6]}"
    await _enqueue(container, org, kind)
    slow = queue("slow", lease=60)
    job = await _claim(container, slow, kind)
    assert job is not None
    async with container.worker_db.transaction(DbContext.anonymous()) as session:
        await session.execute(
            text("UPDATE jobs SET locked_until = now() - interval '1 second' WHERE id = :i"),
            {"i": job.id},
        )
        assert await slow.reclaim_expired(session) >= 1
    fast = queue("fast")
    again = await _claim(container, fast, kind)
    assert again is not None and again.id == job.id and again.attempts == 2
    async with container.worker_db.transaction(DbContext.anonymous()) as session:
        assert not await slow.complete(session, job.id)  # the zombie worker is fenced off
        assert await fast.complete(session, job.id)


async def test_retry_backoff_then_dead_letter(container, factory) -> None:
    org = await factory.org()
    kind = f"t.retry.{uuid.uuid4().hex[:6]}"
    await _enqueue(container, org, kind, max_attempts=2)
    q = queue()
    job = await _claim(container, q, kind)
    async with container.worker_db.transaction(DbContext.anonymous()) as session:
        assert await q.fail(session, job, JobError("boom")) is JobStatus.QUEUED
        await session.execute(
            text("UPDATE jobs SET run_after = now() WHERE id = :i"), {"i": job.id}
        )
    job = await _claim(container, q, kind)
    assert job is not None and job.attempts == 2
    async with container.worker_db.transaction(DbContext.anonymous()) as session:
        assert await q.fail(session, job, JobError("boom again")) is JobStatus.DEAD
    async with container.db.transaction(DbContext(org_id=org)) as session:
        row = (await session.execute(select(Job).where(Job.id == job.id))).scalar_one()
        assert row.status == "dead" and row.last_error_code == "job_failed"
        assert await requeue_dead(session, job.id)


async def test_permanent_error_skips_retries(container, factory) -> None:
    org = await factory.org()
    kind = f"t.perm.{uuid.uuid4().hex[:6]}"
    await _enqueue(container, org, kind, max_attempts=5)
    q = queue()
    job = await _claim(container, q, kind)
    async with container.worker_db.transaction(DbContext.anonymous()) as session:
        assert (
            await q.fail(session, job, PermanentJobError("malformed", code="parse_error"))
            is JobStatus.DEAD
        )


async def test_backoff_grows_and_is_capped() -> None:
    q = JobQueue("w", lease_seconds=60, backoff_base=2.0, backoff_max=10.0)
    assert 1.5 <= q.backoff(1) <= 2.5
    assert 6.0 <= q.backoff(3) <= 10.0
    assert q.backoff(20) <= 12.0
