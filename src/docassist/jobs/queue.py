"""PostgreSQL-backed job queue.

Why not Celery/RQ? Enqueueing happens *inside the business transaction* (transactional
outbox): the document row and its ingestion job commit together or not at all - no lost
jobs, no jobs for rows that rolled back. Claiming uses ``FOR UPDATE SKIP LOCKED`` so any
number of workers can poll without contention.

Reliability rules:

* **leases** - a claimed job carries ``locked_until``; a crashed worker's job becomes
  claimable again when the lease expires (``reclaim_expired``);
* **fencing** - completion/failure updates require ``locked_by = me AND status = running``,
  so a worker whose lease was taken over cannot overwrite the new owner's result;
* **idempotency** - ``idempotency_key`` is unique; enqueueing twice is a no-op;
* **retries** - exponential backoff with jitter; :class:`PermanentJobError` skips retries;
  exhausting ``max_attempts`` moves the job to ``dead`` (dead-letter) for an operator.
* Payloads carry identifiers only - never document content or secrets.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.core.context import current_request_id, utcnow
from docassist.core.enums import JobStatus
from docassist.core.ids import uuid7
from docassist.db.models import Job


class JobError(Exception):
    """Transient failure: the job will be retried with backoff."""

    code = "job_failed"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


class PermanentJobError(JobError):
    """Retrying cannot help (malformed document, missing row...)."""

    code = "permanent_failure"


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    id: uuid.UUID
    kind: str
    organization_id: uuid.UUID | None
    payload: dict[str, Any]
    attempts: int
    max_attempts: int
    request_id: str | None


async def enqueue(
    session: AsyncSession,
    *,
    kind: str,
    organization_id: uuid.UUID | None,
    payload: dict[str, Any],
    idempotency_key: str | None = None,
    priority: int = 100,
    max_attempts: int = 5,
    run_after: datetime | None = None,
    created_by: uuid.UUID | None = None,
) -> uuid.UUID | None:
    """Insert a job in the caller's transaction. Returns ``None`` if the key already exists."""
    job_id = uuid7()
    stmt = (
        insert(Job)
        .values(
            id=job_id,
            organization_id=organization_id,
            kind=kind,
            payload=payload,
            status=JobStatus.QUEUED.value,
            priority=priority,
            attempts=0,
            max_attempts=max_attempts,
            run_after=run_after or utcnow(),
            idempotency_key=idempotency_key,
            request_id=current_request_id(),
            created_by=created_by,
        )
        .on_conflict_do_nothing(index_elements=["idempotency_key"])
        .returning(Job.id)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


_CLAIM_SQL = text(
    """
    UPDATE jobs SET status = 'running', locked_by = :worker, attempts = attempts + 1,
           locked_until = now() + make_interval(secs => :lease),
           started_at = now(), updated_at = now()
     WHERE id = (
        SELECT id FROM jobs
         WHERE status = 'queued' AND run_after <= now() AND (:kinds_all OR kind = ANY(:kinds))
         ORDER BY priority, run_after
         FOR UPDATE SKIP LOCKED
         LIMIT 1)
    RETURNING id, kind, organization_id, payload, attempts, max_attempts, request_id
    """
)


class JobQueue:
    """Worker-side operations. Use with a session from the *worker* database role."""

    def __init__(
        self, worker_id: str, lease_seconds: int, backoff_base: float, backoff_max: float
    ) -> None:
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max

    async def claim(
        self, session: AsyncSession, kinds: list[str] | None = None
    ) -> ClaimedJob | None:
        row = (
            await session.execute(
                _CLAIM_SQL,
                {
                    "worker": self.worker_id,
                    "lease": self.lease_seconds,
                    "kinds_all": kinds is None,
                    "kinds": kinds or [],
                },
            )
        ).first()
        if row is None:
            return None
        return ClaimedJob(
            id=row.id,
            kind=row.kind,
            organization_id=row.organization_id,
            payload=dict(row.payload or {}),
            attempts=row.attempts,
            max_attempts=row.max_attempts,
            request_id=row.request_id,
        )

    async def heartbeat(self, session: AsyncSession, job_id: uuid.UUID) -> bool:
        result = await session.execute(
            update(Job)
            .where(
                Job.id == job_id,
                Job.locked_by == self.worker_id,
                Job.status == JobStatus.RUNNING.value,
            )
            .values(locked_until=func.now() + timedelta(seconds=self.lease_seconds))
        )
        return bool(result.rowcount)  # type: ignore[attr-defined]

    async def complete(
        self, session: AsyncSession, job_id: uuid.UUID, result: dict[str, Any] | None = None
    ) -> bool:
        res = await session.execute(
            update(Job)
            .where(
                Job.id == job_id,
                Job.locked_by == self.worker_id,
                Job.status == JobStatus.RUNNING.value,
            )
            .values(
                status=JobStatus.SUCCEEDED.value,
                result=result,
                finished_at=func.now(),
                locked_by=None,
                locked_until=None,
                last_error=None,
                last_error_code=None,
            )
        )
        return bool(res.rowcount)  # type: ignore[attr-defined]

    def backoff(self, attempts: int) -> float:
        exponent = float(2 ** max(0, attempts - 1))
        base = min(self._backoff_max, self._backoff_base * exponent)
        return float(base * random.uniform(0.8, 1.2))

    async def fail(self, session: AsyncSession, job: ClaimedJob, error: JobError) -> JobStatus:
        permanent = isinstance(error, PermanentJobError)
        exhausted = job.attempts >= job.max_attempts
        status = JobStatus.DEAD if (permanent or exhausted) else JobStatus.QUEUED
        values: dict[str, Any] = {
            "status": status.value,
            "locked_by": None,
            "locked_until": None,
            "last_error_code": error.code[:64],
            "last_error": str(error)[:500],
        }
        if status is JobStatus.QUEUED:
            values["run_after"] = utcnow() + timedelta(seconds=self.backoff(job.attempts))
        else:
            values["finished_at"] = func.now()
        await session.execute(
            update(Job)
            .where(
                Job.id == job.id,
                Job.locked_by == self.worker_id,
                Job.status == JobStatus.RUNNING.value,
            )
            .values(**values)
        )
        return status

    async def reclaim_expired(self, session: AsyncSession) -> int:
        """Return crashed workers' jobs to the queue (or dead-letter them if out of attempts)."""
        result = await session.execute(
            text(
                """
                UPDATE jobs SET
                   status = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'queued' END,
                   locked_by = NULL, locked_until = NULL, updated_at = now(),
                   last_error_code = 'lease_expired',
                   last_error = 'worker lease expired before completion',
                   finished_at = CASE WHEN attempts >= max_attempts THEN now() ELSE NULL END
                 WHERE status = 'running' AND locked_until < now()
                """
            )
        )
        return int(result.rowcount or 0)  # type: ignore[attr-defined]

    async def depth(self, session: AsyncSession) -> dict[str, int]:
        rows = (await session.execute(select(Job.status, func.count()).group_by(Job.status))).all()
        return {status: int(count) for status, count in rows}


async def requeue_dead(session: AsyncSession, job_id: uuid.UUID) -> bool:
    """Operator action: give a dead-lettered job a fresh set of attempts."""
    result = await session.execute(
        update(Job)
        .where(Job.id == job_id, Job.status.in_([JobStatus.DEAD.value, JobStatus.FAILED.value]))
        .values(status=JobStatus.QUEUED.value, attempts=0, run_after=func.now(), finished_at=None)
    )
    return bool(result.rowcount)  # type: ignore[attr-defined]
