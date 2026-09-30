"""Background worker: claims jobs from the PostgreSQL queue and runs their handlers.

* **Least privilege** - :meth:`Worker.run` first checks it is connected as
  ``database.worker_role`` without superuser/BYPASSRLS/table ownership and refuses to start
  otherwise (only an explicit ``database.verify_role_privileges = false`` outside production
  downgrades that to a warning).
* **Claiming** happens in its own short transaction (``FOR UPDATE SKIP LOCKED``); handlers
  never run inside it. Up to ``concurrency`` jobs run at once.
* **Leases** - a heartbeat extends the lease every ``heartbeat_interval`` seconds. If the
  heartbeat finds the lease gone (another worker reclaimed the job), the handler is
  cancelled and nothing is written: the job belongs to someone else now (**fencing**).
  Completion and failure updates are fenced by ``locked_by = me AND status = running``.
* **Outcomes** - success completes the job; :class:`JobError` retries with backoff;
  :class:`PermanentJobError`, an unknown job kind, or exhausted attempts dead-letter it (and
  audit ``job.dead_lettered``); a per-job timeout is a retryable ``job_timeout``; any other
  exception is a retryable ``internal_error`` whose *type* and stack frames - never its
  message, which may quote data - are logged.
* **Graceful shutdown** - SIGTERM/SIGINT (POSIX) or cancellation of :meth:`run` (e.g.
  ``KeyboardInterrupt`` under ``asyncio.run`` on Windows) stops claiming, waits up to
  ``worker.shutdown_grace_seconds`` for running jobs, then cancels the rest and *releases*
  them to the queue immediately (the interrupted attempt is not counted).

:meth:`run_once` (claim + execute one job) and :meth:`drain` serve tests and the CLI.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import socket
import time
import traceback
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select, update

from docassist.audit.service import Actor
from docassist.core.enums import AuditOutcome, JobStatus
from docassist.core.errors import AppError
from docassist.core.logging import get_logger
from docassist.db.models import Job
from docassist.db.session import Database, DbContext, verify_least_privilege
from docassist.jobs.maintenance import Maintenance
from docassist.jobs.queue import ClaimedJob, JobError, JobQueue, PermanentJobError
from docassist.jobs.registry import Handler, JobContext, load_handlers
from docassist.observability import metrics

if TYPE_CHECKING:
    from docassist.api.container import Container

log = get_logger(__name__)

DEFAULT_JOB_TIMEOUT_SECONDS = 900.0
MAX_RESULT_BYTES = 65_536


class WorkerStartupError(RuntimeError):
    """The worker refuses to start (wrong or over-privileged database role, no worker DB)."""


@dataclass(frozen=True, slots=True)
class JobOutcome:
    """What happened to one claimed job.

    ``status``: ``succeeded`` | ``retry`` | ``dead`` | ``fenced`` (the lease was lost, nothing
    written) | ``released`` (returned to the queue by a stopping worker).
    """

    job_id: uuid.UUID
    kind: str
    status: str
    error_code: str | None = None
    result: dict[str, Any] | None = None


def default_worker_id() -> str:
    return f"{socket.gethostname()[:60]}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _frames(exc: BaseException) -> list[str]:
    """File/line/function of each frame - no exception message, no local values."""
    return [
        f"{frame.filename}:{frame.lineno}:{frame.name}"
        for frame in traceback.extract_tb(exc.__traceback__)
    ][-12:]


def _json_safe(result: object) -> dict[str, Any] | None:
    if result is None:
        return None
    if not isinstance(result, Mapping):
        raise TypeError("job handlers must return a mapping or None")
    encoded = json.dumps(dict(result), default=str)
    if len(encoded) > MAX_RESULT_BYTES:
        return {"truncated": True}
    decoded: dict[str, Any] = json.loads(encoded)
    return decoded


class Worker:
    def __init__(
        self,
        container: Container,
        *,
        concurrency: int | None = None,
        poll_interval: float | None = None,
        kinds: Iterable[str] | None = None,
        worker_id: str | None = None,
        handlers: Mapping[str, Handler] | None = None,
        job_timeout: float = DEFAULT_JOB_TIMEOUT_SECONDS,
        heartbeat_interval: float | None = None,
        maintenance: Maintenance | None = None,
        run_maintenance: bool = True,
        db: Database | None = None,
    ) -> None:
        database = db or container.worker_db
        if database is None:
            raise WorkerStartupError(
                "the worker needs the worker database (build_container(role='worker'))"
            )
        settings = container.settings.worker
        self._container = container
        self._db = database
        self.concurrency = concurrency or settings.concurrency
        self.poll_interval = poll_interval or settings.poll_interval_seconds
        self.kinds = sorted(set(kinds)) if kinds is not None else None
        self.worker_id = worker_id or default_worker_id()
        self._handlers: Mapping[str, Handler] = (
            handlers if handlers is not None else load_handlers()
        )
        self.job_timeout = job_timeout
        self.queue = JobQueue(
            self.worker_id,
            settings.lease_seconds,
            settings.backoff_base_seconds,
            settings.backoff_max_seconds,
        )
        self.heartbeat_interval = heartbeat_interval or max(1.0, settings.lease_seconds / 3)
        self.shutdown_grace = settings.shutdown_grace_seconds
        self.maintenance_interval = settings.maintenance_interval_seconds
        self.maintenance = maintenance
        if maintenance is None and run_maintenance:
            self.maintenance = Maintenance(container, db=database)
        self._stop = asyncio.Event()
        self._active: set[asyncio.Task[JobOutcome | None]] = set()
        self._lost: set[uuid.UUID] = set()

    # ------------------------------------------------------------------ lifecycle
    async def verify_role(self) -> None:
        settings = self._container.settings
        problems = await verify_least_privilege(self._db, settings.database.worker_role)
        if not problems:
            return
        if settings.database.verify_role_privileges or settings.is_production:
            raise WorkerStartupError("worker database role check failed: " + "; ".join(problems))
        log.warning("worker_db_role_problems", problems=problems)

    def request_stop(self) -> None:
        """Stop claiming new jobs; :meth:`run` returns once running jobs have finished."""
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    async def run(self) -> None:
        """Run until :meth:`request_stop`, SIGTERM/SIGINT (POSIX) or cancellation."""
        await self.verify_role()
        self._stop.clear()
        installed = self._install_signal_handlers()
        maintenance = asyncio.create_task(self._maintenance_loop()) if self.maintenance else None
        log.info(
            "worker_started",
            worker_id=self.worker_id,
            concurrency=self.concurrency,
            kinds=self.kinds,
        )
        try:
            await self._claim_loop()
        finally:
            await self._shutdown(maintenance)
            self._remove_signal_handlers(installed)
            log.info("worker_stopped", worker_id=self.worker_id)

    async def run_once(self) -> JobOutcome | None:
        """Claim and execute at most one job (``None`` when nothing is claimable)."""
        job = await self._claim()
        if job is None:
            return None
        return await self._execute(job)

    async def drain(self, max_jobs: int = 1_000) -> list[JobOutcome]:
        """Execute claimable jobs one by one until none is left or ``max_jobs`` ran."""
        outcomes: list[JobOutcome] = []
        while len(outcomes) < max_jobs:
            outcome = await self.run_once()
            if outcome is None:
                break
            outcomes.append(outcome)
        return outcomes

    # ------------------------------------------------------------------ loop internals
    def _install_signal_handlers(self) -> list[signal.Signals]:
        loop = asyncio.get_running_loop()
        installed: list[signal.Signals] = []
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except (NotImplementedError, RuntimeError, ValueError):
                continue  # Windows / non-main thread: cancellation still stops the worker
            installed.append(sig)
        return installed

    @staticmethod
    def _remove_signal_handlers(installed: list[signal.Signals]) -> None:
        loop = asyncio.get_running_loop()
        for sig in installed:
            loop.remove_signal_handler(sig)

    async def _pause(self, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)

    async def _claim_loop(self) -> None:
        while not self._stop.is_set():
            if len(self._active) >= self.concurrency:
                await asyncio.wait(
                    self._active, timeout=self.poll_interval, return_when=asyncio.FIRST_COMPLETED
                )
                continue
            try:
                job = await self._claim()
            except Exception as exc:  # noqa: BLE001 - the database may be briefly unavailable
                log.warning("job_claim_failed", error_type=type(exc).__name__)
                await self._pause(self.poll_interval)
                continue
            if job is None:
                await self._pause(self.poll_interval)
                continue
            task = asyncio.create_task(self._execute_logged(job), name=f"job-{job.id}")
            self._active.add(task)
            task.add_done_callback(self._active.discard)

    async def _shutdown(self, maintenance: asyncio.Task[None] | None) -> None:
        self._stop.set()
        if maintenance is not None:
            maintenance.cancel()
            await asyncio.gather(maintenance, return_exceptions=True)
        if not self._active:
            return
        _, pending = await asyncio.wait(set(self._active), timeout=self.shutdown_grace)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _maintenance_loop(self) -> None:
        maintenance = self.maintenance
        if maintenance is None:
            return
        while not self._stop.is_set():
            try:
                await maintenance.run_once()
            except Exception as exc:  # noqa: BLE001 - maintenance must never stop the worker
                log.warning(
                    "maintenance_failed", error_type=type(exc).__name__, frames=_frames(exc)
                )
            await self._pause(self.maintenance_interval)

    # ------------------------------------------------------------------ one job
    async def _claim(self) -> ClaimedJob | None:
        async with self._db.transaction(DbContext.anonymous()) as session:
            return await self.queue.claim(session, self.kinds)

    async def _execute_logged(self, job: ClaimedJob) -> JobOutcome | None:
        try:
            return await self._execute(job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - recording the outcome failed; the lease will expire
            # not log.exception: exception text may quote SQL parameters (document data)
            log.error(  # noqa: TRY400
                "job_outcome_not_recorded", job_id=str(job.id), error_type=type(exc).__name__
            )
            return None

    def _kind_label(self, kind: str) -> str:
        return kind if kind in self._handlers else "unknown"

    async def _execute(self, job: ClaimedJob) -> JobOutcome:
        started = time.perf_counter()
        handler = self._handlers.get(job.kind)
        if handler is None:
            error: JobError = PermanentJobError(
                "no handler is registered for this job kind", code="unknown_kind"
            )
            return await self._fail(job, error, started)
        context = JobContext(container=self._container, worker_db=self._db)

        async def call() -> dict[str, Any] | None:
            return await handler(context, job)

        task = asyncio.create_task(call())
        heartbeat = asyncio.create_task(self._heartbeat(job, task))
        try:
            done, _ = await asyncio.wait({task}, timeout=self.job_timeout)
        except asyncio.CancelledError:
            task.cancel()
            heartbeat.cancel()
            await asyncio.gather(task, heartbeat, return_exceptions=True)
            await self._release(job)
            raise
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
        if job.id in self._lost:
            self._lost.discard(job.id)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return self._outcome(job, "fenced", "lease_lost", started)
        if not done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return await self._fail(
                job, JobError("job exceeded its time limit", code="job_timeout"), started
            )
        try:
            result = _json_safe(task.result())
        except JobError as exc:
            return await self._fail(job, exc, started)
        except asyncio.CancelledError:  # the handler cancelled itself (we were not cancelled)
            return await self._fail(
                job, JobError("handler was cancelled", code="handler_cancelled"), started
            )
        except AppError as exc:
            return await self._fail(job, JobError(exc.public_message, code=exc.code), started)
        except Exception as exc:  # noqa: BLE001 - any handler bug is a retryable internal error
            log.error(  # noqa: TRY400 - type and frames only, never the message
                "job_handler_crashed",
                job_id=str(job.id),
                kind=job.kind,
                error_type=type(exc).__name__,
                frames=_frames(exc),
            )
            error = JobError(f"unexpected {type(exc).__name__}", code="internal_error")
            return await self._fail(job, error, started)
        async with self._db.transaction(DbContext.anonymous()) as session:
            completed = await self.queue.complete(session, job.id, result)
        if not completed:
            return self._outcome(job, "fenced", "lease_lost", started)
        return self._outcome(job, "succeeded", None, started, result)

    async def _heartbeat(self, job: ClaimedJob, task: asyncio.Task[Any]) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                async with self._db.transaction(DbContext.anonymous()) as session:
                    alive = await self.queue.heartbeat(session, job.id)
            except Exception as exc:  # noqa: BLE001 - a missed beat is retried on the next tick
                log.warning(
                    "job_heartbeat_failed", job_id=str(job.id), error_type=type(exc).__name__
                )
                continue
            if not alive:
                log.warning("job_lease_lost", job_id=str(job.id), kind=job.kind)
                self._lost.add(job.id)
                task.cancel()
                return

    async def _fail(self, job: ClaimedJob, error: JobError, started: float) -> JobOutcome:
        ctx = (
            DbContext.system_for_org(job.organization_id)
            if job.organization_id
            else DbContext(org_id=None, platform=True)
        )
        async with self._db.transaction(ctx) as session:
            owner = (
                await session.execute(
                    select(Job.locked_by, Job.status).where(Job.id == job.id).with_for_update()
                )
            ).first()
            if owner is None or owner[0] != self.worker_id or owner[1] != JobStatus.RUNNING.value:
                return self._outcome(job, "fenced", "lease_lost", started)
            status = await self.queue.fail(session, job, error)
            if status is JobStatus.DEAD:
                self._container.audit.record(
                    session,
                    Actor.system(job.organization_id),
                    "job.dead_lettered",
                    outcome=AuditOutcome.FAILURE,
                    resource_type="job",
                    resource_id=job.id,
                    details={
                        "kind": job.kind[:64],
                        "error_code": error.code[:64],
                        "attempts": job.attempts,
                    },
                )
        label = "dead" if status is JobStatus.DEAD else "retry"
        return self._outcome(job, label, error.code, started)

    async def _release(self, job: ClaimedJob) -> None:
        """Give an interrupted job back to the queue without charging the attempt."""
        try:
            async with self._db.transaction(DbContext.anonymous()) as session:
                await session.execute(
                    update(Job)
                    .where(
                        Job.id == job.id,
                        Job.locked_by == self.worker_id,
                        Job.status == JobStatus.RUNNING.value,
                    )
                    .values(
                        status=JobStatus.QUEUED.value,
                        locked_by=None,
                        locked_until=None,
                        attempts=func.greatest(Job.attempts - 1, 0),
                        run_after=func.now(),
                        last_error_code="worker_shutdown",
                        last_error="released by a stopping worker",
                    )
                )
        except Exception as exc:  # noqa: BLE001 - the lease expiry is the fallback
            log.warning("job_release_failed", job_id=str(job.id), error_type=type(exc).__name__)
            return
        metrics.JOBS.labels(kind=self._kind_label(job.kind), outcome="released").inc()
        log.info("job_released", job_id=str(job.id), kind=job.kind)

    def _outcome(
        self,
        job: ClaimedJob,
        status: str,
        error_code: str | None,
        started: float,
        result: dict[str, Any] | None = None,
    ) -> JobOutcome:
        metrics.JOBS.labels(kind=self._kind_label(job.kind), outcome=status).inc()
        log.info(
            "job_finished",
            job_id=str(job.id),
            kind=job.kind[:64],
            status=status,
            error_code=error_code,
            attempts=job.attempts,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        return JobOutcome(job.id, job.kind, status, error_code, result)
