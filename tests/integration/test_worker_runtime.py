"""Worker runtime against the real queue: outcomes, retries, leases, fencing, shutdown.

Each test uses its own job kind so jobs enqueued by other tests are never claimed here.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import uuid
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select, text, update

from docassist.core.context import utcnow
from docassist.core.errors import ServiceUnavailable
from docassist.db.models import AuditEvent, Document, DocumentVersion, Job
from docassist.db.session import DbContext
from docassist.documents.storage import LocalEncryptedStorage
from docassist.embeddings.hashing import HashingEmbedder
from docassist.ingestion.jobs import ingest_version
from docassist.ingestion.ocr import OcrConfig, OcrEngine
from docassist.ingestion.pipeline import IngestionPipeline
from docassist.ingestion.sandbox import SandboxConfig
from docassist.jobs.queue import JobError, PermanentJobError, enqueue
from docassist.jobs.registry import JobContext, load_handlers
from docassist.jobs.worker import Worker, WorkerStartupError
from tests.helpers_ingestion import CONTRACT_TEXT, InlineSandbox, RecordingEmbedder, seed_version

pytestmark = [pytest.mark.db]


def new_kind() -> str:
    return f"test.{uuid.uuid4().hex[:12]}"


async def add_job(
    container: Any,
    org: uuid.UUID,
    kind: str,
    payload: dict[str, Any] | None = None,
    max_attempts: int = 3,
) -> uuid.UUID:
    async with container.db.transaction(DbContext(org_id=org)) as session:
        job_id = await enqueue(
            session,
            kind=kind,
            organization_id=org,
            payload=payload or {},
            max_attempts=max_attempts,
        )
    assert job_id is not None
    return job_id


async def job(container: Any, job_id: uuid.UUID) -> Job:
    async with container.worker_db.session(DbContext.anonymous()) as session:
        return (await session.execute(select(Job).where(Job.id == job_id))).scalar_one()


async def make_claimable(container: Any, job_id: uuid.UUID) -> None:
    async with container.worker_db.transaction(DbContext.anonymous()) as session:
        await session.execute(
            update(Job).where(Job.id == job_id).values(run_after=utcnow() - timedelta(seconds=1))
        )


async def steal(container: Any, job_id: uuid.UUID) -> None:
    """Simulate a lease takeover by another worker."""
    async with container.worker_db.transaction(DbContext.anonymous()) as session:
        await session.execute(
            update(Job).where(Job.id == job_id).values(locked_by="another-worker")
        )


def worker_for(container: Any, kind: str, handler: Any | None, **kwargs: Any) -> Worker:
    handlers = {kind: handler} if handler is not None else {}
    return Worker(container, kinds=[kind], handlers=handlers, run_maintenance=False, **kwargs)


# --------------------------------------------------------------------------- #
async def test_success_completes_the_job(container, factory) -> None:
    org, kind = await factory.org(), new_kind()
    job_id = await add_job(container, org, kind, {"n": 2})

    async def handler(ctx: JobContext, claimed: Any) -> dict[str, Any]:
        assert ctx.worker_db is container.worker_db and claimed.payload == {"n": 2}
        return {"doubled": claimed.payload["n"] * 2, "when": utcnow()}

    outcome = await worker_for(container, kind, handler).run_once()
    assert outcome is not None and outcome.status == "succeeded" and outcome.job_id == job_id
    row = await job(container, job_id)
    assert (
        row.status == "succeeded"
        and row.result["doubled"] == 4
        and isinstance(row.result["when"], str)
    )
    assert row.locked_by is None and row.finished_at is not None and row.attempts == 1


async def test_retries_with_backoff_then_dead_letters(container, factory) -> None:
    org, kind = await factory.org(), new_kind()
    job_id = await add_job(container, org, kind, max_attempts=2)

    async def flaky(ctx: JobContext, claimed: Any) -> None:
        raise JobError("upstream wobble", code="flaky")

    worker = worker_for(container, kind, flaky)
    first = await worker.run_once()
    assert first is not None and (first.status, first.error_code) == ("retry", "flaky")
    row = await job(container, job_id)
    assert row.status == "queued" and row.attempts == 1 and row.run_after > utcnow()
    assert (row.last_error_code, row.last_error) == ("flaky", "upstream wobble")
    assert await worker.run_once() is None  # backing off: not claimable yet
    await make_claimable(container, job_id)
    second = await worker.run_once()
    assert second is not None and second.status == "dead"
    assert (await job(container, job_id)).status == "dead"
    async with container.worker_db.session(DbContext.system_for_org(org)) as session:
        actions = (
            (
                await session.execute(
                    select(AuditEvent.action).where(AuditEvent.resource_id == str(job_id))
                )
            )
            .scalars()
            .all()
        )
    assert actions == ["job.dead_lettered"]


async def test_permanent_error_dead_letters_immediately(container, factory) -> None:
    org, kind = await factory.org(), new_kind()
    job_id = await add_job(container, org, kind, max_attempts=5)

    async def broken(ctx: JobContext, claimed: Any) -> None:
        raise PermanentJobError("cannot ever work", code="bad_input")

    outcome = await worker_for(container, kind, broken).run_once()
    assert outcome is not None and (outcome.status, outcome.error_code) == ("dead", "bad_input")
    row = await job(container, job_id)
    assert row.status == "dead" and row.attempts == 1


async def test_unknown_kind_is_dead_lettered(container, factory) -> None:
    org, kind = await factory.org(), new_kind()
    job_id = await add_job(container, org, kind)
    outcome = await worker_for(container, kind, None).run_once()
    assert outcome is not None and (outcome.status, outcome.error_code) == ("dead", "unknown_kind")
    assert (await job(container, job_id)).status == "dead"


async def test_unexpected_exception_is_retryable_and_message_free(container, factory) -> None:
    org, kind = await factory.org(), new_kind()
    job_id = await add_job(container, org, kind)

    async def buggy(ctx: JobContext, claimed: Any) -> None:
        raise ValueError("confidential clause text from the document")

    outcome = await worker_for(container, kind, buggy).run_once()
    assert outcome is not None and (outcome.status, outcome.error_code) == (
        "retry",
        "internal_error",
    )
    row = await job(container, job_id)
    assert (
        "confidential" not in (row.last_error or "") and row.last_error == "unexpected ValueError"
    )


async def test_app_errors_keep_their_code_and_public_message(container, factory) -> None:
    org, kind = await factory.org(), new_kind()
    job_id = await add_job(container, org, kind)

    async def unavailable(ctx: JobContext, claimed: Any) -> None:
        raise ServiceUnavailable(internal_detail="db host 10.0.0.5 refused")

    outcome = await worker_for(container, kind, unavailable).run_once()
    assert outcome is not None and outcome.error_code == "service_unavailable"
    assert "10.0.0.5" not in ((await job(container, job_id)).last_error or "")


async def test_non_mapping_result_is_an_internal_error(container, factory) -> None:
    org, kind = await factory.org(), new_kind()
    await add_job(container, org, kind)

    async def odd(ctx: JobContext, claimed: Any) -> Any:
        return ["not", "a", "mapping"]

    outcome = await worker_for(container, kind, odd).run_once()
    assert outcome is not None and outcome.error_code == "internal_error"


async def test_job_timeout(container, factory) -> None:
    org, kind = await factory.org(), new_kind()
    job_id = await add_job(container, org, kind)

    async def sleepy(ctx: JobContext, claimed: Any) -> None:
        await asyncio.sleep(60)

    outcome = await worker_for(container, kind, sleepy, job_timeout=0.5).run_once()
    assert outcome is not None and (outcome.status, outcome.error_code) == ("retry", "job_timeout")
    assert (await job(container, job_id)).status == "queued"


async def test_heartbeat_extends_the_lease(container, factory) -> None:
    org, kind = await factory.org(), new_kind()
    job_id = await add_job(container, org, kind)
    seen: list[Any] = []

    async def slow(ctx: JobContext, claimed: Any) -> None:
        seen.append((await job(container, job_id)).locked_until)
        await asyncio.sleep(2.5)
        seen.append((await job(container, job_id)).locked_until)

    outcome = await worker_for(container, kind, slow, heartbeat_interval=0.3).run_once()
    assert outcome is not None and outcome.status == "succeeded"
    assert seen[1] > seen[0]


async def test_lost_lease_cancels_the_handler_and_writes_nothing(container, factory) -> None:
    org, kind = await factory.org(), new_kind()
    job_id = await add_job(container, org, kind)
    finished = False

    async def victim(ctx: JobContext, claimed: Any) -> None:
        nonlocal finished
        await steal(container, job_id)
        await asyncio.sleep(30)
        finished = True

    outcome = await worker_for(container, kind, victim, heartbeat_interval=0.3).run_once()
    assert outcome is not None and (outcome.status, outcome.error_code) == ("fenced", "lease_lost")
    assert not finished
    row = await job(container, job_id)
    assert row.status == "running" and row.locked_by == "another-worker"  # the new owner's row


@pytest.mark.parametrize("fails", [False, True], ids=["complete", "fail"])
async def test_outcomes_are_fenced(container, factory, fails: bool) -> None:
    org, kind = await factory.org(), new_kind()
    job_id = await add_job(container, org, kind)

    async def racer(ctx: JobContext, claimed: Any) -> dict[str, Any]:
        await steal(container, job_id)
        if fails:
            raise JobError("late failure", code="late")
        return {"late": True}

    outcome = await worker_for(container, kind, racer).run_once()
    assert outcome is not None and outcome.status == "fenced"
    row = await job(container, job_id)
    assert row.status == "running" and row.result is None and row.last_error_code is None


async def test_drain_and_empty_queue(container, factory) -> None:
    org, kind = await factory.org(), new_kind()
    worker = worker_for(container, kind, lambda ctx, claimed: asyncio.sleep(0, result={"ok": True}))
    assert await worker.run_once() is None
    for _ in range(3):
        await add_job(container, org, kind)
    outcomes = await worker.drain(max_jobs=10)
    assert [o.status for o in outcomes] == ["succeeded"] * 3
    await add_job(container, org, kind)
    await add_job(container, org, kind)
    assert len(await worker.drain(max_jobs=1)) == 1


async def wait_for(predicate: Any, timeout: float = 60.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not await predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.1)


async def test_run_loop_processes_jobs_until_stopped(container, factory) -> None:
    org, kind = await factory.org(), new_kind()
    ids = [await add_job(container, org, kind) for _ in range(3)]

    async def quick(ctx: JobContext, claimed: Any) -> dict[str, Any]:
        return {"id": str(claimed.id)}

    worker = worker_for(container, kind, quick, poll_interval=0.1, concurrency=2)
    runner = asyncio.create_task(worker.run())

    async def all_done() -> bool:
        statuses = [(await job(container, i)).status for i in ids]
        return all(status == "succeeded" for status in statuses)

    await wait_for(all_done)
    worker.request_stop()
    await asyncio.wait_for(runner, timeout=60)
    assert worker.stopping


async def test_cancelled_worker_releases_running_jobs(container, factory) -> None:
    org, kind = await factory.org(), new_kind()
    job_id = await add_job(container, org, kind)
    started = asyncio.Event()

    async def long_job(ctx: JobContext, claimed: Any) -> None:
        started.set()
        await asyncio.sleep(120)

    worker = worker_for(container, kind, long_job, poll_interval=0.1)
    worker.shutdown_grace = 0.2
    runner = asyncio.create_task(worker.run())
    await asyncio.wait_for(started.wait(), timeout=60)
    runner.cancel()  # what asyncio.run does on Ctrl+C
    with pytest.raises(asyncio.CancelledError):
        await runner
    row = await job(container, job_id)
    assert row.status == "queued" and row.attempts == 0 and row.last_error_code == "worker_shutdown"
    assert row.locked_by is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal handling")
async def test_sigterm_stops_the_worker(container, factory) -> None:
    kind = new_kind()
    worker = worker_for(container, kind, None, poll_interval=0.1)
    runner = asyncio.create_task(worker.run())
    await asyncio.sleep(1.0)
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(runner, timeout=60)
    assert worker.stopping


async def test_worker_refuses_the_api_role(container, monkeypatch) -> None:
    worker = Worker(container, db=container.db, run_maintenance=False, handlers={})
    with pytest.raises(WorkerStartupError, match="docassist_worker"):
        await worker.verify_role()
    relaxed = container.settings.model_copy(
        update={
            "database": container.settings.database.model_copy(
                update={"verify_role_privileges": False}
            )
        }
    )
    monkeypatch.setattr(container, "settings", relaxed)
    await worker.verify_role()  # development opt-out: logged, not fatal


async def test_worker_role_passes_the_check(container) -> None:
    await Worker(container, run_maintenance=False, handlers={}).verify_role()


def test_worker_needs_the_worker_database(container) -> None:
    with pytest.raises(WorkerStartupError):
        Worker(SimpleNamespace(settings=container.settings, worker_db=None), handlers={})


def test_registry_contains_ingestion_handler() -> None:
    assert load_handlers()["ingest_version"] is ingest_version


# --------------------------------------------------------------------------- #
# ingest_version through the worker
# --------------------------------------------------------------------------- #
@pytest.fixture
def storage(container: Any, settings: Any) -> LocalEncryptedStorage:
    return LocalEncryptedStorage(Path(settings.storage.root), container.ring)


def install_pipeline(
    container: Any, storage: Any, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> None:
    kwargs.setdefault(
        "embeddings", HashingEmbedder(dimensions=container.settings.embedding.dimensions)
    )
    pipeline = IngestionPipeline(
        container,
        storage=storage,
        sandbox=InlineSandbox(SandboxConfig.from_settings(container.settings)),
        ocr=OcrEngine(OcrConfig(enabled=False, command=())),
        **kwargs,
    )
    monkeypatch.setattr(container, "pipeline", pipeline)


async def test_ingest_job_end_to_end(container, factory, storage, monkeypatch) -> None:
    install_pipeline(container, storage, monkeypatch)
    org = await factory.org()
    owner = await factory.user(org)
    seeded = await seed_version(
        container,
        storage,
        org_id=org,
        owner_id=owner.id,
        data=CONTRACT_TEXT.encode(),
        extension="md",
    )
    kind = new_kind()
    job_id = await add_job(container, org, kind, {"version_id": str(seeded.version_id)})
    outcome = await worker_for(container, kind, ingest_version).run_once()
    assert outcome is not None and outcome.status == "succeeded"
    assert outcome.result is not None and outcome.result["status"] == "indexed"
    assert (await job(container, job_id)).result["chunks"] >= 1
    async with container.worker_db.session(DbContext.system_for_org(org)) as session:
        document = (
            await session.execute(select(Document).where(Document.id == seeded.document_id))
        ).scalar_one()
    assert document.status == "ready"


@pytest.mark.parametrize(
    "payload",
    [{}, {"version_id": "not-a-uuid"}, {"version_id": 42}],
    ids=["missing", "garbage", "type"],
)
async def test_ingest_job_rejects_bad_payloads(container, factory, payload) -> None:
    org, kind = await factory.org(), new_kind()
    await add_job(container, org, kind, payload)
    outcome = await worker_for(container, kind, ingest_version).run_once()
    assert outcome is not None and (outcome.status, outcome.error_code) == (
        "dead",
        "invalid_payload",
    )


async def test_ingest_job_records_failure_after_last_retry(
    container, factory, storage, monkeypatch
) -> None:
    install_pipeline(
        container, storage, monkeypatch, embeddings=RecordingEmbedder(is_external=False, fail=True)
    )
    org = await factory.org()
    owner = await factory.user(org)
    seeded = await seed_version(
        container,
        storage,
        org_id=org,
        owner_id=owner.id,
        data=CONTRACT_TEXT.encode(),
        extension="md",
    )
    kind = new_kind()
    await add_job(container, org, kind, {"version_id": str(seeded.version_id)}, max_attempts=1)
    outcome = await worker_for(container, kind, ingest_version).run_once()
    assert outcome is not None and (outcome.status, outcome.error_code) == (
        "dead",
        "embedding_unavailable",
    )
    async with container.worker_db.session(DbContext.system_for_org(org)) as session:
        version = (
            await session.execute(
                select(DocumentVersion).where(DocumentVersion.id == seeded.version_id)
            )
        ).scalar_one()
    assert version.status == "failed" and version.error_code == "embedding_unavailable"


async def test_worker_sessions_leave_no_tenant_context_behind(container) -> None:
    async with container.worker_db.session(DbContext.anonymous()) as session:
        value = (await session.execute(text("SELECT current_setting('app.org_id', true)"))).scalar()
    assert value in ("", None)
