"""Worker maintenance: advisory lock, audit sealing, lease reclaim, queue gauge, retention."""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select, text, update

from docassist.audit.service import Actor, verify_chain
from docassist.core.context import utcnow
from docassist.db.models import (
    AuditEvent,
    AuthSession,
    Conversation,
    Document,
    DocumentVersion,
    Export,
    Job,
    LlmUsage,
    Message,
    MfaChallenge,
    PasswordResetToken,
    RefreshToken,
)
from docassist.db.session import DbContext
from docassist.documents.storage import (
    LocalEncryptedStorage,
    StorageError,
    new_object_key,
    storage_context,
)
from docassist.jobs.maintenance import MAINTENANCE_LOCK_KEY, PLATFORM_KEY, Maintenance
from docassist.jobs.queue import enqueue
from docassist.observability import metrics
from tests.helpers_ingestion import seed_version

pytestmark = [pytest.mark.db]


@pytest.fixture
def storage(container: Any, settings: Any) -> LocalEncryptedStorage:
    return LocalEncryptedStorage(Path(settings.storage.root), container.ring)


def maintenance(container: Any, storage: Any, **kwargs: Any) -> Maintenance:
    kwargs.setdefault("retention_on_start", False)
    return Maintenance(container, storage=storage, **kwargs)


async def test_pass_seals_reclaims_and_measures(container, factory, storage) -> None:
    org = await factory.org()
    async with container.db.transaction(DbContext(org_id=org)) as session:
        for i in range(3):
            container.audit.record(session, Actor.system(org), "test.maintenance", details={"i": i})
        job_id = await enqueue(
            session, kind=f"test.{uuid.uuid4().hex}", organization_id=org, payload={}
        )
    async with container.worker_db.transaction(DbContext.anonymous()) as session:
        await session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status="running",
                locked_by="crashed-worker",
                attempts=1,
                locked_until=utcnow() - timedelta(minutes=5),
            )
        )
    report = await maintenance(container, storage).run_once()
    assert report.ran and report.sealed >= 3 and report.reclaimed >= 1 and report.retention is None
    assert report.queue_depth.get("queued", 0) >= 1
    assert (
        metrics.JOB_QUEUE_DEPTH.labels(status="queued")._value.get() == report.queue_depth["queued"]
    )
    async with container.worker_db.session(DbContext.anonymous()) as session:
        row = (await session.execute(select(Job).where(Job.id == job_id))).scalar_one()
        chain = await verify_chain(
            session, container.settings.security.audit_hmac_key.get_secret_value().encode(), org
        )
    assert (
        row.status == "queued" and row.last_error_code == "lease_expired" and row.locked_by is None
    )
    assert chain.valid and chain.unsealed == 0 and chain.checked >= 3


async def test_only_one_replica_runs_maintenance(container, storage) -> None:
    async with container.worker_db.engine.connect() as other:
        assert (
            await other.execute(
                text("SELECT pg_try_advisory_lock(:k)"), {"k": MAINTENANCE_LOCK_KEY}
            )
        ).scalar()
        await other.commit()
        try:
            assert (await maintenance(container, storage).run_once()).ran is False
        finally:
            await other.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": MAINTENANCE_LOCK_KEY})
            await other.commit()
    assert (await maintenance(container, storage).run_once()).ran is True


async def test_retention_schedule(container, storage) -> None:
    first = maintenance(container, storage, retention_on_start=True, retention_interval=10_000)
    assert (await first.run_once()).retention is not None
    assert (await first.run_once()).retention is None  # not due again yet
    assert (await first.run_once(force_retention=True)).retention is not None


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #
async def insert(container: Any, org: uuid.UUID | None, *rows: Any) -> None:
    ctx = DbContext.system_for_org(org) if org else DbContext(org_id=None, platform=True)
    async with container.worker_db.transaction(ctx) as session:
        for row in rows:
            session.add(row)
            await session.flush()


async def exists(container: Any, org: uuid.UUID | None, model: Any, row_id: Any) -> bool:
    ctx = DbContext.system_for_org(org) if org else DbContext(org_id=None, platform=True)
    async with container.worker_db.session(ctx) as session:
        return (await session.execute(select(model).where(model.id == row_id))).first() is not None


async def test_retention_purges_expired_data_per_organisation(container, factory, storage) -> None:
    now = utcnow()
    long_ago = now - timedelta(days=800)
    org = await factory.org()
    user = await factory.user(org)
    retention = container.settings.retention

    old_conversation = Conversation(
        organization_id=org, user_id=user.id, title="old", updated_at=long_ago
    )
    live_conversation = Conversation(organization_id=org, user_id=user.id, title="live")
    await insert(container, org, old_conversation, live_conversation)
    old_message = Message(
        organization_id=org,
        conversation_id=live_conversation.id,
        user_id=user.id,
        role="user",
        content="old",
        created_at=long_ago,
    )
    new_message = Message(
        organization_id=org,
        conversation_id=live_conversation.id,
        user_id=user.id,
        role="user",
        content="new",
    )
    await insert(container, org, old_message, new_message)

    export_key = new_object_key(org)
    export_id = uuid.uuid4()
    await storage.put_bytes(export_key, b"a,b\n1,2\n", storage_context(org, "export", export_id))
    await insert(
        container, org,
        Export(id=export_id, organization_id=org, user_id=user.id, kind="fields", format="csv", status="ready",
               storage_key=export_key, size_bytes=8, expires_at=now - timedelta(hours=1)),
    )  # fmt: skip
    old_usage = LlmUsage(
        organization_id=org,
        task="answer",
        provider="p",
        model="m",
        status="ok",
        cost_usd=Decimal(0),
        created_at=long_ago,
    )
    new_usage = LlmUsage(
        organization_id=org,
        task="answer",
        provider="p",
        model="m",
        status="ok",
        cost_usd=Decimal(0),
    )
    await insert(container, org, old_usage, new_usage)
    old_job = Job(organization_id=org, kind="test.old", status="succeeded", finished_at=long_ago)
    running_job = Job(
        organization_id=org, kind="test.running", status="running", started_at=long_ago
    )
    await insert(container, org, old_job, running_job)

    # sessions / tokens
    expired_session = AuthSession(
        user_id=user.id, organization_id=org, expires_at=now - timedelta(days=1)
    )
    revoked_session = AuthSession(
        user_id=user.id,
        organization_id=org,
        expires_at=now + timedelta(days=1),
        revoked_at=now - timedelta(days=30),
    )
    live_session = AuthSession(
        user_id=user.id, organization_id=org, expires_at=now + timedelta(days=1)
    )
    await insert(container, org, expired_session, revoked_session, live_session)
    token = RefreshToken(
        session_id=live_session.id,
        organization_id=org,
        token_hash=uuid.uuid4().bytes * 2,
        expires_at=now - timedelta(minutes=1),
    )
    reset = PasswordResetToken(
        user_id=user.id,
        organization_id=org,
        token_hash=uuid.uuid4().bytes * 2,
        expires_at=now - timedelta(minutes=1),
    )
    challenge = MfaChallenge(
        user_id=user.id,
        organization_id=org,
        token_hash=uuid.uuid4().bytes * 2,
        expires_at=now - timedelta(minutes=1),
    )
    await insert(container, org, token, reset, challenge)

    # documents: purge-able, legal hold, retention date, recently deleted, active
    purge_days = timedelta(days=retention.deleted_document_purge_days + 1)
    purgeable = await seed_version(
        container, storage, org_id=org, owner_id=user.id, data=b"gone", extension="txt"
    )
    held = await seed_version(
        container, storage, org_id=org, owner_id=user.id, data=b"held", extension="txt"
    )
    retained = await seed_version(
        container, storage, org_id=org, owner_id=user.id, data=b"kept", extension="txt"
    )
    recent = await seed_version(
        container, storage, org_id=org, owner_id=user.id, data=b"recent", extension="txt"
    )
    active = await seed_version(
        container, storage, org_id=org, owner_id=user.id, data=b"active", extension="txt"
    )
    async with container.worker_db.transaction(DbContext.system_for_org(org)) as session:
        for seeded, extra in (
            (purgeable, {}),
            (held, {"legal_hold": True}),
            (retained, {"retention_until": date.today() + timedelta(days=365)}),
        ):
            await session.execute(
                update(Document).where(Document.id == seeded.document_id)
                .values(status="deleted", deleted_at=now - purge_days, current_version_id=seeded.version_id, **extra)
            )  # fmt: skip
        await session.execute(
            update(Document)
            .where(Document.id == recent.document_id)
            .values(status="deleted", deleted_at=now)
        )

    report = await maintenance(container, storage).purge_retention(now=now)
    counts = report[str(org)]
    assert counts["conversations"] == 1 and counts["messages"] == 1
    assert counts["exports"] == 1 and counts["llm_usage"] == 1 and counts["jobs"] == 1
    assert counts["documents"] == 1 and counts["document_blobs"] == 1
    assert counts["sessions"] == 2 and counts["refresh_tokens"] == 1
    assert counts["password_reset_tokens"] == 1 and counts["mfa_challenges"] == 1

    assert not await exists(container, org, Conversation, old_conversation.id)
    assert await exists(container, org, Conversation, live_conversation.id)
    assert not await exists(container, org, Message, old_message.id) and await exists(
        container, org, Message, new_message.id
    )
    assert not await exists(container, org, LlmUsage, old_usage.id) and await exists(
        container, org, LlmUsage, new_usage.id
    )
    assert not await exists(container, org, Job, old_job.id) and await exists(
        container, org, Job, running_job.id
    )
    assert await exists(container, org, AuthSession, live_session.id)
    assert not await exists(container, org, AuthSession, expired_session.id)
    assert not await exists(container, org, AuthSession, revoked_session.id)
    async with container.worker_db.session(DbContext.system_for_org(org)) as session:
        export = (await session.execute(select(Export).where(Export.id == export_id))).scalar_one()
    assert export.status == "expired" and export.storage_key is None
    assert not await storage.exists(export_key)

    assert not await exists(container, org, Document, purgeable.document_id)
    assert not await exists(container, org, DocumentVersion, purgeable.version_id)  # cascaded
    assert not await storage.exists(purgeable.storage_key)
    for kept in (held, retained, recent, active):
        assert await exists(container, org, Document, kept.document_id)
        assert await storage.exists(kept.storage_key)

    async with container.worker_db.session(DbContext.system_for_org(org)) as session:
        purge_events = (
            (
                await session.execute(
                    select(AuditEvent).where(
                        AuditEvent.organization_id == org, AuditEvent.action == "retention.purge"
                    )
                )
            )
            .scalars()
            .all()
        )
    assert len(purge_events) == 1 and purge_events[0].actor_role == "system"
    assert purge_events[0].details["documents"] == 1

    again = await maintenance(container, storage).purge_retention(now=now)
    assert not any(again[str(org)].values())  # idempotent: nothing left to purge


async def test_platform_rows_are_purged_in_the_platform_pass(container, factory, storage) -> None:
    admin = await factory.user(None, "platform_admin")
    expired = AuthSession(
        user_id=admin.id, organization_id=None, expires_at=utcnow() - timedelta(hours=1)
    )
    await insert(container, None, expired)
    report = await maintenance(container, storage).purge_retention()
    assert report[PLATFORM_KEY]["sessions"] >= 1
    assert not await exists(container, None, AuthSession, expired.id)


async def test_blob_failure_keeps_the_document(container, factory, storage) -> None:
    class BrokenStorage:
        async def delete(self, key: str) -> None:
            raise StorageError("disk unavailable")

    org = await factory.org()
    user = await factory.user(org)
    seeded = await seed_version(
        container, storage, org_id=org, owner_id=user.id, data=b"x", extension="txt"
    )
    async with container.worker_db.transaction(DbContext.system_for_org(org)) as session:
        await session.execute(
            update(Document)
            .where(Document.id == seeded.document_id)
            .values(status="deleted", deleted_at=utcnow() - timedelta(days=400))
        )
    report = await Maintenance(
        container, storage=BrokenStorage(), retention_on_start=False
    ).purge_retention()  # type: ignore[arg-type]
    assert report[str(org)]["documents"] == 0
    assert await exists(container, org, Document, seeded.document_id)


async def test_versions_of_dead_ingest_jobs_are_failed(container, factory, storage) -> None:
    org = await factory.org()
    user = await factory.user(org)
    stuck = await seed_version(
        container,
        storage,
        org_id=org,
        owner_id=user.id,
        data=b"x",
        extension="txt",
        version_status="processing",
    )
    done = await seed_version(
        container,
        storage,
        org_id=org,
        owner_id=user.id,
        data=b"y",
        extension="txt",
        version_status="indexed",
    )
    await insert(
        container, org,
        Job(organization_id=org, kind="ingest_version", status="dead", finished_at=utcnow(),
            payload={"version_id": str(stuck.version_id)}, last_error_code="job_timeout"),
        Job(organization_id=org, kind="ingest_version", status="dead", finished_at=utcnow(),
            payload={"version_id": str(done.version_id)}, last_error_code="job_timeout"),
        Job(organization_id=org, kind="ingest_version", status="dead", finished_at=utcnow(),
            payload={"version_id": "garbage"}, last_error_code="invalid_payload"),
    )  # fmt: skip
    sweeper = maintenance(container, storage)
    assert await sweeper.reconcile_ingestion() >= 1
    async with container.worker_db.session(DbContext.system_for_org(org)) as session:
        statuses = dict(
            (
                await session.execute(
                    select(DocumentVersion.id, DocumentVersion.status).where(
                        DocumentVersion.organization_id == org
                    )
                )
            ).all()
        )
        [(document_status,)] = (
            await session.execute(select(Document.status).where(Document.id == stuck.document_id))
        ).all()
    assert statuses[stuck.version_id] == "failed" and statuses[done.version_id] == "indexed"
    assert document_status == "failed"
    assert await sweeper.reconcile_ingestion() == 0  # nothing new since the previous pass
