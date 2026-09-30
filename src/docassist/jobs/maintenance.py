"""Periodic maintenance run by the worker (not queued).

Every run is guarded by a PostgreSQL session advisory lock (``pg_try_advisory_lock``): with
several worker replicas exactly one performs maintenance at a time, the others skip.

Each run
* seals pending audit events into their HMAC hash chains (:class:`AuditSealer`);
* returns jobs whose lease expired to the queue (or dead-letters them when out of attempts);
* marks a document version ``failed`` when its ``ingest_version`` job was dead-lettered
  while the version was still pending (e.g. the last attempt timed out or its worker
  crashed), so no document stays "processing" forever;
* refreshes the ``docassist_job_queue_depth`` gauge.

Retention (on the first pass unless ``retention_on_start=False``, then every
``retention_interval`` seconds; per organisation, plus a platform pass):

* conversations not updated and messages created more than ``retention.conversation_days``
  ago;
* exports past ``expires_at``: blob deleted, row marked ``expired``;
* finished jobs (succeeded / failed / dead / cancelled) older than ``retention.job_days``;
* ``llm_usage`` rows older than ``retention.llm_usage_days``;
* soft-deleted documents older than ``retention.deleted_document_purge_days`` that are not on
  legal hold and past any ``retention_until`` date: the row is locked, the blobs of all its
  versions deleted, then the row hard-deleted (versions, chunks, embeddings, fields and
  grants cascade). The database trigger independently refuses to delete held documents;
* expired sessions (and revoked ones after a grace period), refresh tokens, password-reset
  tokens and MFA challenges;
* sealed audit events older than ``retention.audit_days`` (the chain is anchored first, so it
  stays verifiable).

Every organisation with something purged gets one ``retention.purge`` audit event with the
counts (system actor).
"""

from __future__ import annotations

import os
import socket
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.audit.service import PLATFORM_CHAIN, Actor, AuditSealer
from docassist.core.context import utcnow
from docassist.core.enums import DocumentStatus, ExportStatus, JobStatus
from docassist.core.ids import parse_uuid
from docassist.core.logging import get_logger
from docassist.db.models import (
    AuthSession,
    Conversation,
    Document,
    DocumentVersion,
    Export,
    Job,
    LlmUsage,
    Message,
    MfaChallenge,
    Organization,
    PasswordResetToken,
    RefreshToken,
)
from docassist.db.session import Database, DbContext
from docassist.documents.storage import StorageError
from docassist.jobs.queue import JobQueue
from docassist.observability import metrics

if TYPE_CHECKING:
    from docassist.api.container import Container
    from docassist.documents.storage import ObjectStorage

log = get_logger(__name__)

MAINTENANCE_LOCK_KEY = 724_002
DEFAULT_RETENTION_INTERVAL_SECONDS = 3_600.0
REVOKED_SESSION_GRACE = timedelta(days=7)
DOCUMENT_PURGE_BATCH = 200
EXPORT_BATCH = 500
MAX_SEAL_ROUNDS = 20
RECONCILE_LOOKBACK = timedelta(days=1)
RECONCILE_OVERLAP = timedelta(minutes=5)
RECONCILE_BATCH = 500
PLATFORM_KEY = "platform"
_FINISHED = [
    JobStatus.SUCCEEDED.value,
    JobStatus.FAILED.value,
    JobStatus.DEAD.value,
    JobStatus.CANCELLED.value,
]


def _rowcount(result: Any) -> int:
    return int(getattr(result, "rowcount", 0) or 0)


@dataclass(slots=True)
class MaintenanceReport:
    ran: bool
    sealed: int = 0
    reclaimed: int = 0
    reconciled: int = 0
    queue_depth: dict[str, int] = field(default_factory=dict)
    retention: dict[str, dict[str, int]] | None = None


class Maintenance:
    def __init__(
        self,
        container: Container,
        *,
        db: Database | None = None,
        storage: ObjectStorage | None = None,
        retention_interval: float = DEFAULT_RETENTION_INTERVAL_SECONDS,
        retention_on_start: bool = True,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        database = db or container.worker_db
        if database is None:
            raise RuntimeError("maintenance runs on the worker database role")
        self._container = container
        self._db = database
        self._storage = storage
        self._clock = clock
        self.retention_interval = retention_interval
        # With retention_on_start the first pass purges; otherwise the first purge waits one
        # full interval.
        self._last_retention: float | None = None if retention_on_start else time.monotonic()
        self._reconciled_until: datetime | None = None
        settings = container.settings
        self.sealer = AuditSealer(
            database, settings.security.audit_hmac_key.get_secret_value().encode()
        )
        worker = settings.worker
        self.queue = JobQueue(
            f"maintenance:{socket.gethostname()[:60]}:{os.getpid()}",
            worker.lease_seconds,
            worker.backoff_base_seconds,
            worker.backoff_max_seconds,
        )

    @property
    def storage(self) -> ObjectStorage:
        return self._storage or self._container.storage

    # ------------------------------------------------------------------ orchestration
    async def run_once(self, *, force_retention: bool = False) -> MaintenanceReport:
        """One maintenance pass; ``ran`` is false when another replica holds the lock."""
        async with self._db.engine.connect() as conn:
            got = (
                await conn.execute(
                    text("SELECT pg_try_advisory_lock(:key)"), {"key": MAINTENANCE_LOCK_KEY}
                )
            ).scalar()
            await conn.commit()
            if not got:
                return MaintenanceReport(ran=False)
            try:
                report = MaintenanceReport(ran=True)
                report.sealed = await self.seal_audit()
                report.reclaimed = await self.reclaim_expired()
                report.reconciled = await self.reconcile_ingestion()
                report.queue_depth = await self.update_queue_depth()
                due = self._last_retention is None or (
                    time.monotonic() - self._last_retention >= self.retention_interval
                )
                if force_retention or due:
                    report.retention = await self.purge_retention()
                    self._last_retention = time.monotonic()
                return report
            finally:
                await conn.execute(
                    text("SELECT pg_advisory_unlock(:key)"), {"key": MAINTENANCE_LOCK_KEY}
                )
                await conn.commit()

    async def seal_audit(self) -> int:
        total = 0
        for _ in range(MAX_SEAL_ROUNDS):
            sealed = await self.sealer.seal_pending()
            total += sealed
            if not sealed:
                break
        return total

    async def reclaim_expired(self) -> int:
        async with self._db.transaction(DbContext.anonymous()) as session:
            reclaimed = await self.queue.reclaim_expired(session)
        if reclaimed:
            log.warning("jobs_reclaimed", count=reclaimed)
        return reclaimed

    async def reconcile_ingestion(self) -> int:
        """Fail pending versions whose ``ingest_version`` job ended up dead-lettered.

        Looks at jobs dead-lettered since the previous pass (a day back on the first one);
        versions that are already indexed or failed are left alone.
        """
        now = self._clock()
        since = (self._reconciled_until or now - RECONCILE_LOOKBACK) - RECONCILE_OVERLAP
        async with self._db.session(DbContext.anonymous()) as session:
            rows = (
                await session.execute(
                    select(Job.organization_id, Job.payload, Job.last_error_code)
                    .where(
                        Job.kind == "ingest_version",
                        Job.status == JobStatus.DEAD.value,
                        Job.finished_at >= since,
                    )
                    .order_by(Job.finished_at)
                    .limit(RECONCILE_BATCH)
                )
            ).all()
        marked = 0
        pipeline = self._container.pipeline
        for org_id, payload, code in rows:
            raw = (payload or {}).get("version_id")
            version_id = parse_uuid(raw) if isinstance(raw, str) else None
            if org_id is None or version_id is None:
                continue
            if await pipeline.mark_failed(
                org_id, version_id, code or "internal_error", only_if_pending=True
            ):
                marked += 1
        self._reconciled_until = now
        if marked:
            log.warning("ingestion_versions_reconciled", count=marked)
        return marked

    async def update_queue_depth(self) -> dict[str, int]:
        async with self._db.session(DbContext.anonymous()) as session:
            depth = await self.queue.depth(session)
        for status in JobStatus:
            metrics.JOB_QUEUE_DEPTH.labels(status=status.value).set(depth.get(status.value, 0))
        return depth

    # ------------------------------------------------------------------ retention
    async def purge_retention(self, now: datetime | None = None) -> dict[str, dict[str, int]]:
        """Apply every retention rule; returns counts per organisation id (and ``platform``)."""
        now = now or self._clock()
        retention = self._container.settings.retention
        jobs = await self._purge_jobs(now - timedelta(days=retention.job_days))
        report: dict[str, dict[str, int]] = {}
        async with self._db.session(DbContext.anonymous()) as session:
            orgs = list(
                (await session.execute(select(Organization.id).order_by(Organization.id))).scalars()
            )
        for org_id in orgs:
            counts = await self._purge_organisation(org_id, now)
            counts["jobs"] = jobs.get(str(org_id), 0)
            counts["audit_events"] = await self.sealer.purge_older_than(
                org_id, now - timedelta(days=retention.audit_days)
            )
            report[str(org_id)] = counts
            await self._audit(org_id, counts)
        platform = await self._purge_platform(now)
        platform["jobs"] = jobs.get(PLATFORM_KEY, 0)
        platform["audit_events"] = await self.sealer.purge_older_than(
            PLATFORM_CHAIN, now - timedelta(days=retention.audit_days)
        )
        report[PLATFORM_KEY] = platform
        await self._audit(None, platform)
        return report

    async def _purge_jobs(self, cutoff: datetime) -> dict[str, int]:
        async with self._db.transaction(DbContext.anonymous()) as session:
            owners = (
                (
                    await session.execute(
                        delete(Job)
                        .where(Job.status.in_(_FINISHED), Job.finished_at < cutoff)
                        .returning(Job.organization_id)
                    )
                )
                .scalars()
                .all()
            )
        return dict(Counter(str(owner) if owner else PLATFORM_KEY for owner in owners))

    async def _purge_organisation(self, org_id: uuid.UUID, now: datetime) -> dict[str, int]:
        retention = self._container.settings.retention
        documents, blobs = await self._purge_documents(org_id, now)
        counts = {"documents": documents, "document_blobs": blobs}
        conversation_cutoff = now - timedelta(days=retention.conversation_days)
        async with self._db.transaction(DbContext.system_for_org(org_id)) as session:
            counts["messages"] = _rowcount(
                await session.execute(
                    delete(Message).where(
                        Message.organization_id == org_id, Message.created_at < conversation_cutoff
                    )
                )
            )
            counts["conversations"] = _rowcount(
                await session.execute(
                    delete(Conversation).where(
                        Conversation.organization_id == org_id,
                        Conversation.updated_at < conversation_cutoff,
                    )
                )
            )
            counts["llm_usage"] = _rowcount(
                await session.execute(
                    delete(LlmUsage).where(
                        LlmUsage.organization_id == org_id,
                        LlmUsage.created_at < now - timedelta(days=retention.llm_usage_days),
                    )
                )
            )
            counts.update(await self._purge_identity(session, org_id, now))
        counts["exports"] = await self._expire_exports(org_id, now)
        return counts

    async def _purge_platform(self, now: datetime) -> dict[str, int]:
        async with self._db.transaction(DbContext(org_id=None, platform=True)) as session:
            return await self._purge_identity(session, None, now)

    async def _purge_identity(
        self, session: AsyncSession, org_id: uuid.UUID | None, now: datetime
    ) -> dict[str, int]:
        def owner(column: Any) -> Any:
            return column.is_(None) if org_id is None else column == org_id

        sessions = await session.execute(
            delete(AuthSession).where(
                owner(AuthSession.organization_id),
                or_(
                    AuthSession.expires_at < now,
                    AuthSession.revoked_at < now - REVOKED_SESSION_GRACE,
                ),
            )
        )
        refresh = await session.execute(
            delete(RefreshToken).where(
                owner(RefreshToken.organization_id), RefreshToken.expires_at < now
            )
        )
        resets = await session.execute(
            delete(PasswordResetToken).where(
                owner(PasswordResetToken.organization_id), PasswordResetToken.expires_at < now
            )
        )
        challenges = await session.execute(
            delete(MfaChallenge).where(
                owner(MfaChallenge.organization_id), MfaChallenge.expires_at < now
            )
        )
        return {
            "sessions": _rowcount(sessions),
            "refresh_tokens": _rowcount(refresh),
            "password_reset_tokens": _rowcount(resets),
            "mfa_challenges": _rowcount(challenges),
        }

    async def _purge_documents(self, org_id: uuid.UUID, now: datetime) -> tuple[int, int]:
        retention = self._container.settings.retention
        cutoff = now - timedelta(days=retention.deleted_document_purge_days)
        ctx = DbContext.system_for_org(org_id)
        eligible = (
            Document.organization_id == org_id,
            Document.status == DocumentStatus.DELETED.value,
            Document.deleted_at < cutoff,
            Document.legal_hold.is_(False),
            or_(Document.retention_until.is_(None), Document.retention_until < now.date()),
        )
        async with self._db.session(ctx) as session:
            candidates = list(
                (
                    await session.execute(
                        select(Document.id)
                        .where(*eligible)
                        .order_by(Document.deleted_at)
                        .limit(DOCUMENT_PURGE_BATCH)
                    )
                ).scalars()
            )
        purged = blobs = 0
        for document_id in candidates:
            async with self._db.transaction(ctx) as session:
                # Lock the row: a legal hold set concurrently waits for (or blocks) this purge.
                locked = (
                    await session.execute(
                        select(Document.id)
                        .where(Document.id == document_id, *eligible)
                        .with_for_update(skip_locked=True)
                    )
                ).scalar_one_or_none()
                if locked is None:
                    continue
                keys = (
                    (
                        await session.execute(
                            select(DocumentVersion.storage_key).where(
                                DocumentVersion.organization_id == org_id,
                                DocumentVersion.document_id == document_id,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                try:
                    for key in keys:
                        await self.storage.delete(key)
                        blobs += 1
                except (StorageError, OSError) as exc:
                    log.warning(
                        "document_blob_purge_failed",
                        document_id=str(document_id),
                        error_type=type(exc).__name__,
                    )
                    await session.rollback()
                    continue
                await session.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(current_version_id=None)
                )
                await session.execute(delete(Document).where(Document.id == document_id, *eligible))
                purged += 1
        return purged, blobs

    async def _expire_exports(self, org_id: uuid.UUID, now: datetime) -> int:
        async with self._db.transaction(DbContext.system_for_org(org_id)) as session:
            rows = (
                await session.execute(
                    select(Export.id, Export.storage_key)
                    .where(
                        Export.organization_id == org_id,
                        Export.expires_at < now,
                        Export.status != ExportStatus.EXPIRED.value,
                    )
                    .with_for_update(skip_locked=True)
                    .limit(EXPORT_BATCH)
                )
            ).all()
            expired: list[uuid.UUID] = []
            for export_id, key in rows:
                if key:
                    try:
                        await self.storage.delete(key)
                    except (StorageError, OSError) as exc:
                        log.warning(
                            "export_blob_purge_failed",
                            export_id=str(export_id),
                            error_type=type(exc).__name__,
                        )
                        continue
                expired.append(export_id)
            if expired:
                await session.execute(
                    update(Export)
                    .where(Export.id.in_(expired))
                    .values(status=ExportStatus.EXPIRED.value, storage_key=None, size_bytes=None)
                )
        return len(expired)

    async def _audit(self, org_id: uuid.UUID | None, counts: dict[str, int]) -> None:
        if not any(counts.values()):
            return
        ctx = DbContext.system_for_org(org_id) if org_id else DbContext(org_id=None, platform=True)
        async with self._db.transaction(ctx) as session:
            self._container.audit.record(
                session,
                Actor.system(org_id),
                "retention.purge",
                resource_type="organization" if org_id else "platform",
                resource_id=org_id,
                details=dict(counts),
            )
        log.info("retention_purge", org_id=str(org_id) if org_id else PLATFORM_KEY, **counts)
