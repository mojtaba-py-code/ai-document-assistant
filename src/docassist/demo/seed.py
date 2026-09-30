"""Idempotent demo seeding used by ``docassist seed-demo``.

Running it twice changes nothing the second time: organisations, departments and accounts
are looked up by slug/email, documents by title. New accounts get random passwords that are
returned to the caller (the CLI writes them to an owner-only file) and never logged.
Documents go through the real upload path (``container.documents.upload``: validation,
scanning, encryption, audit, ingestion job) and are then processed by the real worker
(``docassist.jobs.worker.Worker.drain``).
"""

from __future__ import annotations

import asyncio
import importlib
import secrets
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.audit.service import Actor
from docassist.authz.permissions import DEFAULT_CLEARANCE
from docassist.authz.principal import Principal
from docassist.core.context import utcnow
from docassist.core.enums import (
    Classification,
    DocumentStatus,
    OrganizationStatus,
    Role,
    UserStatus,
)
from docassist.core.errors import AppError, ValidationFailed
from docassist.core.ids import uuid7
from docassist.db.models import (
    AuthSession,
    Department,
    Document,
    Organization,
    User,
    UserDepartment,
)
from docassist.db.session import DbContext
from docassist.demo.content import ORGS, USERS, DemoDocument, DemoOrg, DemoUser, build_documents

if TYPE_CHECKING:
    from docassist.api.container import Container

_VIA = {"via": "seed-demo"}
_CHUNK = 64 * 1024


class DemoComponentMissing(Exception):
    """A component the seeder needs (documents service, worker) is not available."""

    def __init__(self, component: str) -> None:
        super().__init__(component)
        self.component = component


@dataclass(frozen=True, slots=True)
class Credential:
    email: str
    role: str
    organization: str
    password: str

    def __repr__(self) -> str:  # never let a password reach a log line or traceback
        return f"Credential(email={self.email!r}, role={self.role!r}, password='***')"


@dataclass
class SeedReport:
    organizations_created: list[str] = field(default_factory=list)
    departments_created: int = 0
    users_created: int = 0
    credentials: list[Credential] = field(default_factory=list)
    documents_uploaded: list[str] = field(default_factory=list)
    documents_existing: list[str] = field(default_factory=list)
    documents_rejected: dict[str, str] = field(default_factory=dict)
    jobs_processed: int | None = None
    jobs_dead: int = 0
    unavailable: str | None = None
    """Component that was missing (documents service or worker), if any."""


@dataclass
class Tenant:
    org_id: uuid.UUID
    departments: dict[str, uuid.UUID]


# --------------------------------------------------------------------------- #
# Identities
# --------------------------------------------------------------------------- #
def _ctx(org_id: uuid.UUID | None) -> DbContext:
    return DbContext(org_id=org_id) if org_id else DbContext(org_id=None, platform=True)


async def ensure_org(container: Container, spec: DemoOrg, report: SeedReport) -> Tenant:
    org_id = await container.admin.find_organization_id(spec.slug)
    if org_id is None:
        org_id = uuid7()
        async with container.db.session(DbContext(org_id=org_id, platform=True)) as session:
            session.add(
                Organization(
                    id=org_id,
                    slug=spec.slug,
                    name=spec.name,
                    status=OrganizationStatus.ACTIVE.value,
                    settings={},
                )
            )
            await session.flush()
            container.audit.record(
                session,
                Actor.system(None),
                "platform.organization_created",
                resource_type="organization",
                resource_id=org_id,
                details={"slug": spec.slug, **_VIA},
            )
            await session.commit()
        report.organizations_created.append(spec.slug)
    async with container.db.session(DbContext(org_id=org_id)) as session:
        rows = await session.execute(
            select(Department.slug, Department.id).where(Department.organization_id == org_id)
        )
        departments: dict[str, uuid.UUID] = {row[0]: row[1] for row in rows.all()}
        for slug, name in spec.departments:
            if slug in departments:
                continue
            dept = Department(organization_id=org_id, slug=slug, name=name)
            session.add(dept)
            await session.flush()
            departments[slug] = dept.id
            report.departments_created += 1
            container.audit.record(
                session,
                Actor.system(org_id),
                "admin.department_created",
                resource_type="department",
                resource_id=dept.id,
                details={"slug": slug, **_VIA},
            )
        await session.commit()
    return Tenant(org_id, departments)


def _new_password(container: Container, spec: DemoUser) -> str:
    while True:
        candidate = secrets.token_urlsafe(18)
        try:
            container.auth.validate_new_password(candidate, email=spec.email, name=spec.full_name)
        except ValidationFailed:
            continue
        return candidate


async def _principal(session: AsyncSession, user: User, org_id: uuid.UUID | None) -> Principal:
    memberships = (
        await session.execute(
            select(UserDepartment.department_id, UserDepartment.is_manager).where(
                UserDepartment.user_id == user.id
            )
        )
    ).all()
    return Principal(
        user_id=user.id,
        org_id=org_id,
        role=Role(user.role),
        clearance=Classification(user.clearance),
        session_id=uuid.uuid4(),  # service-level principal; no interactive session exists
        email=user.email,
        department_ids=frozenset(m[0] for m in memberships),
        managed_department_ids=frozenset(m[0] for m in memberships if m[1]),
    )


async def ensure_user(
    container: Container,
    spec: DemoUser,
    tenant: Tenant | None,
    report: SeedReport,
    *,
    reset_password: bool,
    clearance: Classification | None = None,
    sign_in: bool = True,
) -> Principal:
    """Find or create one account. ``sign_in=False`` creates a service-only persona with an
    unusable password (nothing is added to the report's credentials)."""
    org_id = tenant.org_id if tenant else None
    actor = Actor.system(org_id)
    async with container.db.session(_ctx(org_id)) as session:
        user = (
            await session.execute(select(User).where(User.email == spec.email).with_for_update())
        ).scalar_one_or_none()
        password: str | None = None
        if user is None:
            secret = _new_password(container, spec)
            password = secret if sign_in else None
            user = User(
                organization_id=org_id,
                email=spec.email,
                full_name=spec.full_name,
                password_hash=await asyncio.to_thread(container.passwords.hash, secret),
                role=spec.role.value,
                clearance=(clearance or DEFAULT_CLEARANCE[spec.role]).value,
                status=UserStatus.ACTIVE.value,
            )
            session.add(user)
            await session.flush()
            for slug in spec.departments:
                if tenant is None:
                    raise ValueError("platform accounts cannot belong to departments")
                session.add(
                    UserDepartment(
                        user_id=user.id,
                        department_id=tenant.departments[slug],
                        organization_id=org_id,
                        is_manager=slug in spec.managed,
                    )
                )
            container.audit.record(
                session,
                actor,
                "admin.user_created",
                resource_type="user",
                resource_id=user.id,
                details={"role": spec.role.value, **_VIA},
            )
            report.users_created += 1
        elif reset_password and sign_in:
            password = _new_password(container, spec)
            now = utcnow()
            user.password_hash = await asyncio.to_thread(container.passwords.hash, password)
            user.password_changed_at = now
            user.token_version += 1
            user.failed_login_count = 0
            user.locked_until = None
            await session.execute(
                update(AuthSession)
                .where(AuthSession.user_id == user.id, AuthSession.revoked_at.is_(None))
                .values(revoked_at=now, revoke_reason="password_reset")
            )
            container.audit.record(
                session,
                actor,
                "admin.password_reset",
                resource_type="user",
                resource_id=user.id,
                details=dict(_VIA),
            )
        await session.flush()
        principal = await _principal(session, user, org_id)
        await session.commit()
    if password is not None:
        report.credentials.append(
            Credential(spec.email, spec.role.value, spec.org or "platform", password)
        )
    return principal


async def seed_identities(
    container: Container, report: SeedReport, *, reset_passwords: bool = False
) -> tuple[dict[str, Principal], dict[str, Tenant]]:
    tenants = {org.slug: await ensure_org(container, org, report) for org in ORGS}
    principals: dict[str, Principal] = {}
    for spec in USERS:
        tenant = tenants[spec.org] if spec.org else None
        principals[spec.key] = await ensure_user(
            container, spec, tenant, report, reset_password=reset_passwords
        )
    return principals, tenants


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
async def _chunks(data: bytes) -> AsyncIterator[bytes]:
    for start in range(0, len(data), _CHUNK):
        yield data[start : start + _CHUNK]


async def _document_exists(container: Container, org_id: uuid.UUID, title: str) -> bool:
    async with container.db.session(DbContext(org_id=org_id)) as session:
        found = await session.scalar(
            select(Document.id).where(
                Document.organization_id == org_id,
                Document.title == title,
                Document.status != DocumentStatus.DELETED.value,
            )
        )
    return found is not None


async def seed_documents(
    container: Container,
    documents: list[DemoDocument],
    principals: dict[str, Principal],
    tenants: dict[str, Tenant],
    report: SeedReport,
) -> None:
    service = getattr(container, "documents", None)
    if service is None:
        raise DemoComponentMissing("documents service")
    for doc in documents:
        tenant = tenants[doc.org]
        if await _document_exists(container, tenant.org_id, doc.title):
            report.documents_existing.append(doc.key)
            continue
        try:
            await service.upload(
                principals[doc.uploader],
                stream=_chunks(doc.data),
                filename=doc.filename,
                declared_mime=doc.mime,
                title=doc.title,
                classification=doc.classification,
                department_id=tenant.departments[doc.department] if doc.department else None,
                doc_type=doc.doc_type,
                tags=list(doc.tags),
                allowed_roles=(),
            )
        except AppError as exc:
            report.documents_rejected[doc.key] = exc.code
            continue
        report.documents_uploaded.append(doc.key)


async def process_pending_jobs(
    container: Container, *, max_jobs: int, report: SeedReport | None = None
) -> int:
    """Run the real worker until the queue is empty (or ``max_jobs`` jobs ran).

    The worker's database-role check runs first, exactly as for ``docassist worker``.
    Returns the number of jobs executed; dead-lettered ones are counted in the report.
    """
    try:
        module = importlib.import_module("docassist.jobs.worker")
    except ImportError as exc:
        raise DemoComponentMissing("background worker") from exc
    startup_error: Any = getattr(module, "WorkerStartupError", RuntimeError)
    try:
        worker = module.Worker(
            container,
            concurrency=container.settings.worker.concurrency,
            poll_interval=container.settings.worker.poll_interval_seconds,
        )
        await worker.verify_role()
    except startup_error as exc:
        raise DemoComponentMissing("background worker (database role check failed)") from exc
    outcomes = list(await worker.drain(max_jobs))
    if report is not None:
        report.jobs_dead += sum(1 for o in outcomes if getattr(o, "status", None) == "dead")
    return len(outcomes)


async def seed_demo(
    container: Container,
    *,
    report: SeedReport | None = None,
    today: date | None = None,
    reset_passwords: bool = False,
    documents: bool = True,
    process: bool = True,
    max_jobs: int = 500,
) -> SeedReport:
    """Create (or complete) the demo tenants. Safe to run repeatedly.

    Pass your own ``report`` to keep the issued credentials even if a later step raises. A
    missing documents service or worker does not raise: it is recorded in
    ``report.unavailable`` (accounts and any uploads done so far stay in place, and a later
    run completes the rest).
    """
    report = report if report is not None else SeedReport()
    principals, tenants = await seed_identities(container, report, reset_passwords=reset_passwords)
    if not documents:
        return report
    try:
        corpus = build_documents(today or utcnow().date())
        await seed_documents(container, corpus, principals, tenants, report)
        if process:
            report.jobs_processed = await process_pending_jobs(
                container, max_jobs=max_jobs, report=report
            )
    except DemoComponentMissing as exc:
        report.unavailable = exc.component
    return report
