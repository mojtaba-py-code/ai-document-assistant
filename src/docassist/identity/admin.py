"""Administration service: organisations, settings, departments, users, jobs, audit, usage.

Security rules enforced here (each one has a test):

* **RBAC first, then state checks** - every method that takes a ``Principal`` re-checks its
  permission even though the router already did (defence in depth). The few operator methods
  without a principal (``provision_organization``, ``create_platform_admin``,
  ``verify_chain_as_operator``) exist for the ``docassist`` CLI, which already holds the
  database credentials; no route calls them.
* **Tenant isolation** - tenant methods run in the caller's RLS context *and* filter on the
  caller's organisation explicitly; another tenant's rows are simply invisible, so they
  produce ``404`` exactly like rows that do not exist.
* **No privilege escalation** - an administrator may only assign roles listed in
  ``ASSIGNABLE_ROLES`` (platform admins can never be created through tenant APIs), never a
  clearance above their own, never change their *own* role, clearance or status, and never
  change the privileges of a user whose clearance is higher than their own.
* **Last administrator** - the organisation's active administrators are locked
  (``SELECT ... FOR UPDATE`` in id order) at the start of every user mutation, the acting
  administrator is re-verified against that locked set, and a change that would leave no
  active administrator is refused. Two administrators demoting each other concurrently
  therefore serialise: the second one is no longer an administrator when it runs.
* **Forced re-login** - changing a user's role, clearance, status or departments bumps
  ``token_version`` (outstanding access tokens stop verifying) *and* revokes every session
  (refresh tokens stop working), so the new privileges apply from the next sign-in.
* **No passwords in the API** - new users (and a new organisation's first administrator)
  get an unusable random password and a single-use reset link by email.
* **Everything is audited** - successful changes inside the business transaction; attempts
  refused by the rules above are recorded as ``admin.denied`` in a separate transaction after
  the rollback.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import re
import secrets
import time
import uuid
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Concatenate, Literal

from sqlalchemy import ColumnElement, and_, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from docassist import __version__
from docassist.audit.service import PLATFORM_CHAIN, Actor, verify_chain
from docassist.audit.usage import policy_cache_key
from docassist.authz.permissions import ASSIGNABLE_ROLES, DEFAULT_CLEARANCE, Permission
from docassist.authz.principal import Principal
from docassist.core.context import utcnow
from docassist.core.enums import (
    AuditOutcome,
    Classification,
    JobStatus,
    OrganizationStatus,
    Role,
    UserStatus,
)
from docassist.core.errors import (
    AppError,
    Conflict,
    NotFound,
    PermissionDenied,
    ServiceUnavailable,
    ValidationFailed,
)
from docassist.core.ids import parse_uuid, uuid7
from docassist.core.logging import get_logger
from docassist.core.text import clean_line_text
from docassist.db.models import (
    AuditEvent,
    AuthSession,
    ChunkEmbedding,
    Department,
    Document,
    DocumentVersion,
    Job,
    LlmUsage,
    Organization,
    PasswordResetToken,
    User,
    UserDepartment,
)
from docassist.db.session import DbContext
from docassist.identity.schemas import (
    EXPORT_ROW_CAP,
    AdminHealth,
    AuditEventOut,
    AuditEventPage,
    AuditVerifyOut,
    ComponentHealth,
    CursorError,
    DepartmentCreate,
    DepartmentList,
    DepartmentMembershipIn,
    DepartmentOut,
    DepartmentUpdate,
    DeploymentLimits,
    EffectiveOrgPolicy,
    JobOut,
    JobPage,
    OrganizationCreate,
    OrganizationCreated,
    OrganizationOut,
    OrganizationPage,
    OrganizationSettingsOut,
    OrganizationUpdate,
    OrgSettings,
    OrgSettingsPatch,
    QueueHealth,
    SessionsRevoked,
    UsageBudget,
    UsageReport,
    UsageRow,
    UsageTotals,
    UserCreate,
    UserMembershipOut,
    UserOut,
    UserPage,
    UserUpdate,
    effective_export_rows,
    effective_external_ceiling,
    effective_retention,
    effective_token_budget,
    escape_like,
    int_cursor,
    parse_int_cursor,
    parse_month,
    parse_time_id_cursor,
    slugify,
    time_id_cursor,
    validate_org_settings,
)

if TYPE_CHECKING:
    from docassist.api.container import Container

log = get_logger(__name__)

MAX_PAGE_SIZE = 200
_HEALTH_TIMEOUT_SECONDS = 3.0
_RESOURCE_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,38}_id$")
_ORG_ADMIN = Role.ORGANIZATION_ADMIN
_CONSTRAINT_MESSAGES = {
    "uq_organizations_slug": "An organization with this slug already exists.",
    "uq_users_email": "A user with this email address already exists.",
    "uq_departments_organization_id_slug": "A department with this slug already exists.",
}


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
class _Refusal(Exception):
    """A refused administrative action; audited as ``admin.denied`` after the rollback."""

    def __init__(
        self,
        operation: str,
        reason: str,
        message: str,
        *,
        resource_type: str | None = None,
        resource_id: uuid.UUID | None = None,
        error: type[AppError] = PermissionDenied,
    ) -> None:
        super().__init__(reason)
        self.operation = operation
        self.reason = reason
        self.message = message
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.error = error


def _audits_refusals[**P, R](
    fn: Callable[Concatenate[AdminService, Principal, P], Coroutine[Any, Any, R]],
) -> Callable[Concatenate[AdminService, Principal, P], Coroutine[Any, Any, R]]:
    """Turn a :class:`_Refusal` into an audited ``PermissionDenied`` / ``Conflict``.

    The wrapped method's database session has already been rolled back when the refusal
    reaches this wrapper, so the audit row is written in its own transaction.
    """

    async def wrapper(
        self: AdminService, principal: Principal, /, *args: P.args, **kwargs: P.kwargs
    ) -> R:
        try:
            return await fn(self, principal, *args, **kwargs)
        except _Refusal as refusal:
            await self._audit.record_detached(
                Actor.of(principal),
                "admin.denied",
                outcome=AuditOutcome.DENIED,
                resource_type=refusal.resource_type,
                resource_id=refusal.resource_id,
                details={"operation": refusal.operation, "reason": refusal.reason},
            )
            raise refusal.error(refusal.message, internal_detail=refusal.reason) from None

    return functools.update_wrapper(wrapper, fn)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _constraint_name(exc: IntegrityError) -> str | None:
    candidates = (exc.orig, getattr(exc.orig, "__cause__", None))
    for candidate in candidates:
        name = getattr(candidate, "constraint_name", None)
        if isinstance(name, str):
            return name
    return None


def _page_size(limit: int) -> int:
    return max(1, min(MAX_PAGE_SIZE, limit))


def _time_id_before(created: Any, row_id: Any, cursor: str | None) -> ColumnElement[bool] | None:
    """Keyset predicate for ``ORDER BY created_at DESC, id DESC``."""
    if not cursor:
        return None
    try:
        moment, last_id = parse_time_id_cursor(cursor)
    except CursorError as exc:
        raise ValidationFailed("The cursor is invalid.") from exc
    return or_(created < moment, and_(created == moment, row_id < last_id))


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _min_clearance(a: Classification, b: Classification) -> Classification:
    return a if a.rank <= b.rank else b


def _resource_ids(payload: Mapping[str, Any] | None) -> dict[str, uuid.UUID]:
    """Only well-formed ``*_id`` UUID values leave the job payload (no other internals)."""
    out: dict[str, uuid.UUID] = {}
    for key, value in (payload or {}).items():
        if len(out) >= 5:
            break
        if isinstance(key, str) and _RESOURCE_KEY_RE.fullmatch(key) and isinstance(value, str):
            parsed = parse_uuid(value)
            if parsed is not None:
                out[key] = parsed
    return out


def _job_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        kind=job.kind,
        status=JobStatus(job.status),
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        error_code=job.last_error_code,
        resource_ids=_resource_ids(job.payload),
        created_at=job.created_at,
        updated_at=job.updated_at,
        run_after=job.run_after,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _audit_out(event: AuditEvent) -> AuditEventOut:
    return AuditEventOut(
        id=event.id,
        occurred_at=event.occurred_at,
        actor_user_id=event.actor_user_id,
        actor_role=event.actor_role,
        actor_ip_prefix=event.actor_ip_prefix,
        action=event.action,
        resource_type=event.resource_type,
        resource_id=event.resource_id,
        outcome=AuditOutcome(event.outcome),
        request_id=event.request_id,
        details=dict(event.details or {}),
        sealed=event.hash is not None,
        seal_seq=event.seal_seq,
    )


async def _circuit_states(gateway: Any) -> dict[str, str] | None:
    """Circuit-breaker states exposed by the LLM gateway (``circuit_states()``), if any."""
    getter = getattr(gateway, "circuit_states", None)
    if not callable(getter):
        return None
    try:
        raw = getter()
        if inspect.isawaitable(raw):
            raw = await raw
    except Exception:  # noqa: BLE001 - health reporting must not fail on a broken gateway
        log.warning("llm_circuit_state_unavailable")
        return None
    if not isinstance(raw, Mapping):
        return None
    return {
        clean_line_text(str(name), 64): clean_line_text(str(getattr(state, "value", state)), 32)
        for name, state in list(raw.items())[:20]
    }


@dataclass(frozen=True, slots=True)
class _Invitation:
    email: str
    token: str
    organization_name: str


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class AdminService:
    def __init__(self, container: Container) -> None:
        self._c = container
        self._s = container.settings
        self._db = container.db
        self._audit = container.audit

    # ============================================================ primitives
    async def _unusable_password_hash(self) -> str:
        """Argon2 hash of a discarded random secret: nobody can sign in with it, and a
        login attempt costs the same time as for any other account (no enumeration)."""
        return await asyncio.to_thread(self._c.passwords.hash, secrets.token_urlsafe(48))

    def _reset_token(self, user: User) -> tuple[str, PasswordResetToken]:
        token = self._c.tokens.new_opaque_token()
        row = PasswordResetToken(
            user_id=user.id,
            organization_id=user.organization_id,
            token_hash=self._c.tokens.hash_opaque(token),
            expires_at=utcnow() + timedelta(seconds=self._s.security.password_reset_ttl_seconds),
        )
        return token, row

    async def _send_invitation(self, invitation: _Invitation, *, first_login: bool) -> bool:
        """Email a single-use password link. Failures are logged (never the token)."""
        base = self._s.public_base_url.rstrip("/")
        link = f"{base}/#/reset-password?token={invitation.token}"
        minutes = self._s.security.password_reset_ttl_seconds // 60
        if first_login:
            subject = "Your AI Document Assistant account"
            intro = (
                f"An account has been created for you in {invitation.organization_name} "
                "on AI Document Assistant."
            )
        else:
            subject = "Reset your AI Document Assistant password"
            intro = "An administrator started a password reset for your account."
        body = (
            f"{intro}\n\nOpen this link within {minutes} minutes to choose your password:\n"
            f"{link}\n\nIf the link has expired, use 'Forgot password' on the sign-in page.\n"
            "If you did not expect this message, contact your administrator.\n"
        )
        try:
            await self._c.email.send(invitation.email, subject, body)
        except Exception:  # noqa: BLE001 - the account exists either way; report, don't fail
            log.warning("admin_email_failed", first_login=first_login)
            return False
        return True

    async def _flush(self, session: AsyncSession) -> None:
        """Flush, turning unique/foreign-key violations into a safe ``Conflict``."""
        try:
            await session.flush()
        except IntegrityError as exc:
            name = _constraint_name(exc)
            raise Conflict(
                _CONSTRAINT_MESSAGES.get(name or ""), internal_detail=f"integrity error: {name}"
            ) from None

    async def _revoke_sessions(
        self, session: AsyncSession, user_id: uuid.UUID, reason: str, now: datetime
    ) -> int:
        result = await session.execute(
            update(AuthSession)
            .where(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None))
            .values(revoked_at=now, revoke_reason=reason)
        )
        return int(result.rowcount or 0)  # type: ignore[attr-defined]

    async def _lock_active_admins(
        self, session: AsyncSession, org_id: uuid.UUID
    ) -> dict[uuid.UUID, User]:
        """Lock the organisation's active administrators (always in id order: no deadlocks)."""
        rows = (
            (
                await session.execute(
                    select(User)
                    .where(
                        User.organization_id == org_id,
                        User.role == _ORG_ADMIN.value,
                        User.status == UserStatus.ACTIVE.value,
                    )
                    .order_by(User.id)
                    .with_for_update()
                )
            )
            .scalars()
            .all()
        )
        return {row.id: row for row in rows}

    @staticmethod
    def _acting_clearance(
        principal: Principal, admins: Mapping[uuid.UUID, User], operation: str
    ) -> Classification:
        """Re-verify the actor against the locked administrator set (TOCTOU-safe)."""
        actor = admins.get(principal.user_id)
        if actor is None:
            raise _Refusal(
                operation,
                "actor is not an active organization admin",
                "You don't have permission to perform this action.",
            )
        return Classification(actor.clearance)

    async def _lock_user(
        self, session: AsyncSession, org_id: uuid.UUID, user_id: uuid.UUID
    ) -> User:
        user = (
            await session.execute(
                select(User)
                .where(User.id == user_id, User.organization_id == org_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if user is None:
            raise NotFound()
        return user

    # ============================================================ organisations
    async def create_organization(
        self, principal: Principal, data: OrganizationCreate
    ) -> OrganizationCreated:
        """Platform admins only: organisation + first administrator (invited by email)."""
        principal.require(Permission.ORG_CREATE)
        if not principal.is_platform_admin:
            raise PermissionDenied(internal_detail="org creation requires a platform admin")
        return await self.provision_organization(
            data, actor=Actor.of(principal), acting_user_id=principal.user_id
        )

    async def provision_organization(
        self,
        data: OrganizationCreate,
        *,
        actor: Actor,
        acting_user_id: uuid.UUID | None = None,
        invite: bool = True,
    ) -> OrganizationCreated:
        """Create an organisation and its first ``organization_admin`` atomically.

        Used by the platform API and by the ``docassist create-org`` operator command. The
        transaction runs with the new organisation's id *and* the platform flag, which is
        exactly what row-level security needs to insert both the organisation and its
        first user (and to read back both audit events).
        """
        org_id = uuid7()
        password_hash = await self._unusable_password_hash()
        ctx = DbContext(org_id=org_id, user_id=acting_user_id, platform=True)
        async with self._db.session(ctx) as session:
            taken = await session.scalar(
                select(Organization.id).where(Organization.slug == data.slug)
            )
            if taken is not None:
                raise Conflict(_CONSTRAINT_MESSAGES["uq_organizations_slug"])
            org = Organization(
                id=org_id,
                slug=data.slug,
                name=data.name,
                status=OrganizationStatus.ACTIVE.value,
                settings={},
            )
            session.add(org)
            await self._flush(session)
            clearance = DEFAULT_CLEARANCE[_ORG_ADMIN]
            admin = User(
                organization_id=org_id,
                email=data.admin_email,
                full_name=data.admin_name,
                password_hash=password_hash,
                role=_ORG_ADMIN.value,
                clearance=clearance.value,
                status=UserStatus.ACTIVE.value,
            )
            session.add(admin)
            await self._flush(session)
            token, reset = self._reset_token(admin)
            session.add(reset)
            self._audit.record(
                session,
                actor,
                "platform.organization_created",
                resource_type="organization",
                resource_id=org_id,
                details={"slug": data.slug},
            )
            self._audit.record(
                session,
                actor,
                "admin.user_created",
                resource_type="user",
                resource_id=admin.id,
                details={
                    "role": _ORG_ADMIN.value,
                    "clearance": clearance.value,
                    "first_admin": True,
                    "invited": invite,
                },
                org_id=org_id,
            )
            await self._flush(session)
            out = OrganizationOut.model_validate(org)
            admin_id = admin.id
            await session.commit()
        sent = False
        if invite:
            sent = await self._send_invitation(
                _Invitation(data.admin_email, token, data.name), first_login=True
            )
        return OrganizationCreated(organization=out, admin_user_id=admin_id, invitation_sent=sent)

    async def list_organizations(
        self,
        principal: Principal,
        *,
        status: OrganizationStatus | None = None,
        q: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> OrganizationPage:
        principal.require(Permission.ORG_READ_ANY)
        size = _page_size(limit)
        stmt = select(Organization)
        if status is not None:
            stmt = stmt.where(Organization.status == status.value)
        if q:
            pattern = f"%{escape_like(q)}%"
            stmt = stmt.where(
                or_(
                    Organization.slug.ilike(pattern, escape="\\"),
                    Organization.name.ilike(pattern, escape="\\"),
                )
            )
        keyset = _time_id_before(Organization.created_at, Organization.id, cursor)
        if keyset is not None:
            stmt = stmt.where(keyset)
        stmt = stmt.order_by(Organization.created_at.desc(), Organization.id.desc()).limit(size + 1)
        async with self._db.session(principal.db_context) as session:
            rows = list((await session.execute(stmt)).scalars().all())
        more = len(rows) > size
        rows = rows[:size]
        return OrganizationPage(
            items=[OrganizationOut.model_validate(r) for r in rows],
            next_cursor=time_id_cursor(rows[-1].created_at, rows[-1].id) if more else None,
        )

    async def get_organization(self, principal: Principal, org_id: uuid.UUID) -> OrganizationOut:
        principal.require(Permission.ORG_READ_ANY)
        async with self._db.session(principal.db_context) as session:
            org = await session.get(Organization, org_id)
        if org is None:
            raise NotFound()
        return OrganizationOut.model_validate(org)

    async def update_organization(
        self, principal: Principal, org_id: uuid.UUID, data: OrganizationUpdate
    ) -> OrganizationOut:
        """Rename or suspend/reactivate an organisation. Suspension revokes every session of
        the organisation's users (and sign-in is refused while suspended)."""
        principal.require(Permission.ORG_UPDATE_ANY)
        actor = Actor.of(principal)
        ctx = DbContext(org_id=org_id, user_id=principal.user_id, platform=True)
        async with self._db.session(ctx) as session:
            org = (
                await session.execute(
                    select(Organization).where(Organization.id == org_id).with_for_update()
                )
            ).scalar_one_or_none()
            if org is None:
                raise NotFound()
            changes: dict[str, dict[str, str]] = {}
            if data.name is not None and data.name != org.name:
                changes["name"] = {"before": org.name, "after": data.name}
                org.name = data.name
            revoked = 0
            if data.status is not None and data.status.value != org.status:
                changes["status"] = {"before": org.status, "after": data.status.value}
                org.status = data.status.value
                if data.status is OrganizationStatus.SUSPENDED:
                    result = await session.execute(
                        update(AuthSession)
                        .where(
                            AuthSession.organization_id == org_id,
                            AuthSession.revoked_at.is_(None),
                        )
                        .values(revoked_at=utcnow(), revoke_reason="organization_suspended")
                    )
                    revoked = int(result.rowcount or 0)  # type: ignore[attr-defined]
            if changes:
                self._audit.record(
                    session,
                    actor,
                    "platform.organization_updated",
                    resource_type="organization",
                    resource_id=org_id,
                    details={"changes": changes, "revoked_sessions": revoked},
                )
                if "status" in changes:
                    suspended = data.status is OrganizationStatus.SUSPENDED
                    self._audit.record(
                        session,
                        actor,
                        "organization.suspended" if suspended else "organization.reactivated",
                        resource_type="organization",
                        resource_id=org_id,
                        details={"revoked_sessions": revoked},
                        org_id=org_id,
                    )
                await session.flush()
                await session.refresh(org)
            out = OrganizationOut.model_validate(org)
            await session.commit()
        return out

    # ============================================================ own organisation
    def _settings_view(self, org: Organization) -> OrganizationSettingsOut:
        parsed = OrgSettings.from_stored(org.settings)
        settings = self._s
        return OrganizationSettingsOut(
            organization=OrganizationOut.model_validate(org),
            settings=parsed,
            effective=EffectiveOrgPolicy(
                external_max_classification=effective_external_ceiling(settings, parsed),
                monthly_token_budget=effective_token_budget(settings, parsed),
                retention=effective_retention(settings, parsed),
                export_max_rows=effective_export_rows(parsed),
            ),
            deployment=DeploymentLimits(
                external_max_classification=settings.llm.external_max_classification,
                monthly_token_budget=settings.llm.monthly_token_budget_per_org or None,
                retention=effective_retention(settings, OrgSettings()),
                export_max_rows=EXPORT_ROW_CAP,
            ),
        )

    async def get_organization_settings(self, principal: Principal) -> OrganizationSettingsOut:
        principal.require(Permission.ORG_READ)
        org_id = principal.require_org()
        async with self._db.session(principal.db_context) as session:
            org = await session.get(Organization, org_id)
        if org is None:
            raise NotFound()
        return self._settings_view(org)

    @_audits_refusals
    async def update_organization_settings(
        self, principal: Principal, patch: OrgSettingsPatch
    ) -> OrganizationSettingsOut:
        """Merge ``patch`` into the stored settings; tenants can only tighten AI limits."""
        principal.require(Permission.ORG_UPDATE)
        org_id = principal.require_org()
        changes = patch.changed_fields()
        if not changes:
            raise ValidationFailed("Nothing to update.")
        if "llm" in changes and not principal.has(Permission.LLM_CONFIGURE):
            raise _Refusal(
                "organization.settings",
                "llm:configure required",
                "You don't have permission to change AI settings.",
                resource_type="organization",
                resource_id=org_id,
            )
        problems = validate_org_settings(changes, self._s)
        if problems:
            raise ValidationFailed(
                "The settings would relax a deployment limit.", extra={"errors": problems}
            )
        async with self._db.session(principal.db_context) as session:
            org = (
                await session.execute(
                    select(Organization).where(Organization.id == org_id).with_for_update()
                )
            ).scalar_one_or_none()
            if org is None:
                raise NotFound()
            merged = OrgSettings.from_stored(org.settings)
            before: dict[str, Any] = {}
            after: dict[str, Any] = {}
            for section_name, fields in changes.items():
                section = getattr(merged, section_name)
                for key, value in fields.items():
                    old = getattr(section, key)
                    before[f"{section_name}.{key}"] = getattr(old, "value", old)
                    after[f"{section_name}.{key}"] = getattr(value, "value", value)
                merged = merged.model_copy(update={section_name: section.model_copy(update=fields)})
            org.settings = merged.to_stored(org.settings)
            self._audit.record(
                session,
                Actor.of(principal),
                "admin.org_settings_changed",
                resource_type="organization",
                resource_id=org_id,
                details={"before": before, "after": after},
            )
            await session.flush()
            await session.refresh(org)
            out = self._settings_view(org)
            ceiling_change = after.get("llm.external_max_classification")
            if ceiling_change is not None:
                await self._purge_embeddings_above(session, org_id, str(ceiling_change))
            await session.commit()
        # The gateway caches the organisation's AI policy briefly; a tightened ceiling must
        # apply to the very next request, not after the cache expires.
        await self._c.cache.delete(policy_cache_key(self._c.cache, org_id))
        return out

    async def _purge_embeddings_above(
        self, session: AsyncSession, org_id: uuid.UUID, ceiling: str
    ) -> None:
        """Drop stored embeddings the organisation's (new) policy no longer allows.

        With an external embedding provider, the vectors of documents above the ceiling were
        produced by sending their text outside. Lowering the ceiling removes those vectors in
        the same transaction (the documents stay keyword-searchable) and the pipeline will not
        send such text again.
        """
        if not self._c.embeddings.is_external:
            return
        try:
            allowed = [c.value for c in Classification.at_most(Classification(ceiling))]
        except ValueError:
            return
        above = select(Document.id).where(
            Document.organization_id == org_id, Document.classification.not_in(allowed)
        )
        await session.execute(
            delete(ChunkEmbedding).where(
                ChunkEmbedding.organization_id == org_id, ChunkEmbedding.document_id.in_(above)
            )
        )
        await session.execute(
            update(DocumentVersion)
            .where(
                DocumentVersion.organization_id == org_id, DocumentVersion.document_id.in_(above)
            )
            .values(semantic_indexed=False)
        )

    # ============================================================ departments
    async def _department_out(self, session: AsyncSession, dept: Department) -> DepartmentOut:
        counts = (
            await session.execute(
                select(
                    func.count(),
                    func.count().filter(UserDepartment.is_manager.is_(True)),
                ).where(UserDepartment.department_id == dept.id)
            )
        ).one()
        out = DepartmentOut.model_validate(dept)
        return out.model_copy(
            update={"member_count": int(counts[0]), "manager_count": int(counts[1])}
        )

    async def list_departments(self, principal: Principal) -> DepartmentList:
        """All departments of the caller's organisation (at most 1,000), by name."""
        principal.require(Permission.DEPARTMENT_READ)
        org_id = principal.require_org()
        members = (
            select(func.count())
            .where(UserDepartment.department_id == Department.id)
            .correlate(Department)
            .scalar_subquery()
            .label("member_count")
        )
        managers = (
            select(func.count())
            .where(
                UserDepartment.department_id == Department.id, UserDepartment.is_manager.is_(True)
            )
            .correlate(Department)
            .scalar_subquery()
            .label("manager_count")
        )
        stmt = (
            select(Department, members, managers)
            .where(Department.organization_id == org_id)
            .order_by(Department.name, Department.id)
            .limit(1000)
        )
        async with self._db.session(principal.db_context) as session:
            rows = (await session.execute(stmt)).all()
        items = [
            DepartmentOut.model_validate(dept).model_copy(
                update={"member_count": int(m), "manager_count": int(g)}
            )
            for dept, m, g in rows
        ]
        return DepartmentList(items=items)

    async def create_department(
        self, principal: Principal, data: DepartmentCreate
    ) -> DepartmentOut:
        principal.require(Permission.DEPARTMENT_MANAGE)
        org_id = principal.require_org()
        slug = data.slug or slugify(data.name)
        async with self._db.session(principal.db_context) as session:
            taken = await session.scalar(
                select(Department.id).where(
                    Department.organization_id == org_id, Department.slug == slug
                )
            )
            if taken is not None:
                raise Conflict(_CONSTRAINT_MESSAGES["uq_departments_organization_id_slug"])
            dept = Department(
                organization_id=org_id, name=data.name, slug=slug, description=data.description
            )
            session.add(dept)
            await self._flush(session)
            self._audit.record(
                session,
                Actor.of(principal),
                "admin.department_created",
                resource_type="department",
                resource_id=dept.id,
                details={"slug": slug, "name": data.name},
            )
            await self._flush(session)
            out = DepartmentOut.model_validate(dept)
            await session.commit()
        return out

    async def update_department(
        self, principal: Principal, department_id: uuid.UUID, data: DepartmentUpdate
    ) -> DepartmentOut:
        principal.require(Permission.DEPARTMENT_MANAGE)
        org_id = principal.require_org()
        async with self._db.session(principal.db_context) as session:
            dept = (
                await session.execute(
                    select(Department)
                    .where(Department.id == department_id, Department.organization_id == org_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if dept is None:
                raise NotFound()
            changes: dict[str, dict[str, str | None]] = {}
            for field in ("name", "slug", "description"):
                if field not in data.model_fields_set:
                    continue
                new = getattr(data, field)
                old = getattr(dept, field)
                if new != old:
                    changes[field] = {"before": old, "after": new}
                    setattr(dept, field, new)
            if changes:
                self._audit.record(
                    session,
                    Actor.of(principal),
                    "admin.department_updated",
                    resource_type="department",
                    resource_id=dept.id,
                    details={"changes": changes},
                )
                await self._flush(session)
            out = await self._department_out(session, dept)
            await session.commit()
        return out

    async def delete_department(self, principal: Principal, department_id: uuid.UUID) -> None:
        """Delete a department that no document (not even a soft-deleted one) references."""
        principal.require(Permission.DEPARTMENT_MANAGE)
        org_id = principal.require_org()
        async with self._db.session(principal.db_context) as session:
            dept = (
                await session.execute(
                    select(Department)
                    .where(Department.id == department_id, Department.organization_id == org_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if dept is None:
                raise NotFound()
            documents = int(
                await session.scalar(
                    select(func.count())
                    .select_from(Document)
                    .where(Document.organization_id == org_id, Document.department_id == dept.id)
                )
                or 0
            )
            if documents:
                raise Conflict(
                    "The department still has documents. Move or purge them first.",
                    extra={"document_count": documents},
                )
            members = int(
                await session.scalar(
                    select(func.count())
                    .select_from(UserDepartment)
                    .where(UserDepartment.department_id == dept.id)
                )
                or 0
            )
            self._audit.record(
                session,
                Actor.of(principal),
                "admin.department_deleted",
                resource_type="department",
                resource_id=dept.id,
                details={"slug": dept.slug, "memberships_removed": members},
            )
            try:
                await session.execute(delete(Department).where(Department.id == dept.id))
                await session.commit()
            except IntegrityError:
                # A document was attached concurrently (the foreign key is RESTRICT).
                raise Conflict(
                    "The department still has documents. Move or purge them first."
                ) from None

    # ============================================================ users - reads
    async def _memberships(
        self, session: AsyncSession, user_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, list[UserMembershipOut]]:
        out: dict[uuid.UUID, list[UserMembershipOut]] = {uid: [] for uid in user_ids}
        if not user_ids:
            return out
        rows = await session.execute(
            select(
                UserDepartment.user_id,
                UserDepartment.department_id,
                Department.name,
                UserDepartment.is_manager,
            )
            .join(
                Department,
                and_(
                    Department.id == UserDepartment.department_id,
                    Department.organization_id == UserDepartment.organization_id,
                ),
            )
            .where(UserDepartment.user_id.in_(list(user_ids)))
            .order_by(Department.name, Department.id)
        )
        for user_id, dept_id, dept_name, is_manager in rows:
            out[user_id].append(
                UserMembershipOut(
                    department_id=dept_id, department_name=dept_name, is_manager=is_manager
                )
            )
        return out

    @staticmethod
    def _user_out(user: User, memberships: list[UserMembershipOut], now: datetime) -> UserOut:
        return UserOut(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=Role(user.role),
            clearance=Classification(user.clearance),
            status=UserStatus(user.status),
            mfa_enabled=user.mfa_enabled,
            locked=user.locked_until is not None and user.locked_until > now,
            last_login_at=user.last_login_at,
            created_at=user.created_at,
            updated_at=user.updated_at,
            departments=memberships,
        )

    async def _load_user_out(
        self, session: AsyncSession, org_id: uuid.UUID, user_id: uuid.UUID
    ) -> UserOut:
        user = (
            await session.execute(
                select(User)
                .where(User.id == user_id, User.organization_id == org_id)
                .execution_options(populate_existing=True)
            )
        ).scalar_one_or_none()
        if user is None:
            raise NotFound()
        memberships = await self._memberships(session, [user.id])
        return self._user_out(user, memberships[user.id], utcnow())

    async def list_users(
        self,
        principal: Principal,
        *,
        role: Role | None = None,
        status: UserStatus | None = None,
        department_id: uuid.UUID | None = None,
        q: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> UserPage:
        principal.require(Permission.USER_READ)
        org_id = principal.require_org()
        size = _page_size(limit)
        stmt = select(User).where(User.organization_id == org_id)
        if role is not None:
            stmt = stmt.where(User.role == role.value)
        if status is not None:
            stmt = stmt.where(User.status == status.value)
        if department_id is not None:
            stmt = stmt.where(
                User.id.in_(
                    select(UserDepartment.user_id).where(
                        UserDepartment.department_id == department_id
                    )
                )
            )
        if q:
            pattern = f"%{escape_like(q)}%"
            stmt = stmt.where(
                or_(
                    User.email.ilike(pattern, escape="\\"),
                    User.full_name.ilike(pattern, escape="\\"),
                )
            )
        keyset = _time_id_before(User.created_at, User.id, cursor)
        if keyset is not None:
            stmt = stmt.where(keyset)
        stmt = stmt.order_by(User.created_at.desc(), User.id.desc()).limit(size + 1)
        async with self._db.session(principal.db_context) as session:
            users = list((await session.execute(stmt)).scalars().all())
            more = len(users) > size
            users = users[:size]
            memberships = await self._memberships(session, [u.id for u in users])
        now = utcnow()
        return UserPage(
            items=[self._user_out(u, memberships[u.id], now) for u in users],
            next_cursor=time_id_cursor(users[-1].created_at, users[-1].id) if more else None,
        )

    async def get_user(self, principal: Principal, user_id: uuid.UUID) -> UserOut:
        principal.require(Permission.USER_READ)
        org_id = principal.require_org()
        async with self._db.session(principal.db_context) as session:
            return await self._load_user_out(session, org_id, user_id)

    # ============================================================ users - writes
    async def _check_memberships(
        self,
        session: AsyncSession,
        org_id: uuid.UUID,
        memberships: Sequence[DepartmentMembershipIn],
        role: Role,
    ) -> None:
        if not memberships:
            return
        wanted = {m.department_id for m in memberships}
        found = set(
            (
                await session.execute(
                    select(Department.id).where(
                        Department.organization_id == org_id, Department.id.in_(list(wanted))
                    )
                )
            ).scalars()
        )
        if found != wanted:
            raise ValidationFailed("One or more departments do not exist.")
        if role is not Role.DEPARTMENT_MANAGER and any(m.is_manager for m in memberships):
            raise ValidationFailed("Only department managers can manage a department.")

    @_audits_refusals
    async def create_user(self, principal: Principal, data: UserCreate) -> UserOut:
        """Create a user in the caller's organisation and email them a password link."""
        principal.require(Permission.USER_MANAGE)
        org_id = principal.require_org()
        operation = "user.create"
        # bounds bulk account creation and e-mail probing (see L3 in docs/security-audit.md)
        await self._c.limiter.enforce(
            "admin_user_create", str(principal.user_id), self._s.rate_limit.user_create_per_admin
        )
        if data.role not in ASSIGNABLE_ROLES.get(principal.role, frozenset()):
            raise _Refusal(
                operation,
                f"role {data.role.value} is not assignable by {principal.role.value}",
                "You cannot assign this role.",
            )
        password_hash = await self._unusable_password_hash()
        async with self._db.session(principal.db_context) as session:
            admins = await self._lock_active_admins(session, org_id)
            actor_clearance = self._acting_clearance(principal, admins, operation)
            clearance = data.clearance or _min_clearance(
                DEFAULT_CLEARANCE[data.role], actor_clearance
            )
            if clearance.rank > actor_clearance.rank:
                raise _Refusal(
                    operation,
                    f"clearance {clearance.value} above actor clearance {actor_clearance.value}",
                    "You cannot assign a clearance above your own.",
                )
            await self._check_memberships(session, org_id, data.departments, data.role)
            user = User(
                organization_id=org_id,
                email=data.email,
                full_name=data.full_name,
                password_hash=password_hash,
                role=data.role.value,
                clearance=clearance.value,
                status=UserStatus.ACTIVE.value,
            )
            session.add(user)
            try:
                await self._flush(session)
            except Conflict:
                # Emails are platform-wide login identifiers, so a taken address may belong to
                # another tenant. The conflict cannot be hidden without breaking the admin UX,
                # but every such probe is audited (see docs/threat-model.md, residual risks).
                await self._audit.record_detached(
                    Actor.of(principal),
                    "admin.user_email_conflict",
                    outcome=AuditOutcome.DENIED,
                    resource_type="user",
                    details={"email_domain": data.email.rsplit("@", 1)[-1][:120]},
                )
                raise
            for membership in data.departments:
                session.add(
                    UserDepartment(
                        user_id=user.id,
                        department_id=membership.department_id,
                        organization_id=org_id,
                        is_manager=membership.is_manager,
                    )
                )
            token, reset = self._reset_token(user)
            session.add(reset)
            self._audit.record(
                session,
                Actor.of(principal),
                "admin.user_created",
                resource_type="user",
                resource_id=user.id,
                details={
                    "role": data.role.value,
                    "clearance": clearance.value,
                    "departments": sorted(str(m.department_id) for m in data.departments),
                    "managed_departments": sorted(
                        str(m.department_id) for m in data.departments if m.is_manager
                    ),
                    "invited": True,
                },
            )
            await self._flush(session)
            org_name = await session.scalar(
                select(Organization.name).where(Organization.id == org_id)
            )
            out = await self._load_user_out(session, org_id, user.id)
            await session.commit()
        await self._send_invitation(
            _Invitation(data.email, token, org_name or "your organization"), first_login=True
        )
        return out

    @_audits_refusals
    async def update_user(
        self, principal: Principal, user_id: uuid.UUID, data: UserUpdate
    ) -> UserOut:
        """Change name, role, clearance, status (disable/enable) or department memberships."""
        principal.require(Permission.USER_MANAGE)
        org_id = principal.require_org()
        operation = "user.update"
        async with self._db.session(principal.db_context) as session:
            admins = await self._lock_active_admins(session, org_id)
            actor_clearance = self._acting_clearance(principal, admins, operation)
            user = await self._lock_user(session, org_id, user_id)

            old_role = Role(user.role)
            old_clearance = Classification(user.clearance)
            old_status = UserStatus(user.status)
            new_role = data.role or old_role
            new_clearance = data.clearance or old_clearance
            new_status = data.status or old_status
            current = {
                row.department_id: row.is_manager
                for row in (
                    await session.execute(
                        select(UserDepartment).where(UserDepartment.user_id == user.id)
                    )
                ).scalars()
            }
            wanted: dict[uuid.UUID, bool] | None = None
            if data.departments is not None:
                wanted = {m.department_id: m.is_manager for m in data.departments}
            elif new_role is not Role.DEPARTMENT_MANAGER and any(current.values()):
                wanted = dict.fromkeys(current, False)  # leaving the role ends management

            role_changed = new_role is not old_role
            clearance_changed = new_clearance is not old_clearance
            status_changed = new_status is not old_status
            departments_changed = wanted is not None and wanted != current
            privileged = role_changed or clearance_changed or status_changed or departments_changed

            def refuse(
                reason: str, message: str, error: type[AppError] = PermissionDenied
            ) -> _Refusal:
                return _Refusal(
                    operation,
                    reason,
                    message,
                    resource_type="user",
                    resource_id=user_id,
                    error=error,
                )

            is_self = user.id == principal.user_id
            if is_self and (role_changed or clearance_changed or status_changed):
                raise refuse(
                    "self privilege change",
                    "You cannot change your own role, clearance or status.",
                )
            if privileged and not is_self and old_clearance.rank > actor_clearance.rank:
                raise refuse(
                    "target clearance above actor clearance",
                    "You cannot manage a user whose clearance is higher than yours.",
                )
            assignable = ASSIGNABLE_ROLES.get(principal.role, frozenset())
            if role_changed and (new_role not in assignable or old_role not in assignable):
                raise refuse(
                    f"role change {old_role.value}->{new_role.value} not assignable",
                    "You cannot assign this role.",
                )
            if clearance_changed and new_clearance.rank > actor_clearance.rank:
                raise refuse(
                    f"clearance {new_clearance.value} above actor clearance",
                    "You cannot assign a clearance above your own.",
                )
            if wanted is not None:
                await self._check_memberships(
                    session,
                    org_id,
                    [
                        DepartmentMembershipIn(department_id=k, is_manager=v)
                        for k, v in wanted.items()
                    ],
                    new_role,
                )
            was_admin = old_role is _ORG_ADMIN and old_status is UserStatus.ACTIVE
            stays_admin = new_role is _ORG_ADMIN and new_status is UserStatus.ACTIVE
            if was_admin and not stays_admin and not set(admins) - {user.id}:
                raise refuse(
                    "last active organization admin",
                    "The last active organization administrator cannot be demoted or disabled.",
                    Conflict,
                )

            actor = Actor.of(principal)
            now = utcnow()

            def record(action: str, details: dict[str, Any]) -> None:
                self._audit.record(
                    session,
                    actor,
                    action,
                    resource_type="user",
                    resource_id=user.id,
                    details=details,
                )

            if data.full_name is not None and data.full_name != user.full_name:
                user.full_name = data.full_name
                record("admin.user_updated", {"fields": ["full_name"]})
            if role_changed:
                user.role = new_role.value
                record("admin.role_changed", {"before": old_role.value, "after": new_role.value})
            if clearance_changed:
                user.clearance = new_clearance.value
                record(
                    "admin.clearance_changed",
                    {"before": old_clearance.value, "after": new_clearance.value},
                )
            if status_changed:
                user.status = new_status.value
                record(
                    "admin.user_disabled"
                    if new_status is UserStatus.DISABLED
                    else "admin.user_enabled",
                    {"before": old_status.value, "after": new_status.value},
                )
            if departments_changed and wanted is not None:
                await self._apply_memberships(session, org_id, user.id, current, wanted)
                record(
                    "admin.user_departments_changed",
                    {
                        "added": sorted(str(d) for d in set(wanted) - set(current)),
                        "removed": sorted(str(d) for d in set(current) - set(wanted)),
                        "managed_before": sorted(str(d) for d, m in current.items() if m),
                        "managed_after": sorted(str(d) for d, m in wanted.items() if m),
                    },
                )
            if privileged:
                user.token_version += 1
                revoked = await self._revoke_sessions(session, user.id, "privileges_changed", now)
                record(
                    "admin.sessions_revoked",
                    {"reason": "privileges_changed", "revoked_sessions": revoked},
                )
            await self._flush(session)
            out = await self._load_user_out(session, org_id, user.id)
            await session.commit()
        return out

    @staticmethod
    async def _apply_memberships(
        session: AsyncSession,
        org_id: uuid.UUID,
        user_id: uuid.UUID,
        current: Mapping[uuid.UUID, bool],
        wanted: Mapping[uuid.UUID, bool],
    ) -> None:
        removed = set(current) - set(wanted)
        if removed:
            await session.execute(
                delete(UserDepartment).where(
                    UserDepartment.user_id == user_id,
                    UserDepartment.department_id.in_(list(removed)),
                )
            )
        for dept_id, is_manager in wanted.items():
            if dept_id not in current:
                session.add(
                    UserDepartment(
                        user_id=user_id,
                        department_id=dept_id,
                        organization_id=org_id,
                        is_manager=is_manager,
                    )
                )
            elif current[dept_id] != is_manager:
                await session.execute(
                    update(UserDepartment)
                    .where(
                        UserDepartment.user_id == user_id,
                        UserDepartment.department_id == dept_id,
                    )
                    .values(is_manager=is_manager)
                )

    @_audits_refusals
    async def revoke_user_sessions(
        self, principal: Principal, user_id: uuid.UUID
    ) -> SessionsRevoked:
        """Sign a user out everywhere (access and refresh tokens die with their sessions)."""
        principal.require(Permission.USER_MANAGE)
        org_id = principal.require_org()
        operation = "user.revoke_sessions"
        async with self._db.session(principal.db_context) as session:
            admins = await self._lock_active_admins(session, org_id)
            actor_clearance = self._acting_clearance(principal, admins, operation)
            user = await self._lock_user(session, org_id, user_id)
            if (
                user.id != principal.user_id
                and Classification(user.clearance).rank > actor_clearance.rank
            ):
                raise _Refusal(
                    operation,
                    "target clearance above actor clearance",
                    "You cannot manage a user whose clearance is higher than yours.",
                    resource_type="user",
                    resource_id=user_id,
                )
            revoked = await self._revoke_sessions(session, user.id, "revoked_by_admin", utcnow())
            self._audit.record(
                session,
                Actor.of(principal),
                "admin.sessions_revoked",
                resource_type="user",
                resource_id=user.id,
                details={"reason": "admin_request", "revoked_sessions": revoked},
            )
            await session.commit()
        return SessionsRevoked(revoked_sessions=revoked)

    @_audits_refusals
    async def reset_user_mfa(self, principal: Principal, user_id: uuid.UUID) -> None:
        """Clear another user's two-step verification (lost device or hijacked enrolment).

        The user's sessions are revoked, the change is audited and the user is notified; they
        can enrol again after signing in with their password. Administrators cannot reset
        their own MFA here (that would bypass the password + code check of the profile flow).
        """
        principal.require(Permission.USER_MANAGE)
        org_id = principal.require_org()
        operation = "user.reset_mfa"
        async with self._db.session(principal.db_context) as session:
            admins = await self._lock_active_admins(session, org_id)
            actor_clearance = self._acting_clearance(principal, admins, operation)
            user = await self._lock_user(session, org_id, user_id)
            if user.id == principal.user_id:
                raise _Refusal(
                    operation,
                    "self mfa reset",
                    "Use your profile to change your own two-step verification.",
                    resource_type="user",
                    resource_id=user_id,
                )
            if Classification(user.clearance).rank > actor_clearance.rank:
                raise _Refusal(
                    operation,
                    "target clearance above actor clearance",
                    "You cannot manage a user whose clearance is higher than yours.",
                    resource_type="user",
                    resource_id=user_id,
                )
            if not user.mfa_enabled and user.mfa_secret_enc is None:
                raise Conflict("Two-step verification is not set up for this user.")
            await self._c.auth.admin_reset_mfa(session, principal, user)
            await session.commit()
            email = user.email
        try:
            await self._c.email.send(
                email,
                "Two-step verification was reset",
                "An administrator reset the two-step verification of your account and signed "
                "you out everywhere. Sign in with your password and set it up again.\n\n"
                "If you did not ask for this, contact your administrator immediately.",
            )
        except Exception:  # the reset stands even if the notification fails
            log.exception("mfa_reset_notification_failed")

    @_audits_refusals
    async def send_password_reset(self, principal: Principal, user_id: uuid.UUID) -> bool:
        """Email an active user a single-use password link. Returns whether it was sent."""
        principal.require(Permission.USER_MANAGE)
        org_id = principal.require_org()
        operation = "user.send_reset"
        async with self._db.session(principal.db_context) as session:
            admins = await self._lock_active_admins(session, org_id)
            actor_clearance = self._acting_clearance(principal, admins, operation)
            user = await self._lock_user(session, org_id, user_id)
            if user.status != UserStatus.ACTIVE.value:
                raise Conflict("The account is disabled. Enable it first.")
            if (
                user.id != principal.user_id
                and Classification(user.clearance).rank > actor_clearance.rank
            ):
                raise _Refusal(
                    operation,
                    "target clearance above actor clearance",
                    "You cannot manage a user whose clearance is higher than yours.",
                    resource_type="user",
                    resource_id=user_id,
                )
            await self._c.limiter.enforce(
                "admin_reset", str(user.id), self._s.rate_limit.password_reset_per_ip
            )
            token, reset = self._reset_token(user)
            session.add(reset)
            self._audit.record(
                session,
                Actor.of(principal),
                "admin.password_reset_sent",
                resource_type="user",
                resource_id=user.id,
            )
            await self._flush(session)
            email = user.email
            await session.commit()
        return await self._send_invitation(_Invitation(email, token, ""), first_login=False)

    # ============================================================ jobs
    async def list_jobs(
        self,
        principal: Principal,
        *,
        kind: str | None = None,
        status: JobStatus | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> JobPage:
        """Organisation jobs. Without ``jobs:manage`` a caller sees only jobs they started
        (a job's resource ids would otherwise reveal documents they cannot see)."""
        principal.require(Permission.JOBS_READ)
        org_id = principal.require_org()
        size = _page_size(limit)
        stmt = select(Job).where(Job.organization_id == org_id)
        if not principal.has(Permission.JOBS_MANAGE):
            stmt = stmt.where(Job.created_by == principal.user_id)
        if kind:
            stmt = stmt.where(Job.kind == kind)
        if status is not None:
            stmt = stmt.where(Job.status == status.value)
        keyset = _time_id_before(Job.created_at, Job.id, cursor)
        if keyset is not None:
            stmt = stmt.where(keyset)
        stmt = stmt.order_by(Job.created_at.desc(), Job.id.desc()).limit(size + 1)
        async with self._db.session(principal.db_context) as session:
            jobs = list((await session.execute(stmt)).scalars().all())
        more = len(jobs) > size
        jobs = jobs[:size]
        return JobPage(
            items=[_job_out(j) for j in jobs],
            next_cursor=time_id_cursor(jobs[-1].created_at, jobs[-1].id) if more else None,
        )

    async def _lock_job(self, session: AsyncSession, org_id: uuid.UUID, job_id: uuid.UUID) -> Job:
        job = (
            await session.execute(
                select(Job).where(Job.id == job_id, Job.organization_id == org_id).with_for_update()
            )
        ).scalar_one_or_none()
        if job is None:
            raise NotFound()
        return job

    async def retry_job(self, principal: Principal, job_id: uuid.UUID) -> JobOut:
        """Give a dead-lettered job a fresh set of attempts."""
        principal.require(Permission.JOBS_MANAGE)
        org_id = principal.require_org()
        async with self._db.session(principal.db_context) as session:
            job = await self._lock_job(session, org_id, job_id)
            if job.status != JobStatus.DEAD.value:
                raise Conflict("Only dead-lettered jobs can be retried.")
            previous_error = job.last_error_code
            job.status = JobStatus.QUEUED.value
            job.attempts = 0
            job.run_after = utcnow()
            job.finished_at = None
            job.locked_by = None
            job.locked_until = None
            self._audit.record(
                session,
                Actor.of(principal),
                "jobs.retried",
                resource_type="job",
                resource_id=job.id,
                details={"kind": job.kind, "previous_error_code": previous_error},
            )
            await session.flush()
            await session.refresh(job)
            out = _job_out(job)
            await session.commit()
        return out

    async def cancel_job(self, principal: Principal, job_id: uuid.UUID) -> JobOut:
        """Cancel a job that has not started yet (a running job is never interrupted)."""
        principal.require(Permission.JOBS_MANAGE)
        org_id = principal.require_org()
        async with self._db.session(principal.db_context) as session:
            job = await self._lock_job(session, org_id, job_id)
            if job.status != JobStatus.QUEUED.value:
                raise Conflict("Only queued jobs can be cancelled.")
            job.status = JobStatus.CANCELLED.value
            job.finished_at = utcnow()
            self._audit.record(
                session,
                Actor.of(principal),
                "jobs.cancelled",
                resource_type="job",
                resource_id=job.id,
                details={"kind": job.kind},
            )
            await session.flush()
            await session.refresh(job)
            out = _job_out(job)
            await session.commit()
        return out

    # ============================================================ audit trail
    async def list_audit_events(
        self,
        principal: Principal,
        *,
        action: str | None = None,
        actor_user_id: uuid.UUID | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        outcome: AuditOutcome | None = None,
        occurred_from: datetime | None = None,
        occurred_to: datetime | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> AuditEventPage:
        """Newest first. Tenants see their organisation's chain; platform admins see only
        the platform chain. Reading the trail is itself audited (``audit.events_viewed``)."""
        principal.require(Permission.AUDIT_READ)
        if principal.is_platform_admin:
            chain: ColumnElement[bool] = AuditEvent.organization_id.is_(None)
        else:
            chain = AuditEvent.organization_id == principal.require_org()
        size = _page_size(limit)
        stmt = select(AuditEvent).where(chain)
        used: list[str] = []
        if action:
            used.append("action")
            if action.endswith(".*"):
                stmt = stmt.where(AuditEvent.action.startswith(action[:-1], autoescape=True))
            else:
                stmt = stmt.where(AuditEvent.action == action)
        if actor_user_id is not None:
            used.append("actor")
            stmt = stmt.where(AuditEvent.actor_user_id == actor_user_id)
        if resource_type:
            used.append("resource_type")
            stmt = stmt.where(AuditEvent.resource_type == resource_type)
        if resource_id:
            used.append("resource_id")
            stmt = stmt.where(AuditEvent.resource_id == resource_id)
        if outcome is not None:
            used.append("outcome")
            stmt = stmt.where(AuditEvent.outcome == outcome.value)
        start, end = _as_utc(occurred_from), _as_utc(occurred_to)
        if start is not None:
            used.append("from")
            stmt = stmt.where(AuditEvent.occurred_at >= start)
        if end is not None:
            used.append("to")
            stmt = stmt.where(AuditEvent.occurred_at < end)
        if cursor:
            try:
                stmt = stmt.where(AuditEvent.id < parse_int_cursor(cursor))
            except CursorError as exc:
                raise ValidationFailed("The cursor is invalid.") from exc
        stmt = stmt.order_by(AuditEvent.id.desc()).limit(size + 1)
        async with self._db.session(principal.db_context) as session:
            events = list((await session.execute(stmt)).scalars().all())
            more = len(events) > size
            events = events[:size]
            items = [_audit_out(e) for e in events]
            self._audit.record(
                session,
                Actor.of(principal),
                "audit.events_viewed",
                resource_type="audit_log",
                details={"filters": used, "returned": len(items), "paged": bool(cursor)},
            )
            await session.commit()
        return AuditEventPage(items=items, next_cursor=int_cursor(events[-1].id) if more else None)

    async def verify_audit_chain(self, principal: Principal) -> AuditVerifyOut:
        """Recompute the caller's chain (organisation, or platform for platform admins)."""
        principal.require(Permission.AUDIT_VERIFY)
        org_id = None if principal.is_platform_admin else principal.require_org()
        return await self._verify(principal.db_context, Actor.of(principal), org_id)

    async def verify_chain_as_operator(self, org_id: uuid.UUID | None) -> AuditVerifyOut:
        """CLI (``docassist audit-verify``): verify one chain with a system actor."""
        ctx = DbContext(org_id=org_id) if org_id else DbContext(org_id=None, platform=True)
        return await self._verify(ctx, Actor.system(org_id), org_id)

    async def _verify(
        self, ctx: DbContext, actor: Actor, org_id: uuid.UUID | None
    ) -> AuditVerifyOut:
        key = self._s.security.audit_hmac_key.get_secret_value().encode()
        chain = org_id or PLATFORM_CHAIN
        async with self._db.session(ctx) as session:
            report = await verify_chain(session, key, chain)
            self._audit.record(
                session,
                actor,
                "audit.chain_verified",
                outcome=AuditOutcome.SUCCESS if report.valid else AuditOutcome.FAILURE,
                resource_type="audit_chain",
                resource_id=chain,
                details={
                    "valid": report.valid,
                    "checked": report.checked,
                    "first_bad_seq": report.first_bad_seq,
                    "reason": report.reason,
                },
                org_id=org_id,
            )
            await session.commit()
        if not report.valid:
            log.warning("audit_chain_invalid", chain=str(chain), reason=report.reason)
        return AuditVerifyOut(
            chain="platform" if org_id is None else "organization",
            organization_id=org_id,
            valid=report.valid,
            checked=report.checked,
            head_seq=report.head_seq,
            unsealed=report.unsealed,
            first_bad_seq=report.first_bad_seq,
            reason=report.reason,
            verified_at=utcnow(),
        )

    # ============================================================ usage
    async def usage_report(self, principal: Principal, *, month: str | None = None) -> UsageReport:
        """LLM usage of one UTC calendar month by model and task, plus the token budget.

        Budget consumption counts input + output tokens of every recorded call.
        """
        principal.require(Permission.USAGE_READ)
        org_id = principal.require_org()
        now = utcnow()
        try:
            year, mon = parse_month(month) if month else (now.year, now.month)
        except ValueError as exc:
            raise ValidationFailed("The month must look like YYYY-MM.") from exc
        start = datetime(year, mon, 1, tzinfo=UTC)
        end = (
            datetime(year + 1, 1, 1, tzinfo=UTC)
            if mon == 12
            else datetime(year, mon + 1, 1, tzinfo=UTC)
        )
        stmt = (
            select(
                LlmUsage.model,
                LlmUsage.task,
                func.count(),
                func.coalesce(func.sum(LlmUsage.input_tokens), 0),
                func.coalesce(func.sum(LlmUsage.output_tokens), 0),
                func.coalesce(func.sum(LlmUsage.cost_usd), 0),
            )
            .where(
                LlmUsage.organization_id == org_id,
                LlmUsage.created_at >= start,
                LlmUsage.created_at < end,
            )
            .group_by(LlmUsage.model, LlmUsage.task)
            .order_by(LlmUsage.model, LlmUsage.task)
        )
        async with self._db.session(principal.db_context) as session:
            raw_rows = (await session.execute(stmt)).all()
            org = await session.get(Organization, org_id)
        if org is None:
            raise NotFound()
        rows = [
            UsageRow(
                model=model,
                task=task,
                requests=int(count),
                input_tokens=int(tokens_in),
                output_tokens=int(tokens_out),
                cost_usd=Decimal(cost).quantize(Decimal("0.000001")),
            )
            for model, task, count, tokens_in, tokens_out, cost in raw_rows
        ]
        total_in = sum(r.input_tokens for r in rows)
        total_out = sum(r.output_tokens for r in rows)
        used = total_in + total_out
        parsed = OrgSettings.from_stored(org.settings)
        limit = effective_token_budget(self._s, parsed)
        return UsageReport(
            month=f"{year:04d}-{mon:02d}",
            period_start=start,
            period_end=end,
            rows=rows,
            totals=UsageTotals(
                requests=sum(r.requests for r in rows),
                input_tokens=total_in,
                output_tokens=total_out,
                total_tokens=used,
                cost_usd=sum((r.cost_usd for r in rows), Decimal("0.000000")),
            ),
            budget=UsageBudget(
                deployment_limit=self._s.llm.monthly_token_budget_per_org or None,
                organization_limit=parsed.llm.monthly_token_budget,
                effective_limit=limit,
                used_tokens=used,
                remaining_tokens=None if limit is None else max(0, limit - used),
                exhausted=limit is not None and used >= limit,
            ),
        )

    # ============================================================ health
    @staticmethod
    def may_view_health(principal: Principal) -> bool:
        """Platform admins (``platform:health``) and organisation admins (``org:update``)."""
        return principal.has(Permission.PLATFORM_HEALTH) or (
            principal.org_id is not None and principal.has(Permission.ORG_UPDATE)
        )

    async def health(self, principal: Principal) -> AdminHealth:
        """Dependency status for operators: states and counts only - no hosts, DSNs, keys
        or error strings. Queue depth is scoped by row-level security: an organisation
        admin sees their organisation's jobs, a platform admin the platform-level jobs."""
        if not self.may_view_health(principal):
            raise PermissionDenied(internal_detail="health requires platform:health or org:update")
        components: dict[str, ComponentHealth] = {}

        started = time.perf_counter()
        try:
            await asyncio.wait_for(self._db.ping(), timeout=_HEALTH_TIMEOUT_SECONDS)
            components["database"] = ComponentHealth(
                status="ok",
                detail={"latency_ms": round((time.perf_counter() - started) * 1000, 1)},
            )
        except (TimeoutError, ServiceUnavailable):
            components["database"] = ComponentHealth(status="unavailable")

        if self._c.redis is None:
            components["redis"] = ComponentHealth(status="not_configured")
        else:
            try:
                redis_ok = await asyncio.wait_for(
                    self._c.cache.ping(), timeout=_HEALTH_TIMEOUT_SECONDS
                )
            except TimeoutError:
                redis_ok = False
            components["redis"] = ComponentHealth(status="ok" if redis_ok else "unavailable")

        store = getattr(self._c, "vector_store", None)
        if store is None:
            components["vector_store"] = ComponentHealth(status="not_configured")
        else:
            try:
                store_ok = bool(
                    await asyncio.wait_for(store.health(), timeout=_HEALTH_TIMEOUT_SECONDS)
                )
            except Exception:  # noqa: BLE001 - any failure simply means "unavailable"
                store_ok = False
            components["vector_store"] = ComponentHealth(
                status="ok" if store_ok else "unavailable",
                detail={"backend": clean_line_text(str(getattr(store, "name", "unknown")), 32)},
            )

        embeddings = getattr(self._c, "embeddings", None)
        if embeddings is None:
            components["embeddings"] = ComponentHealth(status="not_configured")
        else:
            components["embeddings"] = ComponentHealth(
                status="ok",
                detail={
                    "provider": clean_line_text(str(getattr(embeddings, "name", "unknown")), 32),
                    "model": clean_line_text(str(getattr(embeddings, "model", "unknown")), 100),
                    "external": bool(getattr(embeddings, "is_external", True)),
                },
            )

        gateway = getattr(self._c, "llm", None)
        if gateway is None:
            components["llm"] = ComponentHealth(status="not_configured")
        else:
            states = await _circuit_states(gateway)
            if states is None:
                components["llm"] = ComponentHealth(status="unknown")
            else:
                tripped = any(state.lower() != "closed" for state in states.values())
                components["llm"] = ComponentHealth(
                    status="degraded" if tripped else "ok", detail={"circuits": states}
                )

        queue: QueueHealth | None = None
        if components["database"].status == "ok":
            scope_filter = (
                Job.organization_id == principal.org_id
                if principal.org_id is not None
                else Job.organization_id.is_(None)
            )
            async with self._db.session(principal.db_context) as session:
                rows = (
                    await session.execute(
                        select(Job.status, func.count()).where(scope_filter).group_by(Job.status)
                    )
                ).all()
            depth = {status.value: 0 for status in JobStatus}
            depth.update({str(status): int(count) for status, count in rows})
            queue = QueueHealth(
                scope="organization" if principal.org_id is not None else "platform", depth=depth
            )

        overall: Literal["ok", "degraded", "unavailable"]
        if components["database"].status != "ok":
            overall = "unavailable"
        elif any(c.status in {"unavailable", "degraded"} for c in components.values()):
            overall = "degraded"
        else:
            overall = "ok"
        return AdminHealth(
            status=overall,
            version=__version__,
            checked_at=utcnow(),
            components=components,
            queue=queue,
        )

    # ============================================================ operator (CLI)
    async def create_platform_admin(
        self, *, email: str, full_name: str, password: str
    ) -> uuid.UUID:
        """``docassist create-platform-admin``: operators are created from the CLI only - no API
        route can create a platform admin."""
        self._c.auth.validate_new_password(password, email=email, name=full_name)
        password_hash = await asyncio.to_thread(self._c.passwords.hash, password)
        async with self._db.session(DbContext(org_id=None, platform=True)) as session:
            user = User(
                organization_id=None,
                email=email,
                full_name=full_name,
                password_hash=password_hash,
                role=Role.PLATFORM_ADMIN.value,
                clearance=DEFAULT_CLEARANCE[Role.PLATFORM_ADMIN].value,
                status=UserStatus.ACTIVE.value,
            )
            session.add(user)
            await self._flush(session)
            self._audit.record(
                session,
                Actor.system(None),
                "platform.admin_created",
                resource_type="user",
                resource_id=user.id,
                details={"via": "cli"},
            )
            await self._flush(session)
            user_id = user.id
            await session.commit()
        return user_id

    async def find_organization_id(self, slug: str) -> uuid.UUID | None:
        """Operator lookup by slug (platform context sees every organisation row)."""
        async with self._db.session(DbContext(org_id=None, platform=True)) as session:
            return await session.scalar(select(Organization.id).where(Organization.slug == slug))

    async def list_organization_ids(self) -> list[tuple[uuid.UUID, str]]:
        async with self._db.session(DbContext(org_id=None, platform=True)) as session:
            rows = await session.execute(
                select(Organization.id, Organization.slug).order_by(Organization.slug)
            )
            return [(row[0], row[1]) for row in rows]
