"""Authentication: login, MFA, session & refresh-token lifecycle, password reset.

Design points
-------------
* **No user enumeration** - unknown email, wrong password, disabled and locked accounts all
  return the same error after the same (Argon2) amount of work.
* **Lockout** - exponential lock after ``lockout_threshold`` consecutive failures, plus
  per-IP and per-account rate limits in front of it (credential stuffing).
* **Sessions** - every login creates an ``auth_sessions`` row; access JWTs carry its id and
  are rejected as soon as the session is revoked (logout is immediate, not "after TTL").
* **Refresh-token rotation with reuse detection** - refresh tokens are single use; presenting
  an already-used token revokes the whole session (the token was stolen or replayed).
* **Password reset** - single-use, peppered-hash token, 30 min TTL, identical response for
  unknown emails; a successful reset revokes every session of the user.
* **MFA (TOTP)** - secret encrypted at rest; login returns an attempt-limited challenge
  instead of tokens when MFA is enabled; codes cannot be replayed.
"""

from __future__ import annotations

import asyncio
import ipaddress
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.audit.service import Actor, AuditLogger
from docassist.authz.principal import Principal
from docassist.cache.ratelimit import RateLimiter
from docassist.core.config import Settings
from docassist.core.context import utcnow
from docassist.core.enums import AuditOutcome, Classification, OrganizationStatus, Role, UserStatus
from docassist.core.errors import AuthenticationFailed, PermissionDenied, ValidationFailed
from docassist.core.logging import get_logger
from docassist.db.models import (
    AuthSession,
    MfaChallenge,
    Organization,
    PasswordResetToken,
    RefreshToken,
    User,
    UserDepartment,
)
from docassist.db.session import Database, DbContext
from docassist.identity.notifications import EmailSender
from docassist.observability import metrics
from docassist.security import totp
from docassist.security.crypto import KeyRing
from docassist.security.passwords import PasswordPolicy, PasswordPolicyError, PasswordService
from docassist.security.tokens import TokenError, TokenService

log = get_logger(__name__)

GENERIC_LOGIN_ERROR = "Invalid email or password, or the account is temporarily locked."
_LAST_SEEN_RESOLUTION = timedelta(minutes=5)


def ip_prefix(ip: str | None) -> str | None:
    """Anonymise an IP to its /24 (IPv4) or /48 (IPv6) network - enough for abuse analysis."""
    if not ip:
        return None
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None
    prefix = 24 if addr.version == 4 else 48
    return str(ipaddress.ip_network(f"{addr}/{prefix}", strict=False))


def normalize_email(email: str) -> str:
    return email.strip().lower()


@dataclass(frozen=True, slots=True)
class TokenPair:
    access_token: str
    access_expires_at: datetime
    refresh_token: str
    refresh_expires_at: datetime
    session_id: uuid.UUID


@dataclass(frozen=True, slots=True)
class LoginResult:
    tokens: TokenPair | None = None
    mfa_challenge: str | None = None


@dataclass(frozen=True, slots=True)
class MfaEnrollment:
    secret: str
    provisioning_uri: str


class AuthService:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        passwords: PasswordService,
        tokens: TokenService,
        ring: KeyRing,
        audit: AuditLogger,
        limiter: RateLimiter,
        email: EmailSender,
    ) -> None:
        self._s = settings
        self._db = db
        self._passwords = passwords
        self._tokens = tokens
        self._ring = ring
        self._audit = audit
        self._limiter = limiter
        self._email = email
        self._background: set[asyncio.Task[None]] = set()
        self.policy = PasswordPolicy(
            min_length=settings.security.password_min_length,
            max_length=settings.security.password_max_length,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _ctx_for(
        org_id: uuid.UUID | None, is_platform: bool, user_id: uuid.UUID | None = None
    ) -> DbContext:
        return DbContext(org_id=org_id, user_id=user_id, platform=is_platform)

    async def _lookup(
        self, sql: str, param: object
    ) -> tuple[uuid.UUID, uuid.UUID | None, bool] | None:
        async with self._db.session(DbContext.anonymous()) as session:
            row = (await session.execute(text(sql), {"p": param})).first()
        return (row[0], row[1], bool(row[2])) if row else None

    def _mfa_context(self, user_id: uuid.UUID) -> bytes:
        return f"mfa-secret:{user_id}".encode()

    def _lockout_delay(self, lockout_count: int) -> timedelta:
        sec = self._s.security
        seconds = min(
            sec.lockout_max_seconds, sec.lockout_base_seconds * (2 ** max(0, lockout_count))
        )
        return timedelta(seconds=seconds)

    async def _record_failure(self, session: AsyncSession, user: User) -> None:
        user.failed_login_count += 1
        if user.failed_login_count >= self._s.security.lockout_threshold:
            user.locked_until = utcnow() + self._lockout_delay(user.lockout_count)
            user.lockout_count += 1
            user.failed_login_count = 0
            self._audit.record(
                session,
                Actor(user.id, user.role, user.organization_id),
                "auth.account_locked",
                outcome=AuditOutcome.DENIED,
                resource_type="user",
                resource_id=user.id,
            )
            metrics.AUTH_EVENTS.labels(event="lockout", outcome="locked").inc()

    @staticmethod
    def _clear_failures(user: User) -> None:
        """Only a *complete* authentication (password and, if enabled, MFA) clears failures."""
        user.failed_login_count = 0
        user.lockout_count = 0
        user.locked_until = None

    @staticmethod
    async def _invalidate_mfa_challenges(
        session: AsyncSession, user_id: uuid.UUID, now: datetime
    ) -> None:
        """At most one live MFA challenge per user (a new login kills the previous one)."""
        await session.execute(
            update(MfaChallenge)
            .where(MfaChallenge.user_id == user_id, MfaChallenge.used_at.is_(None))
            .values(used_at=now)
        )

    @staticmethod
    async def _invalidate_reset_tokens(
        session: AsyncSession, user_id: uuid.UUID, now: datetime
    ) -> None:
        """Every outstanding reset link dies when the password changes by any route."""
        await session.execute(
            update(PasswordResetToken)
            .where(PasswordResetToken.user_id == user_id, PasswordResetToken.used_at.is_(None))
            .values(used_at=now)
        )

    def _locked(self, user: User) -> bool:
        return user.locked_until is not None and user.locked_until > utcnow()

    async def _verify_current_password(
        self, session: AsyncSession, principal: Principal, user: User, password: str, action: str
    ) -> bool:
        """Re-authentication inside a session: rate-limited and subject to the lockout.

        A stolen access token must not become an unthrottled password oracle.
        """
        await self._limiter.enforce(
            "reauth_account", str(user.id), self._s.rate_limit.login_per_account
        )
        if not self._locked(user) and self._passwords.verify(user.password_hash, password):
            return True
        if not self._locked(user):
            await self._record_failure(session, user)
        self._audit.record(
            session,
            Actor.of(principal),
            f"{action}_failed",
            outcome=AuditOutcome.FAILURE,
            resource_type="user",
            resource_id=user.id,
            details={"reason": "reauthentication_failed"},
        )
        await session.commit()
        return False

    async def _notify(self, email: str, subject: str, body: str) -> None:
        try:
            await self._email.send(email, subject, body)
        except Exception:  # a notification must never fail the security change itself
            log.exception("notification_failed", subject=subject)

    async def _issue(
        self,
        session: AsyncSession,
        user: User,
        *,
        user_agent: str | None,
        client_ip: str | None,
        mfa: bool,
    ) -> TokenPair:
        now = utcnow()
        auth_session = AuthSession(
            user_id=user.id,
            organization_id=user.organization_id,
            expires_at=now + timedelta(seconds=self._s.security.session_absolute_ttl_seconds),
            last_seen_at=now,
            user_agent=(user_agent or "")[:256] or None,
            ip_prefix=ip_prefix(client_ip),
            mfa_verified=mfa,
        )
        session.add(auth_session)
        await session.flush()
        return await self._new_token_pair(session, user, auth_session)

    async def _new_token_pair(
        self, session: AsyncSession, user: User, auth_session: AuthSession
    ) -> TokenPair:
        now = utcnow()
        refresh_plain = self._tokens.new_opaque_token()
        refresh_expires = min(
            now + timedelta(seconds=self._s.security.refresh_token_ttl_seconds),
            auth_session.expires_at,
        )
        session.add(
            RefreshToken(
                session_id=auth_session.id,
                organization_id=user.organization_id,
                token_hash=self._tokens.hash_opaque(refresh_plain),
                expires_at=refresh_expires,
            )
        )
        access, access_expires = self._tokens.issue_access_token(
            user_id=user.id,
            org_id=user.organization_id,
            role=user.role,
            session_id=auth_session.id,
            token_version=user.token_version,
        )
        return TokenPair(access, access_expires, refresh_plain, refresh_expires, auth_session.id)

    # -------------------------------------------------------------------- login
    async def login(
        self, email: str, password: str, *, client_ip: str | None, user_agent: str | None
    ) -> LoginResult:
        email_n = normalize_email(email)
        rl = self._s.rate_limit
        await self._limiter.enforce("login_ip", client_ip or "unknown", rl.login_per_ip)
        await self._limiter.enforce("login_account", email_n, rl.login_per_account)

        found = await self._lookup("SELECT * FROM app.auth_find_user(:p)", email_n)
        if found is None:
            self._passwords.verify(None, password)  # equalise timing
            metrics.AUTH_EVENTS.labels(event="login", outcome="unknown_user").inc()
            await self._audit.record_detached(
                Actor(None, None, None, ip_prefix(client_ip)),
                "auth.login_failed",
                outcome=AuditOutcome.FAILURE,
                details={"reason": "unknown_account"},
            )
            raise AuthenticationFailed(GENERIC_LOGIN_ERROR)

        user_id, org_id, is_platform = found
        async with self._db.session(self._ctx_for(org_id, is_platform, user_id)) as session:
            user = (
                await session.execute(select(User).where(User.id == user_id).with_for_update())
            ).scalar_one()
            org_ok = True
            if org_id is not None:
                org = await session.get(Organization, org_id)
                org_ok = org is not None and org.status == OrganizationStatus.ACTIVE.value
            now = utcnow()
            locked = user.locked_until is not None and user.locked_until > now
            password_ok = self._passwords.verify(user.password_hash, password)
            actor = Actor(user.id, user.role, user.organization_id, ip_prefix(client_ip))

            if not password_ok or locked or user.status != UserStatus.ACTIVE.value or not org_ok:
                reason = (
                    "locked"
                    if locked
                    else "disabled"
                    if user.status != UserStatus.ACTIVE.value
                    else "org_suspended"
                    if not org_ok
                    else "bad_password"
                )
                if not password_ok and not locked:
                    await self._record_failure(session, user)
                self._audit.record(
                    session,
                    actor,
                    "auth.login_failed",
                    outcome=AuditOutcome.FAILURE,
                    resource_type="user",
                    resource_id=user.id,
                    details={"reason": reason},
                )
                await session.commit()
                metrics.AUTH_EVENTS.labels(event="login", outcome=reason).inc()
                raise AuthenticationFailed(GENERIC_LOGIN_ERROR)

            if self._passwords.needs_rehash(user.password_hash):
                user.password_hash = self._passwords.hash(password)

            if user.mfa_enabled:
                # The failure counters are NOT reset here: a correct password is only half of
                # the login, and resetting would let an attacker who knows the password
                # brute-force one-time codes across fresh challenges without ever locking.
                await self._invalidate_mfa_challenges(session, user.id, now)
                challenge = self._tokens.new_opaque_token()
                session.add(
                    MfaChallenge(
                        user_id=user.id,
                        organization_id=user.organization_id,
                        token_hash=self._tokens.hash_opaque(challenge),
                        expires_at=now
                        + timedelta(seconds=self._s.security.mfa_challenge_ttl_seconds),
                    )
                )
                self._audit.record(
                    session,
                    actor,
                    "auth.mfa_challenge_issued",
                    resource_type="user",
                    resource_id=user.id,
                )
                await session.commit()
                return LoginResult(mfa_challenge=challenge)

            self._clear_failures(user)
            pair = await self._issue(
                session, user, user_agent=user_agent, client_ip=client_ip, mfa=False
            )
            user.last_login_at = now
            self._audit.record(
                session, actor, "auth.login", resource_type="session", resource_id=pair.session_id
            )
            await session.commit()
        metrics.AUTH_EVENTS.labels(event="login", outcome="success").inc()
        return LoginResult(tokens=pair)

    async def complete_mfa(
        self, challenge: str, code: str, *, client_ip: str | None, user_agent: str | None
    ) -> TokenPair:
        await self._limiter.enforce(
            "login_ip", client_ip or "unknown", self._s.rate_limit.login_per_ip
        )
        digest = self._tokens.hash_opaque(challenge)
        found = await self._lookup("SELECT * FROM app.auth_find_mfa_challenge(:p)", digest)
        if found is None:
            raise AuthenticationFailed("The sign-in challenge is invalid or has expired.")
        user_id, org_id, is_platform = found
        # Per-account limit in addition to the per-IP one: codes cannot be sprayed from many IPs.
        await self._limiter.enforce(
            "mfa_account", str(user_id), self._s.rate_limit.login_per_account
        )
        async with self._db.session(self._ctx_for(org_id, is_platform, user_id)) as session:
            row = (
                await session.execute(
                    select(MfaChallenge).where(MfaChallenge.token_hash == digest).with_for_update()
                )
            ).scalar_one()
            user = (
                await session.execute(select(User).where(User.id == user_id).with_for_update())
            ).scalar_one()
            now = utcnow()
            actor = Actor(user.id, user.role, user.organization_id, ip_prefix(client_ip))
            locked = user.locked_until is not None and user.locked_until > now
            if (
                row.used_at is not None
                or row.expires_at <= now
                or row.attempts >= 5
                or locked
                or user.status != UserStatus.ACTIVE.value
            ):
                raise AuthenticationFailed("The sign-in challenge is invalid or has expired.")
            row.attempts += 1
            secret = self._ring.decrypt_value(
                user.mfa_secret_enc or b"", self._mfa_context(user.id)
            ).decode()
            step = totp.verify(secret, code, last_used_step=user.mfa_last_used_step)
            if step is None:
                # Wrong codes count toward the same exponential lockout as wrong passwords.
                await self._record_failure(session, user)
                if user.locked_until is not None and user.locked_until > now:
                    row.used_at = now  # a locked account's challenge is dead
                self._audit.record(
                    session,
                    actor,
                    "auth.mfa_failed",
                    outcome=AuditOutcome.FAILURE,
                    resource_type="user",
                    resource_id=user.id,
                )
                await session.commit()
                metrics.AUTH_EVENTS.labels(event="mfa", outcome="failure").inc()
                raise AuthenticationFailed("The one-time code is invalid.")
            row.used_at = now
            user.mfa_last_used_step = step
            user.last_login_at = now
            self._clear_failures(user)
            pair = await self._issue(
                session, user, user_agent=user_agent, client_ip=client_ip, mfa=True
            )
            self._audit.record(
                session,
                actor,
                "auth.login",
                resource_type="session",
                resource_id=pair.session_id,
                details={"mfa": True},
            )
            await session.commit()
        metrics.AUTH_EVENTS.labels(event="mfa", outcome="success").inc()
        return pair

    # ------------------------------------------------------------------ refresh
    async def refresh(self, refresh_token: str, *, client_ip: str | None) -> TokenPair:
        digest = self._tokens.hash_opaque(refresh_token)
        found = await self._lookup("SELECT * FROM app.auth_find_refresh_token(:p)", digest)
        if found is None:
            metrics.AUTH_EVENTS.labels(event="refresh", outcome="unknown").inc()
            raise AuthenticationFailed("Your session has expired. Please sign in again.")
        session_id, org_id, is_platform = found
        await self._limiter.enforce(
            "refresh", str(session_id), self._s.rate_limit.refresh_per_session
        )
        async with self._db.session(self._ctx_for(org_id, is_platform)) as session:
            token = (
                await session.execute(
                    select(RefreshToken).where(RefreshToken.token_hash == digest).with_for_update()
                )
            ).scalar_one()
            auth_session = (
                await session.execute(
                    select(AuthSession).where(AuthSession.id == session_id).with_for_update()
                )
            ).scalar_one()
            user = await session.get(User, auth_session.user_id)
            now = utcnow()
            actor = Actor(
                auth_session.user_id, user.role if user else None, org_id, ip_prefix(client_ip)
            )
            if token.used_at is not None:
                # Reuse of a rotated token: assume theft, kill the whole session.
                auth_session.revoked_at = auth_session.revoked_at or now
                auth_session.revoke_reason = "refresh_token_reuse"
                self._audit.record(
                    session,
                    actor,
                    "auth.refresh_token_reuse_detected",
                    outcome=AuditOutcome.DENIED,
                    resource_type="session",
                    resource_id=auth_session.id,
                )
                await session.commit()
                metrics.SECURITY_EVENTS.labels(kind="refresh_token_reuse").inc()
                raise AuthenticationFailed("Your session has expired. Please sign in again.")
            if (
                auth_session.revoked_at is not None
                or token.expires_at <= now
                or auth_session.expires_at <= now
                or user is None
                or user.status != UserStatus.ACTIVE.value
            ):
                raise AuthenticationFailed("Your session has expired. Please sign in again.")
            if user.organization_id is not None:
                org = await session.get(Organization, user.organization_id)
                if org is None or org.status != OrganizationStatus.ACTIVE.value:
                    raise AuthenticationFailed("Your session has expired. Please sign in again.")
            token.used_at = now
            auth_session.last_seen_at = now
            pair = await self._new_token_pair(session, user, auth_session)
            await session.commit()
        metrics.AUTH_EVENTS.labels(event="refresh", outcome="success").inc()
        return pair

    # ------------------------------------------------------------------ logout
    async def logout(self, principal: Principal, *, everywhere: bool = False) -> None:
        async with self._db.session(principal.db_context) as session:
            now = utcnow()
            stmt = update(AuthSession).where(AuthSession.revoked_at.is_(None))
            if everywhere:
                stmt = stmt.where(AuthSession.user_id == principal.user_id)
            else:
                stmt = stmt.where(AuthSession.id == principal.session_id)
            await session.execute(stmt.values(revoked_at=now, revoke_reason="logout"))
            self._audit.record(
                session,
                Actor.of(principal),
                "auth.logout",
                resource_type="session",
                resource_id=principal.session_id,
                details={"everywhere": everywhere},
            )
            await session.commit()

    # ----------------------------------------------------------- authenticate
    async def authenticate(self, access_token: str, *, client_ip: str | None = None) -> Principal:
        try:
            claims = self._tokens.verify_access_token(access_token)
        except TokenError as exc:
            raise AuthenticationFailed("Authentication required.") from exc
        ctx = DbContext(
            org_id=claims.org_id, user_id=claims.user_id, platform=claims.org_id is None
        )
        async with self._db.session(ctx) as session:
            row = (
                await session.execute(
                    select(AuthSession, User)
                    .join(User, User.id == AuthSession.user_id)
                    .where(AuthSession.id == claims.session_id, User.id == claims.user_id)
                )
            ).first()
            if row is None:
                raise AuthenticationFailed("Authentication required.")
            auth_session, user = row
            now = utcnow()
            if (
                auth_session.revoked_at is not None
                or auth_session.expires_at <= now
                or user.status != UserStatus.ACTIVE.value
                or user.token_version != claims.token_version
                or user.role != claims.role
                or user.organization_id != claims.org_id
            ):
                raise AuthenticationFailed("Authentication required.")
            if claims.org_id is not None:
                org_status = (
                    await session.execute(
                        select(Organization.status).where(Organization.id == claims.org_id)
                    )
                ).scalar_one_or_none()
                if org_status != OrganizationStatus.ACTIVE.value:
                    raise AuthenticationFailed("Authentication required.")
            memberships = (
                await session.execute(
                    select(UserDepartment.department_id, UserDepartment.is_manager).where(
                        UserDepartment.user_id == user.id
                    )
                )
            ).all()
            if (
                auth_session.last_seen_at is None
                or now - auth_session.last_seen_at > _LAST_SEEN_RESOLUTION
            ):
                auth_session.last_seen_at = now
                await session.commit()
        return Principal(
            user_id=user.id,
            org_id=user.organization_id,
            role=Role(user.role),
            clearance=Classification(user.clearance),
            session_id=auth_session.id,
            email=user.email,
            department_ids=frozenset(m.department_id for m in memberships),
            managed_department_ids=frozenset(m.department_id for m in memberships if m.is_manager),
            ip_prefix=ip_prefix(client_ip),
        )

    # --------------------------------------------------------------- profile
    async def profile(self, principal: Principal) -> tuple[str, bool]:
        """Return ``(full_name, mfa_enabled)`` for the signed-in user."""
        async with self._db.session(principal.db_context) as session:
            row = (
                await session.execute(
                    select(User.full_name, User.mfa_enabled).where(User.id == principal.user_id)
                )
            ).one()
        return str(row.full_name), bool(row.mfa_enabled)

    # ------------------------------------------------------------ passwords
    def validate_new_password(self, password: str, *, email: str, name: str | None) -> None:
        try:
            self.policy.validate(password, email=email, name=name)
        except PasswordPolicyError as exc:
            raise ValidationFailed(str(exc)) from exc

    async def change_password(self, principal: Principal, current: str, new: str) -> None:
        async with self._db.session(principal.db_context) as session:
            user = (
                await session.execute(
                    select(User).where(User.id == principal.user_id).with_for_update()
                )
            ).scalar_one()
            # Validate the NEW password first: otherwise "wrong current password" (403) vs
            # "weak new password" (422) would reveal whether a guessed current password is right.
            self.validate_new_password(new, email=user.email, name=user.full_name)
            if not await self._verify_current_password(
                session, principal, user, current, "auth.password_change"
            ):
                raise PermissionDenied("The current password is incorrect.")
            now = utcnow()
            user.password_hash = self._passwords.hash(new)
            user.password_changed_at = now
            user.token_version += 1  # invalidates all outstanding access tokens
            await session.execute(
                update(AuthSession)
                .where(AuthSession.user_id == user.id, AuthSession.revoked_at.is_(None))
                .values(revoked_at=now, revoke_reason="password_changed")
            )
            await self._invalidate_reset_tokens(session, user.id, now)
            await self._invalidate_mfa_challenges(session, user.id, now)
            self._audit.record(
                session,
                Actor.of(principal),
                "auth.password_changed",
                resource_type="user",
                resource_id=user.id,
            )
            await session.commit()
            recipient = user.email
        await self._notify(
            recipient,
            "Your AI Document Assistant password was changed",
            "The password of your account was just changed and all sessions were signed out.\n\n"
            "If this was not you, contact your administrator immediately.",
        )

    async def request_password_reset(self, email: str, *, client_ip: str | None) -> None:
        """Always succeeds from the caller's point of view (no account enumeration).

        After the rate limits, the lookup, token creation and e-mail delivery run in a
        background task, so the response time and status are identical for existing and
        unknown accounts, and a mail-delivery error can never surface as a 500 that only
        existing accounts would produce.
        """
        email_n = normalize_email(email)
        await self._limiter.enforce(
            "reset_ip", client_ip or "unknown", self._s.rate_limit.password_reset_per_ip
        )
        await self._limiter.enforce(
            "reset_account", email_n, self._s.rate_limit.password_reset_per_ip
        )
        task = asyncio.create_task(self._send_reset(email_n, client_ip))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def wait_for_background(self) -> None:
        """Await pending background work (tests, graceful shutdown)."""
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    async def _send_reset(self, email_n: str, client_ip: str | None) -> None:
        try:
            found = await self._lookup("SELECT * FROM app.auth_find_user(:p)", email_n)
            if found is None:
                return
            user_id, org_id, is_platform = found
            token = self._tokens.new_opaque_token()
            async with self._db.session(self._ctx_for(org_id, is_platform, user_id)) as session:
                user = await session.get(User, user_id)
                if user is None or user.status != UserStatus.ACTIVE.value:
                    return
                session.add(
                    PasswordResetToken(
                        user_id=user.id,
                        organization_id=user.organization_id,
                        token_hash=self._tokens.hash_opaque(token),
                        expires_at=utcnow()
                        + timedelta(seconds=self._s.security.password_reset_ttl_seconds),
                    )
                )
                self._audit.record(
                    session,
                    Actor(user.id, user.role, user.organization_id, ip_prefix(client_ip)),
                    "auth.password_reset_requested",
                    resource_type="user",
                    resource_id=user.id,
                )
                await session.commit()
                recipient = user.email
            link = f"{self._s.public_base_url.rstrip('/')}/#/reset-password?token={token}"
            minutes = self._s.security.password_reset_ttl_seconds // 60
            await self._email.send(
                recipient,
                "Reset your AI Document Assistant password",
                "A password reset was requested for your account.\n\n"
                f"Open this link within {minutes} minutes:\n"
                f"{link}\n\n"
                "If you did not request this, ignore this message.",
            )
        except Exception:
            log.exception("password_reset_delivery_failed")

    async def reset_password(self, token: str, new_password: str) -> None:
        digest = self._tokens.hash_opaque(token)
        found = await self._lookup("SELECT * FROM app.auth_find_reset_token(:p)", digest)
        invalid = ValidationFailed("The reset link is invalid or has expired.")
        if found is None:
            raise invalid
        user_id, org_id, is_platform = found
        async with self._db.session(self._ctx_for(org_id, is_platform, user_id)) as session:
            row = (
                await session.execute(
                    select(PasswordResetToken)
                    .where(PasswordResetToken.token_hash == digest)
                    .with_for_update()
                )
            ).scalar_one()
            now = utcnow()
            if row.used_at is not None or row.expires_at <= now:
                raise invalid
            user = (
                await session.execute(select(User).where(User.id == user_id).with_for_update())
            ).scalar_one()
            self.validate_new_password(new_password, email=user.email, name=user.full_name)
            user.password_hash = self._passwords.hash(new_password)
            user.password_changed_at = now
            user.token_version += 1
            self._clear_failures(user)
            await session.execute(
                update(AuthSession)
                .where(AuthSession.user_id == user.id, AuthSession.revoked_at.is_(None))
                .values(revoked_at=now, revoke_reason="password_reset")
            )
            # This link AND every other outstanding one (user- or admin-issued) is consumed.
            await self._invalidate_reset_tokens(session, user.id, now)
            await self._invalidate_mfa_challenges(session, user.id, now)
            self._audit.record(
                session,
                Actor(user.id, user.role, user.organization_id),
                "auth.password_reset",
                resource_type="user",
                resource_id=user.id,
            )
            await session.commit()

    # ------------------------------------------------------------------- MFA
    async def begin_mfa_enrollment(self, principal: Principal, password: str) -> MfaEnrollment:
        """Start TOTP enrolment. Requires the current password: a stolen access token alone
        must not be able to bind the attacker's authenticator to the account."""
        secret = totp.new_secret()
        async with self._db.session(principal.db_context) as session:
            user = (
                await session.execute(
                    select(User).where(User.id == principal.user_id).with_for_update()
                )
            ).scalar_one()
            if user.mfa_enabled:
                raise ValidationFailed("Multi-factor authentication is already enabled.")
            if not await self._verify_current_password(
                session, principal, user, password, "auth.mfa_enrollment"
            ):
                raise PermissionDenied("The password is incorrect.")
            user.mfa_secret_enc = self._ring.encrypt_value(
                secret.encode(), self._mfa_context(user.id)
            )
            self._audit.record(
                session,
                Actor.of(principal),
                "auth.mfa_enrollment_started",
                resource_type="user",
                resource_id=user.id,
            )
            await session.commit()
            email = user.email
        return MfaEnrollment(
            secret, totp.provisioning_uri(secret, email, self._s.security.mfa_issuer)
        )

    async def confirm_mfa(self, principal: Principal, code: str) -> None:
        async with self._db.session(principal.db_context) as session:
            user = (
                await session.execute(
                    select(User).where(User.id == principal.user_id).with_for_update()
                )
            ).scalar_one()
            if user.mfa_enabled or user.mfa_secret_enc is None:
                raise ValidationFailed("Start enrollment first.")
            secret = self._ring.decrypt_value(
                user.mfa_secret_enc, self._mfa_context(user.id)
            ).decode()
            step = totp.verify(secret, code, last_used_step=None)
            if step is None:
                raise ValidationFailed("The one-time code is invalid.")
            user.mfa_enabled = True
            user.mfa_last_used_step = step
            self._audit.record(
                session,
                Actor.of(principal),
                "auth.mfa_enabled",
                resource_type="user",
                resource_id=user.id,
            )
            await session.commit()
            recipient = user.email
        await self._notify(
            recipient,
            "Two-step verification was turned on",
            "Two-step verification was just enabled on your account.\n\n"
            "If this was not you, contact your administrator immediately.",
        )

    async def disable_mfa(self, principal: Principal, password: str, code: str) -> None:
        generic = PermissionDenied("The password or one-time code is incorrect.")
        async with self._db.session(principal.db_context) as session:
            user = (
                await session.execute(
                    select(User).where(User.id == principal.user_id).with_for_update()
                )
            ).scalar_one()
            if not user.mfa_enabled or user.mfa_secret_enc is None:
                raise ValidationFailed("Multi-factor authentication is not enabled.")
            if not await self._verify_current_password(
                session, principal, user, password, "auth.mfa_disable"
            ):
                raise generic
            secret = self._ring.decrypt_value(
                user.mfa_secret_enc, self._mfa_context(user.id)
            ).decode()
            if totp.verify(secret, code, last_used_step=user.mfa_last_used_step) is None:
                await self._record_failure(session, user)
                self._audit.record(
                    session,
                    Actor.of(principal),
                    "auth.mfa_disable_failed",
                    outcome=AuditOutcome.FAILURE,
                    resource_type="user",
                    resource_id=user.id,
                    details={"reason": "invalid_code"},
                )
                await session.commit()
                raise generic
            user.mfa_enabled = False
            user.mfa_secret_enc = None
            user.mfa_last_used_step = None
            user.token_version += 1
            self._audit.record(
                session,
                Actor.of(principal),
                "auth.mfa_disabled",
                resource_type="user",
                resource_id=user.id,
            )
            await session.commit()
            recipient = user.email
        await self._notify(
            recipient,
            "Two-step verification was turned off",
            "Two-step verification was just disabled on your account.\n\n"
            "If this was not you, contact your administrator immediately.",
        )

    async def admin_reset_mfa(self, session: AsyncSession, actor: Principal, user: User) -> None:
        """Clear a user's MFA (lost device / hijacked enrolment). Caller enforces authz.

        Every session of the user is revoked; the change is audited and the user notified by
        the caller after commit.
        """
        now = utcnow()
        user.mfa_enabled = False
        user.mfa_secret_enc = None
        user.mfa_last_used_step = None
        user.token_version += 1
        await session.execute(
            update(AuthSession)
            .where(AuthSession.user_id == user.id, AuthSession.revoked_at.is_(None))
            .values(revoked_at=now, revoke_reason="mfa_reset")
        )
        await self._invalidate_mfa_challenges(session, user.id, now)
        self._audit.record(
            session,
            Actor.of(actor),
            "admin.mfa_reset",
            resource_type="user",
            resource_id=user.id,
        )
