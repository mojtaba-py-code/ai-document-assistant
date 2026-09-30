"""Audit trail: recording, sealing (HMAC hash chain) and verification.

Recording is *transactional*: an audit row for a mutation is written in the same
transaction as the mutation, so "the action happened" and "the action was audited" cannot
diverge. Denied/failed attempts (whose business transaction rolls back) are written in a
separate short transaction via :meth:`AuditLogger.record_detached`.

Sealing runs in the worker. It takes unsealed rows in id order, per chain (one chain per
organisation plus one platform chain), and sets

    hash = HMAC-SHA256(audit_key, prev_hash || canonical_json(event))

with a monotonically increasing ``seal_seq``. The key is not stored in the database, so
someone with raw SQL access can delete or edit rows (the triggers make even that hard) but
cannot recompute a valid chain - verification detects it. Chain order is *sealing order*,
which is robust to transactions that commit out of id order.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.authz.principal import Principal
from docassist.core.context import current_request_id, utcnow
from docassist.core.enums import AuditOutcome
from docassist.core.logging import get_logger
from docassist.core.redaction import redact
from docassist.db.models import AuditChainHead, AuditEvent
from docassist.db.session import Database, DbContext

log = get_logger(__name__)

PLATFORM_CHAIN = uuid.UUID(int=0)
GENESIS = hashlib.sha256(b"docassist-audit-genesis").digest()
_MAX_DETAIL_STR = 500
_MAX_DETAIL_KEYS = 40


def _sanitize_details(details: dict[str, Any] | None) -> dict[str, Any]:
    """Keep audit details small and free of secrets/PII (defence in depth)."""
    if not details:
        return {}
    clean: dict[str, Any] = {}
    for index, (key, value) in enumerate(details.items()):
        if index >= _MAX_DETAIL_KEYS:
            clean["_truncated"] = True
            break
        k = str(key)[:64]
        if isinstance(value, str):
            clean[k] = redact(value)[:_MAX_DETAIL_STR]
        elif isinstance(value, bool | int | float) or value is None:
            clean[k] = value
        elif isinstance(value, list | tuple):
            clean[k] = [redact(str(v))[:200] if isinstance(v, str) else v for v in value[:50]]
        elif isinstance(value, dict):
            clean[k] = _sanitize_details(value)
        else:
            clean[k] = redact(str(value))[:_MAX_DETAIL_STR]
    return clean


@dataclass(frozen=True, slots=True)
class Actor:
    user_id: uuid.UUID | None
    role: str | None
    org_id: uuid.UUID | None
    ip_prefix: str | None = None

    @classmethod
    def of(cls, principal: Principal | None, ip_prefix: str | None = None) -> Actor:
        if principal is None:
            return cls(None, None, None, ip_prefix)
        return cls(
            principal.user_id,
            principal.role.value,
            principal.org_id,
            principal.ip_prefix or ip_prefix,
        )

    @classmethod
    def system(cls, org_id: uuid.UUID | None) -> Actor:
        return cls(None, "system", org_id, None)


class AuditLogger:
    def __init__(self, db: Database) -> None:
        self._db = db

    def build(
        self,
        actor: Actor,
        action: str,
        *,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        resource_type: str | None = None,
        resource_id: str | uuid.UUID | None = None,
        details: dict[str, Any] | None = None,
        org_id: uuid.UUID | None = None,
    ) -> AuditEvent:
        return AuditEvent(
            organization_id=org_id if org_id is not None else actor.org_id,
            occurred_at=utcnow(),
            actor_user_id=actor.user_id,
            actor_role=actor.role,
            actor_ip_prefix=actor.ip_prefix,
            action=action[:64],
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            outcome=outcome.value,
            request_id=current_request_id(),
            details=_sanitize_details(details),
        )

    def record(self, session: AsyncSession, actor: Actor, action: str, **kwargs: Any) -> None:
        """Add an audit row to the caller's transaction (committed with the action)."""
        session.add(self.build(actor, action, **kwargs))

    async def record_detached(self, actor: Actor, action: str, **kwargs: Any) -> None:
        """Write an audit row in its own transaction (used for denials and failures).

        A plain ``INSERT`` without ``RETURNING`` is used on purpose: PostgreSQL applies the
        SELECT policy to rows returned by ``RETURNING``, and the API role must be able to
        append events it is not allowed to read back (e.g. a failed login for an unknown
        account, which belongs to no organisation).
        """
        event = self.build(actor, action, **kwargs)
        params = {
            "organization_id": event.organization_id,
            "occurred_at": event.occurred_at,
            "actor_user_id": event.actor_user_id,
            "actor_role": event.actor_role,
            "actor_ip_prefix": event.actor_ip_prefix,
            "action": event.action,
            "resource_type": event.resource_type,
            "resource_id": event.resource_id,
            "outcome": event.outcome,
            "request_id": event.request_id,
            "details": json.dumps(event.details, separators=(",", ":"), default=str),
        }
        ctx = DbContext(org_id=event.organization_id, user_id=actor.user_id)
        try:
            async with self._db.transaction(ctx) as session:
                await session.execute(_INSERT_WITHOUT_RETURNING, params)
        except Exception:  # audit must never take the request down with it
            log.exception("audit_write_failed", action=action)


_INSERT_WITHOUT_RETURNING = text(
    "INSERT INTO audit_events (organization_id, occurred_at, actor_user_id, actor_role,"
    " actor_ip_prefix, action, resource_type, resource_id, outcome, request_id, details)"
    " VALUES (:organization_id, :occurred_at, :actor_user_id, :actor_role, :actor_ip_prefix,"
    " :action, :resource_type, :resource_id, :outcome, :request_id, CAST(:details AS jsonb))"
)


# --------------------------------------------------------------------------- #
# Sealing & verification
# --------------------------------------------------------------------------- #
def canonical_event(event: AuditEvent) -> bytes:
    payload = {
        "id": event.id,
        "org": str(event.organization_id) if event.organization_id else None,
        "at": event.occurred_at.isoformat(),
        "actor": str(event.actor_user_id) if event.actor_user_id else None,
        "role": event.actor_role,
        "ip": event.actor_ip_prefix,
        "action": event.action,
        "rtype": event.resource_type,
        "rid": event.resource_id,
        "outcome": event.outcome,
        "req": event.request_id,
        "details": event.details,
        "seq": event.seal_seq,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def chain_hash(key: bytes, prev_hash: bytes, event: AuditEvent) -> bytes:
    return hmac.new(key, prev_hash + canonical_event(event), hashlib.sha256).digest()


class AuditSealer:
    """Worker-side: seal unsealed events into their per-organisation chains."""

    def __init__(self, worker_db: Database, key: bytes, batch_size: int = 500) -> None:
        self._db = worker_db
        self._key = key
        self._batch = batch_size

    async def seal_pending(self) -> int:
        sealed = 0
        async with self._db.transaction(DbContext.anonymous()) as session:
            # Serialise sealers across worker replicas.
            got = (await session.execute(text("SELECT pg_try_advisory_xact_lock(724001)"))).scalar()
            if not got:
                return 0
            events = (
                (
                    await session.execute(
                        select(AuditEvent)
                        .where(AuditEvent.hash.is_(None))
                        .order_by(AuditEvent.id)
                        .limit(self._batch)
                        .with_for_update()
                    )
                )
                .scalars()
                .all()
            )
            heads: dict[uuid.UUID, AuditChainHead] = {}
            for event in events:
                chain = event.organization_id or PLATFORM_CHAIN
                head = heads.get(chain)
                if head is None:
                    head = await session.get(AuditChainHead, chain, with_for_update=True)
                    if head is None:
                        head = AuditChainHead(
                            chain_key=chain, last_seq=0, last_hash=GENESIS, anchor_seq=0
                        )
                        session.add(head)
                    heads[chain] = head
                event.seal_seq = head.last_seq + 1
                event.prev_hash = head.last_hash
                event.hash = chain_hash(self._key, head.last_hash, event)
                event.sealed_at = utcnow()
                head.last_seq = event.seal_seq
                head.last_hash = event.hash
                sealed += 1
        return sealed

    async def purge_older_than(self, chain: uuid.UUID, before: datetime) -> int:
        async with self._db.transaction(DbContext.anonymous()) as session:
            result = await session.execute(
                text("SELECT app.audit_purge(:chain, :before)"), {"chain": chain, "before": before}
            )
            return int(result.scalar() or 0)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    chain: uuid.UUID
    checked: int
    valid: bool
    first_bad_seq: int | None
    reason: str | None
    head_seq: int
    unsealed: int


async def verify_chain(session: AsyncSession, key: bytes, chain: uuid.UUID) -> VerificationReport:
    """Recompute the chain from its anchor (or genesis) and compare with the stored head."""
    org_filter = (
        AuditEvent.organization_id.is_(None)
        if chain == PLATFORM_CHAIN
        else AuditEvent.organization_id == chain
    )
    head_row = await session.execute(
        text(
            "SELECT last_seq, last_hash, anchor_seq, anchor_hash"
            " FROM audit_chain_heads WHERE chain_key = :c"
        ),
        {"c": chain},
    )
    head = head_row.first()
    unsealed = len(
        (
            await session.execute(
                select(AuditEvent.id).where(org_filter, AuditEvent.hash.is_(None))
            )
        ).all()
    )
    if head is None:
        return VerificationReport(chain, 0, True, None, None, 0, unsealed)
    last_seq, last_hash, anchor_seq, anchor_hash = head
    prev = anchor_hash if anchor_seq else GENESIS
    expected_seq = anchor_seq + 1
    checked = 0
    stream = await session.stream_scalars(
        select(AuditEvent)
        .where(org_filter, AuditEvent.hash.is_not(None))
        .order_by(AuditEvent.seal_seq)
    )
    async for event in stream:
        if event.seal_seq != expected_seq:
            return VerificationReport(
                chain, checked, False, expected_seq, "gap or reordering", last_seq, unsealed
            )
        if event.prev_hash != prev or not hmac.compare_digest(
            chain_hash(key, prev, event), event.hash or b""
        ):
            return VerificationReport(
                chain, checked, False, event.seal_seq, "hash mismatch", last_seq, unsealed
            )
        prev = event.hash or b""
        expected_seq += 1
        checked += 1
    if expected_seq - 1 != last_seq or (checked and prev != last_hash):
        return VerificationReport(
            chain, checked, False, expected_seq, "tail truncated", last_seq, unsealed
        )
    return VerificationReport(chain, checked, True, None, None, last_seq, unsealed)
