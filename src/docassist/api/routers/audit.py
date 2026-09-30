"""Audit trail endpoints: filtered, paginated reading and hash-chain verification.

Tenants (auditors, organisation admins) see only their organisation's events; platform
admins see only the platform chain - row-level security enforces the same split.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query

from docassist.api.container import Container
from docassist.api.deps import get_container, require
from docassist.authz.permissions import Permission
from docassist.authz.principal import Principal
from docassist.core.enums import AuditOutcome
from docassist.core.errors import ValidationFailed
from docassist.identity.schemas import (
    AuditEventPage,
    AuditVerifyOut,
    validate_action_filter,
    validate_resource_type,
)

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])

_audit_read = require(Permission.AUDIT_READ)
_audit_verify = require(Permission.AUDIT_VERIFY)


@router.get("/events", response_model=AuditEventPage)
async def list_events(
    *,
    action: str | None = Query(default=None, max_length=64),
    actor: uuid.UUID | None = Query(default=None),
    resource_type: str | None = Query(default=None, max_length=32),
    resource_id: str | None = Query(default=None, min_length=1, max_length=64),
    outcome: AuditOutcome | None = Query(default=None),
    occurred_from: datetime | None = Query(default=None, alias="from"),
    occurred_to: datetime | None = Query(default=None, alias="to"),
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=50, ge=1, le=200),
    principal: Principal = Depends(_audit_read),
    container: Container = Depends(get_container),
) -> AuditEventPage:
    try:
        action = validate_action_filter(action) if action else None
        resource_type = validate_resource_type(resource_type) if resource_type else None
    except ValueError as exc:
        raise ValidationFailed(str(exc)) from exc
    return await container.admin.list_audit_events(
        principal,
        action=action,
        actor_user_id=actor,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=outcome,
        occurred_from=occurred_from,
        occurred_to=occurred_to,
        cursor=cursor,
        limit=limit,
    )


@router.post("/verify", response_model=AuditVerifyOut)
async def verify(
    principal: Principal = Depends(_audit_verify),
    container: Container = Depends(get_container),
) -> AuditVerifyOut:
    return await container.admin.verify_audit_chain(principal)
