"""Background job visibility and operator actions (retry dead-letters, cancel queued jobs).

Responses expose kind, status, attempts, error *code* and timestamps; of the payload only
well-formed ``*_id`` identifiers are returned - never error text or other internals.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query

from docassist.api.container import Container
from docassist.api.deps import get_container, require
from docassist.authz.permissions import Permission
from docassist.authz.principal import Principal
from docassist.core.enums import JobStatus
from docassist.core.errors import ValidationFailed
from docassist.identity.schemas import JobOut, JobPage, validate_job_kind

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

_jobs_manage = require(Permission.JOBS_MANAGE)
_jobs_read = require(Permission.JOBS_READ)


@router.get("", response_model=JobPage)
async def list_jobs(
    *,
    kind: str | None = Query(default=None, max_length=64),
    status: JobStatus | None = Query(default=None),
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=50, ge=1, le=200),
    principal: Principal = Depends(_jobs_read),
    container: Container = Depends(get_container),
) -> JobPage:
    try:
        kind = validate_job_kind(kind) if kind else None
    except ValueError as exc:
        raise ValidationFailed(str(exc)) from exc
    return await container.admin.list_jobs(
        principal, kind=kind, status=status, cursor=cursor, limit=limit
    )


@router.post("/{job_id}/retry", response_model=JobOut)
async def retry_job(
    job_id: uuid.UUID,
    principal: Principal = Depends(_jobs_manage),
    container: Container = Depends(get_container),
) -> JobOut:
    return await container.admin.retry_job(principal, job_id)


@router.post("/{job_id}/cancel", response_model=JobOut)
async def cancel_job(
    job_id: uuid.UUID,
    principal: Principal = Depends(_jobs_manage),
    container: Container = Depends(get_container),
) -> JobOut:
    return await container.admin.cancel_job(principal, job_id)
