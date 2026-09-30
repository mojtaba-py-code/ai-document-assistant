"""Export endpoints.

Every route except the signed-link download requires ``export:create`` and only ever
touches the caller's own exports (someone else's export is a 404, even in the same
organisation). ``GET /download?token=...`` is unauthenticated by design: the single-use,
five-minute HMAC link *is* the credential; it is rate-limited per client IP.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, Request, Response, status

from docassist.api.container import Container
from docassist.api.deps import client_ip, get_container, require
from docassist.authz.permissions import Permission
from docassist.authz.principal import Principal
from docassist.identity.auth import ip_prefix
from docassist.intelligence.exports import LINK_RATE_BUCKET, ExportFile
from docassist.intelligence.schemas import ExportCreateRequest, ExportLink, ExportList, ExportOut

router = APIRouter(prefix="/api/v1/exports", tags=["exports"])

_export = require(Permission.EXPORT_CREATE)


def _file_response(file: ExportFile) -> Response:
    return Response(
        content=file.data,
        media_type=file.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{file.filename}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; sandbox",
        },
    )


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=ExportOut)
async def create_export(
    payload: ExportCreateRequest,
    principal: Principal = Depends(_export),
    container: Container = Depends(get_container),
) -> ExportOut:
    return await container.intelligence.exports.create(principal, payload)


@router.get("", response_model=ExportList)
async def list_exports(
    limit: int = Query(50, ge=1, le=100),
    principal: Principal = Depends(_export),
    container: Container = Depends(get_container),
) -> ExportList:
    return await container.intelligence.exports.list_own(principal, limit=limit)


# Declared before "/{export_id}" so the literal path wins.
@router.get("/download", response_class=Response)
async def download_with_link(
    request: Request,
    token: str = Query(min_length=1, max_length=512),
    container: Container = Depends(get_container),
) -> Response:
    ip = client_ip(request)
    await container.limiter.enforce(
        LINK_RATE_BUCKET, ip or "unknown", container.settings.rate_limit.api_per_user
    )
    file = await container.intelligence.exports.redeem_link(token, ip_prefix=ip_prefix(ip))
    return _file_response(file)


@router.get("/{export_id}", response_model=ExportOut)
async def get_export(
    export_id: uuid.UUID,
    principal: Principal = Depends(_export),
    container: Container = Depends(get_container),
) -> ExportOut:
    return await container.intelligence.exports.get(principal, export_id)


@router.get("/{export_id}/download", response_class=Response)
async def download_export(
    export_id: uuid.UUID,
    principal: Principal = Depends(_export),
    container: Container = Depends(get_container),
) -> Response:
    return _file_response(await container.intelligence.exports.download(principal, export_id))


@router.post("/{export_id}/link", response_model=ExportLink)
async def create_link(
    export_id: uuid.UUID,
    principal: Principal = Depends(_export),
    container: Container = Depends(get_container),
) -> ExportLink:
    return await container.intelligence.exports.create_link(principal, export_id)
