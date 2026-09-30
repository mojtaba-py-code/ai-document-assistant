"""Liveness, readiness and Prometheus metrics.

Public health responses reveal only ``ok``/``degraded``/``unavailable`` - no versions,
hostnames or dependency error strings (reconnaissance value). ``/metrics`` requires the
metrics bearer token when one is configured and is disabled in production without it.
"""

from __future__ import annotations

import asyncio
import hmac

from fastapi import APIRouter, Depends, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from docassist.api.container import Container
from docassist.api.deps import get_container
from docassist.core.errors import NotFound, ServiceUnavailable
from docassist.observability.metrics import REGISTRY

router = APIRouter(tags=["health"])


@router.get("/health/live", include_in_schema=False)
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", include_in_schema=False)
async def ready(
    response: Response, container: Container = Depends(get_container)
) -> dict[str, str]:
    try:
        await asyncio.wait_for(container.db.ping(), timeout=3)
    except (TimeoutError, ServiceUnavailable):
        response.status_code = 503
        return {"status": "unavailable"}
    redis_ok = True
    if container.redis is not None:
        redis_ok = await container.cache.ping()
    return {"status": "ok" if redis_ok else "degraded"}


@router.get("/metrics", include_in_schema=False)
async def metrics_endpoint(
    request: Request, container: Container = Depends(get_container)
) -> Response:
    settings = container.settings
    if not settings.observability.metrics_enabled:
        raise NotFound()
    token = settings.security.metrics_token
    if token is None:
        if settings.is_production:
            raise NotFound()
    else:
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied.encode(), token.get_secret_value().encode()):
            raise NotFound()
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
