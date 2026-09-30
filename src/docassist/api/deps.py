"""FastAPI dependencies: container access, client IP, authentication, permission gates.

Note: dependency *defaults* (``Depends(...)``) are used instead of ``Annotated`` aliases
because the aliases would be evaluated lazily under ``from __future__ import annotations``.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Awaitable, Callable

from fastapi import Depends, Request

from docassist.api.container import Container
from docassist.audit.service import Actor
from docassist.authz.permissions import Permission
from docassist.authz.principal import Principal
from docassist.core.enums import AuditOutcome
from docassist.core.errors import AuthenticationFailed


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


def client_ip(request: Request) -> str | None:
    """Client address; ``X-Forwarded-For`` is honoured only when sent by a trusted proxy."""
    peer = request.client.host if request.client else None
    container: Container = request.app.state.container
    proxies = [
        ipaddress.ip_network(p, strict=False) for p in container.settings.security.trusted_proxies
    ]
    if not peer or not proxies:
        return peer

    def trusted(addr: str) -> bool:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        return any(ip in net for net in proxies)

    if not trusted(peer):
        return peer
    forwarded = request.headers.get("x-forwarded-for", "")
    hops = [h.strip() for h in forwarded.split(",") if h.strip()][-10:]
    # Walk from the right: the first address not belonging to our proxies is the client.
    for hop in reversed(hops):
        if not trusted(hop):
            try:
                return str(ipaddress.ip_address(hop))
            except ValueError:
                return peer
    return peer


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationFailed("Authentication required.")
    return token.strip()


async def get_principal(
    request: Request, container: Container = Depends(get_container)
) -> Principal:
    cached: Principal | None = getattr(request.state, "principal", None)
    if cached is not None:
        return cached
    principal = await container.auth.authenticate(_bearer(request), client_ip=client_ip(request))
    await container.limiter.enforce(
        "api_user", str(principal.user_id), container.settings.rate_limit.api_per_user
    )
    request.state.principal = principal
    return principal


def require(permission: Permission) -> Callable[..., Awaitable[Principal]]:
    """Dependency factory: authenticated principal that holds ``permission`` (else 403)."""

    async def dependency(
        principal: Principal = Depends(get_principal), container: Container = Depends(get_container)
    ) -> Principal:
        if not principal.has(permission):
            await container.audit.record_detached(
                Actor.of(principal),
                "authz.denied",
                outcome=AuditOutcome.DENIED,
                details={"permission": permission.value},
            )
            principal.require(permission)
        return principal

    dependency.__name__ = f"require_{permission.value.replace(':', '_')}"
    return dependency
