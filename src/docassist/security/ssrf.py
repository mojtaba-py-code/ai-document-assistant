"""Server-side request forgery (SSRF) protection for every outbound HTTP request.

Controls, in order:

1. **URL validation** - scheme allowlist (https; http only for explicitly allowlisted private
   service hosts), no credentials in the URL, canonical IDNA host, no raw-IP tricks
   (decimal/octal/hex forms are rejected because they are not valid DNS names).
2. **Host allowlist** - exact names or ``*.suffix`` wildcards from configuration.
3. **Resolution check** - every address the name resolves to must be globally routable:
   loopback, RFC 1918, link-local (incl. cloud metadata 169.254.169.254 / fd00:ec2::254),
   CGNAT, multicast, reserved, unspecified and IPv4-mapped/6to4 forms are all refused.
4. **DNS pinning** - the connection goes to the IP we vetted (Host header + SNI keep TLS
   valid), so a DNS-rebinding answer between check and connect cannot redirect us.
5. **Redirects** re-run steps 1-4 per hop, capped; **responses** are size- and time-limited
   and content-type checked by :func:`fetch`.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Iterable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

import httpx

_METADATA_V6 = ipaddress.ip_address("fd00:ec2::254")
_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_BENCHMARK = ipaddress.ip_network("198.18.0.0/15")


class EgressDenied(Exception):
    """The destination is not permitted. The message is safe to log, not to show users."""


@dataclass(frozen=True)
class EgressPolicy:
    allowed_hosts: frozenset[str]
    private_allowlist: frozenset[str] = field(default_factory=frozenset)
    max_response_bytes: int = 50 * 1024 * 1024
    timeout_seconds: float = 30.0
    max_redirects: int = 3

    @classmethod
    def build(
        cls,
        allowed_hosts: Iterable[str],
        private_allowlist: Iterable[str] = (),
        **kwargs: float | int,
    ) -> EgressPolicy:
        norm = frozenset(h.strip().lower().rstrip(".") for h in allowed_hosts if h.strip())
        private = frozenset(h.strip().lower().rstrip(".") for h in private_allowlist if h.strip())
        return cls(allowed_hosts=norm | private, private_allowlist=private, **kwargs)  # type: ignore[arg-type]

    def host_allowed(self, host: str) -> bool:
        for entry in self.allowed_hosts:
            if entry.startswith("*."):
                suffix = entry[1:]
                if host.endswith(suffix) and host != suffix.lstrip("."):
                    return True
            elif host == entry:
                return True
        return False


@dataclass(frozen=True, slots=True)
class Target:
    scheme: str
    host: str
    port: int
    private_allowed: bool


def is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return is_public_ip(ip.ipv4_mapped)
        if ip.sixtofour is not None:
            return is_public_ip(ip.sixtofour)
        if ip.teredo is not None:
            return False
        if ip == _METADATA_V6:
            return False
    if isinstance(ip, ipaddress.IPv4Address) and (ip in _CGNAT or ip in _BENCHMARK):
        return False
    return bool(
        ip.is_global
        and not ip.is_private
        and not ip.is_loopback
        and not ip.is_link_local
        and not ip.is_multicast
        and not ip.is_reserved
        and not ip.is_unspecified
    )


def validate_url(url: str, policy: EgressPolicy) -> Target:
    if len(url) > 2048:
        raise EgressDenied("url too long")
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in {"https", "http"}:
        raise EgressDenied(f"scheme {scheme!r} not allowed")
    if parts.username or parts.password or "@" in parts.netloc:
        raise EgressDenied("credentials in url")
    raw_host = parts.hostname
    if not raw_host:
        raise EgressDenied("missing host")
    try:
        host = raw_host.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as exc:
        raise EgressDenied("invalid host") from exc
    try:
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise EgressDenied("invalid port") from exc

    private_allowed = host in policy.private_allowlist
    if not policy.host_allowed(host):
        raise EgressDenied(f"host {host!r} is not on the egress allowlist")
    if scheme == "http" and not private_allowed:
        raise EgressDenied("plain http is only allowed for allowlisted private services")
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None and not private_allowed and not is_public_ip(literal):
        raise EgressDenied("non-public ip literal")
    if any(ch.isdigit() for ch in host.split(".")[-1]) and literal is None:
        raise EgressDenied("numeric top-level label (obfuscated ip?)")
    return Target(scheme=scheme, host=host, port=port, private_allowed=private_allowed)


async def resolve_checked(target: Target) -> str:
    """Resolve and return one vetted IP string. Every resolved address must pass."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(target.host, target.port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise EgressDenied("dns resolution failed") from exc
    addresses = []
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not target.private_allowed and not is_public_ip(ip):
            raise EgressDenied(f"{target.host} resolves to a non-public address")
        addresses.append(str(ip))
    if not addresses:
        raise EgressDenied("no addresses")
    return addresses[0]


class GuardedTransport(httpx.AsyncBaseTransport):
    """httpx transport that validates + pins every request (redirects are never followed here)."""

    def __init__(self, policy: EgressPolicy, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._policy = policy
        self._inner = inner or httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        target = validate_url(str(request.url), self._policy)
        ip = await resolve_checked(target)
        pinned_host = f"[{ip}]" if ":" in ip else ip
        request.url = request.url.copy_with(host=pinned_host)
        request.headers["Host"] = (
            target.host if target.port in (80, 443) else f"{target.host}:{target.port}"
        )
        request.extensions = {**request.extensions, "sni_hostname": target.host}
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def guarded_client(policy: EgressPolicy, **kwargs: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=GuardedTransport(policy),
        follow_redirects=False,
        timeout=httpx.Timeout(policy.timeout_seconds, connect=min(10.0, policy.timeout_seconds)),
        trust_env=False,  # never pick up proxy env vars implicitly
        **kwargs,  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    content_type: str
    content: bytes


async def fetch(
    url: str,
    policy: EgressPolicy,
    *,
    allowed_content_types: Iterable[str],
    client: httpx.AsyncClient | None = None,
) -> FetchResult:
    """GET ``url`` with per-hop validation, redirect cap, size cap and content-type check."""
    allowed = {c.lower() for c in allowed_content_types}
    own_client = client is None
    http = client or guarded_client(policy)
    try:
        current = url
        for _hop in range(policy.max_redirects + 1):
            validate_url(current, policy)
            async with http.stream(
                "GET", current, headers={"Accept": ", ".join(sorted(allowed))}
            ) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise EgressDenied("redirect without location")
                    current = urljoin(current, location)
                    continue
                if resp.status_code != 200:
                    raise EgressDenied(f"unexpected status {resp.status_code}")
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype not in allowed:
                    raise EgressDenied(f"content-type {ctype!r} not allowed")
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > policy.max_response_bytes:
                    raise EgressDenied("response too large")
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > policy.max_response_bytes:
                        raise EgressDenied("response too large")
                return FetchResult(url=current, content_type=ctype, content=bytes(body))
        raise EgressDenied("too many redirects")
    finally:
        if own_client:
            await http.aclose()
