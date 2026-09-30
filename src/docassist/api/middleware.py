"""Pure-ASGI middleware (streaming-safe, no response buffering).

* :class:`RequestContextMiddleware` - correlation ID, structured access log, metrics.
* :class:`SecurityHeadersMiddleware` - CSP, HSTS, anti-framing, no-sniff, cache policy.
* :class:`BodySizeLimitMiddleware` - rejects oversize bodies by header *and* by counting
  streamed bytes (a missing/lying Content-Length cannot bypass it).
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

import structlog

from docassist.core.context import accept_request_id, request_id_var
from docassist.core.logging import get_logger
from docassist.observability import metrics

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

access_log = get_logger("docassist.access")


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            try:
                return str(value.decode("latin-1"))
            except UnicodeDecodeError:
                return None
    return None


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else "unmatched"


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = accept_request_id(_header(scope, b"x-request-id"))
        token = request_id_var.set(request_id)
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        status_holder = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            route = _route_template(scope)
            method = scope.get("method", "GET")
            status = status_holder["status"]
            metrics.HTTP_REQUESTS.labels(method=method, route=route, status=str(status)).inc()
            metrics.HTTP_LATENCY.labels(method=method, route=route).observe(elapsed)
            if route not in {"/health/live", "/health/ready", "/metrics"}:
                access_log.info(
                    "http_request",
                    method=method,
                    route=route,
                    status=status,
                    duration_ms=round(elapsed * 1000, 1),
                )
            structlog.contextvars.unbind_contextvars("request_id")
            request_id_var.reset(token)


_API_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"


def _ui_csp(report_uri: str | None, *, upgrade: bool) -> str:
    policy = (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "font-src 'self'; connect-src 'self'; object-src 'none'; frame-src 'none'; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'self'; "
        "require-trusted-types-for 'script'; trusted-types 'none'"
    )
    if upgrade:
        policy += "; upgrade-insecure-requests"
    if report_uri:
        policy += f"; report-uri {report_uri}"
    return policy


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp, *, hsts: bool, csp_report_uri: str | None = None) -> None:
        self.app = app
        self.hsts = hsts
        self.ui_csp = _ui_csp(csp_report_uri, upgrade=hsts).encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        is_api = path.startswith(("/api/", "/health")) or path == "/metrics"
        is_docs = path in {"/docs", "/redoc", "/openapi.json"} or path.startswith("/docs/")

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"server"]
                existing = {k.lower() for k, _ in headers}
                add: list[tuple[bytes, bytes]] = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"cross-origin-opener-policy", b"same-origin"),
                    (b"cross-origin-resource-policy", b"same-origin"),
                    (
                        b"permissions-policy",
                        b"camera=(), microphone=(), geolocation=(), payment=(), usb=()",
                    ),
                ]
                if not is_docs:
                    add.append(
                        (b"content-security-policy", _API_CSP.encode() if is_api else self.ui_csp)
                    )
                if is_api and b"cache-control" not in existing:
                    add.append((b"cache-control", b"no-store"))
                if self.hsts:
                    add.append(
                        (b"strict-transport-security", b"max-age=63072000; includeSubDomains")
                    )
                headers.extend(h for h in add if h[0] not in existing)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


class _TooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Default JSON limit everywhere; larger limit only for explicitly listed upload routes."""

    def __init__(
        self, app: ASGIApp, *, default_limit: int, upload_limit: int, upload_paths: tuple[str, ...]
    ) -> None:
        self.app = app
        self.default_limit = default_limit
        self.upload_limit = upload_limit
        self.upload_paths = upload_paths

    def _limit_for(self, path: str) -> int:
        if any(path.startswith(p) for p in self.upload_paths):
            return self.upload_limit
        return self.default_limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self._limit_for(scope.get("path", ""))
        declared = _header(scope, b"content-length")
        if declared is not None:
            if not declared.isdigit():
                await _reject(send, 400, "invalid_request", "Invalid Content-Length.")
                return
            if int(declared) > limit:
                await _reject(send, 413, "payload_too_large", "The request body is too large.")
                return
        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _TooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _TooLarge:
            if not response_started:
                await _reject(send, 413, "payload_too_large", "The request body is too large.")


async def _reject(send: Send, status: int, code: str, detail: str) -> None:
    body = json.dumps(
        {
            "type": "about:blank",
            "title": "Request rejected",
            "status": status,
            "detail": detail,
            "code": code,
            "request_id": request_id_var.get(),
        }
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/problem+json"),
                (b"content-length", str(len(body)).encode()),
                (b"cache-control", b"no-store"),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
