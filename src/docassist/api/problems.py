"""RFC 9457 problem responses. Clients get a stable ``code`` and a safe message; internals
(stack traces, SQL, hostnames, tenant IDs) only ever go to the server log."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from docassist.core.context import current_request_id
from docassist.core.errors import AppError, RateLimited
from docassist.core.logging import get_logger

log = get_logger("docassist.api.errors")

PROBLEM_TYPE_BASE = (
    "https://github.com/mojtaba-py-code/ai-document-assistant/blob/main/docs/api.md#"
)


def problem(
    status: int, code: str, title: str, detail: str, *, extra: dict[str, Any] | None = None
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": PROBLEM_TYPE_BASE + code,
        "title": title,
        "status": status,
        "detail": detail,
        "code": code,
        "request_id": current_request_id(),
    }
    if extra:
        body.update(extra)
    headers = {"Cache-Control": "no-store"}
    if status == 401:
        headers["WWW-Authenticate"] = 'Bearer realm="docassist"'
    return JSONResponse(
        body, status_code=status, headers=headers, media_type="application/problem+json"
    )


def _sanitize_validation_errors(errors: list[Any]) -> list[dict[str, Any]]:
    """Pydantic errors echo the rejected *input* (which may be a password) - strip it."""
    safe = []
    for err in errors[:20]:
        loc = [str(part) for part in err.get("loc", ())][:6]
        safe.append(
            {"loc": loc, "msg": str(err.get("msg", "invalid"))[:200], "type": err.get("type")}
        )
    return safe


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            log.error("app_error", code=exc.code, detail=exc.internal_detail)
        elif exc.internal_detail:
            log.info("request_rejected", code=exc.code, detail=exc.internal_detail)
        response = problem(
            exc.status_code, exc.code, exc.title, exc.public_message, extra=exc.extra or None
        )
        if isinstance(exc, RateLimited):
            response.headers["Retry-After"] = str(exc.retry_after)
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return problem(
            422,
            "validation_failed",
            "Validation failed",
            "The request is invalid.",
            extra={"errors": _sanitize_validation_errors(list(exc.errors()))},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        titles = {404: "Not found", 405: "Method not allowed", 413: "Payload too large"}
        code = {404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}.get(
            exc.status_code, "http_error"
        )
        detail = (
            exc.detail
            if isinstance(exc.detail, str) and exc.status_code < 500
            else "Request failed."
        )
        return problem(exc.status_code, code, titles.get(exc.status_code, "Error"), detail)

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_error", error_type=type(exc).__name__, exc_info=exc)
        return problem(500, "internal_error", "Internal error", "An unexpected error occurred.")
