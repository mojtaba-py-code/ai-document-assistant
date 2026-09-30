"""FastAPI application factory."""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from docassist import __version__
from docassist.api.container import Container, build_container
from docassist.api.middleware import (
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from docassist.api.problems import install_error_handlers
from docassist.core.config import Settings, get_settings
from docassist.core.logging import configure_logging, get_logger
from docassist.db.session import verify_least_privilege

log = get_logger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

ROUTER_MODULES = (
    "docassist.api.routers.health",
    "docassist.api.routers.auth",
    "docassist.api.routers.documents",
    "docassist.api.routers.search",
    "docassist.api.routers.assistant",
    "docassist.api.routers.intelligence",
    "docassist.api.routers.exports",
    "docassist.api.routers.admin",
    "docassist.api.routers.audit",
    "docassist.api.routers.jobs",
)

UPLOAD_PATHS = ("/api/v1/documents",)


def create_app(
    settings: Settings | None = None,
    *,
    container: Container | None = None,
    overrides: dict[str, Any] | None = None,
    configure_logs: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    if configure_logs:
        configure_logging(
            settings.observability.log_level,
            settings.observability.log_format,
            settings.service_name,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = container is None
        c = container or build_container(settings, role="api", overrides=overrides)
        app.state.container = c
        if settings.database.verify_role_privileges:
            problems = await verify_least_privilege(c.db, settings.database.app_role)
            if problems:
                message = "database role is over-privileged: " + "; ".join(problems)
                if settings.is_production:
                    raise RuntimeError(message)
                log.warning("db_role_privileges", problems=problems)
        log.info("startup", environment=settings.environment.value, version=__version__)
        try:
            yield
        finally:
            if owned:
                await c.close()

    docs_enabled = settings.security.expose_api_docs
    app = FastAPI(
        title="AI Document Assistant",
        version=__version__,
        description="Secure multi-tenant document intelligence and RAG platform.",
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    install_error_handlers(app)
    for module_name in ROUTER_MODULES:
        app.include_router(importlib.import_module(module_name).router)

    # Middleware executes bottom-up: the last added wraps everything else.
    if settings.security.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.security.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-CSRF-Protection"],
            max_age=600,
        )
    app.add_middleware(
        BodySizeLimitMiddleware,
        default_limit=settings.security.max_json_body_bytes,
        upload_limit=settings.upload.max_upload_bytes + 1_048_576,
        upload_paths=UPLOAD_PATHS,
    )
    app.add_middleware(
        SecurityHeadersMiddleware,
        hsts=settings.is_production,
        csp_report_uri=settings.security.csp_report_uri,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.security.allowed_hosts)
    app.add_middleware(RequestContextMiddleware)

    static_dir = WEB_DIR / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        index = WEB_DIR / "index.html"

        @app.get("/", include_in_schema=False)
        async def spa_index() -> FileResponse:
            return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app
