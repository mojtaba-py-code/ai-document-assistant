"""Composition root: builds every service once per process and wires dependencies.

Feature areas contribute through ``<area>/wiring.py::wire(container)`` so each area owns
its own construction logic while this module stays the single place that knows the order.
Tests build a container with overrides (fake LLM, in-memory email, fakeredis...).
"""

from __future__ import annotations

import importlib
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from redis.asyncio import Redis

from docassist.audit.service import AuditLogger
from docassist.cache.ratelimit import RateLimiter
from docassist.cache.redis import Cache, create_redis
from docassist.core.config import Settings, parse_encryption_keys
from docassist.core.logging import get_logger
from docassist.db.session import Database, create_engine
from docassist.identity.auth import AuthService
from docassist.identity.notifications import EmailSender, OutboxEmailSender
from docassist.security.crypto import KeyRing
from docassist.security.passwords import PasswordService
from docassist.security.ssrf import EgressPolicy
from docassist.security.tokens import TokenService

if TYPE_CHECKING:
    from docassist.documents.service import DocumentService
    from docassist.documents.storage import ObjectStorage
    from docassist.embeddings.base import EmbeddingProvider
    from docassist.identity.admin import AdminService
    from docassist.ingestion.pipeline import IngestionPipeline
    from docassist.intelligence.service import IntelligenceService
    from docassist.llm.gateway import LLMGateway
    from docassist.rag.agent import AgentService
    from docassist.rag.service import AnswerService
    from docassist.search.service import SearchService
    from docassist.search.vector_store import VectorStore

log = get_logger(__name__)

FEATURE_WIRING = (
    "docassist.documents.wiring",
    "docassist.search.wiring",
    "docassist.llm.wiring",
    "docassist.ingestion.wiring",
    "docassist.rag.wiring",
    "docassist.intelligence.wiring",
    "docassist.identity.wiring",
)


@dataclass
class Container:
    settings: Settings
    db: Database
    ring: KeyRing
    egress: EgressPolicy
    redis: Redis | None
    cache: Cache
    limiter: RateLimiter
    passwords: PasswordService
    tokens: TokenService
    audit: AuditLogger
    email: EmailSender
    auth: AuthService
    worker_db: Database | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    # Feature services (set by the wiring modules)
    storage: ObjectStorage = field(init=False)
    embeddings: EmbeddingProvider = field(init=False)
    vector_store: VectorStore = field(init=False)
    documents: DocumentService = field(init=False)
    search: SearchService = field(init=False)
    llm: LLMGateway = field(init=False)
    pipeline: IngestionPipeline = field(init=False)
    answers: AnswerService = field(init=False)
    agent: AgentService = field(init=False)
    intelligence: IntelligenceService = field(init=False)
    admin: AdminService = field(init=False)

    async def close(self) -> None:
        closers = [self.db.dispose()]
        if self.worker_db is not None:
            closers.append(self.worker_db.dispose())
        for closer in closers:
            await closer
        if self.redis is not None:
            await self.redis.aclose()
        for name in ("llm", "embeddings", "vector_store"):
            service = self.__dict__.get(name)
            aclose = getattr(service, "aclose", None)
            if aclose is not None:
                await aclose()


def build_key_ring(settings: Settings) -> KeyRing:
    keys = parse_encryption_keys(settings.security.encryption_keys.get_secret_value())
    return KeyRing(keys=keys, active_kid=settings.security.active_encryption_key_id)


def build_egress_policy(settings: Settings) -> EgressPolicy:
    out = settings.outbound
    return EgressPolicy.build(
        out.allowed_hosts,
        out.private_network_allowlist,
        max_response_bytes=out.max_response_bytes,
        timeout_seconds=out.timeout_seconds,
        max_redirects=out.max_redirects,
    )


def worker_identity() -> str:
    return f"{socket.gethostname()[:60]}:{os.getpid()}"


def build_container(
    settings: Settings,
    *,
    role: str = "api",
    overrides: dict[str, Any] | None = None,
) -> Container:
    """Build all services. ``role='worker'`` also opens the worker-role database."""
    overrides = overrides or {}
    db_settings = settings.database
    db = Database(
        create_engine(
            db_settings.url.get_secret_value(), db_settings, application_name=f"docassist-{role}"
        ),
        role="api",
    )
    worker_db = None
    if role == "worker":
        worker_dsn = (db_settings.worker_url or db_settings.url).get_secret_value()
        worker_db = Database(
            create_engine(worker_dsn, db_settings, application_name="docassist-worker"),
            role="worker",
        )
    redis: Redis | None = overrides.get("redis")
    if redis is None and settings.redis.url is not None:
        redis = create_redis(
            settings.redis.url.get_secret_value(),
            socket_timeout=settings.redis.socket_timeout_seconds,
            max_connections=settings.redis.max_connections,
        )
    ring = build_key_ring(settings)
    sec = settings.security
    passwords = PasswordService(
        time_cost=sec.argon2_time_cost,
        memory_kib=sec.argon2_memory_kib,
        parallelism=sec.argon2_parallelism,
    )
    tokens = TokenService(
        signing_key=sec.jwt_signing_key.get_secret_value(),
        previous_keys=[k.get_secret_value() for k in sec.jwt_previous_signing_keys],
        issuer=sec.jwt_issuer,
        audience=sec.jwt_audience,
        access_ttl_seconds=sec.access_token_ttl_seconds,
        pepper=sec.token_pepper.get_secret_value(),
    )
    audit = AuditLogger(db)
    limiter = RateLimiter(redis, settings.redis.key_prefix, enabled=settings.rate_limit.enabled)
    email: EmailSender = overrides.get("email") or OutboxEmailSender(Path("var/outbox"))
    auth = AuthService(
        settings=settings,
        db=db,
        passwords=passwords,
        tokens=tokens,
        ring=ring,
        audit=audit,
        limiter=limiter,
        email=email,
    )
    container = Container(
        settings=settings,
        db=db,
        ring=ring,
        egress=build_egress_policy(settings),
        redis=redis,
        cache=Cache(redis, settings.redis.key_prefix, ring),
        limiter=limiter,
        passwords=passwords,
        tokens=tokens,
        audit=audit,
        email=email,
        auth=auth,
        worker_db=worker_db,
        overrides=overrides,
    )
    for module_name in FEATURE_WIRING:
        module = importlib.import_module(module_name)
        module.wire(container)
    return container
