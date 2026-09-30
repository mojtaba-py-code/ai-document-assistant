"""What the RAG services need from the application container.

The domain layer must not import the HTTP layer (import-linter contract "Domain services do
not depend on the HTTP layer"), so the services depend on these structural protocols
instead of ``docassist.api.container.Container``, which satisfies them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from docassist.audit.service import AuditLogger
from docassist.authz.principal import Principal
from docassist.cache.ratelimit import RateLimiter
from docassist.cache.redis import Cache
from docassist.core.config import Settings
from docassist.db.session import Database
from docassist.llm.gateway import LLMGateway
from docassist.search.types import RetrievedChunk, SearchFilters


class Retriever(Protocol):
    """``SearchService.retrieve`` (search area): authorised, ranked chunks for a question."""

    async def retrieve(
        self,
        principal: Principal,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
    ) -> Sequence[RetrievedChunk]: ...


class RagDependencies(Protocol):
    @property
    def settings(self) -> Settings: ...

    @property
    def db(self) -> Database: ...

    @property
    def cache(self) -> Cache: ...

    @property
    def limiter(self) -> RateLimiter: ...

    @property
    def audit(self) -> AuditLogger: ...

    @property
    def llm(self) -> LLMGateway: ...

    @property
    def search(self) -> Retriever: ...
