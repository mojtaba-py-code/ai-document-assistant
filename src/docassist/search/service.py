"""Search service: user-facing search and retrieval for grounded answers.

``search()`` (the ``/api/v1/search`` endpoint and read-only agent tools):

1. rate limit ``search_per_user``; validate and sanitise the query;
2. embed the query (``kind="query"``, encrypted Redis cache per organisation, 1 h) unless the
   mode is ``keyword``; if the embedder or the vector store fails, **degrade** to keyword
   search (``degraded=True``, metric ``docassist_degraded_mode_total``);
3. keyword (PostgreSQL FTS) and semantic (vector store) candidates - both with the
   authorisation scope inside the query - fused with Reciprocal Rank Fusion; candidates found
   only semantically with a similarity below ``retrieval.min_relevance`` are dropped (a
   nearest-neighbour search always returns *something*, relevant or not);
4. hydrate the final chunk ids **with the authorisation scope applied again** (defence in
   depth), build plain-text snippets, flag chunks with a notable prompt-injection score;
5. audit ``search.query`` in the same transaction: mode, query length, result count and the
   ids of the documents returned - never document content. ``observability.audit_query_text``
   adds a SHA-256 prefix of the query (``"hash"``), or the prefix plus the PII-redacted query
   text (``"redacted"``); ``"none"`` records neither.

``retrieve()`` (the RAG answer path): hybrid candidates from a pool of
``retrieval.candidate_pool``, hydrated with the same re-check; chunks whose
``injection_score`` reaches ``retrieval.injection_exclude_threshold`` are dropped (counted);
the reranker scores the rest; chunks without enough relevance evidence (best of semantic
similarity, FTS rank and query-term coverage below ``retrieval.min_relevance``) are dropped;
finally MMR (``retrieval.mmr_lambda``) picks ``top_k`` diverse chunks - exact duplicates
(same content hash) and near-duplicates (embedding similarity) are penalised - returned
best score first. Reranking precedes diversification so MMR works on the best relevance
estimate, and the relevance floor precedes it so noise can never be picked "for diversity".
It does not audit or rate-limit - the answer service does both for the whole request.

Result sets are never cached (they depend on ACLs that can change at any moment); only
query embeddings are.
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from docassist.audit.service import Actor, AuditLogger
from docassist.authz.principal import Principal
from docassist.cache.ratelimit import RateLimiter
from docassist.cache.redis import Cache
from docassist.core.config import Settings
from docassist.core.context import utcnow
from docassist.core.errors import ValidationFailed
from docassist.core.logging import get_logger
from docassist.core.redaction import PERSONAL_KINDS, redact
from docassist.db.session import Database
from docassist.embeddings.base import EmbeddingProvider
from docassist.observability import metrics
from docassist.search.hybrid import clamp01, mmr, reciprocal_rank_fusion
from docassist.search.keyword import keyword_search
from docassist.search.rerank import LexicalReranker, Reranker, lexical_features
from docassist.search.sql import ChunkRecord, load_chunks, load_embeddings
from docassist.search.text import make_snippet, normalize_query, truncate_query
from docassist.search.types import (
    SEARCH_MODES,
    RetrievalResult,
    RetrievedChunk,
    SearchFilters,
    SearchMode,
    SearchResponse,
    SearchResult,
)
from docassist.search.vector_store import VectorStore

log = get_logger(__name__)

MAX_SEARCH_LIMIT = 50
MAX_TOP_K = 50
QUERY_EMBEDDING_TTL_SECONDS = 3_600
RATE_LIMIT_BUCKET = "search_user"
_MAX_AUDITED_DOCUMENTS = 20


class SearchDependencies(Protocol):
    """What the service needs from the composition root (``api.container.Container``)."""

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
    def embeddings(self) -> EmbeddingProvider: ...
    @property
    def vector_store(self) -> VectorStore: ...


@dataclass(slots=True)
class _Candidates:
    fused: list[tuple[uuid.UUID, float]]
    keyword: dict[uuid.UUID, float]
    semantic: dict[uuid.UUID, float]
    mode_used: SearchMode
    degraded: bool


class _Timer:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.timings: dict[str, float] = {}
        self._mark = self.started

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.timings[name] = round((now - self._mark) * 1000, 2)
        self._mark = now

    def total(self) -> float:
        self.timings["total"] = round((time.perf_counter() - self.started) * 1000, 2)
        return self.timings["total"]


class SearchService:
    def __init__(
        self,
        deps: SearchDependencies,
        *,
        embeddings: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._deps = deps
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._reranker: Reranker = reranker or LexicalReranker()
        self._clock = clock

    @property
    def settings(self) -> Settings:
        return self._deps.settings

    @property
    def embeddings(self) -> EmbeddingProvider:
        return self._embeddings or self._deps.embeddings

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store or self._deps.vector_store

    @property
    def reranker(self) -> Reranker:
        return self._reranker

    # ------------------------------------------------------------------ #
    # Query embedding (cached, encrypted, org-scoped)
    # ------------------------------------------------------------------ #
    def _valid_vector(self, value: Any) -> list[float] | None:
        dims = self.embeddings.dimensions
        if not isinstance(value, list) or len(value) != dims:
            return None
        if not all(isinstance(v, int | float) and math.isfinite(v) for v in value):
            return None
        return [float(v) for v in value]

    async def _query_vector(self, org_id: uuid.UUID, query: str) -> list[float] | None:
        """The query embedding, or ``None`` when the provider is unavailable (degrade)."""
        embedder = self.embeddings
        cache = self._deps.cache
        key = cache.key("qemb", str(org_id), embedder.model, str(embedder.dimensions), query)
        cached = self._valid_vector(await cache.get_json(key))
        if cached is not None:
            return cached
        text = redact(query, PERSONAL_KINDS) if embedder.is_external else query
        try:
            vectors = await embedder.embed([text], kind="query")
            vector = self._valid_vector(vectors[0] if len(vectors) == 1 else None)
        except Exception as exc:  # noqa: BLE001 - any provider failure degrades to keyword
            log.warning("search_embedding_failed", error_type=type(exc).__name__)
            vector = None
        if vector is None:
            metrics.DEGRADED_MODE.labels(component="embeddings").inc()
            return None
        await cache.set_json(key, vector, QUERY_EMBEDDING_TTL_SECONDS)
        return vector

    async def _semantic(
        self,
        session: AsyncSession,
        principal: Principal,
        vector: list[float],
        *,
        filters: SearchFilters,
        limit: int,
        now: datetime,
    ) -> list[tuple[uuid.UUID, float]] | None:
        store = self.vector_store
        try:
            async with session.begin_nested():  # a failed vector query must not abort the tx
                return await store.query(
                    session, principal, vector, filters=filters, limit=limit, now=now
                )
        except Exception as exc:  # noqa: BLE001 - vector store failure degrades to keyword
            log.warning(
                "search_vector_store_failed", store=store.name, error_type=type(exc).__name__
            )
            metrics.DEGRADED_MODE.labels(component="vector_store").inc()
            return None

    async def _candidates(
        self,
        session: AsyncSession,
        principal: Principal,
        query: str,
        *,
        mode: SearchMode,
        filters: SearchFilters,
        pool: int,
        now: datetime,
        timer: _Timer,
    ) -> _Candidates:
        org_id = principal.require_org()
        degraded = False
        vector: list[float] | None = None
        if mode != "keyword":
            vector = await self._query_vector(org_id, query)
            degraded = vector is None
            timer.lap("embed")

        semantic: list[tuple[uuid.UUID, float]] = []
        if vector is not None:
            found = await self._semantic(
                session, principal, vector, filters=filters, limit=pool, now=now
            )
            if found is None:
                degraded = True
            else:
                semantic = found
            timer.lap("semantic")

        mode_used: SearchMode = "keyword" if (mode == "keyword" or degraded) else mode
        keyword: list[tuple[uuid.UUID, float]] = []
        if mode_used in ("keyword", "hybrid"):
            keyword = (
                await keyword_search(
                    session, principal, query, filters=filters, limit=pool, now=now
                )
            ).hits
            timer.lap("keyword")

        rankings = [[cid for cid, _ in keyword], [cid for cid, _ in semantic]]
        fused = reciprocal_rank_fusion(rankings, k=self.settings.retrieval.rrf_k)
        return _Candidates(fused, dict(keyword), dict(semantic), mode_used, degraded)

    # ------------------------------------------------------------------ #
    # search()
    # ------------------------------------------------------------------ #
    async def search(
        self,
        principal: Principal,
        *,
        query: str,
        mode: SearchMode = "hybrid",
        filters: SearchFilters | None = None,
        limit: int = 10,
    ) -> SearchResponse:
        principal.require_org()
        settings = self.settings
        await self._deps.limiter.enforce(
            RATE_LIMIT_BUCKET, str(principal.user_id), settings.rate_limit.search_per_user
        )
        if mode not in SEARCH_MODES:
            raise ValidationFailed("Unknown search mode.")
        if not 1 <= limit <= MAX_SEARCH_LIMIT:
            raise ValidationFailed(f"'limit' must be between 1 and {MAX_SEARCH_LIMIT}.")
        clean = normalize_query(query, max_chars=settings.retrieval.max_query_chars)
        filters = filters or SearchFilters()
        now = self._clock()
        timer = _Timer()
        pool = min(max(limit * 3, 30), 150)
        floor = settings.retrieval.min_relevance
        warn = settings.retrieval.injection_warn_threshold

        async with self._deps.db.transaction(principal.db_context) as session:
            found = await self._candidates(
                session,
                principal,
                clean,
                mode=mode,
                filters=filters,
                pool=pool,
                now=now,
                timer=timer,
            )
            # Semantic-only candidates below the relevance floor are noise, not results.
            ranked = [
                (cid, score)
                for cid, score in found.fused
                if cid in found.keyword or found.semantic.get(cid, 0.0) >= floor
            ][:limit]
            records = await load_chunks(session, principal, now, filters, [c for c, _ in ranked])
            timer.lap("load")
            results = [
                self._result(records[cid], score, found, clean, warn)
                for cid, score in ranked
                if cid in records
            ]
            self._audit(
                session,
                principal,
                query=clean,
                mode=mode,
                found=found,
                filters=filters,
                results=results,
            )
        metrics.RETRIEVAL_LATENCY.labels(mode=found.mode_used).observe(timer.total() / 1000)
        return SearchResponse(
            results=results,
            mode_used=found.mode_used,
            degraded=found.degraded,
            timings_ms=dict(timer.timings),
        )

    @staticmethod
    def _result(
        record: ChunkRecord, score: float, found: _Candidates, query: str, warn: float
    ) -> SearchResult:
        return SearchResult(
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            version_id=record.version_id,
            document_title=record.document_title,
            version_number=record.version_number,
            is_current=record.is_current,
            classification=record.classification,
            doc_type=record.doc_type,
            page_start=record.page_start,
            page_end=record.page_end,
            section=record.section,
            snippet=make_snippet(record.content, query),
            score=round(clamp01(score), 4),
            keyword_score=_rounded(found.keyword.get(record.chunk_id)),
            semantic_score=_rounded(found.semantic.get(record.chunk_id)),
            flagged=record.injection_score >= warn,
        )

    def _audit(
        self,
        session: AsyncSession,
        principal: Principal,
        *,
        query: str,
        mode: SearchMode,
        found: _Candidates,
        filters: SearchFilters,
        results: Sequence[SearchResult],
    ) -> None:
        policy = self.settings.observability.audit_query_text
        details: dict[str, Any] = {
            "mode": mode,
            "mode_used": found.mode_used,
            "degraded": found.degraded,
            "query_length": len(query),
            "filter_facets": filters.active_facets,
            "include_old_versions": filters.include_old_versions,
            "result_count": len(results),
            "document_ids": list(dict.fromkeys(str(r.document_id) for r in results))[
                :_MAX_AUDITED_DOCUMENTS
            ],
        }
        if policy in ("redacted", "hash"):
            details["query_sha256"] = hashlib.sha256(query.encode()).hexdigest()[:16]
        if policy == "redacted":
            details["query"] = redact(query)[:500]
        self._deps.audit.record(
            session, Actor.of(principal), "search.query", resource_type="search", details=details
        )

    # ------------------------------------------------------------------ #
    # retrieve() - RAG
    # ------------------------------------------------------------------ #
    async def retrieve(
        self,
        principal: Principal,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Authorised, diverse, reranked chunks for answering ``query`` (see module doc).

        Long questions are cut to ``retrieval.max_query_chars`` on a word boundary rather than
        rejected. Raises :class:`ValidationFailed` for an empty query.
        """
        principal.require_org()
        cfg = self.settings.retrieval
        top_k = top_k if top_k is not None else cfg.top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValidationFailed(f"'top_k' must be between 1 and {MAX_TOP_K}.")
        clean = truncate_query(query, max_chars=cfg.max_query_chars)
        if not clean:
            raise ValidationFailed("The question is empty.")
        filters = filters or SearchFilters()
        now = self._clock()
        timer = _Timer()
        pool = max(cfg.candidate_pool, top_k)

        async with self._deps.db.session(principal.db_context) as session:
            found = await self._candidates(
                session,
                principal,
                clean,
                mode="hybrid",
                filters=filters,
                pool=pool,
                now=now,
                timer=timer,
            )
            ids = [cid for cid, _ in found.fused[:pool]]
            records = await load_chunks(session, principal, now, filters, ids)
            safe = [
                (cid, score)
                for cid, score in found.fused[:pool]
                if cid in records and records[cid].injection_score < cfg.injection_exclude_threshold
            ]
            flagged = [
                (cid, score)
                for cid, score in found.fused[:pool]
                if cid in records
                and records[cid].injection_score >= cfg.injection_exclude_threshold
            ]
            embeddings = await load_embeddings(
                session, principal, [c for c, _ in safe], model=self.embeddings.model
            )
            timer.lap("load")

        chunks = [
            records[cid].retrieved(
                score=clamp01(score),
                keyword_score=found.keyword.get(cid),
                semantic_score=found.semantic.get(cid),
            )
            for cid, score in safe
        ]
        reranked = await self._reranker.rerank(principal, clean, chunks)
        timer.lap("rerank")
        relevant = [c for c in reranked if _relevance(clean, c) >= cfg.min_relevance]
        below = len(reranked) - len(relevant)
        by_id = {c.chunk_id: c for c in relevant}
        diverse = mmr(
            [(c.chunk_id, c.score) for c in relevant],
            embeddings,
            cfg.mmr_lambda,
            k=top_k,
            fingerprints={c.chunk_id: c.content_sha256 for c in relevant},
        )
        kept = sorted((by_id[cid] for cid, _ in diverse), key=lambda c: -c.score)
        fused = dict(found.fused)
        excluded = _count_relevant_exclusions(
            clean,
            flagged,
            [fused[c.chunk_id] for c in kept if c.chunk_id in fused],
            cfg.min_relevance,
            lambda cid, score: records[cid].retrieved(
                score=clamp01(score),
                keyword_score=found.keyword.get(cid),
                semantic_score=found.semantic.get(cid),
            ),
            top_k=top_k,
        )
        metrics.RETRIEVAL_LATENCY.labels(mode="rag").observe(timer.total() / 1000)
        if excluded:
            log.info("retrieval_injection_excluded", count=excluded)
        return RetrievalResult(
            kept,
            excluded_injection=excluded,
            below_relevance=below,
            candidates=len(records),
            degraded=found.degraded,
            mode_used=found.mode_used,
            timings_ms=timer.timings,
        )


def _count_relevant_exclusions(
    query: str,
    flagged: list[tuple[uuid.UUID, float]],
    kept_fused_scores: list[float],
    min_relevance: float,
    as_chunk: Callable[[uuid.UUID, float], RetrievedChunk],
    *,
    top_k: int,
) -> int:
    """Count excluded (instruction-like) chunks that would otherwise have been used.

    A chunk counts when it is relevant on its own and it would have made the cut: either
    fewer than ``top_k`` chunks were kept, or its fused score reaches the weakest kept chunk.
    An unrelated injected document elsewhere in the tenant therefore does not raise a
    warning on every answer.
    """
    full = len(kept_fused_scores) >= top_k
    floor = min(kept_fused_scores) if full and kept_fused_scores else float("-inf")
    return sum(
        1
        for cid, score in flagged
        if score >= floor and _relevance(query, as_chunk(cid, score)) >= min_relevance
    )


def _relevance(query: str, chunk: RetrievedChunk) -> float:
    """Absolute relevance evidence: best of semantic similarity, FTS rank and term coverage."""
    return max(
        chunk.semantic_score or 0.0,
        chunk.keyword_score or 0.0,
        lexical_features(query, chunk).coverage,
    )


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)
