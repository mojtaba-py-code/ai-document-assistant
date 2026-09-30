"""Grounded question answering over the caller's documents.

``AnswerService.ask`` (the router checks ``assistant:use`` first):

1. validates the question (sanitised, 1..``llm.max_question_chars``) and applies the
   ``llm_per_user`` rate limit;
2. analyses it deterministically (:mod:`docassist.rag.query`); high-confidence abuse
   (system-prompt extraction, instruction override, cross-tenant or bulk-dump requests,
   exfiltration) is refused **without retrieval or a model call**;
3. "list deadlines" questions take the **deterministic path**: extracted date fields of
   readable documents inside the requested window, answered as a cited table - no model;
4. otherwise retrieves authorised chunks (``container.search.retrieve``); nothing relevant
   means ``insufficient_context`` without a model call;
5. applies data governance: the request's classification is the highest classification of
   the selected chunks; chunks no provider may process are dropped (with a warning) and a
   late :class:`LLMPolicyDenied` is handled the same way, once;
6. builds the spotlighted context, calls the gateway (task ``answer``, strict JSON schema),
   validates every citation against its source, runs the output guard on every
   model-written string and downgrades an "answered" result without a verified citation to
   ``insufficient_context``;
7. computes a confidence score from retrieval strength, citation validity and the model's
   own confidence;
8. persists the question and answer in the caller's private conversation and audits
   ``assistant.ask`` (lengths, hashes, ids - never the question or answer text) in the same
   transaction.

Answers are cached per organisation *and* per principal access fingerprint, keyed by the
question, previous questions, the exact chunk ids + content hashes, the route (provider and
model) and ``PROMPT_VERSION`` (encrypted in Redis, ``llm.answer_cache_ttl_seconds``). A
result that used a RESTRICTED chunk is never cached. Cached payloads are re-validated and
re-guarded on every hit.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from docassist.audit.service import Actor
from docassist.authz.principal import Principal
from docassist.core.context import utcnow
from docassist.core.enums import Classification, MessageRole
from docassist.core.errors import ValidationFailed
from docassist.core.logging import get_logger
from docassist.core.text import clean_line_text, sanitize_text, truncate
from docassist.db.models import Conversation, Message
from docassist.llm.base import (
    ChatMessage,
    LLMOutputInvalid,
    LLMPolicyDenied,
    LLMRefused,
    LLMRequest,
    LLMTask,
)
from docassist.llm.local_extractive import terms
from docassist.observability import metrics
from docassist.rag.citations import validate_citations
from docassist.rag.context import BuiltContext, SourceRef, build_context, render_user_message
from docassist.rag.conversations import (
    ConversationStore,
    load_owned_conversation,
    previous_questions,
)
from docassist.rag.deadlines import find_deadlines
from docassist.rag.guard import REFUSAL_TEXT, OutputGuard
from docassist.rag.ports import RagDependencies
from docassist.rag.prompts import (
    ANSWER_SCHEMA,
    ANSWER_STATUSES,
    PROMPT_VERSION,
    answer_system_prompt,
    canary_token,
)
from docassist.rag.query import QueryAnalysis, TimeWindow, analyze_query, search_topic
from docassist.rag.types import (
    AnswerStatus,
    Citation,
    Evidence,
    GroundedAnswer,
    Usage,
    confidence_label,
)
from docassist.search.types import RetrievedChunk, SearchFilters

log = get_logger(__name__)

HISTORY_QUESTIONS = 3
SHORT_FOLLOW_UP_TERMS = 6
EXCERPT_CHARS = 300
SEARCH_MAX_DOCUMENTS = 10
DEADLINE_ROWS = 25
_MODEL_CONFIDENCE = {"low": 0.3, "medium": 0.6, "high": 0.9}

INSUFFICIENT_TEXT = (
    "I could not find information in the documents you can access that answers this question."
)
UNVERIFIED_TEXT = (
    "An answer was generated but it could not be verified against the source documents, "
    "so it is not shown."
)
REFUSALS = {
    "system_prompt_extraction": "The assistant cannot reveal its configuration or instructions.",
    "instruction_override": (
        "The question contains instructions aimed at the assistant rather than a question "
        "about your documents."
    ),
    "prompt_injection": (
        "The question contains instructions aimed at the assistant rather than a question "
        "about your documents."
    ),
    "cross_tenant_request": (
        "The assistant only uses documents from your own organisation that you are allowed to read."
    ),
    "bulk_data_request": "Bulk export of documents or data is not available through the assistant.",
    "exfiltration_request": "The assistant cannot send data to external destinations.",
}
WARN_EXCLUDED_INJECTION = (
    "{n} passage(s) were excluded because they contained text resembling instructions to an AI."
)
WARN_FLAGGED_SOURCES = (
    "Some sources contained text resembling instructions; it was treated as document data only."
)
WARN_POLICY_EXCLUDED = (
    "Some restricted sources were excluded from AI processing by the data-governance policy."
)
WARN_DEGRADED = "Semantic search was unavailable; results are based on keyword matching only."
WARN_UNVERIFIED = "{n} citation(s) could not be verified against the sources and were removed."
WARN_GUARD = "Parts of the response were removed by the output safety filter."
WARN_OLD_VERSIONS = "Superseded document versions were included on request."
WARN_DEADLINES = (
    "Dates come from automatically extracted document fields; confirm them in the documents."
)


@dataclass(slots=True)
class _Outcome:
    status: AnswerStatus
    answer: str
    path: str
    citations: list[Citation] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0
    model: str | None = None
    provider: str | None = None
    usage: Usage = field(default_factory=Usage)
    cache_hit: bool = False
    source_count: int = 0


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class AnswerService:
    def __init__(self, deps: RagDependencies, *, clock: Callable[[], datetime] = utcnow) -> None:
        self._deps = deps
        self._clock = clock
        self.conversations = ConversationStore(deps.db, deps.audit)

    @property
    def canary(self) -> str:
        return canary_token(self._deps.settings.security.token_pepper.get_secret_value())

    def guard(self) -> OutputGuard:
        return OutputGuard(self.canary)

    # ------------------------------------------------------------------ #
    async def ask(
        self,
        principal: Principal,
        *,
        question: str,
        conversation_id: uuid.UUID | None = None,
        filters: SearchFilters | None = None,
        include_old_versions: bool = False,
    ) -> GroundedAnswer:
        started = time.perf_counter()
        principal.require_org()
        settings = self._deps.settings
        text = " ".join(sanitize_text(question)[0].split())
        if not text:
            raise ValidationFailed("The question is empty.")
        if len(text) > settings.llm.max_question_chars:
            raise ValidationFailed(
                f"The question is too long (at most {settings.llm.max_question_chars} characters)."
            )
        await self._deps.limiter.enforce(
            "llm_user", str(principal.user_id), settings.rate_limit.llm_per_user
        )
        now = self._clock()
        analysis = analyze_query(
            text,
            today=now.date(),
            warn_threshold=settings.retrieval.injection_warn_threshold,
            block_threshold=settings.retrieval.injection_exclude_threshold,
        )
        history: list[str] = []
        if conversation_id is not None:
            async with self._deps.db.session(principal.db_context) as session:
                await load_owned_conversation(session, principal, conversation_id)
                history = await previous_questions(
                    session, principal, conversation_id, limit=HISTORY_QUESTIONS
                )
        if analysis.flags:
            metrics.SECURITY_EVENTS.labels(kind="suspicious_query").inc()

        if analysis.blocked_reason is not None:
            outcome = _Outcome(
                status="refused",
                answer=REFUSALS.get(analysis.blocked_reason, REFUSAL_TEXT),
                path="blocked",
            )
        elif analysis.intent == "list_deadlines" and analysis.window is not None:
            outcome = await self._deadline_answer(
                principal, analysis.window, analysis.doc_types, now
            )
        elif analysis.intent == "search":
            outcome = await self._search_answer(
                principal,
                text=text,
                doc_types=analysis.doc_types,
                filters=filters,
                include_old_versions=include_old_versions,
            )
        else:
            outcome = await self._rag_answer(
                principal,
                text=text,
                history=history,
                filters=filters,
                include_old_versions=include_old_versions,
            )
        if include_old_versions and outcome.path == "rag":
            outcome.warnings.append(WARN_OLD_VERSIONS)

        latency_ms = int((time.perf_counter() - started) * 1000)
        conversation, message = await self._persist(
            principal,
            conversation_id=conversation_id,
            question=text,
            outcome=outcome,
            analysis=analysis,
            latency_ms=latency_ms,
        )
        return GroundedAnswer(
            status=outcome.status,
            answer=outcome.answer,
            confidence=round(outcome.confidence, 3),
            confidence_label=confidence_label(outcome.confidence),
            citations=outcome.citations,
            evidence=outcome.evidence,
            warnings=list(dict.fromkeys(outcome.warnings)),
            model=outcome.model,
            provider=outcome.provider,
            usage=outcome.usage,
            conversation_id=conversation,
            message_id=message,
            latency_ms=latency_ms,
            prompt_version=PROMPT_VERSION,
            cached=outcome.cache_hit,
        )

    # ------------------------------------------------------------------ #
    # Deterministic deadline path
    # ------------------------------------------------------------------ #
    async def _deadline_answer(
        self, principal: Principal, window: TimeWindow, doc_types: tuple[str, ...], now: datetime
    ) -> _Outcome:
        async with self._deps.db.session(principal.db_context) as session:
            rows = await find_deadlines(
                session,
                principal,
                start=window.start,
                end=window.end,
                today=now.date(),
                now=now,
                doc_types=doc_types,
                limit=DEADLINE_ROWS,
            )
        span = f"between {window.start:%d %B %Y} and {window.end:%d %B %Y}"
        if not rows:
            return _Outcome(
                status="insufficient_context",
                answer=f"No documents you can access have a recorded deadline {span}.",
                path="deadlines",
                warnings=[WARN_DEADLINES],
                confidence=0.0,
            )
        lines = [f"{len(rows)} deadline(s) {span}:"]
        citations: list[Citation] = []
        evidence: list[Evidence] = []
        for n, row in enumerate(rows, start=1):
            title = clean_line_text(row.document_title, 200)
            when = "today" if row.days_left == 0 else f"in {row.days_left} day(s)"
            lines.append(f"{n}. {title} - {row.label} on {row.due:%d %B %Y} ({when}) [{n}]")
            quote = clean_line_text(row.evidence or f"{row.field}: {row.due.isoformat()}", 500)
            citations.append(
                Citation(
                    n=n,
                    source_id=f"D{n}",
                    document_id=row.document_id,
                    document_title=title,
                    version_number=row.version_number,
                    page_start=row.page,
                    page_end=row.page,
                    section=None,
                    quote=quote,
                    chunk_id=row.chunk_id,
                )
            )
            evidence.append(
                Evidence(
                    source_id=f"D{n}",
                    document_id=row.document_id,
                    document_title=title,
                    page_start=row.page,
                    section=None,
                    excerpt=truncate(quote, EXCERPT_CHARS),
                )
            )
        confidence = _clamp(sum(r.confidence for r in rows) / len(rows))
        return _Outcome(
            status="answered",
            answer="\n".join(lines),
            path="deadlines",
            citations=citations,
            evidence=evidence,
            warnings=[WARN_DEADLINES],
            confidence=confidence,
            source_count=len(rows),
        )

    # ------------------------------------------------------------------ #
    # Retrieval-augmented path
    # ------------------------------------------------------------------ #
    async def _governed(
        self, chunks: Sequence[RetrievedChunk], org_id: uuid.UUID
    ) -> tuple[list[RetrievedChunk], bool]:
        llm = self._deps.llm
        routable: dict[str, bool] = {}
        for value in {c.classification for c in chunks}:
            routable[value] = await llm.route_for_org(Classification(value), org_id) is not None
        kept = [c for c in chunks if routable[c.classification]]
        return kept, len(kept) < len(chunks)

    # ------------------------------------------------------------------ #
    # Deterministic document-finding path ("find documents about X")
    # ------------------------------------------------------------------ #
    async def _search_answer(
        self,
        principal: Principal,
        *,
        text: str,
        doc_types: tuple[str, ...],
        filters: SearchFilters | None,
        include_old_versions: bool,
    ) -> _Outcome:
        """List the authorised documents that best match the topic, with their best passage.

        No model is involved: the answer is the ranked retrieval result grouped by document,
        so every line is a real document the user can open.
        """
        topic = search_topic(text)
        base = filters or SearchFilters()
        if (doc_types and not base.doc_types) or include_old_versions != base.include_old_versions:
            base = SearchFilters(
                doc_types=base.doc_types or doc_types,
                department_ids=base.department_ids,
                classifications=base.classifications,
                document_ids=base.document_ids,
                tags=base.tags,
                created_after=base.created_after,
                created_before=base.created_before,
                include_old_versions=include_old_versions,
            )
        top_k = max(self._deps.settings.retrieval.top_k, SEARCH_MAX_DOCUMENTS)
        retrieved = await self._deps.search.retrieve(principal, topic, filters=base, top_k=top_k)
        warnings: list[str] = []
        excluded = int(getattr(retrieved, "excluded_injection", 0) or 0)
        if excluded:
            warnings.append(WARN_EXCLUDED_INJECTION.format(n=excluded))
        if getattr(retrieved, "degraded", False):
            warnings.append(WARN_DEGRADED)
        best: dict[uuid.UUID, Any] = {}
        for chunk in retrieved:
            if chunk.document_id not in best:
                best[chunk.document_id] = chunk
            if len(best) >= SEARCH_MAX_DOCUMENTS:
                break
        if not best:
            return _Outcome(
                status="insufficient_context",
                answer=f'No documents you can access match "{truncate(topic, 200)}".',
                path="search",
                warnings=warnings,
            )
        lines = [f'{len(best)} document(s) match "{truncate(topic, 200)}":']
        citations: list[Citation] = []
        evidence: list[Evidence] = []
        for n, chunk in enumerate(best.values(), start=1):
            title = clean_line_text(chunk.document_title, 200)
            where = f"p. {chunk.page_start}" if chunk.page_start else "document"
            if chunk.section:
                where += f", {clean_line_text(chunk.section, 120)}"
            lines.append(f"{n}. {title} ({where}) [{n}]")
            excerpt = truncate(" ".join(chunk.content.split()), EXCERPT_CHARS)
            citations.append(
                Citation(
                    n=n,
                    source_id=f"S{n}",
                    document_id=chunk.document_id,
                    document_title=title,
                    version_number=chunk.version_number,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    quote=excerpt,
                    chunk_id=chunk.chunk_id,
                )
            )
            evidence.append(
                Evidence(
                    source_id=f"S{n}",
                    document_id=chunk.document_id,
                    document_title=title,
                    page_start=chunk.page_start,
                    section=chunk.section,
                    excerpt=excerpt,
                )
            )
        top = [c.score for c in list(best.values())[:3]]
        return _Outcome(
            status="answered",
            answer="\n".join(lines),
            path="search",
            citations=citations,
            evidence=evidence,
            warnings=warnings,
            confidence=_clamp(sum(top) / len(top)),
            source_count=len(best),
        )

    async def _rag_answer(
        self,
        principal: Principal,
        *,
        text: str,
        history: list[str],
        filters: SearchFilters | None,
        include_old_versions: bool,
    ) -> _Outcome:
        settings = self._deps.settings
        org_id = principal.require_org()
        warnings: list[str] = []
        base = filters or SearchFilters()
        if include_old_versions != base.include_old_versions:
            base = SearchFilters(
                doc_types=base.doc_types,
                department_ids=base.department_ids,
                classifications=base.classifications,
                document_ids=base.document_ids,
                tags=base.tags,
                created_after=base.created_after,
                created_before=base.created_before,
                include_old_versions=include_old_versions,
            )
        query = text
        if history and len(terms(text)) < SHORT_FOLLOW_UP_TERMS:
            query = f"{history[0]} {text}"  # resolve a short follow-up with the last question
        retrieved = await self._deps.search.retrieve(
            principal, query, filters=base, top_k=settings.retrieval.top_k
        )
        chunks = list(retrieved)
        excluded = int(getattr(retrieved, "excluded_injection", 0) or 0)
        if excluded:
            warnings.append(WARN_EXCLUDED_INJECTION.format(n=excluded))
        if getattr(retrieved, "degraded", False):
            warnings.append(WARN_DEGRADED)
        if not chunks:
            return _Outcome(
                status="insufficient_context",
                answer=INSUFFICIENT_TEXT,
                path="no_context",
                warnings=warnings,
            )
        selected, dropped = await self._governed(chunks, org_id)
        if dropped:
            warnings.append(WARN_POLICY_EXCLUDED)
        if not selected:
            return _Outcome(
                status="insufficient_context",
                answer=INSUFFICIENT_TEXT,
                path="policy",
                warnings=warnings,
            )
        context = build_context(
            selected,
            max_tokens=settings.llm.max_context_tokens,
            warn_threshold=settings.retrieval.injection_warn_threshold,
        )
        if context.flagged:
            warnings.append(WARN_FLAGGED_SOURCES)
        outcome = await self._generate(
            principal,
            text=text,
            history=history,
            context=context,
            warnings=warnings,
            include_old_versions=include_old_versions,
        )
        if outcome is None:
            # Late policy denial (the organisation's policy changed mid-request): retry once
            # with only the chunks that are still routable.
            selected, _ = await self._governed([ref.chunk for ref in context.sources], org_id)
            warnings.append(WARN_POLICY_EXCLUDED)
            if not selected:
                return _Outcome(
                    status="insufficient_context",
                    answer=INSUFFICIENT_TEXT,
                    path="policy",
                    warnings=warnings,
                )
            context = build_context(
                selected,
                max_tokens=settings.llm.max_context_tokens,
                warn_threshold=settings.retrieval.injection_warn_threshold,
            )
            outcome = await self._generate(
                principal,
                text=text,
                history=history,
                context=context,
                warnings=warnings,
                include_old_versions=include_old_versions,
            )
            if outcome is None:
                return _Outcome(
                    status="insufficient_context",
                    answer=INSUFFICIENT_TEXT,
                    path="policy",
                    warnings=warnings,
                )
        return outcome

    def _cache_key(
        self,
        principal: Principal,
        *,
        text: str,
        history: list[str],
        context: BuiltContext,
        route: str,
        include_old_versions: bool,
    ) -> str:
        signature = ",".join(
            sorted(
                f"{ref.chunk.chunk_id}:{_sha(ref.chunk.content)[:16]}" for ref in context.sources
            )
        )
        return self._deps.cache.key(
            "answer",
            str(principal.require_org()),
            principal.access_fingerprint(),
            PROMPT_VERSION,
            route,
            text.casefold(),
            "\x1e".join(history),
            "old" if include_old_versions else "current",
            signature,
        )

    async def _generate(
        self,
        principal: Principal,
        *,
        text: str,
        history: list[str],
        context: BuiltContext,
        warnings: list[str],
        include_old_versions: bool,
    ) -> _Outcome | None:
        """Model call + validation. ``None`` means the gateway denied the route (policy)."""
        settings = self._deps.settings
        llm = self._deps.llm
        org_id = principal.require_org()
        classification = Classification.highest(
            [Classification(r.chunk.classification) for r in context.sources]
        )
        route = next(
            (row for row in await llm.policy(org_id) if row.classification is classification), None
        )
        cacheable = (
            settings.llm.answer_cache_ttl_seconds > 0
            and classification is not Classification.RESTRICTED
            and route is not None
            and route.allowed
        )
        key = (
            self._cache_key(
                principal,
                text=text,
                history=history,
                context=context,
                route=f"{route.provider}:{route.model}",
                include_old_versions=include_old_versions,
            )
            if cacheable and route is not None
            else None
        )
        payload: dict[str, Any] | None = None
        cache_hit = False
        if key is not None:
            cached = await self._deps.cache.get_json(key)
            if isinstance(cached, dict) and cached.get("status") in ANSWER_STATUSES:
                payload, cache_hit = cached, True
        if payload is None:
            request = LLMRequest(
                task=LLMTask.ANSWER,
                system=answer_system_prompt(self.canary),
                messages=[
                    ChatMessage(role="user", content=render_user_message(context, text, history))
                ],
                output_schema=ANSWER_SCHEMA,
                data_classification=classification,
            )
            try:
                result = await llm.complete(request, org_id=org_id, user_id=principal.user_id)
            except LLMPolicyDenied:
                return None
            except LLMRefused as exc:
                return _Outcome(
                    status="refused",
                    answer="The AI model declined to answer this question.",
                    path="rag",
                    warnings=warnings,
                    model=exc.model,
                    usage=Usage(exc.input_tokens, exc.output_tokens),
                    source_count=len(context.sources),
                )
            except LLMOutputInvalid as exc:
                return _Outcome(
                    status="insufficient_context",
                    answer="The AI response could not be validated, so no answer is shown.",
                    path="rag",
                    warnings=warnings,
                    model=exc.model,
                    usage=Usage(exc.input_tokens, exc.output_tokens),
                    source_count=len(context.sources),
                )
            data = result.data or {}
            payload = {
                "status": data.get("status"),
                "answer": data.get("answer", ""),
                "citations": data.get("citations", []),
                "confidence": data.get("confidence", "low"),
                "missing_information": data.get("missing_information", ""),
                "model": result.model,
                "provider": result.provider,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }
        outcome = self._validate(payload, context, warnings)
        outcome.cache_hit = cache_hit
        if (
            key is not None
            and not cache_hit
            and outcome.path == "rag"
            and WARN_GUARD not in outcome.warnings
        ):
            await self._deps.cache.set_json(key, payload, settings.llm.answer_cache_ttl_seconds)
        return outcome

    def _validate(
        self,
        payload: dict[str, Any],
        context: BuiltContext,
        warnings: list[str],
    ) -> _Outcome:
        guard = self.guard()
        sources_text = [ref.chunk.content for ref in context.sources]
        status_value = payload.get("status")
        status: AnswerStatus = (
            status_value if status_value in ANSWER_STATUSES else "insufficient_context"
        )
        report = validate_citations(
            payload.get("citations"),
            context.by_sid(),
            key_field="source_id",
            text_of=lambda ref: ref.chunk.content,
        )
        answer_check = guard.check(str(payload.get("answer") or ""), sources=sources_text)
        missing_check = guard.check(
            str(payload.get("missing_information") or ""), sources=sources_text, max_chars=500
        )
        outcome = _Outcome(
            status=status,
            answer=answer_check.text,
            path="rag",
            warnings=warnings,
            model=str(payload.get("model") or "") or None,
            provider=str(payload.get("provider") or "") or None,
            usage=Usage(
                int(payload.get("input_tokens") or 0), int(payload.get("output_tokens") or 0)
            ),
            source_count=len(context.sources),
        )
        outcome.evidence = [
            Evidence(
                source_id=ref.sid,
                document_id=ref.chunk.document_id,
                document_title=clean_line_text(ref.chunk.document_title, 200),
                page_start=ref.chunk.page_start,
                section=clean_line_text(ref.chunk.section, 300) if ref.chunk.section else None,
                excerpt=truncate(
                    " ".join(sanitize_text(ref.chunk.content)[0].split()), EXCERPT_CHARS
                ),
            )
            for ref in context.sources
        ]
        if answer_check.blocked or missing_check.blocked:
            outcome.status = "refused"
            outcome.answer = REFUSAL_TEXT
            outcome.warnings.append(WARN_GUARD)
            outcome.path = "guard_blocked"
            return outcome
        if answer_check.actions or missing_check.actions:
            outcome.warnings.append(WARN_GUARD)
        if report.invalid:
            outcome.warnings.append(WARN_UNVERIFIED.format(n=report.invalid))
        outcome.citations = [
            _citation(
                n, valid.target, guard.check(valid.quote, sources=sources_text, max_chars=500).text
            )
            for n, valid in enumerate(report.valid, start=1)
        ]
        if outcome.status == "answered" and not outcome.citations:
            outcome.status = "insufficient_context"
            outcome.answer = UNVERIFIED_TEXT
        if outcome.status == "insufficient_context" and not outcome.answer:
            outcome.answer = INSUFFICIENT_TEXT
        if missing_check.text and outcome.status != "answered":
            outcome.answer = f"{outcome.answer}\n\nMissing information: {missing_check.text}"
        outcome.confidence = _confidence(
            outcome.status,
            [ref.chunk for ref in context.sources],
            report.validity_ratio,
            payload.get("confidence"),
        )
        return outcome

    # ------------------------------------------------------------------ #
    async def _persist(
        self,
        principal: Principal,
        *,
        conversation_id: uuid.UUID | None,
        question: str,
        outcome: _Outcome,
        analysis: QueryAnalysis,
        latency_ms: int,
    ) -> tuple[uuid.UUID, uuid.UUID]:
        org_id = principal.require_org()
        stamp = utcnow()
        async with self._deps.db.transaction(principal.db_context) as session:
            if conversation_id is not None:
                conversation = await load_owned_conversation(session, principal, conversation_id)
                conversation.updated_at = stamp
            else:
                conversation = Conversation(
                    organization_id=org_id,
                    user_id=principal.user_id,
                    title=clean_line_text(question, 120) or "Conversation",
                    created_at=stamp,
                    updated_at=stamp,
                )
                session.add(conversation)
                await session.flush()
            question_row = Message(
                organization_id=org_id,
                conversation_id=conversation.id,
                user_id=principal.user_id,
                role=MessageRole.USER.value,
                content=question,
                created_at=stamp,
            )
            answer_row = Message(
                organization_id=org_id,
                conversation_id=conversation.id,
                user_id=principal.user_id,
                role=MessageRole.ASSISTANT.value,
                content=outcome.answer,
                status=outcome.status,
                confidence=round(outcome.confidence, 3),
                citations=[_citation_json(c) for c in outcome.citations],
                warnings=list(dict.fromkeys(outcome.warnings)),
                model=outcome.model[:100] if outcome.model else None,
                created_at=stamp + timedelta(microseconds=1),
            )
            session.add_all([question_row, answer_row])
            await session.flush()
            self._deps.audit.record(
                session,
                Actor.of(principal),
                "assistant.ask",
                resource_type="conversation",
                resource_id=conversation.id,
                details={
                    "question_chars": len(question),
                    "question_sha256": _sha(question)[:16],
                    "status": outcome.status,
                    "path": outcome.path,
                    "intent": analysis.intent,
                    "flags": list(analysis.flags)[:10],
                    "model": outcome.model,
                    "provider": outcome.provider,
                    "sources": outcome.source_count,
                    "cited_document_ids": sorted({str(c.document_id) for c in outcome.citations})[
                        :10
                    ],
                    # every document whose text was returned as evidence, not only the cited ones
                    "disclosed_document_ids": sorted(
                        {str(e.document_id) for e in outcome.evidence}
                        | {str(c.document_id) for c in outcome.citations}
                    )[:25],
                    "cache_hit": outcome.cache_hit,
                    "prompt_version": PROMPT_VERSION,
                    "latency_ms": latency_ms,
                },
            )
            return conversation.id, answer_row.id


def _citation(n: int, ref: SourceRef, quote: str) -> Citation:
    chunk = ref.chunk
    return Citation(
        n=n,
        source_id=ref.sid,
        document_id=chunk.document_id,
        document_title=clean_line_text(chunk.document_title, 200),
        version_number=chunk.version_number,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section=clean_line_text(chunk.section, 300) if chunk.section else None,
        quote=quote,
        chunk_id=chunk.chunk_id,
    )


def _citation_json(citation: Citation) -> dict[str, Any]:
    return {
        "n": citation.n,
        "source_id": citation.source_id,
        "document_id": str(citation.document_id),
        "document_title": citation.document_title,
        "version_number": citation.version_number,
        "page_start": citation.page_start,
        "page_end": citation.page_end,
        "section": citation.section,
        "quote": citation.quote,
        "chunk_id": str(citation.chunk_id) if citation.chunk_id else None,
    }


def _confidence(
    status: str, chunks: Sequence[RetrievedChunk], validity: float, model_label: Any
) -> float:
    """0.35 x retrieval strength (mean of the top-3 scores) + 0.4 x citation validity ratio
    + 0.25 x the model's own confidence label; 0 for anything but a verified answer."""
    if status != "answered":
        return 0.0
    top = sorted((_clamp(c.score) for c in chunks), reverse=True)[:3]
    retrieval = sum(top) / len(top) if top else 0.0
    model = _MODEL_CONFIDENCE.get(str(model_label), 0.3)
    return _clamp(0.35 * retrieval + 0.4 * validity + 0.25 * model)
