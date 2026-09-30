"""Document summaries: map-reduce over the current version's chunks, with citations.

Flow (``method="llm"``)
    1. load the document with the readable clause, then its current version's chunks;
    2. drop chunks at/above the injection exclusion threshold, give the rest local ids
       ``C1..Cn`` and pack them into batches that fit the context budget;
    3. one batch -> a single call; several -> one *map* call per batch producing a partial
       summary whose key points cite ``C`` ids, then *reduce* calls (hierarchical when the
       partials themselves exceed the budget) that merge partials and keep the citations;
    4. every returned citation id is checked against the ids that were sent; key points
       without a valid citation are dropped.

The data classification of every call is the document's classification, so the gateway's
data-governance routing applies (RESTRICTED text never reaches an external provider).

Fallback (``method="extractive"``)
    When no model is available, the policy forbids AI processing, the quota or rate limit
    is exhausted, or the output is invalid/unverifiable, the summary is built from lead
    sentences of each section - deterministic, cited, and flagged ``method="extractive"``.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docassist.audit.service import Actor
from docassist.authz.principal import Principal
from docassist.core.context import utcnow
from docassist.core.text import estimate_tokens
from docassist.intelligence.access import (
    ChunkView,
    DocView,
    VersionView,
    load_chunks,
    load_readable_document,
    load_version,
)
from docassist.intelligence.llm_contract import ChatMessage, Gateway, LLMRequest, LLMTask
from docassist.intelligence.prompting import (
    PROMPT_VERSION,
    LLMOutcome,
    SourceBatch,
    call_gateway,
    clean_output,
    eligible_chunks,
    fallback_warning,
    gateway_of,
    input_budget,
    new_nonce,
    output_tokens,
    pack_sources,
    security_rules,
)
from docassist.intelligence.schemas import (
    KeyPoint,
    SourceRef,
    SummaryResult,
    SummaryStyle,
    Usage,
)
from docassist.intelligence.textops import (
    escape_prompt_text,
    normalize_for_match,
    split_sentences,
    word_count,
)

if TYPE_CHECKING:
    from docassist.api.container import Container

MAX_SUMMARY_CHUNKS = 3_000
MAX_MAP_BATCHES = 12
MAX_REDUCE_ROUNDS = 3
MAP_CONCURRENCY = 3
MIN_BUDGET_TOKENS = 300
LLM_RATE_BUCKET = "llm_user"


@dataclass(frozen=True, slots=True)
class StyleSpec:
    key_points: int
    words: int
    summary_sentences: int
    instruction: str


STYLES: dict[str, StyleSpec] = {
    "brief": StyleSpec(5, 120, 3, "a brief summary (at most about 120 words)"),
    "executive": StyleSpec(
        7,
        200,
        4,
        "an executive summary (at most about 200 words) focused on decisions, obligations, "
        "money, dates and risks",
    ),
    "detailed": StyleSpec(12, 400, 8, "a detailed summary (at most about 400 words)"),
}


# --------------------------------------------------------------------------- #
# Model output validation
# --------------------------------------------------------------------------- #
class _KeyPointOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=2_000)
    sources: list[str] = Field(max_length=20)


class _SummaryOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(max_length=8_000)
    key_points: list[_KeyPointOut] = Field(max_length=30)


def summary_schema(max_points: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "maxLength": 4_000},
            "key_points": {
                "type": "array",
                "maxItems": max_points,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "maxLength": 600},
                        "sources": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string", "maxLength": 8},
                        },
                    },
                    "required": ["text", "sources"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "key_points"],
        "additionalProperties": False,
    }


@dataclass(slots=True)
class _Partial:
    summary: str
    points: list[tuple[str, list[str]]]


def parse_summary(
    data: dict[str, Any] | None, valid_ids: set[str], max_points: int
) -> _Partial | None:
    """Validate a model summary; unknown citation ids are removed, uncited points dropped."""
    if data is None:
        return None
    try:
        parsed = _SummaryOut.model_validate(data)
    except ValidationError:
        return None
    points: list[tuple[str, list[str]]] = []
    for kp in parsed.key_points:
        text = clean_output(kp.text, 600)
        sources = []
        for raw in kp.sources:
            sid = raw.strip().upper()
            if sid in valid_ids and sid not in sources:
                sources.append(sid)
        if text and sources:
            points.append((text, sources))
        if len(points) >= max_points:
            break
    return _Partial(clean_output(parsed.summary, 4_000), points)


# --------------------------------------------------------------------------- #
# Extractive summary
# --------------------------------------------------------------------------- #
def _section_key(chunk: ChunkView) -> str:
    if chunk.section:
        return chunk.section
    if chunk.heading_path:
        return chunk.heading_path[-1]
    return f"page:{chunk.page_start}"


def _candidate_sentences(chunk: ChunkView) -> list[str]:
    out = []
    for sentence in split_sentences(chunk.content):
        if " | " in sentence or word_count(sentence) < 5:
            continue  # table rows and headings are not lead sentences
        letters = [ch for ch in sentence if ch.isalpha()]
        if letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.8:
            continue  # ALL-CAPS banners
        out.append(clean_output(sentence, 300))
    return out


def extractive_summary(
    chunks: Sequence[ChunkView], style: SummaryStyle
) -> tuple[str, list[KeyPoint]]:
    """Lead sentences of each section (document order), cited to their chunk.

    Round 1 takes the first qualifying sentence of every section, later rounds the next
    ones, until the style's key-point budget is used; duplicates are skipped.
    """
    spec = STYLES[style]
    groups: list[list[tuple[str, ChunkView]]] = []
    last_key: str | None = None
    for chunk in chunks:
        key = _section_key(chunk)
        sentences = [(s, chunk) for s in _candidate_sentences(chunk)]
        if key != last_key or not groups:
            groups.append([])
            last_key = key
        groups[-1].extend(sentences)
    picked: list[tuple[str, ChunkView]] = []
    seen: set[str] = set()
    depth = 0
    while len(picked) < spec.key_points and any(depth < len(g) for g in groups):
        for group in groups:
            if depth < len(group) and len(picked) < spec.key_points:
                sentence, chunk = group[depth]
                norm = normalize_for_match(sentence)
                if norm and norm not in seen:
                    seen.add(norm)
                    picked.append((sentence, chunk))
        depth += 1
    key_points = [KeyPoint(text=text, citations=[source_ref(chunk)]) for text, chunk in picked]
    words: list[str] = []
    for text, _chunk in picked[: spec.summary_sentences]:
        if len(words) + word_count(text) > spec.words and words:
            break
        words.extend(text.split())
    return " ".join(words[: spec.words]), key_points


def source_ref(chunk: ChunkView) -> SourceRef:
    return SourceRef(
        chunk_id=chunk.id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section=chunk.section,
    )


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _Run:
    usage: Usage = field(default_factory=Usage)
    model: str | None = None
    provider: str | None = None

    def add(self, outcome: LLMOutcome) -> None:
        result = outcome.result
        if result is None:
            return
        self.usage = Usage(
            calls=self.usage.calls + 1,
            input_tokens=self.usage.input_tokens + int(result.input_tokens),
            output_tokens=self.usage.output_tokens + int(result.output_tokens),
        )
        self.model, self.provider = result.model, result.provider


@dataclass(frozen=True, slots=True)
class _Job:
    principal: Principal
    doc: DocView
    spec: StyleSpec
    nonce: str
    run: _Run
    gateway: Gateway | None


class Summarizer:
    def __init__(self, container: Container) -> None:
        self._c = container

    async def summarize(
        self, principal: Principal, document_id: uuid.UUID, *, style: SummaryStyle = "brief"
    ) -> SummaryResult:
        """Summarise the current version of a readable document (LLM, else extractive)."""
        principal.require_org()
        now = utcnow()
        async with self._c.db.session(principal.db_context) as session:
            doc = await load_readable_document(session, principal, document_id, now)
            version = await load_version(session, doc, None)
            chunks = await load_chunks(
                session, principal, now, doc, version, limit=MAX_SUMMARY_CHUNKS
            )
        settings = self._c.settings
        await self._c.limiter.enforce(
            LLM_RATE_BUCKET, str(principal.user_id), settings.rate_limit.llm_per_user
        )
        result = await self._llm_summary(principal, doc, version, chunks, style)
        async with self._c.db.transaction(principal.db_context) as session:
            self._c.audit.record(
                session,
                Actor.of(principal),
                "intelligence.summarize",
                resource_type="document",
                resource_id=doc.id,
                details={
                    "version_id": str(version.id),
                    "style": style,
                    "method": result.method,
                    "chunks_used": result.chunks_used,
                    "model": result.model,
                    "classification": doc.classification.value,
                },
            )
        return result

    def extractive(
        self,
        doc: DocView,
        version: VersionView,
        chunks: Sequence[ChunkView],
        style: SummaryStyle,
        *,
        warnings: list[str] | None = None,
    ) -> SummaryResult:
        summary, key_points = extractive_summary(chunks, style)
        return SummaryResult(
            document_id=doc.id,
            version_id=version.id,
            version_number=version.number,
            title=doc.title,
            style=style,
            method="extractive",
            summary=summary,
            key_points=key_points,
            chunks_total=len(chunks),
            chunks_used=len(chunks),
            warnings=list(warnings or []),
        )

    async def _llm_summary(
        self,
        principal: Principal,
        doc: DocView,
        version: VersionView,
        chunks: list[ChunkView],
        style: SummaryStyle,
    ) -> SummaryResult:
        settings = self._c.settings
        warnings: list[str] = []
        if not chunks:
            return self.extractive(
                doc, version, chunks, style, warnings=["The document has no text."]
            )
        eligible, excluded = eligible_chunks(chunks, settings)
        if excluded:
            warnings.append(
                f"{excluded} passage(s) flagged as possible prompt injection were excluded "
                "from AI processing."
            )
        if not eligible:
            warnings.append(fallback_warning("llm_output_unverifiable"))
            return self.extractive(doc, version, chunks, style, warnings=warnings)
        spec = STYLES[style]
        nonce = new_nonce()
        system = self._system_prompt(nonce, spec, stage="map")
        budget = input_budget(settings, system)
        if budget < MIN_BUDGET_TOKENS:
            warnings.append(fallback_warning("context_budget_too_small"))
            return self.extractive(doc, version, chunks, style, warnings=warnings)
        batches = pack_sources(
            eligible, nonce, budget, warn_threshold=settings.retrieval.injection_warn_threshold
        )
        truncated = len(batches) > MAX_MAP_BATCHES
        if truncated:
            batches = batches[:MAX_MAP_BATCHES]
            warnings.append(
                "The document is long; the AI summary covers its first "
                f"{sum(len(b.ids) for b in batches)} of {len(eligible)} passages."
            )
        valid_ids = {sid for batch in batches for sid in batch.ids}
        run = _Run()
        job = _Job(principal, doc, spec, nonce, run, gateway_of(self._c))
        final = await self._map_reduce(job, batches)
        if isinstance(final, str):
            warnings.append(fallback_warning(final))
            return self.extractive(doc, version, chunks, style, warnings=warnings)
        points = final.points[: spec.key_points]
        if not points:
            warnings.append(fallback_warning("llm_output_unverifiable"))
            return self.extractive(doc, version, chunks, style, warnings=warnings)
        id_map = {sid: chunk for batch in batches for sid, chunk in batch.ids.items()}
        key_points = [
            KeyPoint(
                text=text,
                citations=[source_ref(id_map[sid]) for sid in sources if sid in valid_ids],
            )
            for text, sources in points
        ]
        return SummaryResult(
            document_id=doc.id,
            version_id=version.id,
            version_number=version.number,
            title=doc.title,
            style=style,
            method="llm",
            summary=final.summary or " ".join(text for text, _ in points[: spec.summary_sentences]),
            key_points=key_points,
            chunks_total=len(chunks),
            chunks_used=len(valid_ids),
            truncated=truncated,
            model=run.model,
            provider=run.provider,
            prompt_version=PROMPT_VERSION,
            warnings=warnings,
            usage=run.usage,
        )

    # ------------------------------------------------------------------ #
    def _system_prompt(self, nonce: str, spec: StyleSpec, *, stage: str) -> str:
        if stage == "map":
            task = (
                "You summarise excerpts of a business document. Write "
                f"{spec.instruction} of the excerpt and up to {spec.key_points} key points. "
                "Every key point must list the ids (for example C3) of the <source> elements "
                "that state it; only use ids that appear in the excerpt."
            )
        else:
            task = (
                "You merge partial summaries of one business document into "
                f"{spec.instruction} and up to {spec.key_points} key points. Every key point "
                "must keep the source ids (for example C3) cited by the partial key points "
                "it is based on; only use ids that appear in the partials."
            )
        return f"{task}\n\n{security_rules(nonce)}"

    async def _call(self, job: _Job, system: str, user: str) -> LLMOutcome:
        request = LLMRequest(
            task=LLMTask.SUMMARIZE,
            system=system,
            messages=[ChatMessage(role="user", content=user)],
            output_schema=summary_schema(job.spec.key_points),
            max_output_tokens=output_tokens(self._c.settings, 2_000),
            data_classification=job.doc.classification,
        )
        outcome = await call_gateway(job.gateway, request, job.principal, feature="summarize")
        job.run.add(outcome)
        return outcome

    async def _map_reduce(self, job: _Job, batches: list[SourceBatch]) -> _Partial | str:
        """Final partial, or the fallback reason code."""
        spec, nonce = job.spec, job.nonce
        valid_ids = {sid for batch in batches for sid in batch.ids}
        title = escape_prompt_text(job.doc.title)
        map_system = self._system_prompt(nonce, spec, stage="map")
        semaphore = asyncio.Semaphore(MAP_CONCURRENCY)

        async def one(index: int, batch: SourceBatch) -> _Partial | str:
            user = (
                f"Document title (data): {title}\n"
                f"Excerpt {index} of {len(batches)}:\n\n{batch.rendered}"
            )
            async with semaphore:
                outcome = await self._call(job, map_system, user)
            if outcome.reason:
                return outcome.reason
            parsed = parse_summary(outcome.data, set(batch.ids), spec.key_points)
            return parsed if parsed is not None else "llm_output_invalid"

        results = await asyncio.gather(*(one(i, b) for i, b in enumerate(batches, start=1)))
        partials: list[_Partial] = []
        for item in results:
            if isinstance(item, str):
                return item
            partials.append(item)
        if len(partials) == 1:
            return partials[0]
        reduce_system = self._system_prompt(nonce, spec, stage="reduce")
        budget = input_budget(self._c.settings, reduce_system)
        for _round in range(MAX_REDUCE_ROUNDS):
            groups = _group_partials(partials, nonce, budget)
            if all(len(group) == 1 for group in groups):
                # every partial alone fills the budget: merge pairwise to guarantee progress
                flat = [item for group in groups for item in group]
                groups = [flat[i : i + 2] for i in range(0, len(flat), 2)]
            merged: list[_Partial] = []
            for group in groups:
                if len(group) == 1 and len(groups) > 1:
                    merged.append(group[0][1])
                    continue
                rendered = "\n\n".join(block for block, _partial in group)
                user = f"Document title (data): {title}\nPartial summaries:\n\n{rendered}"
                outcome = await self._call(job, reduce_system, user)
                if outcome.reason:
                    return outcome.reason
                parsed = parse_summary(outcome.data, valid_ids, spec.key_points)
                if parsed is None:
                    return "llm_output_invalid"
                merged.append(parsed)
            partials = merged
            if len(partials) == 1:
                return partials[0]
        return "llm_output_invalid"


def render_partial(pid: str, partial: _Partial, nonce: str) -> str:
    lines = [f'<partial id="{pid}" nonce="{nonce}">', escape_prompt_text(partial.summary)]
    for text, sources in partial.points:
        lines.append(f"- {escape_prompt_text(text)} [{', '.join(sources)}]")
    lines.append("</partial>")
    return "\n".join(lines)


def _group_partials(
    partials: list[_Partial], nonce: str, budget: int
) -> list[list[tuple[str, _Partial]]]:
    groups: list[list[tuple[str, _Partial]]] = []
    current: list[tuple[str, _Partial]] = []
    used = 0
    for index, partial in enumerate(partials, start=1):
        block = render_partial(f"P{index}", partial, nonce)
        cost = estimate_tokens(block)
        if current and used + cost > budget:
            groups.append(current)
            current, used = [], 0
        current.append((block, partial))
        used += cost
    if current:
        groups.append(current)
    return groups
