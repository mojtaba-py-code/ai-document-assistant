"""Second-stage reranking of already-authorised candidates.

* :class:`LexicalReranker` (default) - deterministic, offline: blends the first-stage score
  with query-term coverage, an exact-phrase bonus and a heading/section match.
* :class:`LLMReranker` - optional hook that asks a :class:`RelevanceJudge` (e.g. an adapter
  over the LLM gateway with task ``RERANK``) for graded relevance and blends it in. It is off
  by default; any judge failure falls back to the lexical order (reranking must never make
  retrieval fail). The judge receives the highest classification among the passages so a
  gateway adapter can apply the external-provider data policy.

Rerankers only reorder what they are given: they never add candidates, so they cannot widen
what a principal can see.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol, runtime_checkable

from docassist.authz.principal import Principal
from docassist.core.enums import Classification
from docassist.core.logging import get_logger
from docassist.observability import metrics
from docassist.search.hybrid import clamp01
from docassist.search.text import query_terms, stem, tokenize
from docassist.search.types import RetrievedChunk

log = get_logger(__name__)


@runtime_checkable
class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    async def rerank(
        self, principal: Principal, query: str, chunks: Sequence[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Return the same chunks reordered, best first, with ``score`` updated (0..1)."""
        ...


@dataclass(frozen=True, slots=True)
class LexicalFeatures:
    coverage: float  # share of distinct query terms present in the chunk
    phrase: float  # 1.0 exact phrase, else share of query bigrams present
    heading: float  # share of query terms present in section / heading path


def _matches(term: str, term_stem: str, tokens: set[str], stems: set[str]) -> bool:
    if term in tokens or term_stem in stems:
        return True
    return len(term) >= 4 and any(token.startswith(term) for token in tokens)


def lexical_features(query: str, chunk: RetrievedChunk) -> LexicalFeatures:
    terms = query_terms(query)
    if not terms:
        return LexicalFeatures(0.0, 0.0, 0.0)
    content_tokens = tokenize(chunk.content)
    token_set = set(content_tokens)
    stem_set = {stem(t) for t in content_tokens}
    term_stems = [stem(t) for t in terms]
    covered = sum(
        1
        for term, term_stem in zip(terms, term_stems, strict=True)
        if _matches(term, term_stem, token_set, stem_set)
    )
    coverage = covered / len(terms)

    phrase = 0.0
    if len(terms) >= 2:
        stems_seq = [stem(t) for t in content_tokens]
        joined = " " + " ".join(stems_seq) + " "
        if " " + " ".join(term_stems) + " " in joined:
            phrase = 1.0
        else:
            bigrams = list(pairwise(term_stems))
            present = set(pairwise(stems_seq))
            phrase = sum(1 for b in bigrams if b in present) / len(bigrams)

    heading_tokens = tokenize(" ".join([chunk.section or "", *chunk.heading_path]))
    heading = 0.0
    if heading_tokens:
        h_tokens = set(heading_tokens)
        h_stems = {stem(t) for t in heading_tokens}
        heading = sum(
            1
            for term, term_stem in zip(terms, term_stems, strict=True)
            if _matches(term, term_stem, h_tokens, h_stems)
        ) / len(terms)
    return LexicalFeatures(coverage, phrase, heading)


@dataclass(frozen=True, slots=True)
class LexicalReranker:
    """``score = base*w_b + coverage*w_c + phrase*w_p + heading*w_h`` (weights sum to 1)."""

    base_weight: float = 0.5
    coverage_weight: float = 0.3
    phrase_weight: float = 0.1
    heading_weight: float = 0.1
    name: str = "lexical"

    def __post_init__(self) -> None:
        weights = (self.base_weight, self.coverage_weight, self.phrase_weight, self.heading_weight)
        if any(w < 0 for w in weights) or not math.isclose(sum(weights), 1.0):
            raise ValueError("reranker weights must be non-negative and sum to 1")

    def score(self, query: str, chunk: RetrievedChunk) -> float:
        f = lexical_features(query, chunk)
        return clamp01(
            self.base_weight * clamp01(chunk.score)
            + self.coverage_weight * f.coverage
            + self.phrase_weight * f.phrase
            + self.heading_weight * f.heading
        )

    async def rerank(
        self, principal: Principal, query: str, chunks: Sequence[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        scored = [(self.score(query, chunk), index, chunk) for index, chunk in enumerate(chunks)]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [dataclasses.replace(chunk, score=score) for score, _, chunk in scored]


@dataclass(frozen=True, slots=True)
class JudgePassage:
    index: int
    title: str
    section: str | None
    text: str


class RelevanceJudge(Protocol):
    """Grades passages for a query; returns ``{passage index: relevance 0..1}``."""

    async def judge(
        self,
        principal: Principal,
        query: str,
        passages: Sequence[JudgePassage],
        *,
        classification: Classification,
    ) -> Mapping[int, float]: ...


class LLMReranker:
    """Blend lexical order with a relevance judge's grades (lexical fallback on any failure).

    Only the ``max_candidates`` best lexical candidates are sent to the judge (each passage
    capped at ``max_passage_chars``); the remaining candidates keep their lexical order after
    them. Grades that are missing or not finite leave a candidate's lexical score unchanged.
    """

    name = "llm"

    def __init__(
        self,
        judge: RelevanceJudge,
        *,
        fallback: Reranker | None = None,
        blend: float = 0.5,
        max_candidates: int = 20,
        max_passage_chars: int = 2_000,
    ) -> None:
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be within 0..1")
        self._judge = judge
        self._fallback: Reranker = fallback or LexicalReranker()
        self._blend = blend
        self._max_candidates = max(1, max_candidates)
        self._max_chars = max(100, max_passage_chars)

    async def rerank(
        self, principal: Principal, query: str, chunks: Sequence[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        base = await self._fallback.rerank(principal, query, chunks)
        head, tail = base[: self._max_candidates], base[self._max_candidates :]
        if not head:
            return base
        passages = [
            JudgePassage(i, c.document_title, c.section, c.content[: self._max_chars])
            for i, c in enumerate(head)
        ]
        ceiling = Classification.highest([Classification(c.classification) for c in head])
        try:
            grades = await self._judge.judge(principal, query, passages, classification=ceiling)
        except Exception as exc:  # noqa: BLE001 - reranking is best-effort by design
            log.warning("llm_rerank_failed", error_type=type(exc).__name__)
            metrics.DEGRADED_MODE.labels(component="reranker").inc()
            return base
        blended = []
        for index, chunk in enumerate(head):
            grade = grades.get(index)
            if isinstance(grade, int | float) and math.isfinite(grade):
                score = (1 - self._blend) * chunk.score + self._blend * clamp01(float(grade))
            else:
                score = chunk.score
            blended.append((clamp01(score), index, chunk))
        blended.sort(key=lambda item: (-item[0], item[1]))
        return [dataclasses.replace(c, score=s) for s, _, c in blended] + tail
