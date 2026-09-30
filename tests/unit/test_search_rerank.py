"""Lexical reranker and the LLM reranker hook (with fake judges)."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from docassist.authz.principal import Principal
from docassist.core.enums import Classification, Role
from docassist.observability.metrics import REGISTRY
from docassist.search.rerank import (
    JudgePassage,
    LexicalReranker,
    LLMReranker,
    Reranker,
    lexical_features,
)
from docassist.search.types import RetrievedChunk

PRINCIPAL = Principal(
    user_id=uuid.uuid4(),
    org_id=uuid.uuid4(),
    role=Role.EMPLOYEE,
    clearance=Classification.CONFIDENTIAL,
    session_id=uuid.uuid4(),
)


def chunk(
    content: str,
    *,
    score: float = 0.5,
    section: str | None = None,
    heading_path: tuple[str, ...] = (),
    classification: str = "INTERNAL",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        version_number=1,
        is_current=True,
        document_title="Doc",
        classification=classification,
        doc_type="other",
        department_id=None,
        page_start=1,
        page_end=1,
        section=section,
        heading_path=heading_path,
        content=content,
        score=score,
        keyword_score=None,
        semantic_score=None,
        injection_score=0.0,
        injection_flags=(),
        pii_types=(),
    )


def test_lexical_features() -> None:
    features = lexical_features(
        "payment terms", chunk("The payment terms are net 30.", section="Payment")
    )
    assert features.coverage == 1.0
    assert features.phrase == 1.0
    assert features.heading == 0.5
    partial = lexical_features("payment schedule terms", chunk("payment terms apply"))
    assert partial.coverage == pytest.approx(2 / 3)
    assert partial.phrase == 0.0  # neither "payment schedule" nor "schedule terms" occurs
    assert lexical_features("the of", chunk("nothing here")).coverage == 0.0
    assert lexical_features("", chunk("text")) == lexical_features("!!", chunk("text"))


def test_lexical_prefix_and_stem_matching() -> None:
    assert lexical_features("renewal", chunk("Renewals are automatic")).coverage == 1.0
    assert lexical_features("terminat", chunk("termination rights")).coverage == 1.0


async def test_lexical_reranker_orders_by_evidence() -> None:
    reranker = LexicalReranker()
    assert isinstance(reranker, Reranker)
    unrelated = chunk("Office opening hours and parking.", score=0.6)
    partial = chunk("Payment is due on receipt.", score=0.5)
    exact = chunk(
        "Termination fee: the termination fee equals two months.",
        score=0.4,
        section="Termination fee",
    )
    ranked = await reranker.rerank(PRINCIPAL, "termination fee", [unrelated, partial, exact])
    assert ranked[0].chunk_id == exact.chunk_id
    assert ranked[-1].content in {unrelated.content, partial.content}
    assert all(0.0 <= c.score <= 1.0 for c in ranked)
    assert [c.score for c in ranked] == sorted((c.score for c in ranked), reverse=True)


async def test_lexical_reranker_is_deterministic_on_ties() -> None:
    first, second = chunk("alpha", score=0.5), chunk("alpha", score=0.5)
    ranked = await LexicalReranker().rerank(PRINCIPAL, "alpha", [first, second])
    assert [c.chunk_id for c in ranked] == [first.chunk_id, second.chunk_id]


def test_lexical_reranker_weight_validation() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        LexicalReranker(base_weight=0.9)
    with pytest.raises(ValueError, match="non-negative"):
        LexicalReranker(base_weight=1.2, coverage_weight=-0.2, phrase_weight=0, heading_weight=0)
    assert LexicalReranker(1.0, 0.0, 0.0, 0.0).score("x", chunk("x", score=0.3)) == 0.3


class FakeJudge:
    def __init__(self, grades: Mapping[int, Any] | None = None, exc: Exception | None = None):
        self.grades = grades or {}
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    async def judge(
        self,
        principal: Principal,
        query: str,
        passages: Sequence[JudgePassage],
        *,
        classification: Classification,
    ) -> Mapping[int, float]:
        self.calls.append(
            {"query": query, "passages": list(passages), "classification": classification}
        )
        if self.exc:
            raise self.exc
        return self.grades


async def test_llm_reranker_blends_grades() -> None:
    a = chunk("alpha beta", score=0.2)
    b = chunk("gamma delta", score=0.1, classification="CONFIDENTIAL")
    # lexically "delta" puts b first; the judge (grading passage order) strongly prefers a
    base = await LexicalReranker().rerank(PRINCIPAL, "delta", [a, b])
    assert base[0].chunk_id == b.chunk_id
    judge = FakeJudge({1: 1.0})
    reranker = LLMReranker(judge, blend=1.0)
    ranked = await reranker.rerank(PRINCIPAL, "delta", [a, b])
    assert ranked[0].chunk_id == a.chunk_id
    assert ranked[0].score == 1.0
    call = judge.calls[-1]
    assert call["classification"] is Classification.CONFIDENTIAL  # highest among passages
    assert {p.text for p in call["passages"]} == {"alpha beta", "gamma delta"}


async def test_llm_reranker_falls_back_to_lexical_on_failure() -> None:
    before = (
        REGISTRY.get_sample_value("docassist_degraded_mode_total", {"component": "reranker"}) or 0.0
    )
    judge = FakeJudge(exc=RuntimeError("gateway unavailable"))
    chunks = [chunk("unrelated"), chunk("termination fee clause")]
    ranked = await LLMReranker(judge).rerank(PRINCIPAL, "termination fee", chunks)
    lexical = await LexicalReranker().rerank(PRINCIPAL, "termination fee", chunks)
    assert [c.chunk_id for c in ranked] == [c.chunk_id for c in lexical]
    after = REGISTRY.get_sample_value("docassist_degraded_mode_total", {"component": "reranker"})
    assert after == before + 1


async def test_llm_reranker_ignores_invalid_grades_and_caps_passages() -> None:
    chunks = [chunk(f"passage {i} " + "x" * 5_000, score=0.5 - i * 0.01) for i in range(5)]
    judge = FakeJudge({0: math.nan, 1: "high", 2: 7.0, 99: 1.0})
    reranker = LLMReranker(judge, blend=0.5, max_candidates=3, max_passage_chars=200)
    ranked = await reranker.rerank(PRINCIPAL, "passage", chunks)
    passages = judge.calls[-1]["passages"]
    assert len(passages) == 3
    assert all(len(p.text) == 200 for p in passages)
    assert len(ranked) == 5
    assert all(0.0 <= c.score <= 1.0 for c in ranked)
    # grade 7.0 is clamped to 1.0, so passage 2 moves to the top of the judged head
    assert ranked[0].content.startswith("passage 2")
    # the two unjudged candidates keep their lexical order after the head
    assert [c.content[:9] for c in ranked[3:]] == ["passage 3", "passage 4"]


async def test_llm_reranker_empty_and_parameter_validation() -> None:
    assert await LLMReranker(FakeJudge()).rerank(PRINCIPAL, "q", []) == []
    with pytest.raises(ValueError, match="blend"):
        LLMReranker(FakeJudge(), blend=2.0)
