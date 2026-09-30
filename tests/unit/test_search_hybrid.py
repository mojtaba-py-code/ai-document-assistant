"""Reciprocal Rank Fusion, score normalisation and MMR (pure functions)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from docassist.search.hybrid import (
    clamp01,
    cosine_similarity,
    min_max_normalize,
    mmr,
    reciprocal_rank_fusion,
)


# --------------------------------------------------------------------------- #
# RRF
# --------------------------------------------------------------------------- #
def test_rrf_single_list_keeps_order_and_normalises_top_to_one() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert [item for item, _ in fused] == ["a", "b", "c"]
    assert fused[0][1] == pytest.approx(1.0)
    assert fused[1][1] == pytest.approx(61 / 62)


def test_rrf_rewards_agreement_between_lists() -> None:
    keyword = ["exact", "kw_only", "both"]
    semantic = ["sem_only", "both", "exact"]
    fused = dict(reciprocal_rank_fusion([keyword, semantic], k=60))
    assert fused["exact"] > fused["kw_only"]
    assert fused["both"] > fused["sem_only"]
    assert max(fused.values()) < 1.0  # nothing ranked first by both lists


def test_rrf_item_first_everywhere_scores_exactly_one() -> None:
    fused = reciprocal_rank_fusion([["x", "y"], ["x", "z"], ["x"]], k=10)
    assert fused[0] == ("x", pytest.approx(1.0))


def test_rrf_empty_list_does_not_dilute_scores() -> None:
    assert reciprocal_rank_fusion([["a"], []])[0][1] == pytest.approx(1.0)
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_duplicates_inside_a_list_count_once() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=1))
    assert fused["a"] == pytest.approx(1.0)
    assert fused["b"] == pytest.approx((1 / 4) / (1 / 2))


def test_rrf_weights() -> None:
    fused = reciprocal_rank_fusion([["kw"], ["sem"]], weights=[1.0, 3.0])
    assert [item for item, _ in fused] == ["sem", "kw"]
    zero = reciprocal_rank_fusion([["kw"], ["sem"]], weights=[1.0, 0.0])
    assert zero == [("kw", pytest.approx(1.0))]


def test_rrf_ties_are_deterministic() -> None:
    first = reciprocal_rank_fusion([["b", "a"], ["a", "b"]])
    second = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert first == second
    assert [item for item, _ in first] == ["a", "b"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": 0}, "k must"),
        ({"weights": [1.0]}, "weights must match"),
        ({"weights": [1.0, -1.0]}, "non-negative"),
        ({"weights": [1.0, math.inf]}, "finite"),
    ],
)
def test_rrf_rejects_bad_parameters(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        reciprocal_rank_fusion([["a"], ["b"]], **kwargs)


@given(
    st.lists(st.lists(st.integers(min_value=0, max_value=30), max_size=20), min_size=1, max_size=4)
)
def test_rrf_scores_are_bounded_and_sorted(rankings: list[list[int]]) -> None:
    fused = reciprocal_rank_fusion(rankings)
    scores = [score for _, score in fused]
    assert all(0.0 < s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)
    assert {item for item, _ in fused} == {i for ranking in rankings for i in ranking}


# --------------------------------------------------------------------------- #
# normalisation helpers
# --------------------------------------------------------------------------- #
def test_min_max_normalize() -> None:
    assert min_max_normalize({}) == {}
    assert min_max_normalize({"a": 2.0, "b": 4.0, "c": 3.0}) == {"a": 0.0, "b": 1.0, "c": 0.5}
    assert min_max_normalize({"a": 0.3, "b": 0.3}) == {"a": 1.0, "b": 1.0}
    assert min_max_normalize({"a": 0.0}) == {"a": 0.0}


def test_clamp_and_cosine() -> None:
    assert clamp01(1.5) == 1.0
    assert clamp01(-0.2) == 0.0
    assert clamp01(math.nan) == 0.0
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# --------------------------------------------------------------------------- #
# MMR
# --------------------------------------------------------------------------- #
EMB = {
    "a": [1.0, 0.0, 0.0],
    "a_dup": [0.99, 0.01, 0.0],
    "b": [0.0, 1.0, 0.0],
    "c": [0.0, 0.0, 1.0],
}


def test_mmr_lambda_one_is_relevance_order() -> None:
    candidates = [("a", 0.9), ("a_dup", 0.85), ("b", 0.5), ("c", 0.1)]
    assert [i for i, _ in mmr(candidates, EMB, 1.0, k=4)] == ["a", "a_dup", "b", "c"]


def test_mmr_skips_near_duplicates() -> None:
    candidates = [("a", 0.9), ("a_dup", 0.85), ("b", 0.5)]
    selected = mmr(candidates, EMB, 0.7, k=2)
    assert [i for i, _ in selected] == ["a", "b"]
    assert selected[0][1] == pytest.approx(0.9)  # original relevance is returned


def test_mmr_uses_fingerprints_for_exact_duplicates_without_embeddings() -> None:
    candidates = [("x", 1.0), ("x_copy", 0.99), ("y", 0.6)]
    selected = mmr(candidates, {}, 0.7, k=2, fingerprints={"x": "h1", "x_copy": "h1", "y": "h2"})
    assert [i for i, _ in selected] == ["x", "y"]


def test_mmr_without_embeddings_treats_candidates_as_novel() -> None:
    candidates = [("x", 1.0), ("y", 0.8), ("z", 0.7)]
    assert [i for i, _ in mmr(candidates, {}, 0.5, k=3)] == ["x", "y", "z"]


def test_mmr_lambda_zero_maximises_diversity() -> None:
    candidates = [("a", 1.0), ("a_dup", 0.99), ("c", 0.01)]
    assert [i for i, _ in mmr(candidates, EMB, 0.0, k=2)] == ["a", "c"]


def test_mmr_edge_cases() -> None:
    assert mmr([], EMB, 0.5, k=3) == []
    assert mmr([("a", 1.0)], EMB, 0.5, k=0) == []
    assert [i for i, _ in mmr([("a", 0.5), ("a", 0.9), ("b", 0.4)], EMB, 1.0, k=5)] == ["a", "b"]
    assert [i for i, _ in mmr([("a", 0.0), ("b", 0.0)], EMB, 0.7, k=2)] == ["a", "b"]
    with pytest.raises(ValueError, match="lambda_"):
        mmr([("a", 1.0)], EMB, 1.5, k=1)


@given(
    st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=15),
    st.floats(min_value=0, max_value=1),
    st.integers(min_value=1, max_value=20),
)
def test_mmr_returns_distinct_subset(relevances: list[float], lambda_: float, k: int) -> None:
    candidates = [(f"c{i}", r) for i, r in enumerate(relevances)]
    embeddings = {f"c{i}": [math.cos(i), math.sin(i)] for i in range(len(relevances))}
    selected = mmr(candidates, embeddings, lambda_, k=k)
    ids = [i for i, _ in selected]
    assert len(ids) == len(set(ids)) == min(k, len(candidates))
    assert set(ids) <= {c for c, _ in candidates}
