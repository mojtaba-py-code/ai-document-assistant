"""Rank fusion, score normalisation and diversity (MMR) - pure functions, no I/O.

Reciprocal Rank Fusion (Cormack et al., 2009) combines ranked lists without having to
calibrate their raw scores against each other: every list contributes ``w / (k + rank)``.
Scores are normalised by the best achievable fused score, so an item ranked first by every
contributing retriever scores exactly ``1.0``.

Maximal Marginal Relevance (Carbonell & Goldstein, 1998) then picks a diverse subset:
``lambda * relevance - (1 - lambda) * max_similarity_to_already_selected``.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Hashable, Mapping, Sequence


def reciprocal_rank_fusion[K: Hashable](
    rankings: Sequence[Sequence[K]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[K, float]]:
    """Fuse ranked lists (best first) into ``[(item, score 0..1)]``, best first.

    Duplicates inside one list count once, at their best position. Empty lists do not
    dilute the normalisation. Ties are broken by the best single-list rank, then by
    ``str(item)`` - the output is fully deterministic.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    if any(w < 0 or not math.isfinite(w) for w in weights):
        raise ValueError("weights must be finite and non-negative")

    scores: dict[K, float] = {}
    best_rank: dict[K, int] = {}
    achievable = 0.0
    for ranking, weight in zip(rankings, weights, strict=True):
        if not ranking or weight == 0:
            continue
        achievable += weight / (k + 1)
        seen: set[K] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
    if not scores:
        return []
    ordered = sorted(scores, key=lambda item: (-scores[item], best_rank[item], str(item)))
    return [(item, min(1.0, scores[item] / achievable)) for item in ordered]


def min_max_normalize[K: Hashable](values: Mapping[K, float]) -> dict[K, float]:
    """Scale to ``0..1``; a constant (or single) input maps to ``1.0`` for positive values."""
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if math.isclose(high, low):
        return dict.fromkeys(values, 1.0 if high > 0 else 0.0)
    span = high - low
    return {key: (value - low) / span for key, value in values.items()}


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(math.fsum(v * v for v in vector))
    if norm == 0 or not math.isfinite(norm):
        return [0.0] * len(vector)
    return [v / norm for v in vector]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(map(operator.mul, a, b))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in ``-1..1`` (``0.0`` for zero vectors or a length mismatch)."""
    if len(a) != len(b) or not a:
        return 0.0
    return dot(unit(a), unit(b))


def mmr[K: Hashable](
    candidates: Sequence[tuple[K, float]],
    embeddings: Mapping[K, Sequence[float]],
    lambda_: float,
    *,
    k: int,
    fingerprints: Mapping[K, str] | None = None,
) -> list[tuple[K, float]]:
    """Select up to ``k`` candidates balancing relevance and novelty.

    ``candidates`` are ``(item, relevance)``; relevance is rescaled by its maximum so it is
    comparable with cosine similarity. Two items are considered identical (similarity 1)
    when they share a ``fingerprint`` (e.g. the content hash); otherwise similarity is the
    cosine of their embeddings, and ``0`` when either has no embedding (keyword-only chunks).
    ``lambda_ = 1`` reproduces the relevance order; ``lambda_ = 0`` maximises diversity.
    Returns the selected ``(item, relevance)`` pairs in selection order.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be within 0..1")
    if k <= 0 or not candidates:
        return []
    top = max(rel for _, rel in candidates)
    scale = top if top > 0 else 1.0
    relevance: dict[K, float] = {}
    order: dict[K, int] = {}
    for index, (item, rel) in enumerate(candidates):  # first occurrence of a duplicate wins
        relevance.setdefault(item, rel / scale)
        order.setdefault(item, index)
    vectors = {item: unit(embeddings[item]) for item, _ in candidates if item in embeddings}
    prints = fingerprints or {}

    def similarity(a: K, b: K) -> float:
        fa, fb = prints.get(a), prints.get(b)
        if fa is not None and fa == fb:
            return 1.0
        va, vb = vectors.get(a), vectors.get(b)
        if va is None or vb is None or len(va) != len(vb):
            return 0.0
        return dot(va, vb)

    remaining = list(relevance)
    max_sim = dict.fromkeys(remaining, 0.0)
    selected: list[tuple[K, float]] = []
    while remaining and len(selected) < k:
        if selected:
            best = max(
                remaining,
                key=lambda item: (
                    lambda_ * relevance[item] - (1 - lambda_) * max_sim[item],
                    -order[item],
                ),
            )
        else:
            best = max(remaining, key=lambda item: (relevance[item], -order[item]))
        remaining.remove(best)
        selected.append((best, relevance[best] * scale))
        for item in remaining:
            max_sim[item] = max(max_sim[item], similarity(item, best))
    return selected
