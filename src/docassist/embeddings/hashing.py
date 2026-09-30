"""Deterministic, offline feature-hashing embedder (development, tests, air-gapped demos).

It maps word unigrams, word bigrams and character trigrams into a signed hashed vector
space and L2-normalises it. It captures lexical overlap and light morphology (trigrams), not
deep semantics - good enough to exercise the whole hybrid-retrieval path reproducibly, and
it never sends text anywhere. Production uses a real embedding model.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from itertools import pairwise

from docassist.embeddings.base import EmbeddingKind, l2_normalize

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "at",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "as",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "into",
        "than",
        "then",
        "there",
        "their",
        "them",
        "they",
        "we",
        "you",
        "he",
        "she",
        "his",
        "her",
        "our",
        "your",
        "not",
        "no",
    ]
)


class HashingEmbedder:
    name = "hashing"
    is_external = False

    def __init__(self, dimensions: int = 1024, model: str = "hashing-v1") -> None:
        self.dimensions = dimensions
        self.model = model

    def _index(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        sign = 1.0 if value & 1 else -1.0
        return (value >> 1) % self.dimensions, sign

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        words = [w.lower() for w in _WORD.findall(text)]
        content = [w for w in words if w not in _STOP] or words
        for word in content:
            idx, sign = self._index("w:" + word)
            vector[idx] += 2.0 * sign
            padded = f"#{word}#"
            for i in range(len(padded) - 2):
                idx, sign = self._index("c:" + padded[i : i + 3])
                vector[idx] += 0.5 * sign
        for first, second in pairwise(content):
            idx, sign = self._index(f"b:{first} {second}")
            vector[idx] += 1.0 * sign
        return l2_normalize(vector)

    async def embed(self, texts: Sequence[str], *, kind: EmbeddingKind) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    async def aclose(self) -> None:
        return None
