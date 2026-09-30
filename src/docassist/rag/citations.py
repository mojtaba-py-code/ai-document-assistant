"""Citation validation: a citation survives only if its quote is really in its source.

Matching is delegated to :func:`docassist.core.quotes.find_supported_span`: an exact
contiguous match, or an in-order near-exact match whose only differences are spelling
variants, dropped source words or function words - swapped names, invented negations and
changed numbers are rejected. The citation that is kept carries the **source's own text** for
the matched span, never the model's wording.

Citations to unknown source ids are invalid. Duplicates are removed and at most
``max_citations`` are kept. Every outcome is counted in the ``docassist_citations_total``
metric.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from docassist.core.quotes import (
    FUZZY_MIN_TOKENS,
    FUZZY_THRESHOLD,
    MIN_QUOTE_CHARS,
    find_supported_span,
    normalize_for_match,
)
from docassist.observability import metrics

__all__ = [
    "FUZZY_MIN_TOKENS",
    "FUZZY_THRESHOLD",
    "MIN_QUOTE_CHARS",
    "CitationReport",
    "ValidCitation",
    "normalize_for_match",
    "quote_matches",
    "validate_citations",
]


def quote_matches(quote: str, source_text: str) -> bool:
    return find_supported_span(quote, source_text) is not None


@dataclass(frozen=True, slots=True)
class ValidCitation[T]:
    key: str
    quote: str
    target: T


@dataclass(frozen=True, slots=True)
class CitationReport[T]:
    valid: list[ValidCitation[T]]
    invalid: int
    total: int

    @property
    def validity_ratio(self) -> float:
        return len(self.valid) / self.total if self.total else 0.0


def validate_citations[T](
    raw: Any,
    targets: Mapping[str, T],
    *,
    key_field: str,
    text_of: Callable[[T], str],
    max_citations: int = 10,
) -> CitationReport[T]:
    """Keep the citations in ``raw`` whose ``key_field`` names a target and whose quote matches.

    ``text_of(target)`` returns the text a quote must be found in.
    """
    items = raw if isinstance(raw, list) else []
    valid: list[ValidCitation[T]] = []
    seen: set[tuple[str, str]] = set()
    invalid = 0
    for item in items[: max_citations * 2]:
        key = item.get(key_field) if isinstance(item, dict) else None
        quote = item.get("quote") if isinstance(item, dict) else None
        target = targets.get(key) if isinstance(key, str) else None
        match = (
            find_supported_span(quote, text_of(target))
            if target is not None and isinstance(quote, str) and isinstance(key, str)
            else None
        )
        if match is None or target is None or not isinstance(key, str):
            invalid += 1
            metrics.CITATIONS.labels(outcome="invalid").inc()
            continue
        dedupe = (key, normalize_for_match(match.span))
        if dedupe in seen or len(valid) >= max_citations:
            metrics.CITATIONS.labels(outcome="duplicate").inc()
            continue
        seen.add(dedupe)
        valid.append(ValidCitation(key=key, quote=match.span.strip(), target=target))
        metrics.CITATIONS.labels(outcome="valid").inc()
    return CitationReport(valid=valid, invalid=invalid, total=len(valid) + invalid)
