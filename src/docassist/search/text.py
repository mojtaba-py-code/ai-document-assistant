"""Text handling for search: query hygiene, query terms and result snippets.

* :func:`normalize_query` turns raw user input into a safe query string: invisible and
  control characters are removed (Unicode tag characters, bidi overrides, zero-width
  characters, NUL), text is NFKC-normalised, lone surrogates (which PostgreSQL cannot store)
  are dropped and whitespace is collapsed. The decoded content of Unicode *tag* smuggling is
  discarded - hidden text never becomes part of a query.
* :func:`query_terms` / :func:`stem` give the deterministic term view used by the lexical
  reranker, the prefix fallback and snippet selection.
* :func:`make_snippet` picks the window of a chunk that covers the most distinct query terms.
  Output is plain text (no markup is added); clients must render it as text.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from docassist.core.errors import ValidationFailed
from docassist.core.text import sanitize_text

_WORD = re.compile(r"[^\W_]+")
MAX_TERM_CHARS = 64
MAX_PREFIX_TERMS = 8
SNIPPET_CHARS = 400
_ELLIPSIS = "..."

STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    }
)

_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ied", "es", "ed", "ly", "s")


def _drop_surrogates(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cs")


def clean_query(raw: str) -> str:
    """Sanitised, whitespace-collapsed query text (possibly empty)."""
    cleaned, _report = sanitize_text(_drop_surrogates(raw))
    return " ".join(cleaned.split())


def normalize_query(raw: str, *, max_chars: int) -> str:
    """Validate a user search query; raises :class:`ValidationFailed` (never echoes the input)."""
    if not isinstance(raw, str):
        raise ValidationFailed("The search query must be text.")
    if len(raw) > max_chars * 4:
        raise ValidationFailed(f"The search query must be at most {max_chars} characters.")
    query = clean_query(raw)
    if not query:
        raise ValidationFailed("The search query is empty.")
    if len(query) > max_chars:
        raise ValidationFailed(f"The search query must be at most {max_chars} characters.")
    return query


def truncate_query(raw: str, *, max_chars: int) -> str:
    """Clean a (possibly long) question and cut it to ``max_chars`` on a word boundary."""
    query = clean_query(raw)
    if len(query) <= max_chars:
        return query
    cut = query[:max_chars]
    space = cut.rfind(" ")
    return cut[:space] if space > max_chars // 2 else cut


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens (Unicode letters/digits only, capped length)."""
    return [w.lower()[:MAX_TERM_CHARS] for w in _WORD.findall(text)]


def stem(word: str) -> str:
    """Very small, deterministic suffix stripper (``payments`` -> ``payment``).

    Applied identically to query and document tokens, so it only needs to be consistent,
    not linguistically perfect. Words of four characters or fewer are left alone.
    """
    if len(word) <= 4:
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def query_terms(text: str, *, max_terms: int = 32) -> list[str]:
    """Distinct content terms of a query, in order (stop words dropped unless nothing is left)."""
    words = [w for w in tokenize(text) if len(w) >= 2 or w.isdigit()]
    content = [w for w in words if w not in STOP_WORDS] or words
    return list(dict.fromkeys(content))[:max_terms]


def prefix_tsquery(text: str) -> str | None:
    """A ``to_tsquery('simple', ...)`` expression matching every term as a prefix.

    Terms contain only Unicode letters and digits (no tsquery operators or quotes can appear),
    and the expression is still passed as a bound parameter. ``None`` when no usable term.
    """
    terms = [t for t in query_terms(text) if len(t) >= 2][:MAX_PREFIX_TERMS]
    if not terms:
        return None
    return " & ".join(f"'{term}':*" for term in terms)


def like_pattern(text: str, *, max_chars: int = 200) -> str | None:
    """``%text%`` for an ``ILIKE ... ESCAPE '\\'`` substring match, wildcards escaped."""
    needle = " ".join(text.split())[:max_chars]
    if len(needle) < 2:
        return None
    escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


# --------------------------------------------------------------------------- #
# Snippets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _Hit:
    start: int
    end: int
    term: int


def _term_hits(text: str, terms: list[str]) -> list[_Hit]:
    stems = [stem(t) for t in terms]
    hits: list[_Hit] = []
    for match in _WORD.finditer(text):
        token = match.group(0).lower()
        token_stem = stem(token)
        for index, (term, term_stem) in enumerate(zip(terms, stems, strict=True)):
            if token == term or token_stem == term_stem or token.startswith(term):
                hits.append(_Hit(match.start(), match.end(), index))
                break
    return hits


def _snap(text: str, start: int, end: int) -> tuple[int, int]:
    """Move cut points to nearby word boundaries so no word is split at a cut side."""
    new_start, new_end = start, end
    if new_start > 0 and not text[new_start - 1].isspace():
        space = text.find(" ", new_start, min(len(text), new_start + 30))
        if space != -1:
            new_start = space + 1
    if new_end < len(text) and not text[new_end].isspace():
        space = text.rfind(" ", max(new_start, new_end - 30), new_end)
        if space != -1:
            new_end = space
    if new_end - new_start < (end - start) // 2:
        return start, end  # a giant unbroken token: cutting inside it beats an empty snippet
    return new_start, new_end


def make_snippet(content: str, query: str, *, max_chars: int = SNIPPET_CHARS) -> str:
    """Plain-text excerpt of at most ``max_chars`` characters around the query terms.

    The window covering the most *distinct* query terms (then the most occurrences, then the
    earliest) is chosen; cut sides are marked with ``...``. Without any term match the start
    of the chunk is returned.
    """
    text = " ".join(content.split())
    if len(text) <= max_chars:
        return text
    budget = max_chars - 2 * len(_ELLIPSIS)
    hits = _term_hits(text, query_terms(query))
    if not hits:
        _, end = _snap(text, 0, budget)
        return text[:end].rstrip() + _ELLIPSIS

    # Two pointers over the hits: for every left hit, extend right while the span still fits.
    best: tuple[int, int, int] = (0, 0, 0)  # distinct terms, occurrences, -left (earliest wins)
    best_range = (0, 0)
    right = 0
    for left, first in enumerate(hits):
        right = max(right, left)
        while right + 1 < len(hits) and hits[right + 1].end - first.start <= budget:
            right += 1
        window = hits[left : right + 1]
        key = (len({h.term for h in window}), len(window), -left)
        if key > best:
            best, best_range = key, (left, right)
    first_hit, last_hit = hits[best_range[0]], hits[best_range[1]]
    span = last_hit.end - first_hit.start
    slack = max(0, budget - span)
    start = max(0, first_hit.start - slack // 2)
    end = min(len(text), start + budget)
    start = max(0, end - budget)
    start, end = _snap(text, start, end)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = _ELLIPSIS + snippet
    if end < len(text):
        snippet += _ELLIPSIS
    return snippet[:max_chars]
