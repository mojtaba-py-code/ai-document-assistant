"""Deciding whether a quoted passage is really supported by a source text.

Used for citations of AI answers and for the evidence of extracted fields. A quote is
supported when, after normalisation (NFKC, case folding, punctuation ignored):

* **exact** - its tokens occur contiguously in the source, or
* **near-exact** - some source window aligns with it *in order* (``difflib`` opcodes) and
  every difference is harmless:

  - a quote token replaced by a source token only when the two are spelling variants of each
    other (character similarity >= 0.8: a fixed typo, a plural) - ``Acme`` <-> ``Beta`` is
    not a variant, so swapped parties are rejected;
  - quote tokens absent from the source only when they are function words (``the``, ``of``)
    - an invented ``not`` or ``never`` is rejected;
  - source tokens omitted by the quote are tolerated (the model shortened the sentence);
  - **negations and numbers must be identical** in the quote and the window;
  - the overall alignment ratio is at least :data:`FUZZY_THRESHOLD`.

The function returns the *source's own text* for the matched span, so callers display what
the document says, never the model's paraphrase of it.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

FUZZY_THRESHOLD = 0.85
FUZZY_MIN_TOKENS = 5
MIN_QUOTE_CHARS = 4
_TYPO_SIMILARITY = 0.8
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_NEGATIONS = frozenset(
    {"not", "no", "never", "none", "neither", "nor", "without", "cannot", "nothing", "nobody",
     "nowhere", "t", "except", "unless"}
)  # fmt: skip
_FUNCTION_WORDS = frozenset(
    {"a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "from", "with", "and", "or",
     "as", "is", "are", "be", "been", "was", "were", "this", "that", "these", "those", "it",
     "its", "which", "who", "shall", "will"}
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class QuoteMatch:
    span: str  # the source's own text for the matched region
    score: float  # 1.0 exact, otherwise the alignment ratio
    exact: bool


def _prepare(text: str) -> str:
    return unicodedata.normalize("NFKC", html.unescape(text))


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).casefold(), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def normalize_for_match(text: str) -> str:
    """Case-, punctuation- and whitespace-insensitive form (dedupe keys, simple checks)."""
    return " ".join(token for token, _, _ in _tokenize(_prepare(text)))


def _critical(tokens: list[str]) -> Counter[str]:
    return Counter(t for t in tokens if t in _NEGATIONS or any(ch.isdigit() for ch in t))


def _is_word(char: str) -> bool:
    return char.isalnum()


def _variant(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= _TYPO_SIMILARITY


def _harmless(quote: list[str], window: list[str]) -> float | None:
    """Alignment ratio when every difference is harmless, else ``None``."""
    if _critical(quote) != _critical(window):
        return None
    matcher = SequenceMatcher(None, quote, window, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"equal", "insert"}:
            continue  # "insert": source words the quote left out
        if tag == "delete":
            if any(token not in _FUNCTION_WORDS for token in quote[i1:i2]):
                return None
            continue
        # replace: only spelling variants, pairwise and of equal count
        if (i2 - i1) != (j2 - j1) or not all(
            _variant(a, b) for a, b in zip(quote[i1:i2], window[j1:j2], strict=True)
        ):
            return None
    ratio = matcher.ratio()
    return ratio if ratio >= FUZZY_THRESHOLD else None


def find_supported_span(
    quote: str, source: str, *, min_fuzzy_tokens: int = FUZZY_MIN_TOKENS
) -> QuoteMatch | None:
    """The span of ``source`` that supports ``quote``, or ``None`` when it is not supported."""
    source_text = _prepare(source)
    q = [token for token, _, _ in _tokenize(_prepare(quote))]
    if sum(len(t) for t in q) < MIN_QUOTE_CHARS:
        return None
    src = _tokenize(source_text)
    words = [token for token, _, _ in src]
    size = len(q)
    if size == 0 or not words:
        return None

    def span(start: int, stop: int) -> str:
        """Source text of tokens ``start..stop`` plus adjacent punctuation (``(``, ``.``...)."""
        begin, end = src[start][1], src[stop - 1][2]
        while (
            begin > 0
            and not source_text[begin - 1].isspace()
            and not _is_word(source_text[begin - 1])
        ):
            begin -= 1
        while (
            end < len(source_text)
            and not source_text[end].isspace()
            and not _is_word(source_text[end])
        ):
            end += 1
        return source_text[begin:end]

    for i in range(len(words) - size + 1):  # exact, contiguous
        if words[i : i + size] == q:
            return QuoteMatch(span(i, i + size), 1.0, True)
    if size < min_fuzzy_tokens:
        return None

    # Near-exact: cheap multiset prefilter, then ordered alignment around promising windows.
    wanted = Counter(q)
    needed = FUZZY_THRESHOLD * size * 0.8
    best: QuoteMatch | None = None
    width = min(size, len(words))
    window: Counter[str] = Counter(words[:width])
    starts: list[int] = []
    for i in range(len(words) - width + 1):
        if i:
            window[words[i - 1]] -= 1
            window[words[i + width - 1]] += 1
        if sum(min(c, window[t]) for t, c in wanted.items()) >= needed:
            starts.append(i)
    slack = max(2, size // 5)
    seen: set[tuple[int, int]] = set()
    for i in starts:
        for start in range(max(0, i - slack), min(len(words), i + slack) + 1):
            for length in range(max(1, size - slack), size + slack + 1):
                stop = start + length
                if stop > len(words) or (start, stop) in seen:
                    continue
                seen.add((start, stop))
                ratio = _harmless(q, words[start:stop])
                if ratio is not None and (best is None or ratio > best.score):
                    best = QuoteMatch(span(start, stop), round(ratio, 4), False)
    return best


def quote_supported(quote: str, source: str) -> bool:
    return find_supported_span(quote, source) is not None
