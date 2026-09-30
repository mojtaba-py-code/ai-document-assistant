"""Tiny, deterministic language guess for version metadata (and search analyser hints).

Scripts decide first (Arabic/Persian, Cyrillic, Greek, Hebrew, CJK, Hangul, Thai,
Devanagari); Latin text is attributed by stop-word frequency over :data:`LANGUAGES`.
Returns an ISO 639-1 code, or ``None`` when the text is too short or ambiguous.
"""

from __future__ import annotations

import re
from collections import Counter

SAMPLE_CHARS = 20_000
MIN_LATIN_WORDS = 20
_WORD = re.compile(r"[^\W\d_]+")

LANGUAGES: dict[str, frozenset[str]] = {
    "en": frozenset(
        [
            "the",
            "and",
            "of",
            "to",
            "in",
            "is",
            "that",
            "for",
            "with",
            "as",
            "on",
            "by",
            "this",
            "be",
            "are",
            "or",
            "shall",
            "from",
            "which",
        ]
    ),
    "de": frozenset(
        [
            "der",
            "die",
            "das",
            "und",
            "ist",
            "nicht",
            "mit",
            "den",
            "von",
            "zu",
            "ein",
            "eine",
            "auf",
            "für",
            "sich",
            "des",
            "dem",
        ]
    ),
    "fr": frozenset(
        [
            "le",
            "la",
            "les",
            "et",
            "est",
            "des",
            "une",
            "pour",
            "que",
            "dans",
            "du",
            "au",
            "pas",
            "sur",
            "par",
            "avec",
            "qui",
        ]
    ),
    "es": frozenset(
        [
            "el",
            "la",
            "los",
            "las",
            "y",
            "es",
            "que",
            "de",
            "en",
            "por",
            "con",
            "para",
            "una",
            "del",
            "se",
            "al",
            "como",
        ]
    ),
    "it": frozenset(
        [
            "il",
            "lo",
            "la",
            "gli",
            "le",
            "e",
            "che",
            "di",
            "per",
            "con",
            "una",
            "sono",
            "del",
            "della",
            "non",
            "nel",
        ]
    ),
    "pt": frozenset(
        [
            "o",
            "os",
            "as",
            "e",
            "que",
            "de",
            "em",
            "para",
            "com",
            "uma",
            "do",
            "da",
            "dos",
            "não",
            "por",
            "se",
        ]
    ),
    "nl": frozenset(
        [
            "de",
            "het",
            "een",
            "en",
            "van",
            "is",
            "dat",
            "op",
            "te",
            "met",
            "voor",
            "niet",
            "zijn",
            "aan",
        ]
    ),
}

_SCRIPTS: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("ar", ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("ru", ((0x0400, 0x04FF),)),
    ("el", ((0x0370, 0x03FF),)),
    ("he", ((0x0590, 0x05FF),)),
    ("zh", ((0x4E00, 0x9FFF), (0x3400, 0x4DBF))),
    ("ja", ((0x3040, 0x30FF),)),
    ("ko", ((0xAC00, 0xD7AF), (0x1100, 0x11FF))),
    ("th", ((0x0E00, 0x0E7F),)),
    ("hi", ((0x0900, 0x097F),)),
)
_PERSIAN_LETTERS = frozenset(chr(cp) for cp in (0x067E, 0x0686, 0x0698, 0x06AF, 0x06CC, 0x06A9))


def _script_of(ch: str) -> str | None:
    cp = ord(ch)
    for code, ranges in _SCRIPTS:
        if any(low <= cp <= high for low, high in ranges):
            return code
    return None


def guess_language(text: str) -> str | None:
    sample = text[:SAMPLE_CHARS]
    letters = [ch for ch in sample if ch.isalpha()]
    if len(letters) < 40:
        return None
    scripts = Counter(code for ch in letters if (code := _script_of(ch)) is not None)
    if scripts:
        code, count = scripts.most_common(1)[0]
        if count / len(letters) >= 0.3:
            if code == "ja" or (code == "zh" and scripts.get("ja", 0) > 0.1 * count):
                return "ja"
            if code == "ar" and sum(1 for ch in sample if ch in _PERSIAN_LETTERS) > 0.02 * count:
                return "fa"
            return code
    words = [w.casefold() for w in _WORD.findall(sample)]
    if len(words) < MIN_LATIN_WORDS:
        return None
    hits = {lang: sum(1 for w in words if w in stop) for lang, stop in LANGUAGES.items()}
    ranked = sorted(hits.items(), key=lambda item: (-item[1], item[0]))
    (best, best_hits), (_, second_hits) = ranked[0], ranked[1]
    if best_hits < max(3, 0.05 * len(words)) or best_hits < 1.3 * second_hits:
        return None
    return best
