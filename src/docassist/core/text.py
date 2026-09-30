"""Unicode hygiene for untrusted text (document content, OCR output, user queries, metadata).

Invisible characters are a real attack channel against LLM systems: Unicode *tag* characters
(U+E0000-U+E007F) can smuggle ASCII instructions that humans cannot see but models read,
bidi overrides can make a rendered string differ from its logical order, and zero-width
characters split keywords to dodge filters. Everything here is written with ``chr()`` so no
invisible character ever appears literally in the source code.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0xFEFF, 0x180E}
_BIDI = {0x200E, 0x200F, 0x061C, *range(0x202A, 0x202F), *range(0x2066, 0x206A)}
_TAGS = range(0xE0000, 0xE0080)
_OTHER_INVISIBLE = {0x00AD, 0x034F, 0x115F, 0x1160, 0x3164, 0xFFA0, 0xFFF9, 0xFFFA, 0xFFFB}
_VARIATION_SUPPLEMENT = range(0xE0100, 0xE01F0)

_KEEP_CONTROLS = {0x09, 0x0A}
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")


@dataclass(slots=True)
class SanitizeReport:
    zero_width: int = 0
    bidi: int = 0
    tags: int = 0
    controls: int = 0
    other_invisible: int = 0
    decoded_tag_text: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def suspicious(self) -> bool:
        return self.tags > 0 or self.bidi > 0 or self.zero_width > 20

    def as_flags(self) -> list[str]:
        flags = []
        if self.tags:
            flags.append("unicode_tag_smuggling")
        if self.bidi:
            flags.append("bidi_control_characters")
        if self.zero_width > 20:
            flags.append("excessive_zero_width_characters")
        return flags


def sanitize_text(text: str, *, normalize: bool = True) -> tuple[str, SanitizeReport]:
    """Remove invisible / control characters and NFKC-normalise.

    The Unicode tag payload is *decoded* into ``report.decoded_tag_text`` so the injection
    scanner can still inspect what an attacker tried to hide.
    """
    report = SanitizeReport()
    out: list[str] = []
    hidden: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp in _TAGS:
            report.tags += 1
            if 0xE0020 <= cp <= 0xE007E:
                hidden.append(chr(cp - 0xE0000))
            continue
        if cp in _ZERO_WIDTH:
            report.zero_width += 1
            continue
        if cp in _BIDI:
            report.bidi += 1
            continue
        if cp in _OTHER_INVISIBLE or cp in _VARIATION_SUPPLEMENT:
            report.other_invisible += 1
            continue
        if cp == 0x0D:
            continue  # CR / CRLF -> LF handled by dropping CR
        category = unicodedata.category(ch)
        if category == "Cc" and cp not in _KEEP_CONTROLS:
            report.controls += 1
            out.append(" ")
            continue
        if category in {"Zl", "Zp"}:
            out.append("\n")
            continue
        out.append(ch)
    cleaned = "".join(out)
    if normalize:
        cleaned = unicodedata.normalize("NFKC", cleaned)
    report.decoded_tag_text = "".join(hidden)
    return cleaned, report


def clean_line_text(text: str, max_length: int) -> str:
    """Single-line field (titles, filenames, metadata): no newlines, no invisibles, capped."""
    cleaned, _ = sanitize_text(text)
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_length]


def tidy_whitespace(text: str) -> str:
    text = _TRAILING_SPACE.sub("\n", text)
    return _MULTI_BLANK_LINES.sub("\n\n", text).strip()


def estimate_tokens(text: str) -> int:
    """Cheap, provider-agnostic token estimate (~4 chars/token for Latin, denser for CJK)."""
    if not text:
        return 0
    non_ascii = sum(1 for ch in text if ord(ch) > 0x2FFF)
    return max(1, (len(text) - non_ascii) // 4 + non_ascii)


def truncate(text: str, limit: int, marker: str = "...") -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(marker))] + marker
