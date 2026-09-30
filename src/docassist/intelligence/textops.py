"""Deterministic text helpers shared by the intelligence features.

Everything here is pure (no I/O) so it is unit-tested in isolation:

* sentence splitting for extractive summaries;
* quote verification (``quote_score``) - normalised exact match or a sliding-window token
  overlap, used to verify every citation/evidence an LLM returns against the real chunk text;
* date and number *mentions* in evidence text, used to check that an extracted value is
  actually supported by its quote;
* escaping for the three places untrusted text is embedded: LLM prompts (spotlighting),
  Markdown reports and prompt attribute values;
* time-zone parsing for deadline windows (fixed offsets always work; IANA names when the
  platform has a tz database).
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, timedelta, timezone, tzinfo
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from docassist.core.errors import ValidationFailed
from docassist.core.quotes import find_supported_span
from docassist.core.text import clean_line_text

# --------------------------------------------------------------------------- #
# Sentences
# --------------------------------------------------------------------------- #
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_LINE_BREAK = re.compile(r"\n+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences; each line is handled separately (lines never merge)."""
    sentences: list[str] = []
    for line in _LINE_BREAK.split(text):
        compact = " ".join(line.split())
        if not compact:
            continue
        sentences.extend(part.strip() for part in _SENTENCE_BREAK.split(compact) if part.strip())
    return sentences


def word_count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- #
# Quote verification
# --------------------------------------------------------------------------- #
_PUNCT_MAP = {
    **dict.fromkeys((0x2018, 0x2019, 0x201A, 0x201B, 0x2032, 0x00B4, 0x0060), "'"),
    **dict.fromkeys((0x201C, 0x201D, 0x201E, 0x201F, 0x2033, 0x00AB, 0x00BB), '"'),
    **dict.fromkeys((0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212), "-"),
}
_NON_WORD = re.compile(r"[^\w\s]")

FUZZY_THRESHOLD = 0.85
_MIN_FUZZY_TOKENS = 3


def normalize_for_match(text: str) -> str:
    """Case-, whitespace-, punctuation- and entity-insensitive form used for matching."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text).casefold().translate(_PUNCT_MAP)
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())


def match_tokens(text: str) -> list[str]:
    return normalize_for_match(text).split()


def quote_score(quote: str, content: str) -> float:
    """How well ``quote`` is supported by ``content`` (1.0 exact .. 0.0 unsupported).

    Delegates to :func:`docassist.core.quotes.find_supported_span` (in-order alignment;
    swapped names, invented negations and changed numbers are unsupported).
    """
    match = find_supported_span(quote, content, min_fuzzy_tokens=_MIN_FUZZY_TOKENS)
    return 0.0 if match is None else match.score


def token_coverage(value: str, evidence: str) -> float:
    """Share of the value's tokens that appear in the evidence (0..1)."""
    tokens = match_tokens(value)
    if not tokens:
        return 0.0
    available = Counter(match_tokens(evidence))
    hits = 0
    for token in tokens:
        if available[token] > 0:
            available[token] -= 1
            hits += 1
    return hits / len(tokens)


# --------------------------------------------------------------------------- #
# Dates and numbers mentioned in text
# --------------------------------------------------------------------------- #
_MONTHS = {
    name: index
    for index, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DAY_MONTH_YEAR = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.IGNORECASE
)
_MONTH_DAY_YEAR = re.compile(
    rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE
)
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b")
_MIN_YEAR, _MAX_YEAR = 1900, 2200


@dataclass(frozen=True, slots=True)
class DateMention:
    value: date
    ambiguous: bool = False


def _safe_date(year: int, month: int, day: int) -> date | None:
    if not _MIN_YEAR <= year <= _MAX_YEAR:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def date_mentions(text: str) -> list[DateMention]:
    """Every calendar date written in ``text`` (ISO, long forms and numeric d/m or m/d).

    A numeric date whose day and month are both <= 12 is ambiguous: both readings are
    returned, flagged ``ambiguous=True``.
    """
    found: list[DateMention] = []
    for m in _ISO_DATE.finditer(text):
        value = _safe_date(int(m[1]), int(m[2]), int(m[3]))
        if value:
            found.append(DateMention(value))
    for m in _DAY_MONTH_YEAR.finditer(text):
        value = _safe_date(int(m[3]), _MONTHS[m[2].lower()], int(m[1]))
        if value:
            found.append(DateMention(value))
    for m in _MONTH_DAY_YEAR.finditer(text):
        value = _safe_date(int(m[3]), _MONTHS[m[1].lower()], int(m[2]))
        if value:
            found.append(DateMention(value))
    for m in _NUMERIC_DATE.finditer(text):
        first, second, year = int(m[1]), int(m[2]), int(m[3])
        if first > 12:
            candidates = [(first, second)]  # day/month
        elif second > 12:
            candidates = [(second, first)]  # month/day written as m/d
        else:
            candidates = [(first, second), (second, first)]
        ambiguous = len(candidates) > 1 and first != second
        for day, month in candidates:
            value = _safe_date(year, month, day)
            if value:
                found.append(DateMention(value, ambiguous=ambiguous))
    return found


def parse_iso_date(value: str) -> date | None:
    """Strict ``YYYY-MM-DD`` (the only date format the models are asked to produce)."""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value.strip())
    if not m:
        return None
    return _safe_date(int(m[1]), int(m[2]), int(m[3]))


_US_NUMBER = re.compile(
    r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.,])-?\d+(?:\.\d+)?(?![\w])"
)
_EU_NUMBER = re.compile(r"^-?\d{1,3}(?:\.\d{3})+(?:,\d+)?$|^-?\d+,\d{1,2}$")
MAX_ABS_NUMBER = Decimal("1e16")


def parse_decimal(text: str) -> Decimal | None:
    """Parse the first number in ``text`` (``"USD 12,500.00"`` -> ``Decimal("12500.00")``).

    European notation (``1.234,56``) is accepted when the whole value uses it. Values whose
    magnitude does not fit the ``NUMERIC(20, 4)`` column are rejected.
    """
    raw = text.strip()
    stripped = re.sub(r"^[^\d\-]+|[^\d]+$", "", raw)
    if _EU_NUMBER.match(stripped):
        candidate = stripped.replace(".", "").replace(",", ".")
    else:
        m = _US_NUMBER.search(raw)
        if not m:
            return None
        candidate = m.group(0).replace(",", "")
    try:
        value = Decimal(candidate)
    except InvalidOperation:
        return None
    if not value.is_finite() or abs(value) >= MAX_ABS_NUMBER:
        return None
    return value


def number_mentions(text: str) -> set[Decimal]:
    out: set[Decimal] = set()
    for m in _US_NUMBER.finditer(text):
        try:
            out.add(Decimal(m.group(0).replace(",", "")))
        except InvalidOperation:  # pragma: no cover - regex only matches digits
            continue
    return out


def format_decimal(value: Decimal) -> str:
    """Canonical text for a number: no exponent, no trailing zeros (``12500``, ``0.5``)."""
    normalized = value.normalize()
    text = format(normalized, "f")
    return text if text != "-0" else "0"


# --------------------------------------------------------------------------- #
# Escaping untrusted text
# --------------------------------------------------------------------------- #
def escape_prompt_text(text: str) -> str:
    """Content placed inside a spotlighting tag: ``&``, ``<`` and ``>`` are entity-escaped,
    so document text can never close (or open) a tag of the prompt structure."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def prompt_attr(text: str | None, max_length: int = 200) -> str:
    """A single-line attribute value for a prompt tag (quotes and angle brackets removed)."""
    if not text:
        return ""
    cleaned = clean_line_text(text, max_length)
    return cleaned.replace('"', "'").replace("<", "(").replace(">", ")")


_MD_SPECIAL = set("\\`*_{}[]()<>#+-.!|~:&")


def escape_markdown(text: str | None, max_length: int = 2_000) -> str:
    """Inline Markdown-safe text: one line, every Markdown/HTML-significant ASCII
    punctuation character backslash-escaped (no links, images, HTML, headings or tables
    can be formed from document or model text)."""
    if not text:
        return ""
    cleaned = clean_line_text(text, max_length)
    return "".join("\\" + ch if ch in _MD_SPECIAL else ch for ch in cleaned)


# --------------------------------------------------------------------------- #
# Time zones
# --------------------------------------------------------------------------- #
_OFFSET = re.compile(r"^(?:UTC|GMT)?([+-])(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE)
_IANA = re.compile(r"^[A-Za-z][A-Za-z0-9_+\-]*(?:/[A-Za-z0-9_+\-]+){0,2}$")
_UTC_NAMES = {"UTC", "Z", "GMT", "ETC/UTC", "ETC/GMT"}
_MAX_OFFSET = timedelta(hours=14)


def parse_timezone(name: str | None) -> tzinfo:
    """``"UTC"``, a fixed offset (``"+03:00"``, ``"-0530"``, ``"UTC+01:00"``) or an IANA
    zone name (``"Europe/Berlin"``; needs a tz database on the host).

    Raises :class:`ValidationFailed` for anything else.
    """
    if name is None or not name.strip():
        return UTC
    value = name.strip()
    if len(value) > 64:
        raise ValidationFailed("Unknown time zone.")
    if value.upper() in _UTC_NAMES:
        return UTC
    offset = _OFFSET.fullmatch(value)
    if offset:
        sign = -1 if offset[1] == "-" else 1
        hours, minutes = int(offset[2]), int(offset[3] or 0)
        delta = timedelta(hours=hours, minutes=minutes)
        if minutes >= 60 or delta > _MAX_OFFSET:
            raise ValidationFailed("Unknown time zone.")
        return timezone(sign * delta)
    if _IANA.fullmatch(value):
        try:
            return ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
            raise ValidationFailed("Unknown time zone.") from exc
    raise ValidationFailed("Unknown time zone.")
