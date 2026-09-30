"""Deterministic field extraction (``method = "rules"``) over a document's canonical text.

Fields
------
* dates with context - ``effective_date``, ``expiration_date``, ``termination_date``,
  ``renewal_date``, ``due_date``, ``invoice_date``: the label nearest *before* the date in
  the same sentence/line (looked for within :data:`LABEL_WINDOW` chars, and ending at most
  80 chars before the date) decides the field; unlabelled dates are ignored;
* ``payment_terms`` - "Net 30", "payable within 45 days", "due on receipt" (days in
  ``value_number``);
* amounts - ``amount_due``, ``total_amount``, ``subtotal``, ``tax_amount``,
  ``contract_value`` when labelled, else ``amount`` (at most :data:`MAX_UNLABELLED_AMOUNTS`);
  currency from ISO codes (upper case only, so the word "try" is not Turkish lira) or
  symbols (``$`` is reported as USD with lower confidence);
* ``party`` - the two names in "(by and) between X and Y" within the first
  :data:`PARTY_SCAN_CHARS` characters;
* ``invoice_number`` - "Invoice No./Number/#: ABC-123" (the value must contain a digit).

Dates: ISO (``2026-10-14``), "14 October 2026", "October 14, 2026" and numeric
``14/10/2026``. Numeric day/month order is only trusted when unambiguous (one part > 12, or
both equal); otherwise ``value_date`` stays empty, ``value_text`` keeps the original and the
confidence is low.

Every field carries ``evidence`` (at most :data:`MAX_EVIDENCE_CHARS` characters of verbatim
context) and its offsets in the canonical text, which the pipeline maps to a chunk and page.
All patterns use bounded quantifiers only (no catastrophic backtracking on hostile input),
overlap checks are logarithmic and at most :data:`MAX_MENTIONS` dates / amounts are examined.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

MAX_EVIDENCE_CHARS = 300
MAX_FIELDS = 200
MAX_MENTIONS = 20_000  # dates / amounts examined per document (hostile-input bound)
MAX_UNLABELLED_AMOUNTS = 25
LABEL_WINDOW = 120
PARTY_SCAN_CHARS = 20_000
MAX_AMOUNT = Decimal(10) ** 15

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip
_MONTH = r"(?P<month>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")"
_ORDINAL = r"(?:st|nd|rd|th)?"

_DATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("iso", re.compile(r"\b(?P<year>\d{4})-(?P<m>\d{2})-(?P<day>\d{2})\b")),
    (
        "long",
        re.compile(
            r"\b(?P<day>\d{1,2})"
            + _ORDINAL
            + r"\s{1,3}(?:day\s{1,3}of\s{1,3})?"
            + _MONTH
            + r"\.?,?\s{1,3}(?P<year>\d{4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "long",
        re.compile(
            r"\b"
            + _MONTH
            + r"\.?\s{1,3}(?P<day>\d{1,2})"
            + _ORDINAL
            + r",?\s{1,3}(?P<year>\d{4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "numeric",
        re.compile(r"\b(?P<a>\d{1,2})(?P<sep>[/.\-])(?P<b>\d{1,2})(?P=sep)(?P<year>\d{4})\b"),
    ),
)

_DATE_LABELS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(source))
    for name, source in (
        (
            "effective_date",
            (
                r"\beffective(?:\s{1,3}(?:date|as\s{1,3}of|on|from))?\b|\bcommencement\s{1,3}date\b"
                r"|\bcommenc(?:e|es|ing)\s{1,3}on\b|\bdated\s{1,3}as\s{1,3}of\b|\bstart\s{1,3}date\b"
                r"|\bentered\s{1,3}into\s{1,3}(?:as\s{1,3}of|on)\b"
            ),
        ),
        (
            "expiration_date",
            (
                r"\bexpir(?:es|ation|y|ing|e)\b|\bend\s{1,3}date\b|\bvalid\s{1,3}(?:until|through)\b"
                r"|\b(?:force|effect)\s{1,3}until\b"
            ),
        ),
        ("termination_date", r"\bterminat(?:ion\s{1,3}date|es\s{1,3}on|e\s{1,3}on|es)\b"),
        ("renewal_date", r"\brenew(?:al|s|ed)?\b"),
        (
            "due_date",
            (
                r"\bdue\s{1,3}(?:date|on|by)\b|\bpayment\s{1,3}due\b|\bpayable\s{1,3}(?:on|by)\b"
                r"|\bno\s{1,3}later\s{1,3}than\b|\bpay\s{1,3}by\b"
            ),
        ),
        (
            "invoice_date",
            (
                r"\binvoice\s{1,3}date\b|\bdate\s{1,3}of\s{1,3}invoice\b|\binvoice\s{1,3}issued\b"
                r"|\bissue\s{1,3}date\b|\bdate\s{1,3}issued\b|\bissued\s{1,3}on\b"
            ),
        ),
    )
)

_NET_TERMS = re.compile(
    r"\bnet\s{0,2}(?P<days>\d{1,3})\b(?!\s{0,2}(?:%|[.,]\d|million|billion|thousand|[km]\b))"
)
_WITHIN_TERMS = re.compile(
    r"\b(?:payable|payment|paid|due|settled)\b[^.\n]{0,40}?\bwithin\s{1,3}(?P<days>\d{1,3})\s{1,3}"
    r"(?:calendar\s{1,3}|business\s{1,3}|working\s{1,3})?days?\b"
)
_RECEIPT_TERMS = re.compile(r"\bdue\s{1,3}(?:up)?on\s{1,3}receipt\b")

_ISO_CODES = (
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "INR", "CNY", "SEK", "NOK", "DKK",
    "PLN", "TRY", "AED", "SAR", "ZAR", "BRL", "MXN", "SGD", "HKD", "CZK", "HUF", "ILS", "KRW",
)  # fmt: skip
_SYMBOLS = {
    "US$": "USD", "CA$": "CAD", "C$": "CAD", "AU$": "AUD", "A$": "AUD", "NZ$": "NZD",
    "HK$": "HKD", "S$": "SGD", "$": "USD", chr(0x20AC): "EUR", chr(0x00A3): "GBP",
    chr(0x00A5): "JPY", chr(0x20B9): "INR",
}  # fmt: skip
_NUMBER = r"(?P<num>\d{1,3}(?:[,.']\d{3}){1,5}(?:[.,]\d{1,2})?|\d{1,15}(?:[.,]\d{1,2})?)"
_SYMBOL_RE = "|".join(re.escape(s) for s in sorted(_SYMBOLS, key=len, reverse=True))
_CODE_RE = "|".join(_ISO_CODES)
_AMOUNT_PATTERNS = (
    re.compile(r"(?P<sym>" + _SYMBOL_RE + r")\s?" + _NUMBER + r"(?![\d%])"),
    re.compile(r"\b(?P<code>" + _CODE_RE + r")\s?" + _NUMBER + r"(?![\d%])"),
    re.compile(r"(?<![\d.,])" + _NUMBER + r"\s?(?P<code>" + _CODE_RE + r")\b"),
    re.compile(r"(?<![\d.,])" + _NUMBER + r"\s?(?P<sym>" + chr(0x20AC) + "|" + chr(0x00A3) + r")"),
)
_AMOUNT_LABELS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(source))
    for name, source in (
        (
            "amount_due",
            (
                r"\b(?:amount|balance|total)\s{1,3}(?:due|payable|outstanding)\b|\bplease\s{1,3}pay\b"
                r"|\bamount\s{1,3}owing\b"
            ),
        ),
        ("subtotal", r"\bsub-?\s?total\b"),
        ("tax_amount", r"\b(?:sales\s{1,3})?tax(?:es)?\b|\bvat\b|\bgst\b|\bhst\b"),
        (
            "contract_value",
            (
                r"\bcontract\s{1,3}(?:value|price|sum|amount)\b|\btotal\s{1,3}(?:fees?|consideration)\b"
                r"|\bpurchase\s{1,3}price\b|\bfees?\s{1,3}of\b"
            ),
        ),
        ("total_amount", r"\b(?:grand\s{1,3})?total(?:\s{1,3}amount)?\b|\binvoice\s{1,3}total\b"),
    )
)
_INVOICE_NUMBER = re.compile(
    r"\binvoice\s{0,2}(?:number|no\.?|num\.?|#|id)\s{0,2}[:#.]?\s{0,2}(?P<value>[A-Z0-9][A-Z0-9\-_/]{1,30})\b",
    re.IGNORECASE,
)
_BETWEEN = re.compile(r"\b(?:by\s{1,3}and\s{1,3})?between\s{1,3}", re.IGNORECASE)
_PARTY_STOP = re.compile(r"[,(;\n\"]|\s(?:and|&)\s|\.\s|\.$")
_AND = re.compile(r"\s(?:and|&)\s{1,3}")
_ABBREVIATION_END = re.compile(
    r"\b(?:ltd|inc|co|corp|llc|l\.l\.c|plc|s\.a|n\.v|b\.v|gmbh|ag)\.?$", re.I
)
_TIME_LIKE = re.compile(r"^\d{1,2}(?::\d{2})?\s?(?:am|pm)?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class FieldCandidate:
    field: str
    value_text: str | None
    value_date: date | None
    value_number: Decimal | None
    currency: str | None
    confidence: float
    evidence: str
    start: int
    end: int


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def evidence(text: str, start: int, end: int) -> str:
    """Verbatim context around ``[start, end)``: its line, trimmed to the evidence budget."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end < 0 else line_end
    left = max(line_start, start - 150)
    right = min(line_end, end + 150)
    snippet = text[left:right]
    if len(snippet) > MAX_EVIDENCE_CHARS:
        room = max(0, MAX_EVIDENCE_CHARS - (end - start)) // 2
        snippet = text[max(left, start - room) : min(right, end + room)]
    return " ".join(snippet.split())[:MAX_EVIDENCE_CHARS]


def _clause_window(text: str, start: int) -> tuple[str, int]:
    """Lower-cased text before ``start`` in the same sentence/line (bounded)."""
    low = max(0, start - LABEL_WINDOW)
    window = text[low:start]
    cut = max(window.rfind("\n"), window.rfind(". "), window.rfind("; "))
    offset = low + cut + 1 if cut >= 0 else low
    return text[offset:start].casefold(), offset


def _nearest_label(
    window: str, labels: tuple[tuple[str, re.Pattern[str]], ...]
) -> tuple[str, int] | None:
    best: tuple[str, int] | None = None
    for name, pattern in labels:
        last_end = -1
        for match in pattern.finditer(window):
            last_end = match.end()
        if last_end >= 0:
            distance = len(window) - last_end
            if best is None or distance < best[1]:
                best = (name, distance)
    return best


def parse_amount(raw: str) -> Decimal | None:
    """Parse ``1,234.56`` / ``1.234,56`` / ``1'234.5`` / ``1234,56`` into a Decimal."""
    value = raw.replace("'", "")
    if "," in value and "." in value:
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")
    elif "," in value:
        head, _, tail = value.rpartition(",")
        value = (
            head.replace(",", "") + "." + tail
            if value.count(",") == 1 and len(tail) <= 2
            else value.replace(",", "")
        )
    elif "." in value and (value.count(".") > 1 or len(value.rpartition(".")[2]) == 3):
        value = value.replace(".", "")
    try:
        amount = Decimal(value)
    except InvalidOperation:
        return None
    if not amount.is_finite() or abs(amount) >= MAX_AMOUNT:
        return None
    return amount


# --------------------------------------------------------------------------- #
# extractors
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _DateMention:
    start: int
    end: int
    raw: str
    value: date | None
    kind: str


@dataclass(slots=True)
class _Taken:
    """Disjoint ``[start, end)`` spans already claimed by a match; O(log n) overlap test."""

    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)

    def overlaps(self, start: int, end: int) -> bool:
        index = bisect.bisect_left(self.starts, end)  # spans starting before ``end``
        return index > 0 and self.ends[index - 1] > start

    def add(self, start: int, end: int) -> None:
        index = bisect.bisect_left(self.starts, start)
        self.starts.insert(index, start)
        self.ends.insert(index, end)


def _date_mentions(text: str) -> list[_DateMention]:
    mentions: list[_DateMention] = []
    taken = _Taken()
    for kind, pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            if len(mentions) >= MAX_MENTIONS:
                break
            start, end = match.span()
            if taken.overlaps(start, end):
                continue
            valid, value = _date_value(kind, match)
            if not valid:
                continue
            taken.add(start, end)
            mentions.append(_DateMention(start, end, match.group(0), value, kind))
    mentions.sort(key=lambda m: m.start)
    return mentions


def _date_value(kind: str, match: re.Match[str]) -> tuple[bool, date | None]:
    """``(True, date)``; ``(True, None)`` for an ambiguous numeric date; ``(False, None)`` if
    the text is not a real date."""
    groups = match.groupdict()
    year = int(groups["year"])
    try:
        if kind == "iso":
            return True, date(year, int(groups["m"]), int(groups["day"]))
        if kind == "long":
            return True, date(year, _MONTHS[groups["month"].casefold()], int(groups["day"]))
        a, b = int(groups["a"]), int(groups["b"])
        if a > 12 >= b:
            return True, date(year, b, a)  # day/month
        if b > 12 >= a:
            return True, date(year, a, b)  # month/day
        if a == b:
            return True, date(year, a, b)
        date(year, a, b)  # both readings must at least be real dates
        date(year, b, a)
        return True, None
    except ValueError:
        return False, None


def _dates(text: str) -> list[FieldCandidate]:
    out: list[FieldCandidate] = []
    seen: set[tuple[str, str]] = set()
    for mention in _date_mentions(text):
        window, _ = _clause_window(text, mention.start)
        label = _nearest_label(window, _DATE_LABELS)
        if label is None or label[1] > 80:
            continue
        field, distance = label
        if mention.value is None:
            confidence = 0.4
        elif mention.kind == "numeric":
            confidence = 0.75
        else:
            confidence = 0.9 if distance <= 30 else 0.8
        key = (field, mention.value.isoformat() if mention.value else mention.raw)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            FieldCandidate(
                field=field,
                value_text=mention.raw,
                value_date=mention.value,
                value_number=None,
                currency=None,
                confidence=confidence,
                evidence=evidence(text, mention.start, mention.end),
                start=mention.start,
                end=mention.end,
            )
        )
    return out


def _payment_terms(text: str) -> list[FieldCandidate]:
    lowered = text.casefold()
    found: list[tuple[int, int, str, int, float]] = []
    for match in _NET_TERMS.finditer(lowered):
        days = int(match.group("days"))
        found.append((match.start(), match.end(), f"Net {days}", days, 0.9))
    for match in _WITHIN_TERMS.finditer(lowered):
        days = int(match.group("days"))
        found.append((match.start(), match.end(), f"Within {days} days", days, 0.8))
    found.extend(
        (match.start(), match.end(), "Due on receipt", 0, 0.85)
        for match in _RECEIPT_TERMS.finditer(lowered)
    )
    out: list[FieldCandidate] = []
    seen: set[str] = set()
    for start, end, label, days, confidence in sorted(found):
        if label in seen:
            continue
        seen.add(label)
        out.append(
            FieldCandidate(
                field="payment_terms",
                value_text=label,
                value_date=None,
                value_number=Decimal(days),
                currency=None,
                confidence=confidence,
                evidence=evidence(text, start, end),
                start=start,
                end=end,
            )
        )
    return out


def _amounts(text: str) -> list[FieldCandidate]:
    mentions: list[tuple[int, int, Decimal, str, bool]] = []
    taken = _Taken()
    for pattern in _AMOUNT_PATTERNS:
        for match in pattern.finditer(text):
            if len(mentions) >= MAX_MENTIONS:
                break
            start, end = match.span()
            if taken.overlaps(start, end):
                continue
            amount = parse_amount(match.group("num"))
            if amount is None:
                continue
            groups = match.groupdict()
            symbol = groups.get("sym")
            currency = _SYMBOLS[symbol] if symbol else str(groups["code"])
            taken.add(start, end)
            mentions.append((start, end, amount, currency, symbol == "$"))
    out: list[FieldCandidate] = []
    seen: set[tuple[str, Decimal, str]] = set()
    unlabelled = 0
    for start, end, amount, currency, ambiguous_symbol in sorted(mentions):
        line_start = text.rfind("\n", 0, start) + 1
        window = text[max(line_start, start - 60) : start].casefold()
        label = _nearest_label(window, _AMOUNT_LABELS)
        if label is not None:
            field, confidence = label[0], 0.85
        else:
            if unlabelled >= MAX_UNLABELLED_AMOUNTS:
                continue
            unlabelled += 1
            field, confidence = "amount", 0.5
        if ambiguous_symbol:
            confidence -= 0.1
        key = (field, amount, currency)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            FieldCandidate(
                field=field,
                value_text=text[start:end],
                value_date=None,
                value_number=amount,
                currency=currency,
                confidence=round(confidence, 2),
                evidence=evidence(text, start, end),
                start=start,
                end=end,
            )
        )
    return out


def _clean_party(raw: str) -> str | None:
    name = raw.strip().strip("\"'").strip()
    if name.casefold().startswith("the "):
        name = name[4:].strip()
    if name.endswith(".") and not _ABBREVIATION_END.search(name):
        name = name[:-1].rstrip()
    if not 2 <= len(name) <= 120 or len(name.split()) > 12:
        return None
    if not name[0].isupper() or _TIME_LIKE.match(name) or not any(ch.isalpha() for ch in name):
        return None
    return name


def _party_end(segment: str) -> int:
    stop = _PARTY_STOP.search(segment)
    return stop.start() if stop else len(segment)


def _parties(text: str) -> list[FieldCandidate]:
    out: list[FieldCandidate] = []
    seen: set[str] = set()
    scan = text[:PARTY_SCAN_CHARS]
    for match in _BETWEEN.finditer(scan):
        if len(out) >= 4:
            break
        segment = scan[match.end() : match.end() + 400].split("\n\n")[0]
        first_end = _party_end(segment)
        first = _clean_party(segment[:first_end])
        if first is None:
            continue
        depth, cut = 0, -1
        for index in range(first_end, len(segment)):
            char = segment[index]
            depth += (char == "(") - (char == ")")
            if depth == 0 and (and_match := _AND.match(segment, index)):
                cut = and_match.end()
                break
        if cut < 0:
            continue
        second = _clean_party(segment[cut : cut + _party_end(segment[cut:])])
        if second is None:
            continue
        start, end = match.start(), match.end() + cut + len(second)
        for name in (first, second):
            if name.casefold() in seen:
                continue
            seen.add(name.casefold())
            out.append(
                FieldCandidate(
                    field="party",
                    value_text=name,
                    value_date=None,
                    value_number=None,
                    currency=None,
                    confidence=0.7,
                    evidence=evidence(text, start, min(end, len(text))),
                    start=start,
                    end=min(end, len(text)),
                )
            )
    return out


def _invoice_numbers(text: str) -> list[FieldCandidate]:
    out: list[FieldCandidate] = []
    seen: set[str] = set()
    for match in _INVOICE_NUMBER.finditer(text):
        value = match.group("value")
        if not any(ch.isdigit() for ch in value) or value.upper() in seen or len(out) >= 3:
            continue
        seen.add(value.upper())
        out.append(
            FieldCandidate(
                field="invoice_number",
                value_text=value,
                value_date=None,
                value_number=None,
                currency=None,
                confidence=0.85,
                evidence=evidence(text, match.start(), match.end()),
                start=match.start("value"),
                end=match.end("value"),
            )
        )
    return out


def extract_fields(text: str) -> list[FieldCandidate]:
    """All rule-based fields found in ``text``, ordered by position (at most :data:`MAX_FIELDS`)."""
    fields = [
        *_dates(text),
        *_payment_terms(text),
        *_amounts(text),
        *_parties(text),
        *_invoice_numbers(text),
    ]
    fields.sort(key=lambda f: (f.start, f.field))
    return fields[:MAX_FIELDS]
