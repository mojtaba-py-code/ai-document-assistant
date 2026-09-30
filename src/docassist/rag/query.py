"""Deterministic analysis of a user question (no model involved).

Produces:

* ``normalized`` - the question with invisible characters removed and whitespace collapsed;
* ``intent`` - ``qa`` | ``list_deadlines`` | ``summarize`` | ``compare`` | ``search``;
* ``window`` - a date range from phrases such as "next 90 days", "within 3 months",
  "this year", "this month", "in December 2026", "expire before 1 March 2027";
* ``doc_types`` - document-type hints ("contracts", "invoices", "policy", "reports");
* ``flags`` / ``injection_score`` - suspicious-input evidence from the shared prompt-injection
  scanner (:func:`docassist.ingestion.injection.scan_for_injection`) plus patterns specific
  to an assistant: system-prompt extraction, cross-tenant requests, bulk data dumps,
  exfiltration to URLs/addresses, tool/command abuse;
* ``blocked_reason`` - set for high-confidence abuse categories; the answer service then
  refuses without retrieving anything or calling a model.

A "list deadlines" intent needs a deadline word (expire, due, renew, terminate, deadline)
*and* either a time window or a listing cue ("which", "list", "upcoming", "soon"...);
without an explicit window the next :data:`DEFAULT_DEADLINE_DAYS` days are used.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from docassist.core.enums import DocumentType
from docassist.core.text import sanitize_text
from docassist.ingestion.injection import scan_for_injection

Intent = Literal["qa", "list_deadlines", "summarize", "compare", "search"]

DEFAULT_DEADLINE_DAYS = 90
MAX_WINDOW_DAYS = 3650

FLAG_SYSTEM_PROMPT = "system_prompt_extraction"
FLAG_INSTRUCTION_OVERRIDE = "instruction_override"
FLAG_CROSS_TENANT = "cross_tenant_request"
FLAG_DATA_DUMP = "bulk_data_request"
FLAG_EXFILTRATION = "exfiltration_request"
FLAG_TOOL_ABUSE = "command_or_tool_abuse"
FLAG_CONTAINS_URL = "contains_url"
FLAG_PROMPT_INJECTION = "prompt_injection"

BLOCKING_FLAGS = (
    FLAG_SYSTEM_PROMPT,
    FLAG_INSTRUCTION_OVERRIDE,
    FLAG_CROSS_TENANT,
    FLAG_DATA_DUMP,
    FLAG_EXFILTRATION,
    FLAG_PROMPT_INJECTION,
)
"""Categories confident enough to refuse the request outright (first match wins)."""

_I = re.IGNORECASE
_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    FLAG_SYSTEM_PROMPT: (
        re.compile(
            r"\b(system|initial|hidden|original|developer)\s+(prompt|instructions?|message)s?\b", _I
        ),
        re.compile(
            r"\b(reveal|print|show|repeat|output|display|dump|leak|disclose|recite)\b.{0,40}"
            r"\b(your|the)\s+(prompt|instructions?|rules|guidelines|configuration|canary)\b",
            _I,
        ),
        re.compile(r"\bwhat\s+(were|are)\s+you\s+(told|instructed)\b", _I),
        re.compile(r"\b(text|words|everything)\s+(above|before)\s+(this|my)\b", _I),
    ),
    FLAG_INSTRUCTION_OVERRIDE: (
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b.{0,30}\b(previous|prior|above|all|any|your|the|these)\b"
            r".{0,20}\b(instructions?|rules|prompts?|guidelines|constraints|restrictions|policies)\b",
            _I,
        ),
        re.compile(r"\byou\s+are\s+now\b|\bfrom\s+now\s+on\s+you\b", _I),
        re.compile(
            r"\b(developer|dan|god|jailbreak|unrestricted)\s+mode\b|\bdo\s+anything\s+now\b", _I
        ),
    ),
    FLAG_CROSS_TENANT: (
        re.compile(
            r"\b(other|another|different|all|every|foreign)\s+(organi[sz]ations?|tenants?|workspaces?)\b",
            _I,
        ),
        re.compile(r"\b(organi[sz]ation|tenant|org)[_\s-]?ids?\b|\bcross[\s-]?tenant\b", _I),
    ),
    FLAG_DATA_DUMP: (
        re.compile(
            r"\b(dump|export|exfiltrate|extract|leak|download)\b.{0,30}"
            r"\b(the\s+)?(database|db|tables?|all\s+(documents|data|records|files|users|chunks)|everything)\b",
            _I,
        ),
        re.compile(r"\bselect\s+\*\s+from\b|\bdrop\s+table\b|\bunion\s+select\b", _I),
        re.compile(r"\ball\s+(passwords|api\s+keys|secrets|credentials|tokens)\b", _I),
    ),
    FLAG_EXFILTRATION: (
        re.compile(
            r"\b(send|forward|e-?mail|post|upload|transmit|submit|share)\b.{0,60}\b(to|at|into)\b.{0,30}"
            r"(https?://|www\.|[\w.+-]+@[\w-]+\.[\w.]+)",
            _I,
        ),
        re.compile(r"!\[[^\]]*\]\(\s*https?://", _I),
    ),
    FLAG_TOOL_ABUSE: (
        re.compile(
            r"\b(curl|wget|rm\s+-rf|powershell|cmd\.exe|/bin/sh|subprocess|os\.system)\b", _I
        ),
        re.compile(
            r"\b(run|execute|call)\s+(the\s+)?(following\s+)?(command|shell|script|tool)s?\b", _I
        ),
    ),
    FLAG_CONTAINS_URL: (re.compile(r"https?://|www\.", _I),),
}

_DEADLINE = re.compile(
    r"\b(expir\w*|deadlines?|due|renew\w*|terminat\w*|lapse\w*|end\s+dates?|notice\s+periods?)\b",
    _I,
)
_LIST_CUE = re.compile(
    r"\b(which|list|show|all|any|upcoming|coming\s+up|soon|pending|outstanding)\b", _I
)
_SUMMARIZE = re.compile(
    r"^\s*(please\s+)?(summari[sz]e|give\s+(me\s+)?(a|an)\s+(summary|overview)|tl;?dr)\b|\bsummary\s+of\b",
    _I,
)
_COMPARE = re.compile(
    r"\b(compare|comparison|differences?\s+between|differ|what\s+changed|changes\s+between|versus|vs\.?)\b",
    _I,
)
_SEARCH = re.compile(
    r"^\s*(find|search(\s+for)?|locate|show\s+me|list)\b.{0,60}\b(documents?|files?|contracts?|invoices?|policies)\b",
    _I,
)
_DOC_TYPES: tuple[tuple[re.Pattern[str], DocumentType], ...] = (
    (re.compile(r"\b(contracts?|agreements?|ndas?|msas?)\b", _I), DocumentType.CONTRACT),
    (re.compile(r"\binvoices?\b", _I), DocumentType.INVOICE),
    (re.compile(r"\bpolic(y|ies)\b", _I), DocumentType.POLICY),
    (re.compile(r"\breports?\b", _I), DocumentType.REPORT),
)

_MONTHS = {
    name.lower(): index
    for index in range(1, 13)
    for name in (calendar.month_name[index], calendar.month_abbr[index])
}
_MONTHS["sept"] = 9
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?,?\s+(\d{{4}})\b", _I), "dmy"),
    (re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", _I), "mdy"),
    (re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{4}})\b", _I), "my"),
)
_RELATIVE = re.compile(
    r"\b(?:next|coming|following|within(?:\s+the\s+next)?|in\s+the\s+next|over\s+the\s+next|in)\s+"
    r"(\d{1,4}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s*"
    r"(days?|weeks?|months?|years?)\b",
    _I,
)
_RELATIVE_TERM = re.compile(r"\s+(of|after|from|following|upon|prior)\b", _I)
_NEXT_UNIT = re.compile(r"\bnext\s+(week|month|year)\b", _I)
_BEFORE = re.compile(
    r"\b(before|by|until|till|prior\s+to|no\s+later\s+than)\s+(?P<rest>.{4,40})", _I
)
_IN_MONTH = re.compile(rf"\bin\s+({_MONTH_RE})\.?\s+(\d{{4}})\b", _I)
_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
}  # fmt: skip


@dataclass(frozen=True, slots=True)
class TimeWindow:
    start: date
    end: date  # inclusive
    label: str

    @property
    def days(self) -> int:
        return (self.end - self.start).days


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    normalized: str
    intent: Intent
    window: TimeWindow | None
    doc_types: tuple[str, ...]
    flags: tuple[str, ...]
    injection_score: float
    blocked_reason: str | None

    @property
    def suspicious(self) -> bool:
        return bool(self.flags)


def add_months(day: date, months: int) -> date:
    month_index = day.month - 1 + months
    year = day.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _cap(today: date, end: date) -> date:
    return min(end, today + timedelta(days=MAX_WINDOW_DAYS))


def parse_date(text: str) -> tuple[date, str] | None:
    """First absolute date in ``text`` -> (date, precision) with precision "day" or "month"."""
    for pattern, kind in _DATE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        try:
            if kind == "iso":
                year, month, day = (int(g) for g in match.groups())
                return date(year, month, day), "day"
            if kind == "dmy":
                return date(int(match[3]), _MONTHS[match[2].lower()], int(match[1])), "day"
            if kind == "mdy":
                return date(int(match[3]), _MONTHS[match[1].lower()], int(match[2])), "day"
            return date(int(match[2]), _MONTHS[match[1].lower()], 1), "month"
        except (ValueError, KeyError):
            continue
    return None


def parse_window(text: str, today: date) -> TimeWindow | None:
    relative = _RELATIVE.search(text)
    if relative is not None and _RELATIVE_TERM.match(text, relative.end()):
        relative = None  # "within 30 days of receipt" is a contract term, not a date range
    if relative is not None:
        raw, unit = relative.group(1).lower(), relative.group(2).lower().rstrip("s")
        amount = int(raw) if raw.isdigit() else _WORD_NUMBERS[raw]
        if amount <= 0:
            return None
        if unit == "day":
            end = today + timedelta(days=amount)
        elif unit == "week":
            end = today + timedelta(weeks=amount)
        elif unit == "month":
            end = add_months(today, amount)
        else:
            end = add_months(today, 12 * amount)
        return TimeWindow(today, _cap(today, end), f"next {amount} {unit}(s)")
    lowered = text.lower()
    if "this year" in lowered or "end of the year" in lowered or "end of year" in lowered:
        return TimeWindow(today, date(today.year, 12, 31), "this year")
    if "this month" in lowered:
        last = calendar.monthrange(today.year, today.month)[1]
        return TimeWindow(today, date(today.year, today.month, last), "this month")
    in_month = _IN_MONTH.search(text)
    if in_month is not None:
        year, month = int(in_month.group(2)), _MONTHS[in_month.group(1).lower()]
        start = date(year, month, 1)
        end = date(year, month, calendar.monthrange(year, month)[1])
        if end >= today:
            return TimeWindow(
                max(start, today), _cap(today, end), f"{calendar.month_name[month]} {year}"
            )
    before = _BEFORE.search(text)
    if before is not None:
        parsed = parse_date(before.group("rest"))
        if parsed is not None:
            target, precision = parsed
            exclusive = before.group(1).lower() in {"before", "prior to"}
            if precision == "month" and not exclusive:
                target = date(
                    target.year, target.month, calendar.monthrange(target.year, target.month)[1]
                )
            end = target - timedelta(days=1) if exclusive else target
            if end >= today:
                return TimeWindow(today, _cap(today, end), f"until {end.isoformat()}")
    unit_match = _NEXT_UNIT.search(text)
    if unit_match is not None:
        unit = unit_match.group(1).lower()
        end = {
            "week": today + timedelta(weeks=1),
            "month": add_months(today, 1),
            "year": add_months(today, 12),
        }[unit]
        return TimeWindow(today, end, f"next {unit}")
    return None


def detect_flags(text: str) -> list[str]:
    return [flag for flag, patterns in _PATTERNS.items() if any(p.search(text) for p in patterns)]


_SEARCH_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:find|search(?:\s+for)?|locate|show\s+me|list)\s+"
    r"(?:(?:all|any|the|our|my)\s+)*"
    r"(?:documents?|docs?|files?|contracts?|invoices?|polic(?:y|ies)|reports?)\s*"
    r"(?:(?:that|which)\s+)?"
    r"(?:describ\w*|about|on|regarding|relat\w*\s+to|cover\w*|mention\w*|discuss\w*|"
    r"explain\w*|with|for|of|from|by|containing)?\s*",
    _I,
)


def search_topic(question: str) -> str:
    """The subject of a "find documents about X" request (the command words removed)."""
    topic = _SEARCH_PREFIX.sub("", question, count=1).strip(" .?!:;")
    return topic or question.strip()


def analyze_query(
    question: str,
    *,
    today: date,
    warn_threshold: float = 0.4,
    block_threshold: float = 0.8,
) -> QueryAnalysis:
    cleaned, report = sanitize_text(question)
    normalized = " ".join(cleaned.split())
    injection = scan_for_injection(
        normalized, report.decoded_tag_text, channel_flags=report.as_flags()
    )
    flags = detect_flags(normalized)
    if injection.score >= warn_threshold:
        flags.extend(f"injection:{flag}" for flag in injection.flags)
    if injection.score >= block_threshold:
        flags.append(FLAG_PROMPT_INJECTION)
    blocked = next((flag for flag in BLOCKING_FLAGS if flag in flags), None)

    window = parse_window(normalized, today)
    doc_types = tuple(
        dict.fromkeys(t.value for pattern, t in _DOC_TYPES if pattern.search(normalized))
    )
    intent: Intent = "qa"
    if _DEADLINE.search(normalized) and (window is not None or _LIST_CUE.search(normalized)):
        intent = "list_deadlines"
        if window is None:
            window = TimeWindow(today, today + timedelta(days=DEFAULT_DEADLINE_DAYS), "upcoming")
    elif _SUMMARIZE.search(normalized):
        intent = "summarize"
    elif _COMPARE.search(normalized):
        intent = "compare"
    elif _SEARCH.search(normalized):
        intent = "search"
    return QueryAnalysis(
        normalized=normalized,
        intent=intent,
        window=window,
        doc_types=doc_types,
        flags=tuple(dict.fromkeys(flags)),
        injection_score=round(injection.score, 4),
        blocked_reason=blocked,
    )
