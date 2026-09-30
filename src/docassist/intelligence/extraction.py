"""LLM structured extraction for contracts and invoices, with evidence verification.

Every value the model returns must come with an ``evidence`` quote and the id of the
source passage it was quoted from. A value is kept only when

1. the quote is found in the cited passage (normalised exact match, or >= 0.85 token
   overlap) - or, with a confidence penalty, in another passage of the same request;
2. the value parses as its declared type (ISO date, number, integer, ISO currency code);

and its confidence is lowered further when the value itself is not visibly supported by
the quote (a date/number not written in it, text tokens not covered). Unverifiable values
are dropped and counted.

Kept values are merged with the deterministic rules extractor's fields (rules fill the
gaps; disagreements are reported as conflicts) and, when ``persist`` is requested and at
least one value was verified, they replace the version's previous ``method="llm"`` rows
in one transaction. Reading the document is enough to extract (the output derives from
content the caller can already read); persisting requires ``intelligence:use``.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import and_, delete, select

from docassist.audit.service import Actor
from docassist.authz.permissions import Permission
from docassist.authz.policy import readable_clause
from docassist.authz.principal import Principal
from docassist.core.context import utcnow
from docassist.core.errors import Conflict, NotFound, ValidationFailed
from docassist.core.text import clean_line_text, estimate_tokens, truncate
from docassist.db.models import Document, DocumentVersion, ExtractedField
from docassist.intelligence.access import (
    ChunkView,
    DocView,
    FieldRow,
    VersionView,
    load_chunks,
    load_fields,
    load_readable_document,
    load_version,
)
from docassist.intelligence.llm_contract import ChatMessage, LLMRequest, LLMTask
from docassist.intelligence.prompting import (
    PROMPT_VERSION,
    call_gateway,
    clean_output,
    eligible_chunks,
    fallback_warning,
    gateway_of,
    input_budget,
    new_nonce,
    output_tokens,
    pack_sources,
    security_rules,
)
from docassist.intelligence.schemas import (
    ExtractedValue,
    ExtractionKind,
    ExtractionResult,
    FieldConflict,
    LineItem,
    Usage,
)
from docassist.intelligence.textops import (
    date_mentions,
    format_decimal,
    normalize_for_match,
    number_mentions,
    parse_decimal,
    parse_iso_date,
    quote_score,
    token_coverage,
)

if TYPE_CHECKING:
    from docassist.api.container import Container

ValueKind = Literal["text", "date", "number", "integer", "currency", "party"]

MAX_EXTRACT_CHUNKS = 3_000
MAX_EXTRACT_BATCHES = 4
EXTRACT_CONCURRENCY = 2
MAX_PARTIES = 10
MAX_LINE_ITEMS = 50
MAX_INTEGER = 36_500
TEXT_SUPPORT = 0.6
LLM_RATE_BUCKET = "llm_user"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    kind: ValueKind
    hint: str


CONTRACT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("parties", "party", "legal name of each contracting party (one entry per party)"),
    FieldSpec("effective_date", "date", "date the contract takes effect"),
    FieldSpec("expiration_date", "date", "date the contract ends or expires"),
    FieldSpec("renewal_terms", "text", "how and when the contract renews"),
    FieldSpec("payment_terms", "text", "payment terms, for example 'net 30 days'"),
    FieldSpec("payment_deadline_days", "integer", "number of days allowed for payment"),
    FieldSpec("late_penalty", "text", "fee, penalty or interest charged for late payment"),
    FieldSpec("currency", "currency", "ISO 4217 code of the contract currency"),
    FieldSpec("total_value", "number", "total contract value"),
    FieldSpec("governing_law", "text", "governing law or jurisdiction"),
    FieldSpec("termination_notice_days", "integer", "termination notice period in days"),
)
INVOICE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("invoice_number", "text", "invoice number or identifier"),
    FieldSpec("vendor", "text", "name of the issuing vendor/supplier"),
    FieldSpec("customer", "text", "name of the billed customer"),
    FieldSpec("issue_date", "date", "invoice issue date"),
    FieldSpec("due_date", "date", "payment due date"),
    FieldSpec("currency", "currency", "ISO 4217 code of the invoice currency"),
    FieldSpec("subtotal", "number", "amount before tax"),
    FieldSpec("tax", "number", "tax amount"),
    FieldSpec("total", "number", "total amount due"),
)
FIELDS: dict[str, tuple[FieldSpec, ...]] = {"contract": CONTRACT_FIELDS, "invoice": INVOICE_FIELDS}
RULE_ALIASES: dict[str, dict[str, str]] = {
    "contract": {"party": "parties", "total_amount": "total_value"},
    "invoice": {"invoice_date": "issue_date", "total_amount": "total", "amount_due": "total"},
}
"""Rules-extractor field names mapped onto this module's schema fields, per kind."""
MONEY_FIELDS = frozenset({"total_value", "subtotal", "tax", "total"})

_CURRENCY_SYMBOLS = {
    "$": "USD",
    chr(0x20AC): "EUR",
    chr(0xA3): "GBP",
    chr(0xA5): "JPY",
    chr(0x20BA): "TRY",
    chr(0x20B9): "INR",
}
_ISO_CODE = re.compile(r"^[A-Z]{3}$")
_CODE_IN_TEXT = re.compile(r"\b([A-Z]{3})\b")
_KNOWN_CODES = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK", "PLN",
        "CZK", "HUF", "TRY", "INR", "CNY", "HKD", "SGD", "KRW", "BRL", "MXN", "ZAR", "AED",
        "SAR", "ILS", "RUB", "IRR",
    }
)  # fmt: skip

_KEYWORDS = {
    "contract": (
        "agreement", "between", "party", "parties", "effective", "term", "expir", "renew",
        "terminat", "payment", "invoice", "days", "late", "penalt", "interest", "fee",
        "total", "value", "governing law", "jurisdiction", "notice",
    ),
    "invoice": (
        "invoice", "bill to", "vendor", "supplier", "customer", "issue", "date", "due",
        "payment", "subtotal", "tax", "vat", "total", "amount", "qty", "quantity", "price",
    ),
}  # fmt: skip


# --------------------------------------------------------------------------- #
# Schema and prompt
# --------------------------------------------------------------------------- #
_VALUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "value": {"type": "string", "maxLength": 300},
        "evidence": {"type": "string", "maxLength": 400},
        "source": {"type": "string", "maxLength": 8},
    },
    "required": ["found", "value", "evidence", "source"],
    "additionalProperties": False,
}
_LINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "maxLength": 300},
        "quantity": {"type": "string", "maxLength": 32},
        "unit_price": {"type": "string", "maxLength": 32},
        "amount": {"type": "string", "maxLength": 32},
        "evidence": {"type": "string", "maxLength": 400},
        "source": {"type": "string", "maxLength": 8},
    },
    "required": ["description", "quantity", "unit_price", "amount", "evidence", "source"],
    "additionalProperties": False,
}


def extraction_schema(kind: ExtractionKind) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for spec in FIELDS[kind]:
        properties[spec.name] = (
            {"type": "array", "maxItems": MAX_PARTIES, "items": _VALUE_SCHEMA}
            if spec.kind == "party"
            else _VALUE_SCHEMA
        )
    if kind == "invoice":
        properties["line_items"] = {
            "type": "array",
            "maxItems": MAX_LINE_ITEMS,
            "items": _LINE_SCHEMA,
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_FORMATS = {
    "text": "short text copied from the document",
    "party": "a party's name as written",
    "date": "YYYY-MM-DD",
    "number": "plain number such as 12500.00 (no currency symbol, no thousands separator)",
    "integer": "whole number of days such as 30",
    "currency": "three-letter ISO 4217 code such as USD",
}


def system_prompt(kind: ExtractionKind, nonce: str) -> str:
    lines = [
        f"You extract structured data from a {kind}. Fields:",
        *(f"- {s.name}: {s.hint} (format: {_FORMATS[s.kind]})" for s in FIELDS[kind]),
    ]
    if kind == "invoice":
        lines.append(
            "- line_items: each billed line with description, quantity, unit_price and amount"
            " (plain numbers), plus evidence and source"
        )
    lines += [
        (
            "For every value set found=true, copy into evidence the exact sentence or phrase "
            "of the document (at most 300 characters, verbatim) that states it, and set source "
            "to the id of the <source> element it comes from (for example C2)."
        ),
        (
            "If a field is not stated, set found=false and use empty strings. Never guess, "
            "compute or infer values that are not written in the document."
        ),
        "",
        security_rules(nonce),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Output parsing and verification
# --------------------------------------------------------------------------- #
class _ValueOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    found: bool
    value: str = Field(max_length=2_000)
    evidence: str = Field(max_length=4_000)
    source: str = Field(max_length=64)


class _LineOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(max_length=2_000)
    quantity: str = Field(max_length=64)
    unit_price: str = Field(max_length=64)
    amount: str = Field(max_length=64)
    evidence: str = Field(max_length=4_000)
    source: str = Field(max_length=64)


@dataclass(frozen=True, slots=True)
class Parsed:
    display: str
    text: str | None = None
    date_value: date | None = None
    number: Decimal | None = None
    currency: str | None = None


def normalize_currency(value: str) -> str | None:
    raw = value.strip()
    upper = raw.upper()
    if _ISO_CODE.fullmatch(upper):
        return upper
    return _CURRENCY_SYMBOLS.get(raw)


def currency_in(text: str) -> str | None:
    """First currency written in ``text`` (known ISO code or symbol)."""
    for m in _CODE_IN_TEXT.finditer(text):
        if m.group(1) in _KNOWN_CODES:
            return m.group(1)
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    return None


def parse_value(kind: ValueKind, raw: str) -> Parsed | None:
    """Parse a model-produced value into its declared type (``None`` when invalid)."""
    if kind in ("text", "party"):
        text = clean_line_text(raw, 300)
        return Parsed(display=text, text=text) if text else None
    if kind == "date":
        value = parse_iso_date(raw)
        if value is None:
            mentions = [m for m in date_mentions(raw) if not m.ambiguous]
            value = mentions[0].value if len(mentions) == 1 else None
        return Parsed(display=value.isoformat(), date_value=value) if value else None
    if kind == "currency":
        code = normalize_currency(raw)
        return Parsed(display=code, text=code, currency=code) if code else None
    number = parse_decimal(raw)
    if number is None:
        return None
    if kind == "integer":
        if number != number.to_integral_value() or not 0 <= number <= MAX_INTEGER:
            return None
        number = Decimal(int(number))
    return Parsed(display=format_decimal(number), number=number)


def value_support(kind: ValueKind, parsed: Parsed, evidence: str) -> tuple[bool, bool]:
    """``(supported, ambiguous)``: is the value visibly written in its evidence quote?"""
    if kind == "date":
        matches = [m for m in date_mentions(evidence) if m.value == parsed.date_value]
        return bool(matches), bool(matches) and all(m.ambiguous for m in matches)
    if kind in ("number", "integer"):
        return parsed.number in number_mentions(evidence), False
    if kind == "currency":
        code = parsed.currency or ""
        written = re.search(rf"\b{re.escape(code)}\b", evidence.upper()) is not None
        symbols = any(sym in evidence for sym, c in _CURRENCY_SYMBOLS.items() if c == code)
        return written or symbols, False
    return token_coverage(parsed.display, evidence) >= TEXT_SUPPORT, False


def locate_evidence(
    evidence: str, source: str, batch: dict[str, ChunkView]
) -> tuple[ChunkView | None, float, bool]:
    """``(chunk, score, moved)`` - the cited chunk if it contains the quote, else any other
    chunk of the request that does (``moved=True``), else ``(None, 0, False)``."""
    cited = batch.get(source.strip().upper())
    if cited is not None:
        score = quote_score(evidence, cited.content)
        if score > 0:
            return cited, score, False
    for chunk in batch.values():
        if chunk is cited:
            continue
        score = quote_score(evidence, chunk.content)
        if score > 0:
            return chunk, score, True
    return None, 0.0, False


def confidence_for(score: float, *, moved: bool, supported: bool, ambiguous: bool) -> float:
    confidence = 0.9 if score >= 1.0 else 0.8
    if moved:
        confidence -= 0.1
    if ambiguous:
        confidence = min(confidence, 0.6)
    if not supported:
        confidence = min(confidence, 0.5)
    return round(confidence, 2)


@dataclass(frozen=True, slots=True)
class Candidate:
    field: str
    parsed: Parsed
    evidence: str
    chunk: ChunkView
    confidence: float
    supported: bool


@dataclass(slots=True)
class BatchOutcome:
    candidates: list[Candidate] = field(default_factory=list)
    line_items: list[LineItem] = field(default_factory=list)
    unverified: int = 0
    invalid: int = 0


def _verify(
    spec: FieldSpec, raw: _ValueOut, batch: dict[str, ChunkView], out: BatchOutcome
) -> None:
    if not raw.found or not raw.value.strip():
        return
    evidence = clean_output(raw.evidence, 300)
    chunk, score, moved = (
        locate_evidence(evidence, raw.source, batch) if evidence else (None, 0.0, False)
    )
    if chunk is None:
        out.unverified += 1
        return
    parsed = parse_value(spec.kind, raw.value)
    if parsed is None:
        out.invalid += 1
        return
    supported, ambiguous = value_support(spec.kind, parsed, evidence)
    out.candidates.append(
        Candidate(
            field=spec.name,
            parsed=parsed,
            evidence=evidence,
            chunk=chunk,
            confidence=confidence_for(score, moved=moved, supported=supported, ambiguous=ambiguous),
            supported=supported,
        )
    )


def _verify_line(raw: _LineOut, batch: dict[str, ChunkView], out: BatchOutcome) -> None:
    description = clean_line_text(raw.description, 300)
    evidence = clean_output(raw.evidence, 300)
    if not description:
        return
    chunk, score, moved = (
        locate_evidence(evidence, raw.source, batch) if evidence else (None, 0.0, False)
    )
    if chunk is None:
        out.unverified += 1
        return
    numbers: dict[str, str | None] = {}
    for name in ("quantity", "unit_price", "amount"):
        text = getattr(raw, name).strip()
        value = parse_decimal(text) if text else None
        if text and value is None:
            out.invalid += 1
            return
        numbers[name] = format_decimal(value) if value is not None else None
    amount = parse_decimal(raw.amount) if numbers["amount"] else None
    supported = token_coverage(description, evidence) >= TEXT_SUPPORT and (
        amount is None or amount in number_mentions(evidence)
    )
    out.line_items.append(
        LineItem(
            description=description,
            quantity=numbers["quantity"],
            unit_price=numbers["unit_price"],
            amount=numbers["amount"],
            evidence=evidence,
            chunk_id=chunk.id,
            page=chunk.page_start,
            confidence=confidence_for(score, moved=moved, supported=supported, ambiguous=False),
        )
    )


def verify_output(
    kind: ExtractionKind, data: dict[str, Any] | None, batch: dict[str, ChunkView]
) -> BatchOutcome | None:
    """Validate and verify one model response against the passages it was given."""
    if not isinstance(data, dict):
        return None
    out = BatchOutcome()
    for spec in FIELDS[kind]:
        raw = data.get(spec.name)
        entries = raw if spec.kind == "party" and isinstance(raw, list) else [raw]
        for entry in entries[:MAX_PARTIES]:
            if entry is None:
                continue
            try:
                value = _ValueOut.model_validate(entry)
            except ValidationError:
                out.invalid += 1
                continue
            _verify(spec, value, batch, out)
    if kind == "invoice" and isinstance(data.get("line_items"), list):
        for entry in data["line_items"][:MAX_LINE_ITEMS]:
            try:
                line = _LineOut.model_validate(entry)
            except ValidationError:
                out.invalid += 1
                continue
            _verify_line(line, batch, out)
    return out


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #
def merge_candidates(outcomes: Sequence[BatchOutcome]) -> tuple[list[Candidate], list[LineItem]]:
    """Best candidate per single-valued field (first wins ties); de-duplicated parties and
    line items in document order."""
    best: dict[str, Candidate] = {}
    parties: dict[str, Candidate] = {}
    lines: dict[tuple[str, str | None], LineItem] = {}
    for outcome in outcomes:
        for candidate in outcome.candidates:
            if candidate.field == "parties":
                key = normalize_for_match(candidate.parsed.display)
                if key and key not in parties and len(parties) < MAX_PARTIES:
                    parties[key] = candidate
                continue
            current = best.get(candidate.field)
            if current is None or candidate.confidence > current.confidence:
                best[candidate.field] = candidate
        for line in outcome.line_items:
            line_key = (normalize_for_match(line.description), line.amount)
            if line_key not in lines and len(lines) < MAX_LINE_ITEMS:
                lines[line_key] = line
    return [*parties.values(), *best.values()], list(lines.values())


def candidate_value(candidate: Candidate, currency: str | None) -> ExtractedValue:
    parsed = candidate.parsed
    money_currency = None
    if candidate.field in MONEY_FIELDS:
        money_currency = currency or currency_in(candidate.evidence)
    return ExtractedValue(
        field=candidate.field,
        value=f"{parsed.display} {money_currency}" if money_currency else parsed.display,
        value_text=parsed.text,
        value_date=parsed.date_value,
        value_number=format_decimal(parsed.number) if parsed.number is not None else None,
        currency=parsed.currency or money_currency,
        evidence=candidate.evidence,
        chunk_id=candidate.chunk.id,
        page=candidate.chunk.page_start,
        confidence=candidate.confidence,
        method="llm",
        value_supported=candidate.supported,
    )


def rules_value(name: str, row: FieldRow) -> ExtractedValue:
    value = row.to_value()
    return ExtractedValue(
        field=name,
        value=value.value or "",
        value_text=value.value_text,
        value_date=value.value_date,
        value_number=value.value_number,
        currency=value.currency,
        evidence=value.evidence,
        chunk_id=value.chunk_id,
        page=value.page,
        confidence=value.confidence,
        method="rules",
    )


def merge_with_rules(
    kind: ExtractionKind, llm_values: list[ExtractedValue], rules: Sequence[FieldRow]
) -> tuple[list[ExtractedValue], list[FieldConflict]]:
    """LLM values first; rules values fill fields the LLM did not verify; single-valued
    fields where both disagree are reported as conflicts (the LLM value is kept)."""
    names = [spec.name for spec in FIELDS[kind]]
    by_rule: dict[str, list[FieldRow]] = {}
    for row in rules:
        name = RULE_ALIASES[kind].get(row.field, row.field)
        if name in names and row.display():
            by_rule.setdefault(name, []).append(row)
    have = {v.field for v in llm_values}
    merged = list(llm_values)
    conflicts: list[FieldConflict] = []
    for name in names:
        rows = sorted(by_rule.get(name, []), key=lambda r: -r.confidence)
        if not rows:
            continue
        if name not in have:
            merged.extend(rules_value(name, r) for r in (rows if name == "parties" else rows[:1]))
            continue
        if name == "parties":
            continue
        llm = next(v for v in llm_values if v.field == name)
        rule = rows[0]
        if _comparable(llm) != _comparable_row(rule):
            conflicts.append(
                FieldConflict(field=name, llm_value=llm.value, rules_value=rule.display() or "")
            )
    order = {name: index for index, name in enumerate(names)}
    merged.sort(key=lambda v: order.get(v.field, len(order)))
    return merged, conflicts


def _comparable(value: ExtractedValue) -> str:
    if value.value_date is not None:
        return value.value_date.isoformat()
    if value.value_number is not None:
        return value.value_number
    return normalize_for_match(value.value)


def _comparable_row(row: FieldRow) -> str:
    if row.value_date is not None:
        return row.value_date.isoformat()
    if row.value_number is not None:
        return format_decimal(row.value_number)
    return normalize_for_match(row.display() or "")


# --------------------------------------------------------------------------- #
# Chunk selection
# --------------------------------------------------------------------------- #
def select_chunks(
    eligible: list[tuple[str, ChunkView]], kind: ExtractionKind, capacity_tokens: int
) -> list[tuple[str, ChunkView]]:
    """All chunks if they fit ``capacity_tokens``; else the first two, the last one and the
    highest keyword-scoring rest, restored to document order."""
    costs = [estimate_tokens(chunk.content) + 30 for _sid, chunk in eligible]
    if sum(costs) <= capacity_tokens:
        return eligible
    keywords = _KEYWORDS[kind]

    def score(chunk: ChunkView) -> int:
        lowered = chunk.content.lower()
        hits = sum(lowered.count(k) for k in keywords)
        return (
            hits
            + (3 if date_mentions(chunk.content) else 0)
            + (1 if number_mentions(chunk.content) else 0)
        )

    must = {0, 1, len(eligible) - 1}
    ranked = sorted(range(len(eligible)), key=lambda i: (i not in must, -score(eligible[i][1]), i))
    chosen: set[int] = set()
    used = 0
    for index in ranked:
        if used + costs[index] > capacity_tokens:
            continue
        chosen.add(index)
        used += costs[index]
    return [eligible[i] for i in sorted(chosen)]


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
def kind_for(doc: DocView, requested: ExtractionKind | None) -> ExtractionKind:
    if requested is not None:
        return requested
    if doc.doc_type == "contract":
        return "contract"
    if doc.doc_type == "invoice":
        return "invoice"
    raise ValidationFailed(
        "Structured extraction supports contracts and invoices; specify the kind to use."
    )


@dataclass(slots=True)
class _LLMRun:
    outcomes: list[BatchOutcome] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    reasons: list[str] = field(default_factory=list)
    model: str | None = None
    provider: str | None = None


class Extractor:
    def __init__(self, container: Container) -> None:
        self._c = container

    async def extract(
        self,
        principal: Principal,
        document_id: uuid.UUID,
        *,
        kind: ExtractionKind | None = None,
        persist: bool = True,
    ) -> ExtractionResult:
        org_id = principal.require_org()
        if persist:
            principal.require(Permission.INTELLIGENCE_USE)
        now = utcnow()
        async with self._c.db.session(principal.db_context) as session:
            doc = await load_readable_document(session, principal, document_id, now)
            version = await load_version(session, doc, None)
            chunks = await load_chunks(
                session, principal, now, doc, version, limit=MAX_EXTRACT_CHUNKS
            )
            rules = await load_fields(session, principal, now, doc, version.id, method="rules")
        chosen_kind = kind_for(doc, kind)
        settings = self._c.settings
        await self._c.limiter.enforce(
            LLM_RATE_BUCKET, str(principal.user_id), settings.rate_limit.llm_per_user
        )
        warnings: list[str] = []
        run = await self._run_llm(principal, doc, chunks, chosen_kind, warnings)
        candidates, line_items = merge_candidates(run.outcomes)
        currency = next((c.parsed.currency for c in candidates if c.field == "currency"), None)
        llm_values = [candidate_value(c, currency) for c in candidates]
        fields, conflicts = merge_with_rules(chosen_kind, llm_values, rules)
        llm_ok = bool(run.outcomes)
        persisted_count = 0
        if persist and llm_ok and (llm_values or line_items):
            persisted_count = await self._persist(
                principal,
                org_id=org_id,
                doc=doc,
                version=version,
                values=llm_values,
                line_items=line_items,
                now=now,
                kind=chosen_kind,
                run=run,
            )
        else:
            if persist and llm_ok:
                warnings.append(
                    "No value could be verified against the document; previously stored AI "
                    "values were kept."
                )
            await self._audit(principal, doc=doc, version=version, kind=chosen_kind, run=run)
        unverified = sum(o.unverified for o in run.outcomes)
        if unverified:
            warnings.append(
                f"{unverified} value(s) were dropped because their evidence was not found."
            )
        return ExtractionResult(
            document_id=doc.id,
            version_id=version.id,
            version_number=version.number,
            kind=chosen_kind,
            method="llm" if llm_ok else "rules_only",
            fields=fields,
            line_items=line_items,
            conflicts=conflicts,
            llm_values=len(llm_values),
            dropped_unverified=unverified,
            dropped_invalid=sum(o.invalid for o in run.outcomes),
            persisted=persisted_count > 0,
            persisted_count=persisted_count,
            model=run.model,
            provider=run.provider,
            prompt_version=PROMPT_VERSION if llm_ok else None,
            warnings=warnings,
            usage=run.usage,
        )

    async def _run_llm(
        self,
        principal: Principal,
        doc: DocView,
        chunks: list[ChunkView],
        kind: ExtractionKind,
        warnings: list[str],
    ) -> _LLMRun:
        settings = self._c.settings
        run = _LLMRun()
        eligible, excluded = eligible_chunks(chunks, settings)
        if excluded:
            warnings.append(
                f"{excluded} passage(s) flagged as possible prompt injection were excluded "
                "from AI processing."
            )
        if not eligible:
            warnings.append(fallback_warning("llm_output_unverifiable"))
            return run
        nonce = new_nonce()
        system = system_prompt(kind, nonce)
        budget = input_budget(settings, system)
        if budget < 300:
            warnings.append(fallback_warning("context_budget_too_small"))
            return run
        selected = select_chunks(eligible, kind, int(budget * MAX_EXTRACT_BATCHES * 0.9))
        if len(selected) < len(eligible):
            warnings.append(
                f"The document is long; extraction used {len(selected)} of {len(eligible)} "
                "passages (the most relevant ones)."
            )
        warn = settings.retrieval.injection_warn_threshold
        batches = pack_sources(selected, nonce, budget, warn_threshold=warn)[:MAX_EXTRACT_BATCHES]
        gateway = gateway_of(self._c)
        semaphore = asyncio.Semaphore(EXTRACT_CONCURRENCY)

        async def one(batch_ids: dict[str, ChunkView], rendered: str) -> None:
            request = LLMRequest(
                task=LLMTask.EXTRACT,
                system=system,
                messages=[ChatMessage(role="user", content=f"Document passages:\n\n{rendered}")],
                output_schema=extraction_schema(kind),
                max_output_tokens=output_tokens(settings, 4_000),
                data_classification=doc.classification,
            )
            async with semaphore:
                outcome = await call_gateway(gateway, request, principal, feature="extract")
            if outcome.result is None:
                run.reasons.append(outcome.reason or "llm_unavailable")
                return
            run.usage = Usage(
                calls=run.usage.calls + 1,
                input_tokens=run.usage.input_tokens + int(outcome.result.input_tokens),
                output_tokens=run.usage.output_tokens + int(outcome.result.output_tokens),
            )
            run.model, run.provider = outcome.result.model, outcome.result.provider
            verified = verify_output(kind, outcome.data, batch_ids)
            if verified is None:
                run.reasons.append("llm_output_invalid")
                return
            run.outcomes.append(verified)

        await asyncio.gather(*(one(b.ids, b.rendered) for b in batches))
        if run.reasons:
            reason = run.reasons[0]
            if run.outcomes:
                warnings.append(
                    f"{len(run.reasons)} of {len(batches)} extraction request(s) failed; "
                    "results may be incomplete."
                )
            else:
                warnings.append(
                    fallback_warning(reason).replace(
                        "a deterministic result is shown", "only rules-based fields are shown"
                    )
                )
        return run

    async def _persist(
        self,
        principal: Principal,
        *,
        org_id: uuid.UUID,
        doc: DocView,
        version: VersionView,
        values: list[ExtractedValue],
        line_items: list[LineItem],
        now: datetime,
        kind: ExtractionKind,
        run: _LLMRun,
    ) -> int:
        rows = [_row(org_id, doc.id, version.id, v) for v in values]
        currency = next((v.currency for v in values if v.field == "currency"), None)
        rows += [_line_row(org_id, doc.id, version.id, item, currency) for item in line_items]
        async with self._c.db.transaction(principal.db_context) as session:
            locked = (
                await session.execute(
                    select(DocumentVersion.id)
                    .where(
                        DocumentVersion.id == version.id, DocumentVersion.organization_id == org_id
                    )
                    .with_for_update()
                )
            ).scalar_one_or_none()
            current = (
                await session.execute(
                    select(Document.current_version_id).where(
                        Document.id == doc.id, readable_clause(principal, now)
                    )
                )
            ).scalar_one_or_none()
            if locked is None or current is None:
                raise NotFound(internal_detail="document no longer readable")
            if current != version.id:
                raise Conflict("The document changed during extraction; please retry.")
            await session.execute(
                delete(ExtractedField).where(
                    and_(
                        ExtractedField.organization_id == org_id,
                        ExtractedField.version_id == version.id,
                        ExtractedField.method == "llm",
                    )
                )
            )
            session.add_all(rows)
            self._c.audit.record(
                session,
                Actor.of(principal),
                "intelligence.extract",
                resource_type="document",
                resource_id=doc.id,
                details=_audit_details(version, kind, run, len(values), persisted=len(rows)),
            )
        return len(rows)

    async def _audit(
        self,
        principal: Principal,
        *,
        doc: DocView,
        version: VersionView,
        kind: ExtractionKind,
        run: _LLMRun,
    ) -> None:
        values = sum(len(o.candidates) for o in run.outcomes)
        async with self._c.db.transaction(principal.db_context) as session:
            self._c.audit.record(
                session,
                Actor.of(principal),
                "intelligence.extract",
                resource_type="document",
                resource_id=doc.id,
                details=_audit_details(version, kind, run, values, persisted=0),
            )


def _audit_details(
    version: VersionView, kind: str, run: _LLMRun, values: int, *, persisted: int
) -> dict[str, Any]:
    return {
        "version_id": str(version.id),
        "kind": kind,
        "values": values,
        "persisted": persisted,
        "unverified": sum(o.unverified for o in run.outcomes),
        "model": run.model,
        "fallback": run.reasons[0] if run.reasons and not run.outcomes else None,
    }


def _row(
    org_id: uuid.UUID, document_id: uuid.UUID, version_id: uuid.UUID, value: ExtractedValue
) -> ExtractedField:
    return ExtractedField(
        organization_id=org_id,
        document_id=document_id,
        version_id=version_id,
        field=value.field,
        value_text=truncate(value.value_text, 1_000) if value.value_text else None,
        value_date=value.value_date,
        value_number=Decimal(value.value_number) if value.value_number is not None else None,
        currency=value.currency,
        confidence=value.confidence,
        method="llm",
        chunk_id=value.chunk_id,
        page=value.page,
        evidence=truncate(value.evidence, 500) if value.evidence else None,
    )


def _line_row(
    org_id: uuid.UUID,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    item: LineItem,
    currency: str | None,
) -> ExtractedField:
    return ExtractedField(
        organization_id=org_id,
        document_id=document_id,
        version_id=version_id,
        field="line_item",
        value_text=truncate(item.description, 1_000),
        value_number=Decimal(item.amount) if item.amount is not None else None,
        currency=currency,
        confidence=item.confidence,
        method="llm",
        chunk_id=item.chunk_id,
        page=item.page,
        evidence=truncate(item.evidence, 500),
    )
