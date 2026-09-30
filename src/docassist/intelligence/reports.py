"""Document reports: metadata, summary, key fields, deadlines and risk flags.

The report is built from the current version only and returned both as JSON and as
Markdown. Every string taken from the document, its fields or a model is passed through
:func:`~docassist.intelligence.textops.escape_markdown` before it enters the Markdown, so a
malicious title or clause cannot inject links, images, HTML or table structure.

Risk flags are deterministic heuristics (regular expressions over the chunk text plus the
extracted fields); each carries the first matching passage as evidence with its page:

* ``auto_renewal``, ``late_penalty``, ``unlimited_liability``, ``indemnity``,
  ``termination_for_convenience``, ``exclusivity`` - clause patterns;
* ``missing_signature_block`` / ``unsigned_signature_lines`` - contracts/legal documents
  without a signature block, or with blank signature lines;
* ``expired`` / ``expiring_soon`` (<= 90 days) / ``overdue`` - from date fields;
* ``missing_key_fields`` - contract/invoice without the fields that matter most;
* ``embedded_instructions`` - passages flagged by the ingestion injection scanner;
* ``personal_data`` - PII kinds detected at ingestion.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from docassist.audit.service import Actor
from docassist.authz.principal import Principal
from docassist.core.context import utcnow
from docassist.core.text import clean_line_text
from docassist.intelligence.access import (
    ChunkView,
    DocView,
    FieldRow,
    VersionView,
    best_per_field,
    load_chunks,
    load_fields,
    load_readable_document,
    load_version,
)
from docassist.intelligence.deadlines import DEADLINE_FIELDS
from docassist.intelligence.schemas import (
    DeadlineItem,
    DocumentInfo,
    DocumentReport,
    FieldValue,
    ReportResponse,
    RiskFlag,
    SummaryResult,
)
from docassist.intelligence.textops import escape_markdown, split_sentences

if TYPE_CHECKING:
    from docassist.api.container import Container
    from docassist.intelligence.summarize import Summarizer

MAX_REPORT_CHUNKS = 3_000
EXPIRING_SOON_DAYS = 90
Severity = Literal["info", "warning", "high"]


@dataclass(frozen=True, slots=True)
class ClauseRule:
    code: str
    severity: Severity
    title: str
    pattern: re.Pattern[str]


_I = re.IGNORECASE
CLAUSE_RULES: tuple[ClauseRule, ...] = (
    ClauseRule(
        "auto_renewal",
        "warning",
        "Automatic renewal clause",
        re.compile(
            r"\b(?:automatic(?:ally)?\s+renew\w*|auto[- ]?renew\w*|renew\w*\s+automatically|"
            r"evergreen)\b",
            _I,
        ),
    ),
    ClauseRule(
        "late_penalty",
        "warning",
        "Late payment penalty",
        re.compile(
            r"\b(?:late\s+(?:payments?\s+(?:\w+\s+){0,5}?)?(?:fees?|penalt\w+|charges?|interest)|"
            r"liquidated\s+damages|penalt(?:y|ies))\b",
            _I,
        ),
    ),
    ClauseRule(
        "unlimited_liability",
        "high",
        "Unlimited liability",
        re.compile(r"\bunlimited\s+liabilit\w+|liabilit\w+\s+(?:is|shall\s+be)\s+unlimited\b", _I),
    ),
    ClauseRule("indemnity", "info", "Indemnification obligation", re.compile(r"\bindemnif\w+", _I)),
    ClauseRule(
        "termination_for_convenience",
        "info",
        "Termination for convenience",
        re.compile(r"\bterminat\w*\b(?:\W+\w+){0,4}?\W+for\s+convenience\b", _I),
    ),
    ClauseRule(
        "exclusivity",
        "info",
        "Exclusivity or non-compete",
        re.compile(r"\b(?:exclusiv(?:e|ity)\b|non[- ]?compet\w*)", _I),
    ),
)
_SIGNATURE_BLOCK = re.compile(
    r"\bin\s+witness\s+whereof\b|\bsignature\b|\bsigned\s+by\b|/s/|\bauthori[sz]ed\s+signatory\b",
    _I,
)
_BLANK_SIGNATURE = re.compile(r"\b(?:signature|signed|by)\s*:?\s*_{3,}", _I)
_SIGNED_MARK = re.compile(r"/s/\s*\w+|\bsigned\s+by\s+[A-Z][a-z]+", _I)
_SIGNATURE_DOC_TYPES = frozenset({"contract", "legal"})
_KEY_FIELDS = {
    "contract": ("expiration_date", "payment_terms", "parties"),
    "invoice": ("due_date", "total", "invoice_number"),
}


def _evidence(chunk: ChunkView, start: int) -> str:
    """The sentence of ``chunk`` that contains character ``start`` (<= 300 chars)."""
    consumed = 0
    for sentence in split_sentences(chunk.content):
        position = chunk.content.find(sentence[:40], consumed)
        if position == -1:
            continue
        consumed = position
        if position <= start < position + len(sentence) + 1:
            return clean_line_text(sentence, 300)
    window = chunk.content[max(0, start - 120) : start + 180]
    return clean_line_text(window, 300)


def clause_flags(chunks: Sequence[ChunkView]) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    for rule in CLAUSE_RULES:
        first: tuple[ChunkView, int] | None = None
        count = 0
        for chunk in chunks:
            matches = list(rule.pattern.finditer(chunk.content))
            if matches and first is None:
                first = (chunk, matches[0].start())
            count += len(matches)
        if first is None:
            continue
        chunk, start = first
        flags.append(
            RiskFlag(
                code=rule.code,
                severity=rule.severity,
                title=rule.title,
                detail=f"{count} mention(s) found.",
                evidence=_evidence(chunk, start),
                page=chunk.page_start,
                chunk_id=chunk.id,
            )
        )
    return flags


def signature_flags(doc_type: str, chunks: Sequence[ChunkView]) -> list[RiskFlag]:
    if doc_type not in _SIGNATURE_DOC_TYPES or not chunks:
        return []
    block = next(
        ((c, m.start()) for c in chunks if (m := _SIGNATURE_BLOCK.search(c.content))), None
    )
    if block is None:
        return [
            RiskFlag(
                code="missing_signature_block",
                severity="warning",
                title="No signature block detected",
                detail="No signature block or execution clause was found in the text.",
            )
        ]
    blank = next(
        ((c, m.start()) for c in chunks if (m := _BLANK_SIGNATURE.search(c.content))), None
    )
    signed = any(_SIGNED_MARK.search(c.content) for c in chunks)
    if blank is not None and not signed:
        chunk, start = blank
        return [
            RiskFlag(
                code="unsigned_signature_lines",
                severity="warning",
                title="Signature lines appear blank",
                detail="Signature lines are present but no signature was detected in the text.",
                evidence=_evidence(chunk, start),
                page=chunk.page_start,
                chunk_id=chunk.id,
            )
        ]
    return []


def date_flags(deadlines: Sequence[DeadlineItem]) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    for item in deadlines:
        if item.field in {"expiration_date", "termination_date"}:
            if item.days_left < 0:
                code, severity, title = "expired", "warning", "Expired"
                detail = f"{item.field} was {item.date.isoformat()} ({-item.days_left} days ago)."
            elif item.days_left <= EXPIRING_SOON_DAYS:
                code, severity, title = "expiring_soon", "warning", "Expiring soon"
                detail = f"{item.field} is {item.date.isoformat()} (in {item.days_left} days)."
            else:
                continue
        elif item.field in {"due_date", "payment_due_date"} and item.days_left < 0:
            code, severity, title = "overdue", "high", "Payment overdue"
            detail = f"{item.field} was {item.date.isoformat()} ({-item.days_left} days ago)."
        else:
            continue
        flags.append(
            RiskFlag(
                code=code,
                severity=severity,  # type: ignore[arg-type]
                title=title,
                detail=detail,
                evidence=item.evidence,
                page=item.page,
                chunk_id=item.chunk_id,
            )
        )
    return flags


def missing_field_flags(doc_type: str, fields: dict[str, FieldRow]) -> list[RiskFlag]:
    wanted = _KEY_FIELDS.get(doc_type)
    if not wanted:
        return []
    missing = [name for name in wanted if name not in fields]
    if not missing:
        return []
    return [
        RiskFlag(
            code="missing_key_fields",
            severity="info",
            title="Key fields not found",
            detail="Not found in the extracted fields: " + ", ".join(missing) + ".",
        )
    ]


def security_flags(chunks: Sequence[ChunkView], warn_threshold: float) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    flagged = [c for c in chunks if c.injection_score >= warn_threshold]
    if flagged:
        first = flagged[0]
        flags.append(
            RiskFlag(
                code="embedded_instructions",
                severity="high",
                title="Possible embedded instructions",
                detail=(
                    f"{len(flagged)} passage(s) look like instructions aimed at AI systems "
                    "(prompt injection); treat AI output about this document with care."
                ),
                page=first.page_start,
                chunk_id=first.id,
            )
        )
    kinds = sorted({kind for c in chunks for kind in c.pii_types})
    if kinds:
        flags.append(
            RiskFlag(
                code="personal_data",
                severity="info",
                title="Personal data detected",
                detail="Detected kinds: " + ", ".join(kinds[:10]) + ".",
            )
        )
    return flags


def document_deadlines(
    doc: DocView, fields: Sequence[FieldRow], today: date, *, title: str
) -> list[DeadlineItem]:
    """Every deadline-type date of the document (past and future), soonest first."""
    seen: set[tuple[str, date]] = set()
    items: list[DeadlineItem] = []
    ordered = sorted(fields, key=lambda r: (-r.confidence, r.method != "llm"))
    for row in ordered:
        if row.field not in DEADLINE_FIELDS or row.value_date is None:
            continue
        key = (row.field, row.value_date)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            DeadlineItem(
                document_id=doc.id,
                document_title=title,
                doc_type=doc.doc_type,
                classification=doc.classification.value,
                version_id=row.version_id,
                field=row.field,
                date=row.value_date,
                days_left=(row.value_date - today).days,
                value_text=clean_line_text(row.value_text, 300) if row.value_text else None,
                evidence=clean_line_text(row.evidence, 500) if row.evidence else None,
                page=row.page,
                chunk_id=row.chunk_id,
                confidence=round(row.confidence, 4),
                method="llm" if row.method == "llm" else "rules",
            )
        )
    return sorted(items, key=lambda i: (i.date, i.field))


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _page(page: int | None) -> str:
    return f" (p. {page})" if page is not None else ""


def render_markdown(report: DocumentReport) -> str:
    """Markdown rendering in which every untrusted string is escaped."""
    doc = report.document
    esc = escape_markdown
    lines = [
        f"# Document report: {esc(doc.title, 300)}",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| Type | {esc(doc.doc_type)} |",
        f"| Classification | {esc(doc.classification)} |",
        f"| Version | {doc.version_number} of {doc.version_count} |",
        f"| Pages | {doc.page_count if doc.page_count is not None else 'unknown'} |",
        f"| Tags | {esc(', '.join(doc.tags)) or 'none'} |",
        f"| Document id | {doc.id} |",
        "",
        "## Summary",
        "",
        esc(report.summary.summary, 6_000) or "_No summary available._",
        "",
        f"_Method: {report.summary.method}._",
    ]
    if report.summary.key_points:
        lines += ["", "### Key points", ""]
        for point in report.summary.key_points:
            page = point.citations[0].page_start if point.citations else None
            lines.append(f"- {esc(point.text, 600)}{_page(page)}")
    lines += ["", "## Key fields", ""]
    if report.key_fields:
        lines += ["| Field | Value | Confidence | Method | Page |", "|---|---|---|---|---|"]
        lines += [
            f"| {esc(f.field, 64)} | {esc(f.value, 300)} | {f.confidence:.2f} | {f.method} | "
            f"{f.page if f.page is not None else ''} |"
            for f in report.key_fields
        ]
    else:
        lines.append("_No fields were extracted._")
    lines += ["", "## Deadlines", ""]
    if report.deadlines:
        lines += ["| Field | Date | Days left | Page |", "|---|---|---|---|"]
        lines += [
            f"| {esc(d.field, 64)} | {d.date.isoformat()} | {d.days_left} | "
            f"{d.page if d.page is not None else ''} |"
            for d in report.deadlines
        ]
    else:
        lines.append("_No deadlines found._")
    lines += ["", "## Risk flags", ""]
    if report.risk_flags:
        for flag in report.risk_flags:
            lines.append(
                f"- **{flag.severity.upper()}** {esc(flag.title, 200)}: "
                f"{esc(flag.detail, 500)}{_page(flag.page)}"
            )
            if flag.evidence:
                lines.append(f"  > {esc(flag.evidence, 300)}")
    else:
        lines.append("_No risk flags._")
    lines += ["", f"_Generated {report.generated_at.isoformat()}._", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
class ReportBuilder:
    def __init__(self, container: Container, summarizer: Summarizer) -> None:
        self._c = container
        self._summarizer = summarizer

    async def report(
        self,
        principal: Principal,
        document_id: uuid.UUID,
        *,
        summary_mode: Literal["extractive", "llm"] = "extractive",
    ) -> ReportResponse:
        principal.require_org()
        now = utcnow()
        async with self._c.db.session(principal.db_context) as session:
            report = await self.build(session, principal, document_id, now=now)
        if summary_mode == "llm":
            summary = await self._summarizer.summarize(principal, document_id, style="executive")
            report = report.model_copy(update={"summary": summary})
        async with self._c.db.transaction(principal.db_context) as session:
            self._c.audit.record(
                session,
                Actor.of(principal),
                "intelligence.report",
                resource_type="document",
                resource_id=report.document.id,
                details={
                    "version_id": str(report.document.version_id),
                    "summary_method": report.summary.method,
                    "risk_flags": len(report.risk_flags),
                },
            )
        return ReportResponse(report=report, markdown=render_markdown(report))

    async def build(
        self,
        session: AsyncSession,
        principal: Principal,
        document_id: uuid.UUID,
        *,
        now: datetime,
    ) -> DocumentReport:
        """Deterministic report (extractive summary) inside the caller's session."""
        doc = await load_readable_document(session, principal, document_id, now)
        version = await load_version(session, doc, None)
        chunks = await load_chunks(session, principal, now, doc, version, limit=MAX_REPORT_CHUNKS)
        fields = await load_fields(session, principal, now, doc, version.id)
        summary: SummaryResult = self._summarizer.extractive(doc, version, chunks, "executive")
        best = best_per_field(fields)
        today = now.date()
        deadlines = document_deadlines(doc, fields, today, title=doc.title)
        flags = [
            *security_flags(chunks, self._c.settings.retrieval.injection_warn_threshold),
            *clause_flags(chunks),
            *signature_flags(doc.doc_type, chunks),
            *date_flags(deadlines),
            *missing_field_flags(doc.doc_type, best),
        ]
        order = {"high": 0, "warning": 1, "info": 2}
        flags.sort(key=lambda f: order[f.severity])
        key_fields: list[FieldValue] = [best[name].to_value() for name in sorted(best)]
        return DocumentReport(
            document=_info(doc, version, len(chunks)),
            summary=summary,
            key_fields=key_fields,
            deadlines=deadlines,
            risk_flags=flags,
            generated_at=now,
        )


def _info(doc: DocView, version: VersionView, chunk_count: int) -> DocumentInfo:
    return DocumentInfo(
        id=doc.id,
        title=doc.title,
        doc_type=doc.doc_type,
        classification=doc.classification.value,
        status=doc.status,
        version_id=version.id,
        version_number=version.number,
        version_count=doc.version_count,
        page_count=version.page_count,
        chunk_count=chunk_count,
        language=version.language,
        tags=list(doc.tags),
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        processed_at=version.processed_at,
    )
