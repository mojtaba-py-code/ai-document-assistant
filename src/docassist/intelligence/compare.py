"""Comparison of two versions of a document, or of two documents.

Deterministic part (always returned)
    * the chunks of each version are stitched back into one text - the chunker's overlap is
      removed using the recorded character offsets, or, when those are missing, the longest
      suffix/prefix overlap found with a KMP prefix function;
    * the text is split into lines ("paragraphs"), each mapped back to the chunk it came
      from (page range, section, chunk id);
    * ``difflib.SequenceMatcher`` aligns the two paragraph sequences; every non-equal
      opcode becomes an ``added`` / ``removed`` / ``changed`` hunk with page references,
      and changed hunks carry a word-level inline diff;
    * extracted fields of both versions are compared per field name (payment terms
      changed, expiration moved by N days, ...).

Optional LLM change summary
    The model sees ONLY the diff hunks and field changes (spotlighted, with ids ``H#`` /
    ``F#``), never the full documents, and every change it reports must reference ids that
    were sent - anything else is dropped. The data classification of the call is the
    higher of the two documents' classifications.
"""

from __future__ import annotations

import asyncio
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.audit.service import Actor
from docassist.authz.principal import Principal
from docassist.core.context import utcnow
from docassist.core.enums import Classification, VersionStatus
from docassist.core.errors import ValidationFailed
from docassist.core.text import estimate_tokens, truncate
from docassist.db.models import DocumentVersion
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
    fallback_warning,
    gateway_of,
    input_budget,
    new_nonce,
    output_tokens,
    security_rules,
)
from docassist.intelligence.schemas import (
    ChangeSummary,
    ChangeSummaryItem,
    CompareRequest,
    ComparisonResult,
    DiffHunk,
    DiffStats,
    FieldChange,
    InlineChange,
    PageRef,
    Usage,
    VersionRef,
)
from docassist.intelligence.textops import escape_prompt_text, format_decimal, prompt_attr

if TYPE_CHECKING:
    from docassist.api.container import Container

MIN_OVERLAP = 12
OVERLAP_WINDOW = 2_400
MAX_PARAGRAPHS = 20_000
MAX_COMPARE_CHUNKS = 5_000
MAX_HUNK_TEXT = 2_000
MAX_INLINE_OPS = 200
INLINE_MAX_WORDS = 1_500
INLINE_EQUAL_KEEP = 8
LLM_RATE_BUCKET = "llm_user"


# --------------------------------------------------------------------------- #
# Re-assembling a version's text
# --------------------------------------------------------------------------- #
def overlap_length(tail: str, head: str) -> int:
    """Length of the longest prefix of ``head`` that is also a suffix of ``tail`` (KMP)."""
    size = min(len(tail), len(head), OVERLAP_WINDOW)
    if size == 0:
        return 0
    probe = head[:size].replace("\x00", " ") + "\x00" + tail[-size:].replace("\x00", " ")
    prefix = [0] * len(probe)
    for i in range(1, len(probe)):
        k = prefix[i - 1]
        while k and probe[i] != probe[k]:
            k = prefix[k - 1]
        if probe[i] == probe[k]:
            k += 1
        prefix[i] = k
    return prefix[-1]


def _declared_overlap(prev: ChunkView, chunk: ChunkView, tail: str) -> int | None:
    """Overlap from the chunker's character offsets, if they are present and consistent."""
    if prev.char_end is None or chunk.char_start is None:
        return None
    declared = prev.char_end - chunk.char_start
    if declared <= 0:
        return 0
    if declared <= len(chunk.content) and tail.endswith(chunk.content[:declared]):
        return declared
    return None


def merge_chunks(chunks: Sequence[ChunkView]) -> tuple[str, list[tuple[int, ChunkView]]]:
    """The version's text without chunk overlap, plus (start offset, chunk) segments."""
    parts: list[str] = []
    segments: list[tuple[int, ChunkView]] = []
    length = 0
    tail = ""
    prev: ChunkView | None = None
    for chunk in chunks:
        content = chunk.content
        skip = 0
        if prev is not None:
            declared = _declared_overlap(prev, chunk, tail)
            skip = declared if declared is not None else overlap_length(tail, content)
            if skip < MIN_OVERLAP:
                skip = 0
        piece = content[skip:]
        if prev is not None and skip == 0:
            parts.append("\n")
            length += 1
        segments.append((length, chunk))
        parts.append(piece)
        length += len(piece)
        tail = (tail + piece)[-OVERLAP_WINDOW:]
        prev = chunk
    return "".join(parts), segments


@dataclass(frozen=True, slots=True)
class Paragraph:
    text: str
    ref: PageRef


def paragraphs(chunks: Sequence[ChunkView]) -> tuple[list[Paragraph], bool]:
    """Non-empty lines of the merged text with their chunk/page reference (+ truncated)."""
    text, segments = merge_chunks(chunks)
    if not segments:
        return [], False
    starts = [start for start, _ in segments]
    out: list[Paragraph] = []
    offset = 0
    for line in text.split("\n"):
        start = offset
        offset += len(line) + 1
        display = " ".join(line.split())
        if not display:
            continue
        first = segments[max(0, bisect_right(starts, start) - 1)][1]
        last = segments[max(0, bisect_right(starts, start + len(line) - 1) - 1)][1]
        out.append(
            Paragraph(
                display,
                PageRef(
                    chunk_id=first.id,
                    page_start=first.page_start,
                    page_end=last.page_end if last.page_end is not None else last.page_start,
                    section=first.section,
                ),
            )
        )
        if len(out) >= MAX_PARAGRAPHS:
            return out, True
    return out, False


# --------------------------------------------------------------------------- #
# Paragraph diff
# --------------------------------------------------------------------------- #
def inline_diff(before: str, after: str) -> tuple[list[InlineChange], float]:
    """Word-level diff (merged runs, long equal runs elided) and similarity ratio."""
    words_a = before.split()[:INLINE_MAX_WORDS]
    words_b = after.split()[:INLINE_MAX_WORDS]
    matcher = SequenceMatcher(None, words_a, words_b, autojunk=False)
    ops: list[InlineChange] = []

    def emit(op: str, words: list[str]) -> None:
        if not words:
            return
        if op == "equal" and len(words) > 2 * INLINE_EQUAL_KEEP + 1:
            words = [*words[:INLINE_EQUAL_KEEP], "...", *words[-INLINE_EQUAL_KEEP:]]
        text = truncate(" ".join(words), 300)
        if ops and ops[-1].op == op:
            text = truncate(ops[-1].text + " " + text, 300)
            ops[-1] = InlineChange(op=ops[-1].op, text=text)
        elif len(ops) < MAX_INLINE_OPS:
            ops.append(InlineChange(op=op, text=text))  # type: ignore[arg-type]

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            emit("equal", words_a[i1:i2])
        else:
            emit("delete", words_a[i1:i2])
            emit("insert", words_b[j1:j2])
    return ops, round(matcher.ratio(), 4)


def _span_ref(items: Sequence[Paragraph]) -> PageRef:
    first, last = items[0].ref, items[-1].ref
    return PageRef(
        chunk_id=first.chunk_id,
        page_start=first.page_start,
        page_end=last.page_end,
        section=first.section,
    )


def _joined(items: Sequence[Paragraph]) -> str:
    return truncate("\n".join(p.text for p in items), MAX_HUNK_TEXT)


@dataclass(frozen=True, slots=True)
class DiffOutcome:
    hunks: list[DiffHunk]
    hunks_total: int
    stats: DiffStats
    truncated: bool


def diff_paragraphs(
    before: Sequence[Paragraph], after: Sequence[Paragraph], *, max_hunks: int
) -> DiffOutcome:
    """Align two paragraph sequences and describe every difference as a hunk.

    Stats count hunks per kind (``added``/``removed``/``changed``) and equal paragraphs
    (``unchanged``); ``similarity`` is the paragraph-level ``SequenceMatcher`` ratio. A
    replaced block with the same number of paragraphs on both sides is split into one
    ``changed`` hunk per paragraph pair; otherwise it stays a single hunk.
    """
    matcher = SequenceMatcher(
        None, [p.text for p in before], [p.text for p in after], autojunk=False
    )
    hunks: list[DiffHunk] = []
    counts = {"added": 0, "removed": 0, "changed": 0}
    unchanged = 0
    total = 0

    def add(kind: str, old: Sequence[Paragraph], new: Sequence[Paragraph]) -> None:
        nonlocal total
        total += 1
        counts[kind] += 1
        if len(hunks) >= max_hunks:
            return
        hunk_id = f"H{total}"
        if kind == "removed":
            hunks.append(
                DiffHunk(id=hunk_id, kind="removed", before=_joined(old), before_ref=_span_ref(old))
            )
        elif kind == "added":
            hunks.append(
                DiffHunk(id=hunk_id, kind="added", after=_joined(new), after_ref=_span_ref(new))
            )
        else:
            old_text, new_text = _joined(old), _joined(new)
            inline, similarity = inline_diff(old_text, new_text)
            hunks.append(
                DiffHunk(
                    id=hunk_id,
                    kind="changed",
                    before=old_text,
                    after=new_text,
                    before_ref=_span_ref(old),
                    after_ref=_span_ref(new),
                    similarity=similarity,
                    inline=inline,
                )
            )

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            unchanged += i2 - i1
        elif tag == "delete":
            add("removed", before[i1:i2], [])
        elif tag == "insert":
            add("added", [], after[j1:j2])
        elif i2 - i1 == j2 - j1:
            for offset in range(i2 - i1):
                add(
                    "changed",
                    before[i1 + offset : i1 + offset + 1],
                    after[j1 + offset : j1 + offset + 1],
                )
        else:
            add("changed", before[i1:i2], after[j1:j2])
    stats = DiffStats(
        paragraphs_before=len(before),
        paragraphs_after=len(after),
        added=counts["added"],
        removed=counts["removed"],
        changed=counts["changed"],
        unchanged=unchanged,
        similarity=round(matcher.ratio(), 4) if (before or after) else 1.0,
    )
    return DiffOutcome(hunks, total, stats, total > len(hunks))


# --------------------------------------------------------------------------- #
# Field diff
# --------------------------------------------------------------------------- #
def _field_key(row: FieldRow) -> str:
    display = row.display() or ""
    return " ".join(display.casefold().split())


def field_changes(before: Sequence[FieldRow], after: Sequence[FieldRow]) -> list[FieldChange]:
    """Per field name: values only in ``after`` (added), only in ``before`` (removed), or a
    different set of values on each side (changed). Rules and LLM rows with the same value
    count once (the higher-confidence row represents it). Single-date and single-number
    changes carry ``delta_days`` / ``delta_number``."""

    def grouped(rows: Sequence[FieldRow]) -> dict[str, dict[str, FieldRow]]:
        out: dict[str, dict[str, FieldRow]] = {}
        for row in rows:
            key = _field_key(row)
            if not key:
                continue
            values = out.setdefault(row.field, {})
            current = values.get(key)
            if current is None or row.confidence > current.confidence:
                values[key] = row
        return out

    old, new = grouped(before), grouped(after)
    changes: list[FieldChange] = []
    for name in sorted(set(old) | set(new)):
        a, b = old.get(name, {}), new.get(name, {})
        if set(a) == set(b):
            continue
        kind = "changed" if a and b else ("added" if b else "removed")
        before_rows = [a[k] for k in sorted(a)]
        after_rows = [b[k] for k in sorted(b)]
        delta_days = delta_number = None
        if len(before_rows) == 1 and len(after_rows) == 1:
            x, y = before_rows[0], after_rows[0]
            if x.value_date is not None and y.value_date is not None:
                delta_days = (y.value_date - x.value_date).days
            elif x.value_number is not None and y.value_number is not None:
                delta_number = format_decimal(y.value_number - x.value_number)
        changes.append(
            FieldChange(
                id=f"F{len(changes) + 1}",
                field=name,
                change=kind,  # type: ignore[arg-type]
                before=[r.to_value() for r in before_rows],
                after=[r.to_value() for r in after_rows],
                delta_days=delta_days,
                delta_number=delta_number,
            )
        )
    return changes


# --------------------------------------------------------------------------- #
# LLM change summary
# --------------------------------------------------------------------------- #
class _ChangeOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=2_000)
    refs: list[str] = Field(max_length=20)


class _SummaryOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(max_length=6_000)
    changes: list[_ChangeOut] = Field(max_length=30)


CHANGE_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "maxLength": 2_000},
        "changes": {
            "type": "array",
            "maxItems": 15,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": 500},
                    "refs": {
                        "type": "array",
                        "maxItems": 10,
                        "items": {"type": "string", "maxLength": 8},
                    },
                },
                "required": ["text", "refs"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "changes"],
    "additionalProperties": False,
}


def render_hunk(hunk: DiffHunk, nonce: str) -> str:
    ref = hunk.after_ref or hunk.before_ref
    attrs = [f'id="{hunk.id}"', f'nonce="{nonce}"', f'kind="{hunk.kind}"']
    if ref is not None and ref.page_start is not None:
        attrs.append(f'page="{ref.page_start}"')
    if ref is not None and ref.section:
        attrs.append(f'section="{prompt_attr(ref.section, 120)}"')
    lines = [f"<hunk {' '.join(attrs)}>"]
    if hunk.before is not None:
        lines.append(f"<before>{escape_prompt_text(hunk.before)}</before>")
    if hunk.after is not None:
        lines.append(f"<after>{escape_prompt_text(hunk.after)}</after>")
    lines.append("</hunk>")
    return "\n".join(lines)


def render_field_change(change: FieldChange, nonce: str) -> str:
    before = "; ".join(v.value or "" for v in change.before)
    after = "; ".join(v.value or "" for v in change.after)
    return (
        f'<field-change id="{change.id}" nonce="{nonce}" field="{prompt_attr(change.field, 64)}" '
        f'change="{change.change}">before: {escape_prompt_text(before)} | after: '
        f"{escape_prompt_text(after)}</field-change>"
    )


def parse_change_summary(data: dict[str, Any] | None, valid_refs: set[str]) -> _SummaryOut | None:
    if data is None:
        return None
    try:
        parsed = _SummaryOut.model_validate(data)
    except ValidationError:
        return None
    changes = []
    for change in parsed.changes:
        refs = []
        for raw in change.refs:
            ref = raw.strip().upper()
            if ref in valid_refs and ref not in refs:
                refs.append(ref)
        text = clean_output(change.text, 500)
        if refs and text:
            changes.append(_ChangeOut(text=text, refs=refs))
    return _SummaryOut(summary=clean_output(parsed.summary, 2_000), changes=changes[:15])


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
def _version_ref(doc: DocView, version: VersionView) -> VersionRef:
    return VersionRef(
        document_id=doc.id,
        document_title=doc.title,
        version_id=version.id,
        version_number=version.number,
        classification=doc.classification.value,
    )


async def _previous_version(session: AsyncSession, doc: DocView, number: int) -> VersionView | None:
    row = (
        await session.execute(
            select(DocumentVersion.version_number)
            .where(
                DocumentVersion.document_id == doc.id,
                DocumentVersion.organization_id == doc.org_id,
                DocumentVersion.version_number < number,
                DocumentVersion.status == VersionStatus.INDEXED.value,
            )
            .order_by(DocumentVersion.version_number.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return None if row is None else await load_version(session, doc, row)


def _compute(
    chunks_a: list[ChunkView], chunks_b: list[ChunkView], max_hunks: int
) -> tuple[DiffOutcome, bool]:
    paras_a, cut_a = paragraphs(chunks_a)
    paras_b, cut_b = paragraphs(chunks_b)
    return diff_paragraphs(paras_a, paras_b, max_hunks=max_hunks), cut_a or cut_b


class Comparer:
    def __init__(self, container: Container) -> None:
        self._c = container

    async def compare(self, principal: Principal, request: CompareRequest) -> ComparisonResult:
        """Deterministic diff (+ optional constrained LLM summary) of two readable versions."""
        principal.require_org()
        now = utcnow()
        async with self._c.db.session(principal.db_context) as session:
            doc_a = await load_readable_document(session, principal, request.document_id, now)
            if request.other_document_id is not None:
                doc_b = await load_readable_document(
                    session, principal, request.other_document_id, now
                )
                ver_a = await load_version(session, doc_a, request.base_version)
                ver_b = await load_version(session, doc_b, request.target_version)
                mode = "documents"
            else:
                doc_b = doc_a
                ver_b = await load_version(session, doc_a, request.target_version)
                if request.base_version is None:
                    previous = await _previous_version(session, doc_a, ver_b.number)
                    if previous is None:
                        raise ValidationFailed(
                            "The document has no earlier indexed version to compare with."
                        )
                    ver_a = previous
                else:
                    ver_a = await load_version(session, doc_a, request.base_version)
                if ver_a.id == ver_b.id:
                    raise ValidationFailed("Choose two different versions to compare.")
                mode = "versions"
            chunks_a = await load_chunks(
                session, principal, now, doc_a, ver_a, limit=MAX_COMPARE_CHUNKS
            )
            chunks_b = await load_chunks(
                session, principal, now, doc_b, ver_b, limit=MAX_COMPARE_CHUNKS
            )
            fields_a = await load_fields(session, principal, now, doc_a, ver_a.id)
            fields_b = await load_fields(session, principal, now, doc_b, ver_b.id)
        diff, cut = await asyncio.to_thread(_compute, chunks_a, chunks_b, request.max_hunks)
        changes = field_changes(fields_a, fields_b)
        warnings: list[str] = []
        if cut or len(chunks_a) >= MAX_COMPARE_CHUNKS or len(chunks_b) >= MAX_COMPARE_CHUNKS:
            warnings.append("The documents are very long; only their beginning was compared.")
        if diff.truncated:
            warnings.append(
                f"Showing the first {len(diff.hunks)} of {diff.hunks_total} differences."
            )
        summary: ChangeSummary | None = None
        usage = Usage()
        if request.include_change_summary:
            classification = Classification.highest([doc_a.classification, doc_b.classification])
            summary, usage, warning = await self._change_summary(
                principal, diff.hunks, changes, classification
            )
            if warning:
                warnings.append(warning)
        async with self._c.db.transaction(principal.db_context) as session:
            self._c.audit.record(
                session,
                Actor.of(principal),
                "intelligence.compare",
                resource_type="document",
                resource_id=doc_a.id,
                details={
                    "mode": mode,
                    "base_version_id": str(ver_a.id),
                    "target_document_id": str(doc_b.id),
                    "target_version_id": str(ver_b.id),
                    "hunks": diff.hunks_total,
                    "field_changes": len(changes),
                    "change_summary": summary is not None and summary.model is not None,
                },
            )
        return ComparisonResult(
            mode=mode,  # type: ignore[arg-type]
            base=_version_ref(doc_a, ver_a),
            target=_version_ref(doc_b, ver_b),
            stats=diff.stats,
            hunks=diff.hunks,
            hunks_total=diff.hunks_total,
            truncated=diff.truncated,
            field_changes=changes,
            change_summary=summary,
            warnings=warnings,
            usage=usage,
        )

    async def _change_summary(
        self,
        principal: Principal,
        hunks: list[DiffHunk],
        changes: list[FieldChange],
        classification: Classification,
    ) -> tuple[ChangeSummary | None, Usage, str | None]:
        if not hunks and not changes:
            return ChangeSummary(summary="No differences were found.", changes=[]), Usage(), None
        settings = self._c.settings
        await self._c.limiter.enforce(
            LLM_RATE_BUCKET, str(principal.user_id), settings.rate_limit.llm_per_user
        )
        nonce = new_nonce()
        system = (
            "You describe the differences between two versions of a business document. "
            "You only see the differences: <hunk> elements (removed, added or changed text) "
            "and <field-change> elements (extracted values that changed). Summarise what "
            "changed and why it matters in at most 150 words, then list the most important "
            "changes; every change must cite the ids (for example H2 or F1) of the elements "
            "it is based on. Do not speculate about text you were not shown.\n\n"
            + security_rules(nonce)
        )
        budget = input_budget(settings, system)
        blocks: list[str] = []
        refs: set[str] = set()
        used = 0
        omitted = 0
        items: list[tuple[str, str]] = [(c.id, render_field_change(c, nonce)) for c in changes]
        items += [(h.id, render_hunk(h, nonce)) for h in hunks]
        for ref, block in items:
            cost = estimate_tokens(block)
            if used + cost > budget:
                omitted += 1
                continue
            blocks.append(block)
            refs.add(ref)
            used += cost
        if not blocks:
            return None, Usage(), fallback_warning("context_budget_too_small")
        user = "Differences:\n\n" + "\n\n".join(blocks)
        request = LLMRequest(
            task=LLMTask.COMPARE,
            system=system,
            messages=[ChatMessage(role="user", content=user)],
            output_schema=CHANGE_SUMMARY_SCHEMA,
            max_output_tokens=output_tokens(settings, 1_500),
            data_classification=classification,
        )
        outcome = await call_gateway(gateway_of(self._c), request, principal, feature="compare")
        if outcome.result is None:
            reason = outcome.reason or "llm_unavailable"
            return None, Usage(), _summary_unavailable(reason)
        usage = Usage(
            calls=1,
            input_tokens=int(outcome.result.input_tokens),
            output_tokens=int(outcome.result.output_tokens),
        )
        parsed = parse_change_summary(outcome.data, refs)
        if parsed is None or (not parsed.summary and not parsed.changes):
            return None, usage, _summary_unavailable("llm_output_unverifiable")
        summary = ChangeSummary(
            summary=parsed.summary,
            changes=[ChangeSummaryItem(text=c.text, refs=c.refs) for c in parsed.changes],
            model=outcome.result.model,
            provider=outcome.result.provider,
            prompt_version=PROMPT_VERSION,
        )
        warning = (
            f"{omitted} difference(s) did not fit the AI context and are not covered by the "
            "change summary."
            if omitted
            else None
        )
        return summary, usage, warning


def _summary_unavailable(reason: str) -> str:
    return "Change summary unavailable: " + fallback_warning(reason).split(";")[0] + "."
