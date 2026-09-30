"""Structure-aware, deterministic chunking.

Model
-----
The normalised blocks are joined into one *canonical text* (blocks separated by a blank
line, table rows by a newline). Every chunk is a contiguous span of that text,
``text[char_start:char_end]``, optionally preceded by a *prefix*: the table header line
repeated at the top of a chunk that continues a table. So ``chunk.text == prefix +
canonical[char_start:char_end]`` always holds, and offsets map any position in a chunk
back to the document.

Guarantees (property-tested)
----------------------------
* **nothing is lost** - every non-whitespace character of the canonical text lies inside
  at least one chunk span;
* **bounded size** - ``chunk.token_count == estimate_tokens(chunk.text) <= max_tokens``;
* **order** - chunk starts and ends are strictly increasing; consecutive chunks overlap by
  at most ``overlap_tokens``;
* **word integrity** - a chunk never starts or ends inside a word, except inside a run of
  non-space characters that alone exceeds the size budget (a base64 blob, unspaced CJK);
* **deterministic** - same blocks and parameters, same chunks.

Strategy
--------
Headings start a new chunk (their text opens it; ``heading_path``/``section`` describe it),
unless the chunk so far is tiny, in which case small sections share a chunk. A block that
fits in ``max_tokens`` is not split - unless its heading(s) plus the block exceed
``max_tokens``, in which case the headings stay with the block's beginning. Blocks are packed
greedily up to ``target_tokens``.
Larger blocks are split at sentence, then word boundaries into ``target``-sized chunks with
``overlap_tokens`` of trailing context carried into the next chunk. Tables are split
between rows only, each continuation chunk repeating the header line. A tiny trailing chunk
is merged into its predecessor when both belong to the same section and the result fits.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from docassist.core.text import estimate_tokens
from docassist.ingestion.normalize import TextBlock

if TYPE_CHECKING:
    from docassist.core.config import ChunkingSettings

BLOCK_SEPARATOR = "\n\n"
MAX_HEADING_CHARS = 300
MAX_HEADING_DEPTH = 10
MAX_CHUNK_HIDDEN_CHARS = 4_000
_WIDE_FROM = 0x2FFF  # estimate_tokens counts characters above this as one token each
_TEXT_KINDS = ("paragraph", "list")
_SENTENCE_BREAK = re.compile(
    r"[.!?"
    + chr(0x2026)
    + chr(0x3002)
    + r"]+[\"'"
    + chr(0x201D)
    + chr(0x2019)
    + r")\]]*(\s+)|(\n\s*)"
)
_WORD = re.compile(r"\S+")


def _est(length: int, wide: int) -> int:
    """``estimate_tokens`` for a string of ``length`` chars with ``wide`` chars > U+2FFF."""
    if length <= 0:
        return 0
    return max(1, (length - wide) // 4 + wide)


def _wide(text: str, start: int, end: int) -> int:
    return sum(1 for ch in text[start:end] if ord(ch) > _WIDE_FROM)


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink ``[start, end)`` to exclude surrounding whitespace; ``None`` if nothing is left."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


@dataclass(frozen=True, slots=True)
class ChunkingParams:
    target_tokens: int
    max_tokens: int
    overlap_tokens: int

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_tokens < self.target_tokens <= self.max_tokens:
            raise ValueError("require 0 <= overlap_tokens < target_tokens <= max_tokens")

    @classmethod
    def from_settings(cls, settings: ChunkingSettings) -> ChunkingParams:
        return cls(settings.target_tokens, settings.max_tokens, settings.overlap_tokens)

    @property
    def min_tokens(self) -> int:
        return max(1, self.target_tokens // 4)


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    text: str
    char_start: int
    char_end: int
    prefix_chars: int
    page_start: int
    page_end: int
    heading_path: tuple[str, ...]
    section: str | None
    block_types: tuple[str, ...]
    token_count: int
    hidden_text: str = ""
    channel_flags: tuple[str, ...] = ()


@dataclass(slots=True)
class _Unit:
    start: int
    end: int
    block: int
    kind: str
    page: int
    path: tuple[str, ...]
    wide: int
    heading: bool = False
    table: int | None = None
    header: bool = False
    first_of_block: bool = False


@dataclass(slots=True)
class _Block:
    start: int
    end: int
    kind: str
    tokens: int = 0
    content_end: int = 0
    wide_from_first: int = 0
    header: str | None = None  # table header line repeated in continuation chunks


@dataclass(slots=True)
class _Open:
    start: int
    prefix: str
    prefix_wide: int
    units: list[_Unit] = field(default_factory=list)
    wide: int = 0

    @property
    def end(self) -> int:
        return self.units[-1].end

    def tokens(self) -> int:
        return _est(len(self.prefix) + self.end - self.start, self.prefix_wide + self.wide)

    @property
    def only_headings(self) -> bool:
        return all(unit.heading for unit in self.units)


@dataclass(slots=True)
class _Span:
    start: int
    end: int
    prefix: str
    units: list[_Unit]


@dataclass(frozen=True, slots=True)
class ChunkedDocument:
    """Chunks plus the canonical text and position lookups used by field extraction."""

    text: str
    chunks: tuple[Chunk, ...]
    unit_starts: tuple[int, ...] = ()
    unit_pages: tuple[int, ...] = ()
    chunk_ends: tuple[int, ...] = ()

    def page_at(self, offset: int) -> int | None:
        index = bisect.bisect_right(self.unit_starts, offset) - 1
        if index < 0:
            return self.unit_pages[0] if self.unit_pages else None
        return self.unit_pages[index]

    def chunk_at(self, offset: int) -> Chunk | None:
        """The earliest chunk whose span contains ``offset`` (``None`` in a separator gap)."""
        index = bisect.bisect_right(self.chunk_ends, offset)
        if index < len(self.chunks) and self.chunks[index].char_start <= offset:
            return self.chunks[index]
        return None


class Chunker:
    def __init__(self, params: ChunkingParams) -> None:
        self.p = params

    # ------------------------------------------------------------------ units
    def _pieces(self, text: str, start: int, end: int, budget: int) -> list[tuple[int, int, int]]:
        """Split ``text[start:end]`` into (start, end, wide) pieces of at most ``budget`` tokens."""
        wide = _wide(text, start, end)
        if _est(end - start, wide) <= budget:
            return [(start, end, wide)]
        pieces: list[tuple[int, int, int]] = []
        cur_start = cur_end = -1
        cur_wide = 0
        for match in _WORD.finditer(text, start, end):
            w_start, w_end = match.span()
            w_wide = _wide(text, w_start, w_end)
            if cur_start >= 0:
                if _est(w_end - cur_start, cur_wide + w_wide) <= budget:
                    cur_end, cur_wide = w_end, cur_wide + w_wide
                    continue
                pieces.append((cur_start, cur_end, cur_wide))
                cur_start = -1
            if _est(w_end - w_start, w_wide) <= budget:
                cur_start, cur_end, cur_wide = w_start, w_end, w_wide
                continue
            pieces.extend(self._hard_split(text, w_start, w_end, budget))
        if cur_start >= 0:
            pieces.append((cur_start, cur_end, cur_wide))
        return pieces

    @staticmethod
    def _hard_split(text: str, start: int, end: int, budget: int) -> list[tuple[int, int, int]]:
        """Cut a single over-long word into budget-sized runs (the only intra-word split)."""
        pieces: list[tuple[int, int, int]] = []
        piece_start, wide = start, 0
        for pos in range(start, end):
            char_wide = 1 if ord(text[pos]) > _WIDE_FROM else 0
            if _est(pos + 1 - piece_start, wide + char_wide) > budget:
                pieces.append((piece_start, pos, wide))
                piece_start, wide = pos, 0
            wide += char_wide
        pieces.append((piece_start, end, wide))
        return pieces

    def _sentences(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = start
        for match in _SENTENCE_BREAK.finditer(text, start, end):
            group = 1 if match.group(1) is not None else 2
            gap_start, gap_end = match.span(group)
            if gap_start > cursor:
                spans.append((cursor, gap_start))
            cursor = gap_end
        if cursor < end:
            spans.append((cursor, end))
        return [trimmed for s, e in spans if (trimmed := _trim(text, s, e))]

    def _build(
        self, blocks: Sequence[TextBlock]
    ) -> tuple[str, list[_Unit], list[_Block], list[TextBlock]]:
        parts: list[str] = []
        pos = 0
        kept: list[TextBlock] = []
        spans: list[tuple[int, int]] = []
        for block in blocks:
            if not block.text.strip():
                continue
            if kept:
                parts.append(BLOCK_SEPARATOR)
                pos += len(BLOCK_SEPARATOR)
            parts.append(block.text)
            spans.append((pos, pos + len(block.text)))
            pos += len(block.text)
            kept.append(block)
        text = "".join(parts)

        units: list[_Unit] = []
        infos: list[_Block] = []
        stack: list[tuple[int, str]] = []
        path: tuple[str, ...] = ()
        for index, (block, (b_start, b_end)) in enumerate(zip(kept, spans, strict=True)):
            info = _Block(b_start, b_end, block.kind)
            infos.append(info)
            first = len(units)
            kind = block.kind
            if kind == "heading":
                title = " ".join(block.text.split())
                wide = _wide(text, b_start, b_end)
                if _est(b_end - b_start, wide) <= max(1, self.p.target_tokens // 2):
                    level = block.level or 1
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title[:MAX_HEADING_CHARS]))
                    path = tuple(t for _, t in stack)[:MAX_HEADING_DEPTH]
                    units.append(
                        _Unit(b_start, b_end, index, kind, block.page, path, wide, heading=True)
                    )
                else:
                    kind = "paragraph"  # a "heading" too long to be one is body text
            if kind == "table":
                self._table_units(text, info, units, index=index, page=block.page, path=path)
            elif kind != "heading":
                for s, e in self._sentences(text, b_start, b_end):
                    for ps, pe, pw in self._pieces(text, s, e, self.p.target_tokens):
                        units.append(_Unit(ps, pe, index, kind, block.page, path, pw))
            if len(units) > first:
                units[first].first_of_block = True
                info.content_end = units[-1].end
                info.wide_from_first = _wide(text, units[first].start, info.content_end)
                info.tokens = _est(info.content_end - units[first].start, info.wide_from_first)
        return text, units, infos, kept

    def _table_units(
        self,
        text: str,
        info: _Block,
        units: list[_Unit],
        *,
        index: int,
        page: int,
        path: tuple[str, ...],
    ) -> None:
        lines: list[tuple[int, int]] = []
        cursor = info.start
        for line in text[info.start : info.end].split("\n"):
            trimmed = _trim(text, cursor, cursor + len(line))
            cursor += len(line) + 1
            if trimmed:
                lines.append(trimmed)
        if not lines:
            return
        h_start, h_end = lines[0]
        header = text[h_start:h_end]
        prefix_tokens = estimate_tokens(header + "\n")
        if len(lines) > 1 and prefix_tokens <= self.p.max_tokens // 4:
            info.header = header
        budget = self.p.target_tokens
        if info.header is not None:
            budget = max(1, min(budget, self.p.max_tokens - prefix_tokens - 1))
        for n, (s, e) in enumerate(lines):
            for ps, pe, pw in self._pieces(text, s, e, budget):
                units.append(
                    _Unit(ps, pe, index, "table", page, path, pw, table=index, header=n == 0)
                )

    # ------------------------------------------------------------------ packing
    def _fits(self, text: str, cur: _Open, end: int, extra_wide: int, limit: int) -> bool:
        gap = _wide(text, cur.end, end) if extra_wide < 0 else extra_wide
        return _est(len(cur.prefix) + end - cur.start, cur.prefix_wide + cur.wide + gap) <= limit

    def _add(self, text: str, cur: _Open, unit: _Unit) -> None:
        cur.wide += _wide(text, cur.end, unit.start) + unit.wide
        cur.units.append(unit)

    def _overlap_start(self, text: str, low: int, high: int, budget: int) -> int | None:
        """Smallest word start ``p`` in ``[low, high)`` with ``tokens(text[p:high]) <= budget``."""
        if budget <= 0 or low >= high:
            return None
        window = max(low, high - budget * 4 - 8)
        wide_after = [0] * (high - window + 1)
        for pos in range(high - 1, window - 1, -1):
            wide_after[pos - window] = wide_after[pos - window + 1] + (
                1 if ord(text[pos]) > _WIDE_FROM else 0
            )
        for pos in range(window, high):
            if text[pos].isspace() or (pos > 0 and not text[pos - 1].isspace()):
                continue
            if _est(high - pos, wide_after[pos - window]) <= budget:
                return pos
        return None

    def _open(
        self,
        text: str,
        unit: _Unit,
        infos: list[_Block],
        previous: _Span | None,
        *,
        reason: str,
        carry: list[_Unit],
    ) -> _Open:
        members = [*carry, unit]
        info = infos[unit.block]
        prefix = ""
        if not carry and unit.table is not None and not unit.header and info.header is not None:
            prefix = info.header + "\n"
        cur = _Open(
            start=members[0].start, prefix=prefix, prefix_wide=_wide(prefix, 0, len(prefix))
        )
        cur.units = [members[0]]
        cur.wide = members[0].wide
        for extra in members[1:]:
            self._add(text, cur, extra)
        if (
            reason in ("block", "continuation")
            and not carry
            and previous is not None
            and self.p.overlap_tokens > 0
            and unit.kind in _TEXT_KINDS
        ):
            last = previous.units[-1]
            if last.kind in _TEXT_KINDS and last.path == unit.path and not last.heading:
                anticipated = (
                    info.tokens
                    if reason == "block" and info.tokens <= self.p.max_tokens
                    else _est(unit.end - unit.start, unit.wide)
                )
                budget = min(self.p.overlap_tokens, self.p.max_tokens - anticipated - 1)
                low = max(previous.start + 1, infos[last.block].start)
                start = self._overlap_start(text, low, unit.start, budget)
                if start is not None:
                    cur.wide += _wide(text, start, unit.start)
                    cur.start = start
        return cur

    def _span_tokens(self, text: str, span: _Span) -> int:
        return _est(
            len(span.prefix) + span.end - span.start,
            _wide(span.prefix, 0, len(span.prefix)) + _wide(text, span.start, span.end),
        )

    def _close(self, text: str, cur: _Open, spans: list[_Span]) -> _Span:
        span = _Span(cur.start, cur.end, cur.prefix, list(cur.units))
        if spans and not span.units[0].heading and cur.tokens() < self.p.min_tokens:
            previous = spans[-1]
            if previous.units[-1].path == span.units[0].path:
                merged = _Span(
                    previous.start, span.end, previous.prefix, previous.units + span.units
                )
                if self._span_tokens(text, merged) <= self.p.max_tokens:
                    spans[-1] = merged
                    return merged
        spans.append(span)
        return span

    def _pack(self, text: str, units: list[_Unit], infos: list[_Block]) -> list[_Span]:
        spans: list[_Span] = []
        cur: _Open | None = None
        previous: _Span | None = None
        p = self.p
        for unit in units:
            if cur is None:
                cur = self._open(text, unit, infos, previous, reason="start", carry=[])
                continue
            if unit.heading:
                if not cur.only_headings and cur.tokens() >= p.min_tokens:
                    previous = self._close(text, cur, spans)
                    cur = self._open(text, unit, infos, previous, reason="section", carry=[])
                elif self._fits(text, cur, unit.end, -1, p.max_tokens):
                    self._add(text, cur, unit)
                else:
                    previous = self._close(text, cur, spans)
                    cur = self._open(text, unit, infos, previous, reason="section", carry=[])
                continue

            info = infos[unit.block]
            if unit.first_of_block:
                gap = _wide(text, cur.end, unit.start) + info.wide_from_first
                if self._fits(text, cur, info.content_end, gap, p.target_tokens):
                    self._add(text, cur, unit)
                    continue
                if cur.tokens() < p.min_tokens and self._fits(
                    text, cur, info.content_end, gap, p.max_tokens
                ):
                    self._add(text, cur, unit)
                    continue
                if cur.only_headings and self._fits(text, cur, unit.end, -1, p.max_tokens):
                    self._add(text, cur, unit)
                    continue
                if info.tokens > p.max_tokens and self._fits(
                    text, cur, unit.end, -1, p.target_tokens
                ):
                    self._add(text, cur, unit)  # a big block starts filling the current chunk
                    continue
                reason = "block"
            else:
                limit = p.target_tokens if info.tokens > p.max_tokens else p.max_tokens
                if self._fits(text, cur, unit.end, -1, limit):
                    self._add(text, cur, unit)
                    continue
                reason = "continuation"
            carry: list[_Unit] = []
            if not cur.only_headings:
                # headings at the end of a full chunk belong to the content that follows them
                while cur.units[-1].heading:
                    carry.insert(0, cur.units.pop())
                if carry:
                    cur.wide = _wide(text, cur.start, cur.end)
            previous = self._close(text, cur, spans)
            cur = self._open(text, unit, infos, previous, reason=reason, carry=carry)
            if cur.tokens() > p.max_tokens and carry:
                # the carried headings do not fit with this unit: give them their own chunk
                heading_only = _Open(
                    carry[0].start, "", 0, list(carry), _wide(text, carry[0].start, carry[-1].end)
                )
                previous = self._close(text, heading_only, spans)
                cur = self._open(text, unit, infos, previous, reason=reason, carry=[])
        if cur is not None:
            self._close(text, cur, spans)
        return spans

    # ------------------------------------------------------------------ public
    def chunk(self, blocks: Sequence[TextBlock]) -> ChunkedDocument:
        text, units, infos, kept = self._build(blocks)
        if not units:
            return ChunkedDocument(text=text, chunks=())
        spans = self._pack(text, units, infos)
        unit_starts = [unit.start for unit in units]
        chunks: list[Chunk] = []
        for index, span in enumerate(spans):
            body = span.prefix + text[span.start : span.end]
            first = bisect.bisect_right(unit_starts, span.start) - 1
            covered = units[max(0, first) : bisect.bisect_left(unit_starts, span.end)]
            content = [u for u in span.units if not u.heading]
            path = content[0].path if content else span.units[-1].path
            block_ids = sorted({u.block for u in covered})
            hidden = "\n".join(
                dict.fromkeys(kept[b].hidden_text for b in block_ids if kept[b].hidden_text)
            )
            flags = sorted({flag for b in block_ids for flag in kept[b].channel_flags})
            chunks.append(
                Chunk(
                    index=index,
                    text=body,
                    char_start=span.start,
                    char_end=span.end,
                    prefix_chars=len(span.prefix),
                    page_start=covered[0].page if covered else span.units[0].page,
                    page_end=span.units[-1].page,
                    heading_path=path,
                    section=path[-1] if path else None,
                    block_types=tuple(sorted({u.kind for u in covered})),
                    token_count=estimate_tokens(body),
                    hidden_text=hidden[:MAX_CHUNK_HIDDEN_CHARS],
                    channel_flags=tuple(flags),
                )
            )
        return ChunkedDocument(
            text=text,
            chunks=tuple(chunks),
            unit_starts=tuple(unit_starts),
            unit_pages=tuple(unit.page for unit in units),
            chunk_ends=tuple(chunk.char_end for chunk in chunks),
        )


def chunk_blocks(blocks: Sequence[TextBlock], params: ChunkingParams) -> ChunkedDocument:
    return Chunker(params).chunk(blocks)
