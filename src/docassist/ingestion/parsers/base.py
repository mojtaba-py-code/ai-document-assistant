"""Shared parser machinery: errors, budget-enforcing document builder, text helpers.

Runs inside the sandbox child: standard library and :mod:`docassist.ingestion.model` only.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass, field

from docassist.ingestion.model import (
    ERROR_CODES,
    MAX_HEADING_LEVEL,
    MAX_METADATA_CHARS,
    Block,
    DocumentMetadata,
    Limits,
    OcrImage,
    ParsedDocument,
    ParsedPage,
)

TABLE_CELL_SEPARATOR = " | "
_WHITESPACE = re.compile(r"\s+")


class ParseError(Exception):
    """A document cannot be parsed. ``code`` is one of :data:`model.ERROR_CODES`."""

    def __init__(self, code: str, detail: str = "") -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown parse error code {code!r}")
        super().__init__(detail or code)
        self.code = code


def one_line(text: str) -> str:
    """Collapse every whitespace run (including newlines) to one space."""
    return _WHITESPACE.sub(" ", text).strip()


def clip(text: str | None, limit: int = MAX_METADATA_CHARS) -> str | None:
    if text is None:
        return None
    value = one_line(str(text))
    return value[:limit] or None


def split_at_whitespace(text: str, limit: int) -> list[str]:
    """Split ``text`` into pieces of at most ``limit`` chars, preferring whitespace cuts."""
    pieces: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = rest.rfind("\n", 0, limit + 1)
        if cut <= 0:
            cut = limit  # a single "word" longer than the budget
        pieces.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest)
    return [piece for piece in pieces if piece]


def render_row(cells: list[str]) -> str:
    return TABLE_CELL_SEPARATOR.join(one_line(cell) for cell in cells)


@dataclass(slots=True)
class _PageState:
    number: int
    blocks: list[Block] = field(default_factory=list)
    needs_ocr: bool = False


class DocumentBuilder:
    """Collects pages and blocks while enforcing :class:`Limits`.

    When the total text budget is reached the builder records ``text_truncated`` and
    ``exhausted`` becomes true; parsers stop early instead of doing useless work.
    """

    def __init__(self, fmt: str, limits: Limits) -> None:
        self.format = fmt
        self.limits = limits
        self._pages: list[_PageState] = []
        self._warnings: list[str] = []
        self._chars = 0
        self._hidden_chars = 0
        self._blocks = 0
        self._images: list[OcrImage] = []
        self._image_bytes = 0
        self.exhausted = False

    # ------------------------------------------------------------------ pages
    def new_page(self, number: int | None = None) -> None:
        if len(self._pages) >= self.limits.max_pages:
            self.warn("pages_truncated")
            self.exhausted = True
            return
        previous = self._pages[-1].number if self._pages else 0
        self._pages.append(_PageState(number=max(previous + 1, number or 0)))

    @property
    def page_number(self) -> int:
        if not self._pages:
            self.new_page(1)
        return self._pages[-1].number

    def mark_needs_ocr(self) -> None:
        if self._pages:
            self._pages[-1].needs_ocr = True

    def warn(self, code: str) -> None:
        if code not in self._warnings and len(self._warnings) < self.limits.max_warnings:
            self._warnings.append(code)

    # ------------------------------------------------------------------ blocks
    def _take_hidden(self, hidden: str) -> str:
        room = self.limits.max_hidden_chars - self._hidden_chars
        if room <= 0 or not hidden:
            return ""
        taken = hidden[:room]
        self._hidden_chars += len(taken)
        return taken

    def _append(self, kind: str, text: str, level: int | None, hidden: str) -> bool:
        if self.exhausted:
            return False
        if not self._pages:
            self.new_page(1)
        if self._blocks >= self.limits.max_blocks:
            self.warn("text_truncated")
            self.exhausted = True
            return False
        room = self.limits.max_total_chars - self._chars
        if len(text) > room:
            text = text[:room]
            self.warn("text_truncated")
            self.exhausted = True
        if not text.strip():
            return not self.exhausted
        self._pages[-1].blocks.append(Block(kind, text, level, self._take_hidden(hidden)))
        self._chars += len(text)
        self._blocks += 1
        return not self.exhausted

    def add_text(self, kind: str, text: str, *, hidden: str = "") -> bool:
        """Add a paragraph or list block, split at whitespace if it exceeds the block cap."""
        pieces = split_at_whitespace(text.strip(), self.limits.max_block_chars)
        if len(pieces) > 1:
            self.warn("block_split")
        for index, piece in enumerate(pieces):
            if not self._append(kind, piece, None, hidden if index == 0 else ""):
                return False
        return True

    def add_heading(self, text: str, level: int, *, hidden: str = "") -> bool:
        title = one_line(text)
        if not title:
            return True
        if len(title) > 300:  # a "heading" this long is body text
            return self.add_text("paragraph", title, hidden=hidden)
        return self._append("heading", title, min(max(level, 1), MAX_HEADING_LEVEL), hidden)

    def add_table(self, rows: list[list[str]], *, hidden: str = "") -> bool:
        """Render rows as ``a | b`` lines; the first row is the header, repeated per block."""
        lines = [render_row(row) for row in rows]
        lines = [line for line in lines if line.replace("|", "").strip()]
        if not lines:
            return True
        header, body = lines[0], lines[1:]
        cap = self.limits.max_block_chars
        if len(header) > cap // 2:
            header = header[: cap // 2]
            self.warn("table_header_truncated")
        current = [header]
        size = len(header)
        first = True
        for raw_line in body:
            line = raw_line
            if len(line) > cap - len(header) - 1:
                line = line[: cap - len(header) - 1]
                self.warn("table_row_truncated")
            if size + 1 + len(line) > cap:
                if not self._append("table", "\n".join(current), None, hidden if first else ""):
                    return False
                first = False
                current = [header]
                size = len(header)
            current.append(line)
            size += 1 + len(line)
        return self._append("table", "\n".join(current), None, hidden if first else "")

    def add_ocr_image(self, fmt: str, data: bytes) -> bool:
        if len(self._images) >= self.limits.max_ocr_images:
            self.warn("ocr_images_truncated")
            return False
        if self._image_bytes + len(data) > self.limits.max_ocr_bytes:
            self.warn("ocr_images_truncated")
            return False
        self._images.append(OcrImage(page=self.page_number, format=fmt, data=data))
        self._image_bytes += len(data)
        return True

    # ------------------------------------------------------------------ result
    @property
    def char_count(self) -> int:
        return self._chars

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def build(self, metadata: DocumentMetadata) -> ParsedDocument:
        pages = tuple(
            ParsedPage(number=p.number, blocks=tuple(p.blocks), needs_ocr=p.needs_ocr)
            for p in self._pages
        )
        return ParsedDocument(
            format=self.format,
            pages=pages,
            metadata=metadata,
            warnings=tuple(self._warnings),
            needs_ocr=any(p.needs_ocr for p in pages),
            ocr_images=tuple(self._images),
        )


# --------------------------------------------------------------------------- #
# Plain-text helpers (TXT, MD, CSV, PDF text layer)
# --------------------------------------------------------------------------- #
def decode_text(data: bytes, builder: DocumentBuilder) -> str:
    """Decode UTF-8 (with or without BOM) or BOM-marked UTF-16; never fails.

    Invalid UTF-8 is decoded with replacement characters and a ``decoding_replaced``
    warning, so a slightly damaged file still yields its readable text.
    """
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            builder.warn("decoding_replaced")
            return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        builder.warn("decoding_replaced")
        return data.decode("utf-8-sig", errors="replace")


_BULLETS = "".join(
    chr(cp)
    for cp in (0x2022, 0x2013, 0x2014, 0x00B7, 0x25AA, 0x25CF, 0x25E6, 0x2043, 0x2219, 0x25A0)
)
BULLET_LINE = re.compile(r"^\s*(?:[-*+" + _BULLETS + r"]|\(?\d{1,3}[.)]|\(?[a-z][.)])\s+\S")
_NUMBERED_HEADING = re.compile(
    r"^(?:(?P<num>\d{1,2}(?:\.\d{1,2}){0,4})\.?|(?P<roman>[IVXLC]{1,6})\.|"
    r"(?P<word>article|section|chapter|part|schedule|appendix|annex|exhibit)\s+"
    r"(?:\d{1,3}(?:\.\d{1,2})*|[IVXLC]{1,6}|[A-Z])[.:]?)\s+\S",
    re.IGNORECASE,
)


def heading_level(line: str) -> int | None:
    """Conservative heading detector for text without markup; returns a level or ``None``.

    Accepts numbered headings (``2.1 Scope``, ``ARTICLE IV - TERM``, ``Section 3: Fees``) and
    short ALL-CAPS lines. Sentences (terminal ``.``/``,``/``;``), long lines and lines with
    few letters are rejected - a missed heading costs little, a false one mislabels sections.
    """
    text = line.strip()
    if not 3 <= len(text) <= 120 or len(text.split()) > 14:
        return None
    if text.endswith((".", ",", ";")):
        return None
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 3:
        return None
    match = _NUMBERED_HEADING.match(text)
    if match:
        rest = text[match.end() - 1 :]
        if rest[:1].islower():
            return None  # "1 apple, 2 pears" style enumerations
        if match.group("num"):
            return min(match.group("num").count(".") + 1, MAX_HEADING_LEVEL)
        return 1
    if all(not ch.islower() for ch in letters) and len(letters) >= 4 and len(text.split()) <= 10:
        return 1
    return None


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines into one paragraph, removing end-of-line hyphenation."""
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}"
    return out
