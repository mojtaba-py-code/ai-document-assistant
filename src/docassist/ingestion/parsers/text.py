"""Plain text (``.txt``) and Markdown (``.md``) parsers.

* TXT - form feeds (``\\f``) separate pages; blank lines separate paragraphs; bullet lines
  become list blocks; standalone short numbered/ALL-CAPS lines become headings; groups of
  ``|``-separated lines become tables; wrapped prose is re-joined while short structured
  lines keep their line breaks.
* MD  - ATX (``# Title``) and setext headings, fenced code kept verbatim, bullet/numbered
  lists, pipe tables (the delimiter row is dropped), block quotes; inline markup is kept as
  written so the injection scanner still sees e.g. markdown image URLs.
"""

from __future__ import annotations

import re

from docassist.ingestion.model import DocumentMetadata, Limits, ParsedDocument
from docassist.ingestion.parsers.base import (
    BULLET_LINE,
    DocumentBuilder,
    ParseError,
    decode_text,
    heading_level,
    join_lines,
)

_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_ATX_EMPTY = re.compile(r"^ {0,3}(#{1,6})[ \t]*$")
_SETEXT_1 = re.compile(r"^ {0,3}=+[ \t]*$")
_SETEXT_2 = re.compile(r"^ {0,3}-+[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_RULE = re.compile(r"^ {0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$")
_TABLE_DELIMITER = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
_QUOTE = re.compile(r"^ {0,3}>[ ]?")


def _finish(builder: DocumentBuilder, basis: str) -> ParsedDocument:
    if builder.char_count == 0:
        raise ParseError("empty_document")
    return builder.build(DocumentMetadata(page_count=builder.page_count, page_basis=basis))


# --------------------------------------------------------------------------- #
# TXT
# --------------------------------------------------------------------------- #
WRAPPED_LINE_CHARS = 60


def _txt_paragraph(builder: DocumentBuilder, lines: list[str]) -> bool:
    """One blank-line-separated group of lines -> heading, list, table or paragraph.

    Lines are re-joined only when the group looks like wrapped prose (some line at least
    :data:`WRAPPED_LINE_CHARS` long); short structured lines (``Invoice No: 42``) keep their
    line breaks, and groups where every line has a ``|`` become a table.
    """
    if not lines:
        return True
    if len(lines) == 1:
        level = heading_level(lines[0])
        if level is not None:
            return builder.add_heading(lines[0], level)
    if all(BULLET_LINE.match(line) for line in lines):
        return builder.add_text("list", "\n".join(line.strip() for line in lines))
    if len(lines) > 1 and all("|" in line for line in lines):
        return builder.add_table([_table_cells(line) for line in lines])
    if any(len(line.strip()) >= WRAPPED_LINE_CHARS for line in lines):
        return builder.add_text("paragraph", join_lines(lines))
    return builder.add_text("paragraph", "\n".join(line.strip() for line in lines))


def parse_txt(data: bytes, limits: Limits) -> ParsedDocument:
    builder = DocumentBuilder("txt", limits)
    text = decode_text(data, builder)
    pages = text.replace("\r\n", "\n").replace("\r", "\n").split("\f")
    for page_text in pages:
        builder.new_page()
        if builder.exhausted:
            break
        paragraph: list[str] = []
        for line in page_text.split("\n"):
            if line.strip():
                paragraph.append(line)
                continue
            if not _txt_paragraph(builder, paragraph):
                return _finish(builder, "pages")
            paragraph = []
        if not _txt_paragraph(builder, paragraph):
            break
    basis = "pages" if len(pages) > 1 else "single"
    return _finish(builder, basis)


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped)]


class _MarkdownParser:
    def __init__(self, builder: DocumentBuilder) -> None:
        self.b = builder
        self.paragraph: list[str] = []
        self.items: list[str] = []

    def flush(self) -> bool:
        ok = True
        if self.paragraph:
            ok = self.b.add_text("paragraph", join_lines(self.paragraph))
            self.paragraph = []
        if ok and self.items:
            ok = self.b.add_text("list", "\n".join(self.items))
            self.items = []
        return ok

    def run(self, lines: list[str]) -> None:
        index = 0
        while index < len(lines) and not self.b.exhausted:
            line = lines[index]
            nxt = lines[index + 1] if index + 1 < len(lines) else ""
            fence = _FENCE.match(line)
            if fence:
                self.flush()
                marker = fence.group(1)
                body: list[str] = []
                index += 1
                while index < len(lines) and not lines[index].strip().startswith(marker):
                    body.append(lines[index])
                    index += 1
                self.b.add_text("paragraph", "\n".join(body))
                index += 1
                continue
            atx = _ATX.match(line)
            if atx or _ATX_EMPTY.match(line):
                self.flush()
                if atx:
                    self.b.add_heading(atx.group(2), len(atx.group(1)))
                index += 1
                continue
            if not line.strip():
                self.flush()
                index += 1
                continue
            if "|" in line and _TABLE_DELIMITER.match(nxt) and not self.paragraph:
                self.flush()
                rows = [_table_cells(line)]
                index += 2
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    rows.append(_table_cells(lines[index]))
                    index += 1
                self.b.add_table(rows)
                continue
            setext = _SETEXT_1.match(nxt) or _SETEXT_2.match(nxt)
            if setext and not self.items and not BULLET_LINE.match(line) and not _RULE.match(line):
                self.paragraph.append(_QUOTE.sub("", line))
                heading = join_lines(self.paragraph)
                self.paragraph = []
                self.b.add_heading(heading, 1 if _SETEXT_1.match(nxt) else 2)
                index += 2
                continue
            if _RULE.match(line):
                self.flush()
                index += 1
                continue
            if BULLET_LINE.match(line):
                if self.paragraph:
                    self.flush()
                self.items.append(line.strip())
                index += 1
                continue
            if self.items and line.startswith((" ", "\t")):
                self.items[-1] = f"{self.items[-1]} {line.strip()}"
                index += 1
                continue
            if self.items:
                self.flush()
            self.paragraph.append(_QUOTE.sub("", line))
            index += 1
        self.flush()


def parse_md(data: bytes, limits: Limits) -> ParsedDocument:
    builder = DocumentBuilder("md", limits)
    text = decode_text(data, builder)
    builder.new_page(1)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    _MarkdownParser(builder).run(lines)
    return _finish(builder, "single")
