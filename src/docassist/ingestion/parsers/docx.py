"""DOCX parser (python-docx; lxml with entity resolution disabled).

* Body paragraphs and tables are read in document order.
* ``Heading N`` / ``Title`` styles and explicit outline levels become heading blocks;
  ``List ...`` styles and numbered paragraphs become list blocks; tables become table
  blocks (header row repeated per block, horizontally merged cells de-duplicated, nested table text
  flattened into its cell).
* Pages: Word stores *rendered* page breaks when it saves a file; when present they give
  real page numbers (``page_basis = rendered_breaks``). Otherwise hard page breaks are used
  (``page_breaks``), otherwise one page per top-level heading group (``sections``), else 1.
* Text hidden from a reader - ``w:vanish`` runs, white text, fonts of 2pt or less - stays
  in the block text (it is part of the document) but is also reported in ``Block.hidden`` so
  the injection scanner treats it as a hidden channel.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from docassist.ingestion.model import DocumentMetadata, Limits, ParsedDocument
from docassist.ingestion.parsers.base import DocumentBuilder, ParseError, clip, one_line

_W_P = qn("w:p")
_W_T = qn("w:t")
_W_R = qn("w:r")
_W_BR = qn("w:br")
_W_TYPE = qn("w:type")
_W_VAL = qn("w:val")
_W_RENDERED = qn("w:lastRenderedPageBreak")
_HIDDEN_COLORS = frozenset({"FFFFFF", "FEFEFE", "FDFDFD"})
_TINY_FONT_HALF_POINTS = 4


@dataclass(slots=True)
class _Paragraph:
    kind: str
    text: str
    level: int | None
    hidden: str
    breaks_before: int
    breaks_after: int


def _style_name(paragraph: Paragraph) -> str:
    try:
        style = paragraph.style
    except (KeyError, ValueError, AttributeError):
        return ""
    return str(style.name or "") if style is not None else ""


def _outline_level(paragraph: Paragraph) -> int | None:
    ppr = paragraph._p.pPr
    if ppr is None:
        return None
    node = ppr.find(qn("w:outlineLvl"))
    if node is None:
        return None
    value = node.get(_W_VAL, "")
    return int(value) + 1 if value.isdigit() and int(value) < 9 else None


def _run_is_hidden(run: Any) -> bool:
    rpr = run.find(qn("w:rPr"))
    if rpr is None:
        return False
    vanish = rpr.find(qn("w:vanish"))
    if vanish is not None and vanish.get(_W_VAL, "true").lower() not in ("0", "false", "off"):
        return True
    color = rpr.find(qn("w:color"))
    if color is not None and color.get(_W_VAL, "").upper() in _HIDDEN_COLORS:
        return True
    size = rpr.find(qn("w:sz"))
    value = size.get(_W_VAL, "") if size is not None else ""
    return value.isdigit() and int(value) <= _TINY_FONT_HALF_POINTS


def _hidden_text(element: Any) -> str:
    parts = [
        "".join(t.text or "" for t in run.iter(_W_T))
        for run in element.iter(_W_R)
        if _run_is_hidden(run)
    ]
    return one_line(" ".join(parts))


def _breaks(element: Any, rendered: bool) -> tuple[int, int]:
    """Page breaks in a paragraph: (breaks before any text, breaks after text started)."""
    before = after = 0
    seen_text = False
    for node in element.iter():
        if node.tag == _W_T and (node.text or "").strip():
            seen_text = True
        is_break = (
            node.tag == _W_RENDERED
            if rendered
            else node.tag == _W_BR and node.get(_W_TYPE) == "page"
        )
        if is_break:
            if seen_text:
                after += 1
            else:
                before += 1
    return before, after


def _classify(paragraph: Paragraph) -> tuple[str, int | None]:
    style = _style_name(paragraph)
    lowered = style.lower()
    if lowered == "title":
        return "heading", 1
    if lowered.startswith("heading"):
        suffix = lowered.removeprefix("heading").strip()
        if suffix.isdigit() and 1 <= int(suffix) <= 9:
            return "heading", int(suffix)
    outline = _outline_level(paragraph)
    if outline is not None:
        return "heading", outline
    ppr = paragraph._p.pPr
    if lowered.startswith("list") or (ppr is not None and ppr.find(qn("w:numPr")) is not None):
        return "list", None
    return "paragraph", None


def _table_rows(table: Table, limits: Limits, builder: DocumentBuilder) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_index, row in enumerate(table.rows):
        if row_index >= limits.max_sheet_rows:
            builder.warn("rows_truncated")
            break
        cells: list[str] = []
        previous = None
        for cell in row.cells:
            if cell._tc is previous:
                continue  # horizontally merged cell repeated by python-docx
            previous = cell._tc
            if len(cells) >= limits.max_sheet_cols:
                builder.warn("columns_truncated")
                break
            texts = ("".join(t.text or "" for t in p.iter(_W_T)) for p in cell._tc.iter(_W_P))
            cells.append(one_line(" ".join(texts)))
        rows.append(cells)
    return rows


def _paragraphs(document: Any, rendered: bool) -> Iterator[_Paragraph | Table]:
    for item in document.iter_inner_content():
        if isinstance(item, Table):
            yield item
            continue
        kind, level = _classify(item)
        before, after = _breaks(item._p, rendered)
        yield _Paragraph(kind, item.text, level, _hidden_text(item._p), before, after)


def _has(document: Any, tag: str, **attrs: str) -> bool:
    for node in document.element.body.iter(tag):
        if all(node.get(k) == v for k, v in attrs.items()):
            return True
    return False


def parse_docx(data: bytes, limits: Limits) -> ParsedDocument:
    builder = DocumentBuilder("docx", limits)
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ParseError("parse_error", "unreadable docx") from exc

    rendered = _has(document, _W_RENDERED)
    hard = not rendered and _has(document, _W_BR, **{_W_TYPE: "page"})
    items = list(_paragraphs(document, rendered))
    heading_levels = [
        i.level for i in items if isinstance(i, _Paragraph) and i.kind == "heading" and i.level
    ]
    if rendered or hard:
        basis = "rendered_breaks" if rendered else "page_breaks"
    else:
        basis = "sections" if heading_levels else "single"
    top_level = min(heading_levels, default=1)

    page = 1
    content_on_page = False
    builder.new_page(page)
    items_list: list[str] = []

    def flush_list() -> bool:
        if not items_list:
            return True
        ok = builder.add_text("list", "\n".join(items_list))
        items_list.clear()
        return ok

    def move_to(number: int) -> None:
        nonlocal page, content_on_page
        if number != page:
            page = number
            builder.new_page(page)
            content_on_page = False

    for item in items:
        if builder.exhausted:
            break
        if isinstance(item, Table):
            if not flush_list():
                break
            builder.add_table(_table_rows(item, limits, builder))
            content_on_page = True
            continue
        if (
            basis == "sections"
            and item.kind == "heading"
            and item.level == top_level
            and content_on_page
        ):
            flush_list()
            move_to(page + 1)
        if item.breaks_before and basis in ("rendered_breaks", "page_breaks"):
            flush_list()
            move_to(page + item.breaks_before)
        text = item.text.strip()
        if text:
            if item.kind == "list":
                items_list.append(one_line(text))
            else:
                if not flush_list():
                    break
                if item.kind == "heading":
                    builder.add_heading(text, item.level or 1, hidden=item.hidden)
                else:
                    builder.add_text("paragraph", text, hidden=item.hidden)
            content_on_page = True
        if item.breaks_after and basis in ("rendered_breaks", "page_breaks"):
            flush_list()
            move_to(page + item.breaks_after)
    flush_list()

    if builder.char_count == 0:
        raise ParseError("empty_document")
    return builder.build(_metadata(document, builder.page_count, basis))


def _metadata(document: Any, page_count: int, basis: str) -> DocumentMetadata:
    try:
        props = document.core_properties
        return DocumentMetadata(
            title=clip(props.title),
            author=clip(props.author),
            subject=clip(props.subject),
            created=props.created.isoformat() if props.created else None,
            modified=props.modified.isoformat() if props.modified else None,
            page_count=page_count,
            page_basis=basis,
        )
    except Exception:  # noqa: BLE001 - a malformed core.xml must not sink the text
        return DocumentMetadata(page_count=page_count, page_basis=basis)
