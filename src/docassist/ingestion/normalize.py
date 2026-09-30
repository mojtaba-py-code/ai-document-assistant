"""Normalisation of parser output: Unicode hygiene per block and a hidden-channel report.

Every block goes through :func:`docassist.core.text.sanitize_text` (invisible characters
removed, NFKC). Lone surrogates, which a hostile or broken text layer can smuggle through
JSON escapes and which PostgreSQL would reject, are dropped as well. What was removed is not
forgotten: the per-block counts become hidden-channel flags for the injection scanner; text
decoded from Unicode tag characters, the parser's "hidden" text (e.g. vanished DOCX runs)
and the *displayed* form of text after a right-to-left override (which differs from its
logical order) become the block's ``hidden_text``; and the document-level
:class:`UnicodeReport` is stored with the version and surfaced as a security finding when
suspicious.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from docassist.core.text import SanitizeReport, sanitize_text
from docassist.ingestion.injection import rlo_visual_segments
from docassist.ingestion.model import ParsedDocument

MAX_HIDDEN_TEXT_CHARS = 4_000
_SURROGATES = re.compile("[" + chr(0xD800) + "-" + chr(0xDFFF) + "]")
_HSPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class TextBlock:
    """A normalised block, ready for chunking."""

    kind: str
    text: str
    page: int
    level: int | None = None
    hidden_text: str = ""
    channel_flags: tuple[str, ...] = ()


@dataclass(slots=True)
class UnicodeReport:
    zero_width: int = 0
    bidi: int = 0
    tags: int = 0
    controls: int = 0
    other_invisible: int = 0
    hidden_blocks: int = 0
    pages_with_hidden_channels: list[int] = field(default_factory=list)

    def add(self, report: SanitizeReport, page: int) -> None:
        self.zero_width += report.zero_width
        self.bidi += report.bidi
        self.tags += report.tags
        self.controls += report.controls
        self.other_invisible += report.other_invisible
        if report.suspicious and page not in self.pages_with_hidden_channels:
            self.pages_with_hidden_channels.append(page)

    @property
    def suspicious(self) -> bool:
        return (
            self.tags > 0
            or self.bidi > 0
            or bool(self.pages_with_hidden_channels)
            or self.hidden_blocks > 0
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "zero_width": self.zero_width,
            "bidi": self.bidi,
            "tags": self.tags,
            "controls": self.controls,
            "other_invisible": self.other_invisible,
            "hidden_blocks": self.hidden_blocks,
            "pages_with_hidden_channels": self.pages_with_hidden_channels[:50],
        }


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    blocks: tuple[TextBlock, ...]
    report: UnicodeReport
    page_numbers: tuple[int, ...]

    @property
    def char_count(self) -> int:
        return sum(len(block.text) for block in self.blocks)


def clean_text(text: str) -> tuple[str, SanitizeReport]:
    cleaned, report = sanitize_text(text)
    return _SURROGATES.sub("", cleaned), report


def _layout(kind: str, text: str) -> str:
    if kind == "heading":
        return " ".join(text.split())
    if kind in ("table", "list"):
        lines = (" ".join(line.split()) for line in text.split("\n"))
        return "\n".join(line for line in lines if line)
    text = _HSPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def normalize_block(
    kind: str, text: str, *, page: int, level: int | None, hidden: str, report: UnicodeReport
) -> TextBlock | None:
    """Sanitise one block; ``None`` when nothing readable is left."""
    cleaned, block_report = clean_text(text)
    report.add(block_report, page)
    body = _layout(kind, cleaned)
    if not body:
        return None
    hidden_parts = []
    if hidden:
        hidden_clean, _ = clean_text(hidden)
        hidden_parts.append(" ".join(hidden_clean.split()))
        report.hidden_blocks += 1
    if block_report.decoded_tag_text:
        hidden_parts.append(block_report.decoded_tag_text)
    for segment in rlo_visual_segments(text):  # what a reader sees after a bidi override
        visual, _ = clean_text(segment)
        hidden_parts.append(" ".join(visual.split()))
    hidden_text = "\n".join(part for part in hidden_parts if part)[:MAX_HIDDEN_TEXT_CHARS]
    return TextBlock(
        kind=kind,
        text=body,
        page=page,
        level=level if kind == "heading" else None,
        hidden_text=hidden_text,
        channel_flags=tuple(block_report.as_flags()),
    )


def normalize_document(parsed: ParsedDocument) -> NormalizedDocument:
    """Sanitise every block of ``parsed``; empty blocks are dropped, order is preserved."""
    report = UnicodeReport()
    blocks: list[TextBlock] = []
    for page in parsed.pages:
        for block in page.blocks:
            normalized = normalize_block(
                block.kind,
                block.text,
                page=page.number,
                level=block.level,
                hidden=block.hidden,
                report=report,
            )
            if normalized is not None:
                blocks.append(normalized)
    return NormalizedDocument(
        blocks=tuple(blocks),
        report=report,
        page_numbers=tuple(page.number for page in parsed.pages),
    )
