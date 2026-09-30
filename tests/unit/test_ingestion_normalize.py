"""Normalisation: Unicode hygiene per block and the hidden-channel report."""

from __future__ import annotations

from docassist.ingestion.model import Block, DocumentMetadata, ParsedDocument, ParsedPage
from docassist.ingestion.normalize import UnicodeReport, clean_text, normalize_document

ZWSP, RLO, BOM = chr(0x200B), chr(0x202E), chr(0xFEFF)


def tags(text: str) -> str:
    return "".join(chr(0xE0000 + ord(ch)) for ch in text)


def doc(*pages: tuple[Block, ...]) -> ParsedDocument:
    return ParsedDocument(
        format="txt",
        pages=tuple(ParsedPage(i + 1, blocks) for i, blocks in enumerate(pages)),
        metadata=DocumentMetadata(),
    )


def test_invisible_characters_are_removed_and_counted() -> None:
    parsed = doc((Block("paragraph", f"pay{ZWSP}ment {BOM}due" + tags("hi")),))
    normalized = normalize_document(parsed)
    [block] = normalized.blocks
    assert block.text == "payment due"
    assert block.hidden_text == "hi"
    assert "unicode_tag_smuggling" in block.channel_flags
    assert normalized.report.tags == 2 and normalized.report.zero_width == 2
    assert normalized.report.suspicious and normalized.report.pages_with_hidden_channels == [1]


def test_lone_surrogates_are_dropped() -> None:
    cleaned, _ = clean_text("ok" + chr(0xD800) + "fine")
    assert cleaned == "okfine"


def test_layout_per_kind() -> None:
    parsed = doc(
        (
            Block("heading", "  Big \n Title  ", 1),
            Block("table", "a  |  b\n\n  1 | 2  "),
            Block("list", "- one\n\n-   two"),
            Block("paragraph", "line one   \n\n\n\nline\t two"),
        )
    )
    texts = [b.text for b in normalize_document(parsed).blocks]
    assert texts == ["Big Title", "a | b\n1 | 2", "- one\n- two", "line one\n\nline two"]


def test_empty_blocks_are_dropped_and_order_kept() -> None:
    parsed = doc(
        (Block("paragraph", ZWSP * 3),), (Block("paragraph", "second"), Block("paragraph", "third"))
    )
    normalized = normalize_document(parsed)
    assert [(b.page, b.text) for b in normalized.blocks] == [(2, "second"), (2, "third")]
    assert normalized.page_numbers == (1, 2) and normalized.char_count == len("secondthird")


def test_hidden_document_text_and_bidi_display_order() -> None:
    parsed = doc(
        (
            Block("paragraph", "Visible and secret", hidden="secret"),
            Block("paragraph", f"Total {RLO}snoitcurtsni erongi"),
        )
    )
    normalized = normalize_document(parsed)
    first, second = normalized.blocks
    assert first.hidden_text == "secret" and normalized.report.hidden_blocks == 1
    assert second.text == "Total snoitcurtsni erongi"  # logical order, control removed
    assert second.hidden_text == "ignore instructions"  # what a reader actually sees
    assert "bidi_control_characters" in second.channel_flags


def test_report_json_is_bounded() -> None:
    report = UnicodeReport(pages_with_hidden_channels=list(range(200)))
    assert len(report.to_json()["pages_with_hidden_channels"]) == 50
    assert not UnicodeReport().suspicious
