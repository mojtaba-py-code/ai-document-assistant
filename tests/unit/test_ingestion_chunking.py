"""Structure-aware chunker: unit behaviour and hypothesis property tests."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from docassist.core.config import ChunkingSettings
from docassist.core.text import estimate_tokens
from docassist.ingestion.chunking import ChunkedDocument, ChunkingParams, chunk_blocks
from docassist.ingestion.normalize import TextBlock


def para(text: str, page: int = 1, **kw: object) -> TextBlock:
    return TextBlock("paragraph", text, page, **kw)  # type: ignore[arg-type]


def heading(text: str, level: int = 1, page: int = 1) -> TextBlock:
    return TextBlock("heading", text, page, level)


def table(rows: list[str], page: int = 1) -> TextBlock:
    return TextBlock("table", "\n".join(rows), page)


def sentence(n: int, words: int = 12) -> str:
    return " ".join(f"word{n}x{i}" for i in range(words)) + "."


def check_invariants(
    blocks: list[TextBlock], params: ChunkingParams, result: ChunkedDocument
) -> None:
    canonical = result.text
    assert canonical == "\n\n".join(b.text for b in blocks if b.text.strip())
    covered = bytearray(len(canonical))
    previous = None
    split_floor = min(params.target_tokens, params.max_tokens - params.max_tokens // 4 - 1)
    for index, chunk in enumerate(result.chunks):
        assert chunk.index == index
        assert chunk.text[chunk.prefix_chars :] == canonical[chunk.char_start : chunk.char_end]
        assert chunk.token_count == estimate_tokens(chunk.text) <= params.max_tokens
        assert 0 <= chunk.char_start < chunk.char_end <= len(canonical)
        assert chunk.page_start <= chunk.page_end
        if previous is not None:
            assert chunk.char_start > previous.char_start and chunk.char_end > previous.char_end
            if chunk.char_start < previous.char_end:
                overlap = canonical[chunk.char_start : previous.char_end]
                assert estimate_tokens(overlap) <= params.overlap_tokens
        for boundary in (chunk.char_start, chunk.char_end):
            inside_word = 0 < boundary < len(canonical) and not (
                canonical[boundary - 1].isspace() or canonical[boundary].isspace()
            )
            if inside_word:  # only runs longer than the size budget may be cut
                lo = boundary
                while lo > 0 and not canonical[lo - 1].isspace():
                    lo -= 1
                hi = boundary
                while hi < len(canonical) and not canonical[hi].isspace():
                    hi += 1
                assert estimate_tokens(canonical[lo:hi]) > split_floor
        covered[chunk.char_start : chunk.char_end] = b"\x01" * (chunk.char_end - chunk.char_start)
        previous = chunk
    for position, char in enumerate(canonical):
        if not char.isspace():
            assert covered[position], f"character {position} is in no chunk"


# --------------------------------------------------------------------------- #
# Unit behaviour
# --------------------------------------------------------------------------- #
def test_params_validation_and_settings() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkingParams(target_tokens=100, max_tokens=90, overlap_tokens=10)
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkingParams(target_tokens=100, max_tokens=200, overlap_tokens=100)
    params = ChunkingParams.from_settings(ChunkingSettings())
    assert (params.target_tokens, params.max_tokens, params.overlap_tokens) == (450, 800, 60)
    assert params.min_tokens == 112


def test_empty_input_has_no_chunks() -> None:
    result = chunk_blocks([para("   "), para("")], ChunkingParams(50, 80, 10))
    assert result.chunks == () and result.text == ""


def test_headings_open_sections_with_paths() -> None:
    blocks = [
        heading("Agreement", 1),
        para(" ".join(sentence(i) for i in range(4))),
        heading("Payment", 2),
        para(" ".join(sentence(i) for i in range(4, 8))),
        heading("Schedule", 3),
        para(sentence(9)),
    ]
    params = ChunkingParams(40, 80, 0)
    result = chunk_blocks(blocks, params)
    check_invariants(blocks, params, result)
    assert result.chunks[0].text.startswith("Agreement")
    payment = next(c for c in result.chunks if c.text.startswith("Payment"))
    assert payment.heading_path == ("Agreement", "Payment") and payment.section == "Payment"
    schedule = next(c for c in result.chunks if "Schedule" in c.text)
    assert schedule.heading_path[-1] in {"Schedule", "Payment"}
    assert "heading" in payment.block_types and "paragraph" in payment.block_types


def test_sibling_heading_replaces_path_level() -> None:
    blocks = [
        heading("A", 1),
        heading("B", 2),
        para(sentence(1) * 3),
        heading("C", 2),
        para(sentence(2) * 3),
    ]
    result = chunk_blocks(blocks, ChunkingParams(20, 40, 0))
    paths = {c.heading_path for c in result.chunks}
    assert ("A", "C") in paths and ("A", "B", "C") not in paths


def test_small_blocks_are_packed_and_fitting_block_is_never_split() -> None:
    blocks = [para(f"Short paragraph number {i}.") for i in range(6)]
    blocks.append(para(" ".join(sentence(i) for i in range(10))))  # ~ 250 tokens
    params = ChunkingParams(100, 400, 20)
    result = chunk_blocks(blocks, params)
    check_invariants(blocks, params, result)
    assert "Short paragraph number 0." in result.chunks[0].text
    assert "Short paragraph number 5." in result.chunks[0].text  # packed together
    big = blocks[-1].text
    assert any(big in c.text for c in result.chunks)  # fits in max_tokens: kept whole


def test_long_paragraph_splits_on_sentences_with_overlap() -> None:
    text = " ".join(sentence(i) for i in range(30))
    params = ChunkingParams(60, 90, 15)
    result = chunk_blocks([para(text)], params)
    check_invariants([para(text)], params, result)
    assert len(result.chunks) > 3
    for previous, chunk in zip(result.chunks, result.chunks[1:], strict=False):
        assert chunk.char_start < previous.char_end  # overlapping context
        assert estimate_tokens(result.text[chunk.char_start : previous.char_end]) <= 15


def test_no_overlap_across_sections() -> None:
    blocks = [
        heading("One"),
        para(" ".join(sentence(i) for i in range(12))),
        heading("Two"),
        para(sentence(99)),
    ]
    params = ChunkingParams(40, 60, 10)
    result = chunk_blocks(blocks, params)
    two = next(c for c in result.chunks if c.text.startswith("Two"))
    before = result.chunks[two.index - 1]
    assert two.char_start >= before.char_end


def test_table_rows_stay_whole_and_header_repeats() -> None:
    rows = ["Item | Qty | Price"] + [f"Widget {i} | {i} | USD {i * 10}.00" for i in range(60)]
    params = ChunkingParams(40, 60, 5)
    result = chunk_blocks([table(rows)], params)
    check_invariants([table(rows)], params, result)
    assert len(result.chunks) > 3
    for chunk in result.chunks:
        lines = chunk.text.split("\n")
        assert lines[0] == "Item | Qty | Price"
        for line in lines[1:]:
            assert line in rows  # never a partial row
    continued = [c for c in result.chunks if c.prefix_chars]
    assert continued and all(c.prefix_chars == len("Item | Qty | Price\n") for c in continued)


def test_small_table_is_one_block_chunk() -> None:
    rows = ["A | B", "1 | 2", "3 | 4"]
    result = chunk_blocks([table(rows)], ChunkingParams(64, 128, 8))
    assert len(result.chunks) == 1 and result.chunks[0].text == "A | B\n1 | 2\n3 | 4"
    assert result.chunks[0].prefix_chars == 0 and result.chunks[0].block_types == ("table",)


def test_unbreakable_run_is_cut_at_the_budget() -> None:
    blob = "Q" * 2_000
    params = ChunkingParams(50, 80, 10)
    result = chunk_blocks([para(f"prefix {blob} suffix")], params)
    check_invariants([para(f"prefix {blob} suffix")], params, result)
    assert "".join(c.text for c in result.chunks).count("Q") >= 2_000


def test_cjk_text_is_bounded() -> None:
    text = "".join(chr(0x4E00 + (i % 500)) for i in range(900))
    params = ChunkingParams(64, 128, 16)
    result = chunk_blocks([para(text)], params)
    check_invariants([para(text)], params, result)
    assert all(c.token_count <= 128 for c in result.chunks) and len(result.chunks) >= 8


def test_pages_hidden_text_and_flags_are_carried() -> None:
    blocks = [
        para("Page one text.", page=1),
        para(
            "Page two text.",
            page=2,
            hidden_text="secret run",
            channel_flags=("bidi_control_characters",),
        ),
    ]
    result = chunk_blocks(blocks, ChunkingParams(64, 128, 8))
    [chunk] = result.chunks
    assert (chunk.page_start, chunk.page_end) == (1, 2)
    assert chunk.hidden_text == "secret run" and chunk.channel_flags == ("bidi_control_characters",)


def test_tiny_tail_is_merged_into_previous_chunk() -> None:
    text = " ".join(sentence(i) for i in range(6)) + " End."
    params = ChunkingParams(40, 200, 0)
    result = chunk_blocks([para(text)], params)
    assert result.chunks[-1].token_count >= params.min_tokens or len(result.chunks) == 1
    assert result.chunks[-1].text.endswith("End.")


def test_lookups_map_offsets_to_chunks_and_pages() -> None:
    blocks = [
        para(" ".join(sentence(i) for i in range(10)), page=3),
        para("Last words here.", page=4),
    ]
    result = chunk_blocks(blocks, ChunkingParams(40, 60, 0))
    last = result.text.index("Last words")
    assert result.page_at(last) == 4 and result.page_at(0) == 3
    chunk = result.chunk_at(last)
    assert chunk is not None and chunk.char_start <= last < chunk.char_end
    assert result.chunk_at(10**9) is None


def test_chunking_is_deterministic() -> None:
    blocks = [heading("T"), para(" ".join(sentence(i) for i in range(40)))]
    params = ChunkingParams(50, 90, 12)
    assert chunk_blocks(blocks, params) == chunk_blocks(blocks, params)


# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #
_LATIN = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJ0123456789.,;:!?-'", min_size=1, max_size=14
)
_CJK = st.text(alphabet=[chr(cp) for cp in range(0x4E00, 0x4E40)], min_size=1, max_size=30)
_LONG = st.text(alphabet="xyzXYZ0189+/=", min_size=40, max_size=400)
_WORD = st.one_of(_LATIN, _LATIN, _LATIN, _CJK, _LONG)
_SPACE = st.sampled_from([" ", " ", " ", "  ", "\n", " \n "])


@st.composite
def _text(draw: st.DrawFn, max_words: int = 60) -> str:
    words = draw(st.lists(_WORD, min_size=1, max_size=max_words))
    out = [words[0]]
    for word in words[1:]:
        out.append(draw(_SPACE))
        out.append(word)
    return "".join(out)


@st.composite
def _block(draw: st.DrawFn) -> tuple[str, str, int | None]:
    kind = draw(st.sampled_from(["paragraph", "paragraph", "list", "heading", "table"]))
    if kind == "heading":
        return kind, draw(_text(max_words=6)), draw(st.integers(1, 4))
    if kind == "table":
        rows = draw(st.lists(st.lists(_LATIN, min_size=1, max_size=5), min_size=1, max_size=25))
        return kind, "\n".join(" | ".join(r) for r in rows), None
    return kind, draw(_text()), None


@st.composite
def _document(draw: st.DrawFn) -> list[TextBlock]:
    raw = draw(st.lists(_block(), min_size=1, max_size=14))
    blocks: list[TextBlock] = []
    page = 1
    for kind, text, level in raw:
        page += draw(st.integers(0, 1))
        blocks.append(TextBlock(kind, text, page, level))
    return blocks


@st.composite
def _params(draw: st.DrawFn) -> ChunkingParams:
    target = draw(st.integers(4, 120))
    maximum = draw(st.integers(target, target * 3))
    overlap = draw(st.integers(0, target - 1))
    return ChunkingParams(target, maximum, overlap)


@settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(blocks=_document(), params=_params())
def test_chunking_properties(blocks: list[TextBlock], params: ChunkingParams) -> None:
    """No text lost, sizes bounded, order preserved, words intact, deterministic."""
    result = chunk_blocks(blocks, params)
    check_invariants(blocks, params, result)
    assert chunk_blocks(blocks, params) == result


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=_text(max_words=400), params=_params())
def test_single_long_paragraph_properties(text: str, params: ChunkingParams) -> None:
    blocks = [para(text)]
    check_invariants(blocks, params, chunk_blocks(blocks, params))
