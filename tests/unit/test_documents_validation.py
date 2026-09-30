"""Upload validation: filenames, spooling, sniffing, text checks and ZIP/OOXML inspection."""

from __future__ import annotations

import codecs
import hashlib
import io
import os
import sys
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from docassist.core.errors import (
    PayloadTooLarge,
    RejectedContent,
    UnsupportedMediaType,
    ValidationFailed,
)
from docassist.documents import validation as v
from docassist.documents.validation import (
    ZipLimits,
    extension_of,
    inspect_zip,
    sanitize_filename,
    sniff,
    spool_upload,
    validate_content,
    validate_text,
)
from tests.fixtures import sample_files as sf
from tests.helpers_documents import stream_of

LIMITS = ZipLimits(max_entries=2_000, max_uncompressed_bytes=200 * sf.MIB, max_ratio=100)


def _reason(exc: pytest.ExceptionInfo[BaseException]) -> str:
    value = exc.value
    assert isinstance(value, (RejectedContent, UnsupportedMediaType))
    return str(value.extra["reason"])


# --------------------------------------------------------------------------- #
# Filenames
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\evil.PDF", "evil.pdf"),
        ("C:\\Users\\x\\invoice.docx", "invoice.docx"),
        ("/absolute/path/notes.md", "notes.md"),
        ("a" + chr(0) + "b.txt", "a b.txt"),
        ("line\nbreak\ttab.csv", "line break tab.csv"),
        ("  .hidden.txt  ", "hidden.txt"),
        ("....pdf", "document.pdf"),
        ("", "document"),
        ("..", "document"),
        ("CON.txt", "_CON.txt"),
        ("lpt1.tar.pdf", "_lpt1.tar.pdf"),
        ('in<va>l"i|d?*.pdf', "in_va_l_i_d__.pdf"),
        (
            "r" + chr(0xE9) + "sum" + chr(0xE9) + ".pdf",
            "r" + chr(0xE9) + "sum" + chr(0xE9) + ".pdf",
        ),
        ("name.withaverylongextension", "name.withaverylongextension"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_invisible_and_bidi_characters_are_removed() -> None:
    rtl_override = chr(0x202E)
    zero_width = chr(0x200B)
    tag_a = chr(0xE0041)
    name = f"inv{zero_width}oice{rtl_override}fdp{tag_a}.exe"
    assert sanitize_filename(name) == "invoicefdp.exe"


def test_fullwidth_solidus_cannot_smuggle_a_path() -> None:
    fullwidth_solidus = chr(0xFF0F)
    assert (
        sanitize_filename(f"..{fullwidth_solidus}..{fullwidth_solidus}secret.pdf") == "secret.pdf"
    )


def test_long_names_are_capped_but_keep_extension() -> None:
    result = sanitize_filename("x" * 500 + ".docx")
    assert len(result) == v.MAX_FILENAME_CHARS
    assert result.endswith(".docx")


def test_extension_of() -> None:
    assert extension_of("a.PDF") == "pdf"
    assert extension_of("archive.tar.gz") == "gz"
    assert extension_of("noext") == ""
    assert extension_of("weird.ex e") == ""


# --------------------------------------------------------------------------- #
# Spooling
# --------------------------------------------------------------------------- #
async def test_spool_hashes_and_keeps_small_uploads_in_memory() -> None:
    data = b"hello world" * 100
    with await spool_upload(stream_of(data, 7), max_bytes=10_000) as spool:
        assert spool.in_memory and spool.path is None
        assert spool.size == len(data)
        assert spool.sha256 == hashlib.sha256(data).hexdigest()
        assert spool.head == data[: v.HEAD_BYTES]
        assert b"".join(spool.iter_chunks(5)) == data


async def test_spool_rolls_over_to_private_temp_file_and_deletes_it(tmp_path: Path) -> None:
    data = os.urandom(300_000)
    spool = await spool_upload(
        stream_of(data, 50_000), max_bytes=1_000_000, temp_dir=tmp_path, memory_limit=100_000
    )
    try:
        assert not spool.in_memory
        assert spool.path is not None and spool.path.parent == tmp_path
        if sys.platform != "win32":
            assert (spool.path.stat().st_mode & 0o777) == 0o600
        chunks = [c async for c in spool.aiter_chunks(64_000)]
        assert b"".join(chunks) == data
        with spool.open() as first, spool.open() as second:
            assert first.read(10) == second.read(10) == data[:10]
    finally:
        spool.close()
    assert list(tmp_path.iterdir()) == []
    spool.close()  # idempotent


async def test_spool_aborts_at_the_first_chunk_over_the_limit(tmp_path: Path) -> None:
    consumed = 0

    async def endless() -> AsyncIterator[bytes]:
        nonlocal consumed
        while True:
            consumed += 1
            yield b"x" * 1000

    with pytest.raises(PayloadTooLarge):
        await spool_upload(endless(), max_bytes=5_500, temp_dir=tmp_path, memory_limit=1_000)
    assert consumed == 6  # stopped right at the chunk that crossed the limit
    assert list(tmp_path.iterdir()) == []  # the partial temp file is gone


async def test_empty_upload_rejected() -> None:
    with pytest.raises(ValidationFailed):
        await spool_upload(stream_of(b""), max_bytes=100)


# --------------------------------------------------------------------------- #
# Sniffing
# --------------------------------------------------------------------------- #
def test_sniff_accepts_matching_types() -> None:
    assert sniff(sf.build_pdf(), "pdf").mime == "application/pdf"
    assert sniff(sf.build_docx()[:8192], "docx").family == "ooxml"
    assert sniff(b"a,b\n1,2\n", "csv").mime == "text/csv"
    detected = sniff(codecs.BOM_UTF8 + b"# Title\n", "md")
    assert (detected.mime, detected.text_encoding) == ("text/markdown", "utf-8-sig")
    assert sniff("hello".encode("utf-16"), "txt").text_encoding == "utf-16"


@pytest.mark.parametrize(
    ("head", "extension", "reason"),
    [
        (b"PK\x03\x04rest", "pdf", "type_mismatch"),
        (b"  %PDF-1.7", "pdf", "type_mismatch"),  # leading junk (polyglot trick)
        (b"%PDF-1.7\n", "docx", "type_mismatch"),
        (b"%PDF-1.7\n", "txt", "type_mismatch"),
        (b"PK\x03\x04", "csv", "type_mismatch"),
        (b"{\\rtf1\\ansi hello}", "txt", "type_mismatch"),
        (b"\x7fELF\x02\x01", "md", "type_mismatch"),
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "docx", "legacy_or_encrypted_office"),
        (codecs.BOM_UTF32_LE + "hi".encode("utf-32-le"), "txt", "unsupported_text_encoding"),
        (b"MZ\x00\x00\x03\x00\x00\x00", "txt", "binary_content"),
        (b"caf\xe9 au lait", "txt", "invalid_text_encoding"),
        (b"hello", "exe", "extension_not_supported"),
    ],
)
def test_sniff_rejects(head: bytes, extension: str, reason: str) -> None:
    with pytest.raises((UnsupportedMediaType, RejectedContent)) as exc:
        sniff(head, extension)
    assert _reason(exc) == reason


def test_declared_mime_is_irrelevant_to_sniffing() -> None:
    # sniff() never sees the declared type; only bytes and extension matter.
    assert sniff(b"%PDF-1.4\n", "pdf").extension == "pdf"


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
def test_text_with_binary_tail_is_rejected() -> None:
    body = b"valid text\n" * 2000 + b"\x00\x01\x02 binary"
    with pytest.raises(UnsupportedMediaType) as exc:
        validate_text([body[i : i + 1000] for i in range(0, len(body), 1000)], "utf-8")
    assert _reason(exc) == "binary_content"


def test_text_multibyte_characters_split_across_chunks_are_fine() -> None:
    body = ("na" + chr(0xEF) + "ve " * 1000).encode("utf-8")
    chunks = [body[i : i + 7] for i in range(0, len(body), 7)]
    stats = validate_text(chunks, "utf-8")
    assert stats.control_characters == 0


def test_control_character_ratio() -> None:
    few = b"a" * 10_000 + b"\x07" * 20  # 20 controls in 10k chars: under 1 percent
    assert validate_text([few], "utf-8").control_characters == 20
    many = b"a" * 100 + b"\x07" * 20
    with pytest.raises(UnsupportedMediaType) as exc:
        validate_text([many], "utf-8")
    assert _reason(exc) == "control_characters"
    tabs_and_newlines = b"a\tb\r\nc\x0cd\n" * 100
    assert validate_text([tabs_and_newlines], "utf-8").control_characters == 0


def test_truncated_utf8_at_end_of_file_is_rejected() -> None:
    with pytest.raises(UnsupportedMediaType):
        validate_text([b"abc", b"\xe2\x82"], "utf-8")


def test_utf16_nul_is_rejected() -> None:
    with pytest.raises(UnsupportedMediaType):
        validate_text([("ab" + chr(0) + "c").encode("utf-16")], "utf-16")


# --------------------------------------------------------------------------- #
# ZIP / OOXML
# --------------------------------------------------------------------------- #
def test_real_docx_and_xlsx_pass() -> None:
    assert inspect_zip(io.BytesIO(sf.build_docx()), LIMITS).package_type == "docx"
    assert inspect_zip(io.BytesIO(sf.build_xlsx()), LIMITS).package_type == "xlsx"


def test_inspect_zip_accepts_a_path(tmp_path: Path) -> None:
    path = tmp_path / "x.docx"
    path.write_bytes(sf.build_docx())
    assert inspect_zip(path, LIMITS).entry_count > 3


def test_macro_enabled_package_is_rejected() -> None:
    with pytest.raises(RejectedContent) as exc:
        inspect_zip(io.BytesIO(sf.docm_disguised_as_docx()), LIMITS)
    assert _reason(exc) == "ooxml_macro_enabled"


async def _validate_bytes(data: bytes, extension: str) -> v.ValidatedContent:
    with await spool_upload(stream_of(data), max_bytes=50 * sf.MIB) as spool:
        return validate_content(spool, extension, LIMITS)


async def test_xlsx_renamed_docx_is_a_mismatch() -> None:
    with pytest.raises(UnsupportedMediaType) as exc:
        await _validate_bytes(sf.build_xlsx(), "docx")
    assert _reason(exc) == "type_mismatch"


def test_zip_bomb_by_ratio() -> None:
    bomb = sf.raw_zip([("word/document.xml", b"\x00" * (20 * sf.MIB))])
    assert len(bomb) < 100_000
    with pytest.raises(RejectedContent) as exc:
        inspect_zip(io.BytesIO(bomb), LIMITS)
    assert _reason(exc) == "zip_ratio_exceeded"


def test_zip_bomb_by_total_size() -> None:
    limits = ZipLimits(max_entries=100, max_uncompressed_bytes=2 * sf.MIB, max_ratio=1_000)
    data = sf.raw_zip([(f"word/p{i}.xml", os.urandom(900_000)) for i in range(3)])
    with pytest.raises(RejectedContent) as exc:
        inspect_zip(io.BytesIO(data), limits)
    assert _reason(exc) == "zip_too_large_uncompressed"


def test_too_many_entries_detected_from_the_end_record() -> None:
    data = sf.raw_zip([(f"word/p{i}.xml", b"x") for i in range(30)])
    with pytest.raises(RejectedContent) as exc:
        inspect_zip(
            io.BytesIO(data),
            ZipLimits(max_entries=10, max_uncompressed_bytes=sf.MIB, max_ratio=100),
        )
    assert _reason(exc) == "zip_too_many_entries"
    assert v.declared_entry_count(io.BytesIO(data)) == 31


@pytest.mark.parametrize(
    "name",
    ["../evil.xml", "word/../../evil.xml", "/etc/passwd", "C:evil.xml", "word\\evil.xml"],
)
def test_path_traversal_entries(name: str) -> None:
    info = zipfile.ZipInfo("placeholder-entry-name.xml")
    data = sf.raw_zip([(info, b"x")])
    # Patch the stored name in place (zipfile would normalise some of these on write).
    placeholder = b"placeholder-entry-name.xml"
    encoded = name.encode()
    assert len(encoded) <= len(placeholder)
    data = data.replace(placeholder, encoded.ljust(len(placeholder), b"_"))
    with pytest.raises(RejectedContent) as exc:
        inspect_zip(io.BytesIO(data), LIMITS)
    assert _reason(exc) == "zip_path_traversal"


def test_nul_in_entry_name_is_rejected() -> None:
    data = sf.raw_zip([("word/aaaa.xml", b"x")])
    data = data.replace(b"word/aaaa.xml", b"word/a\x00aa.xml")  # same length: offsets hold
    with pytest.raises(RejectedContent) as exc:
        inspect_zip(io.BytesIO(data), LIMITS)
    assert _reason(exc) == "zip_bad_entry_name"


def test_encrypted_entry_rejected() -> None:
    data = sf.set_zip_flag(sf.raw_zip([("word/document.xml", b"<x/>")]), "word/document.xml", 0x1)
    with pytest.raises(RejectedContent) as exc:
        inspect_zip(io.BytesIO(data), LIMITS)
    assert _reason(exc) == "zip_encrypted_entry"


def test_symlink_entry_rejected() -> None:
    data = sf.raw_zip([(sf.symlink_entry("word/link.xml"), b"/etc/passwd")])
    with pytest.raises(RejectedContent) as exc:
        inspect_zip(io.BytesIO(data), LIMITS)
    assert _reason(exc) == "zip_symlink_entry"


@pytest.mark.parametrize("name", ["word/embeddings/inner.zip", "word/embeddings/Sheet.xlsx"])
def test_nested_archive_rejected(name: str) -> None:
    data = sf.raw_zip([(name, b"PK\x03\x04")])
    with pytest.raises(RejectedContent) as exc:
        inspect_zip(io.BytesIO(data), LIMITS)
    assert _reason(exc) == "zip_nested_archive"


def test_case_insensitive_duplicate_parts_rejected() -> None:
    data = sf.raw_zip([("word/document.xml", b"a"), ("WORD/Document.xml", b"b")])
    with pytest.raises(RejectedContent) as exc:
        inspect_zip(io.BytesIO(data), LIMITS)
    assert _reason(exc) == "zip_duplicate_entry"


def test_unsupported_compression_rejected() -> None:
    data = sf.raw_zip([("word/document.xml", b"abc" * 100)], compression=zipfile.ZIP_BZIP2)
    with pytest.raises(RejectedContent) as exc:
        inspect_zip(io.BytesIO(data), LIMITS)
    assert _reason(exc) == "zip_unsupported_compression"


def test_archive_without_content_types_is_not_office() -> None:
    data = sf.raw_zip([("readme.txt", b"hi")], content_types=False)
    with pytest.raises(UnsupportedMediaType) as exc:
        inspect_zip(io.BytesIO(data), LIMITS)
    assert _reason(exc) == "not_office_document"


def test_content_types_with_dtd_is_rejected() -> None:
    bad = b'<?xml version="1.0"?><!DOCTYPE t [<!ENTITY e "x">]><Types>&e;</Types>'
    data = sf.raw_zip([("[Content_Types].xml", bad)], content_types=False)
    with pytest.raises(RejectedContent) as exc:
        inspect_zip(io.BytesIO(data), LIMITS)
    assert _reason(exc) == "corrupt_archive"


def test_truncated_archive_is_corrupt() -> None:
    data = sf.build_docx()[:-40]
    with pytest.raises(RejectedContent) as exc:
        inspect_zip(io.BytesIO(data), LIMITS)
    assert _reason(exc) == "corrupt_archive"


async def test_validate_content_for_each_format() -> None:
    assert (await _validate_bytes(sf.build_pdf(), "pdf")).detected.family == "pdf"
    assert (await _validate_bytes(sf.build_docx(), "docx")).archive is not None
    assert (await _validate_bytes(sf.build_xlsx(), "xlsx")).archive is not None
    assert (await _validate_bytes(b"a;b\n1;2\n", "csv")).text is not None
    utf16 = await _validate_bytes("# Title\n\nbody".encode("utf-16"), "md")
    assert utf16.detected.text_encoding == "utf-16"


async def test_docm_renamed_docx_is_rejected_end_to_end() -> None:
    with pytest.raises(RejectedContent) as exc:
        await _validate_bytes(sf.docm_disguised_as_docx(), "docx")
    assert _reason(exc) == "ooxml_macro_enabled"
