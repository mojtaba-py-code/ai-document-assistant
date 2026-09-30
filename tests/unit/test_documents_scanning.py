"""Malware scanner protocol, EICAR heuristic, PDF/OOXML/CSV active-content detection, policy."""

from __future__ import annotations

import asyncio
import hashlib
import struct
import zlib

import pytest

from docassist.core.errors import ServiceUnavailable
from docassist.documents.scanning import (
    EICAR_TEST_SIGNATURE,
    ActiveContentLimits,
    ActiveContentScanner,
    ClamAvScanner,
    Finding,
    NullScanner,
    Severity,
    Verdict,
    contains_eicar,
    decide,
    parse_clamd_reply,
)
from tests.fixtures import sample_files as sf
from tests.helpers_documents import BytesSource, FakeClamd

SCANNER = ActiveContentScanner()


def codes(data: bytes, family: str, **kwargs: str) -> dict[str, str]:
    return {f.code: f.severity.value for f in SCANNER.scan(BytesSource(data), family, **kwargs)}


# --------------------------------------------------------------------------- #
# EICAR
# --------------------------------------------------------------------------- #
def test_eicar_constant_is_the_official_test_file() -> None:
    assert len(EICAR_TEST_SIGNATURE) == 68
    assert (
        hashlib.sha256(EICAR_TEST_SIGNATURE).hexdigest()
        == "275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f"
    )


@pytest.mark.parametrize("split", [1, 10, 34, 67])
def test_eicar_found_across_chunk_boundaries(split: int) -> None:
    body = b"x" * 100 + EICAR_TEST_SIGNATURE + b"y" * 100
    cut = 100 + split
    assert contains_eicar([body[:cut], body[cut:]])
    assert contains_eicar(body[i : i + 3] for i in range(0, len(body), 3))


def test_no_eicar_in_clean_data() -> None:
    assert not contains_eicar([b"harmless " * 1000, EICAR_TEST_SIGNATURE[:40]])


@pytest.mark.parametrize("family", ["pdf", "ooxml", "text"])
def test_eicar_is_critical_in_every_format(family: str) -> None:
    data = {
        "pdf": sf.build_pdf(extra_objects=[b"(" + EICAR_TEST_SIGNATURE + b")"]),
        # stored (not deflated) so the signature appears in the raw archive bytes
        "ooxml": sf.raw_zip([("customXml/eicar.txt", EICAR_TEST_SIGNATURE)], compression=0),
        "text": EICAR_TEST_SIGNATURE,
    }[family]
    assert codes(data, family)["eicar_test_signature"] == "critical"


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def test_clean_pdf_has_no_findings() -> None:
    assert codes(sf.build_pdf(["Hello", "World"]), "pdf") == {}


def test_javascript_open_action() -> None:
    found = codes(sf.build_pdf(catalog_extra=sf.JS_OPEN_ACTION), "pdf")
    assert found["pdf_javascript"] == "critical"
    assert found["pdf_open_action"] == "warning"


def test_hex_escaped_javascript_is_decoded_and_flagged() -> None:
    found = codes(sf.build_pdf(catalog_extra=sf.HEX_ESCAPED_JS), "pdf")
    assert found["pdf_javascript"] == "critical"
    assert found["pdf_obfuscated_name"] == "high"


def test_javascript_hidden_in_compressed_object_stream() -> None:
    pdf = sf.build_pdf(object_stream=[b"<< /S /JavaScript /JS (app.alert\\(1\\)) >>"])
    assert b"JavaScript" not in pdf  # invisible without inflating the stream
    assert codes(pdf, "pdf")["pdf_javascript"] == "critical"


def test_escaped_name_inside_object_stream() -> None:
    pdf = sf.build_pdf(object_stream=[b"<< /S /Launch /F (cmd.exe) /L#61unch 1 >>"])
    found = codes(pdf, "pdf")
    assert found["pdf_launch_action"] == "critical"
    assert found["pdf_obfuscated_name"] == "high"


def test_object_stream_with_unscannable_filter_is_high() -> None:
    pdf = sf.build_pdf(
        object_stream=[b"<< /JS (x) >>"],
        object_stream_dict_extra=b"/Filter /ASCIIHexDecode",
        object_stream_encoder="hex",
    )
    assert codes(pdf, "pdf")["pdf_unscannable_stream"] == "high"


def test_object_stream_predictor_is_unscannable() -> None:
    pdf = sf.build_pdf(
        object_stream=[b"<< /A 1 >>"],
        object_stream_dict_extra=b"/Filter /FlateDecode /DecodeParms << /Predictor 12 /Columns 4 >>",
    )
    assert codes(pdf, "pdf")["pdf_unscannable_stream"] == "high"


def test_oversized_object_stream_hits_the_scan_limit() -> None:
    scanner = ActiveContentScanner(ActiveContentLimits(max_pdf_stream_decoded_bytes=1_000))
    pdf = sf.build_pdf(object_stream=[b"<< /Pad (" + b"a" * 5_000 + b") >>"])
    found = {f.code for f in scanner.scan(BytesSource(pdf), "pdf")}
    assert "pdf_scan_limit" in found


def test_corrupt_object_stream_is_a_warning() -> None:
    body = zlib.compress(b"1 0 << /A 1 >>")[:-6] + b"garbage!"
    pdf = sf.build_pdf(
        extra_objects=[
            b"<< /Type /ObjStm /N 1 /First 4 /Filter /FlateDecode /Length %d >>\nstream\n"
            % len(body)
            + body
            + b"\nendstream"
        ]
    )
    assert codes(pdf, "pdf").get("pdf_corrupt_stream") == "warning"


@pytest.mark.parametrize(
    ("extra", "code", "severity"),
    [
        (b"/OpenAction << /S /Launch /F (calc.exe) >>", "pdf_launch_action", "critical"),
        (b"/Names << /EmbeddedFiles 9 0 R >>", "pdf_embedded_file", "high"),
        (b"/AcroForm << /XFA 9 0 R >>", "pdf_xfa_form", "high"),
        (b"/AA << /O << /S /GoTo >> >>", "pdf_additional_actions", "high"),
        (b"/OpenAction << /S /SubmitForm /F (https://x.example) >>", "pdf_submit_form", "high"),
        (b"/OpenAction << /S /GoToR /F (other.pdf) >>", "pdf_remote_goto", "warning"),
        (b"/OpenAction << /S /URI /URI (https://example.com) >>", "pdf_uri", "info"),
        (b"/RichMedia << >>", "pdf_rich_media", "high"),
    ],
)
def test_pdf_action_catalogue(extra: bytes, code: str, severity: str) -> None:
    assert codes(sf.build_pdf(catalog_extra=extra), "pdf")[code] == severity


def test_encrypted_pdf() -> None:
    found = codes(sf.build_pdf(trailer_extra=sf.ENCRYPT_TRAILER), "pdf")
    assert found["pdf_encrypted"] == "critical"


def test_encrypt_word_inside_a_string_is_not_encryption() -> None:
    pdf = sf.build_pdf(extra_objects=[b"<< /Title (About /Encrypt) >>"])
    assert "pdf_encrypted" not in codes(pdf, "pdf")


def test_binary_stream_data_does_not_raise_false_alarms() -> None:
    noise = (b"\x00/JS (x) /JavaScript << /Launch /AA << >>" * 200) + bytes(range(256)) * 50
    assert codes(sf.build_pdf(binary_stream=noise), "pdf") == {}


def test_indirect_lengths_are_resolved_so_page_text_is_skipped() -> None:
    pdf = sf.build_pdf(["Read (/JavaScript) and /JS in this manual"], indirect_lengths=True)
    assert codes(pdf, "pdf") == {}


def test_stream_with_wrong_length_scans_the_gap_conservatively() -> None:
    # A strict reader parses the bytes after /Length as objects, so the scanner does too:
    # a dangerous-looking name there is flagged (fail closed) and the mismatch is reported.
    content = b"BT (/JS text) Tj ET"
    obj = b"<< /Length 3 >>\nstream\n" + content + b"\nendstream"
    found = codes(sf.build_pdf(extra_objects=[obj]), "pdf")
    assert found["pdf_length_mismatch"] == "warning"
    assert found["pdf_javascript"] == "critical"


def test_dictionary_after_a_stream_is_still_scanned() -> None:
    stream = b"<< /Length 4 >>\nstream\nabcd\nendstream"
    pdf = sf.build_pdf(extra_objects=[stream, b"<< /S /JavaScript /JS (1) >>"])
    assert codes(pdf, "pdf")["pdf_javascript"] == "critical"


# --------------------------------------------------------------------------- #
# OOXML
# --------------------------------------------------------------------------- #
def test_clean_office_files_have_no_findings() -> None:
    assert codes(sf.build_docx(), "ooxml") == {}
    assert codes(sf.build_xlsx(), "ooxml") == {}


@pytest.mark.parametrize(
    ("builder", "code", "severity"),
    [
        (sf.docx_with_undeclared_vba, "ooxml_macro", "critical"),
        (sf.docx_with_remote_template, "ooxml_remote_template", "high"),
        (
            lambda: sf.docx_with_external_relationship("oleObject", "https://x.example/o.html"),
            "ooxml_external_ole_object",
            "high",
        ),
        (
            lambda: sf.docx_with_external_relationship("hyperlink", "https://example.com/doc"),
            "ooxml_external_hyperlink",
            "info",
        ),
        (
            lambda: sf.docx_with_external_relationship("image", "\\\\attacker\\share\\a.png"),
            "ooxml_unc_target",
            "high",
        ),
        (
            lambda: sf.docx_with_external_relationship("hyperlink", "ms-msdt:/id PCWDiagnostic"),
            "ooxml_protocol_handler",
            "high",
        ),
        (
            lambda: sf.docx_with_field(["DD", "EAUTO c:\\\\windows\\\\cmd.exe /c calc"]),
            "ooxml_dde_field",
            "high",
        ),
        (lambda: sf.docx_with_field(["DDEAUTO cmd x"], simple=True), "ooxml_dde_field", "high"),
        (lambda: sf.docx_with_field(["&#68;DE cmd x"]), "ooxml_dde_field", "high"),
        (sf.docx_with_rels_dtd, "ooxml_xml_dtd", "high"),
        (lambda: sf.xlsx_with_formula("=cmd|' /C calc'!A0"), "ooxml_dde_formula", "high"),
        (
            lambda: sf.rewrite_zip(sf.build_docx(), add={"word/activeX/activeX1.xml": b"<x/>"}),
            "ooxml_activex",
            "high",
        ),
        (
            lambda: sf.rewrite_zip(
                sf.build_docx(), add={"word/embeddings/oleObject1.bin": b"\xd0\xcf"}
            ),
            "ooxml_ole_object",
            "high",
        ),
        (
            lambda: sf.rewrite_zip(sf.build_xlsx(), add={"xl/macrosheets/sheet1.xml": b"<x/>"}),
            "ooxml_xlm_macro",
            "critical",
        ),
        (
            lambda: sf.rewrite_zip(
                sf.build_xlsx(),
                add={
                    "xl/externalLinks/externalLink1.xml": b"<externalLink><ddeLink/></externalLink>"
                },
            ),
            "ooxml_dde_link",
            "high",
        ),
    ],
)
def test_ooxml_catalogue(builder, code: str, severity: str) -> None:  # type: ignore[no-untyped-def]
    assert codes(builder(), "ooxml")[code] == severity


def test_benign_fields_and_formulas_are_not_flagged() -> None:
    assert codes(sf.docx_with_field([' TOC \\o "1-3" ']), "ooxml") == {}
    assert codes(sf.docx_with_field(["PAGE"], simple=True), "ooxml") == {}
    assert codes(sf.xlsx_with_formula('=CONCAT("a|b!", "c")'), "ooxml") == {}
    assert codes(sf.xlsx_with_formula("=SUM(A1:A3)"), "ooxml") == {}


def test_remote_template_detail_names_the_host_not_the_path() -> None:
    [finding] = SCANNER.scan(
        BytesSource(sf.docx_with_remote_template("https://evil.example/secret-token/t.dotm")),
        "ooxml",
    )
    assert "evil.example" in finding.detail
    assert "secret-token" not in finding.detail


def test_repeated_findings_are_counted_once() -> None:
    data = sf.docx_with_external_relationship("hyperlink", "https://a.example")
    rels = sf.read_part(data, "word/_rels/document.xml.rels").decode()
    extra = "".join(
        f'<Relationship Id="rIdH{i}" Type="{sf.OFFICE_REL}/hyperlink" '
        f'Target="https://b{i}.example" TargetMode="External"/>'
        for i in range(3)
    )
    data = sf.rewrite_zip(
        data,
        replace={
            "word/_rels/document.xml.rels": rels.replace(
                "</Relationships>", extra + "</Relationships>"
            ).encode()
        },
    )
    [finding] = SCANNER.scan(BytesSource(data), "ooxml")
    assert finding.detail.endswith("(4 occurrences)")


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def test_csv_dde_formula() -> None:
    data = b"name,value\nalice,=cmd|' /C calc'!A0\n"
    assert codes(data, "text", extension="csv")["csv_dde_formula"] == "high"


def test_csv_dde_in_utf16() -> None:
    data = "a;b\n1;@SUM|'x'!Z9\n".encode("utf-16")
    assert "csv_dde_formula" in codes(data, "text", extension="csv", text_encoding="utf-16")


def test_plain_csv_and_txt_are_clean() -> None:
    assert codes(b"a,b\n=1+2,x|y\n", "text", extension="csv") == {}
    assert codes(b"=cmd|' /C calc'!A0", "text", extension="txt") == {}


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
def _f(code: str, severity: Severity) -> Finding:
    return Finding(code, severity, "d")


@pytest.mark.parametrize(
    ("findings", "reject_active", "verdict"),
    [
        ([], True, Verdict.ACCEPT),
        (
            [_f("pdf_uri", Severity.INFO), _f("pdf_open_action", Severity.WARNING)],
            True,
            Verdict.ACCEPT,
        ),
        ([_f("pdf_embedded_file", Severity.HIGH)], True, Verdict.QUARANTINE),
        ([_f("pdf_javascript", Severity.CRITICAL)], True, Verdict.QUARANTINE),
        ([_f("pdf_javascript", Severity.CRITICAL)], False, Verdict.ACCEPT),
        ([_f("eicar_test_signature", Severity.CRITICAL)], False, Verdict.QUARANTINE),
        ([_f("malware_detected", Severity.CRITICAL)], False, Verdict.QUARANTINE),
        (
            [_f("pdf_encrypted", Severity.CRITICAL), _f("malware_detected", Severity.CRITICAL)],
            True,
            Verdict.REJECT,
        ),
    ],
)
def test_decide(findings: list[Finding], reject_active: bool, verdict: Verdict) -> None:
    assert decide(findings, reject_active_content=reject_active) is verdict


# --------------------------------------------------------------------------- #
# Malware scanners
# --------------------------------------------------------------------------- #
async def test_null_scanner_reports_not_scanned() -> None:
    result = await NullScanner().scan(BytesSource(b"x"))
    assert result.clean and not result.scanned


@pytest.mark.parametrize(
    ("reply", "clean", "signature"),
    [
        (b"stream: OK\x00", True, None),
        (b"stream: Win.Test.EICAR_HDB-1 FOUND\x00", False, "Win.Test.EICAR_HDB-1"),
        (b"stream: Evil;rm -rf FOUND\x00", False, "Evil_rm_-rf"),
    ],
)
def test_parse_clamd_reply(reply: bytes, clean: bool, signature: str | None) -> None:
    result = parse_clamd_reply(reply)
    assert (result.clean, result.signature) == (clean, signature)


@pytest.mark.parametrize(
    "reply", [b"INSTREAM size limit exceeded. ERROR\x00", b"UNKNOWN COMMAND\x00", b"\x00"]
)
def test_unexpected_clamd_reply_fails_closed(reply: bytes) -> None:
    with pytest.raises(ServiceUnavailable):
        parse_clamd_reply(reply)


async def test_clamav_instream_protocol() -> None:
    payload = bytes(range(256)) * 700  # ~175 KiB -> several chunks
    clamd = FakeClamd(reply=b"stream: OK\x00")
    async with clamd.running():
        scanner = ClamAvScanner("127.0.0.1", clamd.port, timeout_seconds=5, chunk_size=64 * 1024)
        result = await scanner.scan(BytesSource(payload))
        assert await scanner.ping()
    assert result.clean and result.scanned
    assert clamd.commands[0] == b"zINSTREAM\x00"
    assert bytes(clamd.received) == payload
    assert clamd.chunk_sizes[-1] == 0  # zero-length terminator
    assert all(size <= 64 * 1024 for size in clamd.chunk_sizes)


async def test_clamav_found() -> None:
    clamd = FakeClamd(reply=b"stream: Eicar-Signature FOUND\x00")
    async with clamd.running():
        result = await ClamAvScanner("127.0.0.1", clamd.port, 5).scan(BytesSource(b"payload"))
    assert not result.clean
    assert result.signature == "Eicar-Signature"


async def test_clamav_unreachable_fails_closed() -> None:
    clamd = FakeClamd()
    async with clamd.running():
        port = clamd.port
    scanner = ClamAvScanner("127.0.0.1", port, timeout_seconds=2)
    with pytest.raises(ServiceUnavailable):
        await scanner.scan(BytesSource(b"payload"))
    assert not await scanner.ping()


async def test_clamav_timeout_fails_closed() -> None:
    clamd = FakeClamd(delay=2.0)
    async with clamd.running():
        with pytest.raises(ServiceUnavailable):
            await ClamAvScanner("127.0.0.1", clamd.port, timeout_seconds=0.2).scan(
                BytesSource(b"payload")
            )


async def test_clamav_reply_without_terminator_fails_closed() -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\x00")
        while (struct.unpack("!I", await reader.readexactly(4)))[0]:
            pass
        writer.write(b"x" * 10_000)  # never null-terminated, over the reply limit
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        with pytest.raises(ServiceUnavailable):
            await ClamAvScanner("127.0.0.1", port, 5).scan(BytesSource(b""))
    finally:
        server.close()
        await server.wait_closed()


# --------------------------------------------------------------------------- #
# Review findings (2026-09-30): scanner evasions that must stay closed
# --------------------------------------------------------------------------- #
def _raw_pdf(*objects: bytes) -> bytes:
    return b"%PDF-1.7\n" + b"\n".join(objects) + b"\ntrailer << /Root 1 0 R >>\n%%EOF\n"


def test_object_stream_with_indirect_filter_fails_closed() -> None:
    hidden = zlib.compress(b"5 0 << /Type /Action /S /JavaScript /JS (app.alert(1)) >>")
    assert b"JavaScript" not in hidden
    pdf = _raw_pdf(
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        b"2 0 obj << /Type /Pages /Kids [] /Count 0 >> endobj",
        b"3 0 obj /FlateDecode endobj",
        b"4 0 obj << /Type /ObjStm /N 1 /First 4 /Filter 3 0 R /Length "
        + str(len(hidden)).encode()
        + b" >>\nstream\n"
        + hidden
        + b"\nendstream\nendobj",
    )
    found = codes(pdf, "pdf")
    assert found["pdf_unscannable_stream"] == "high"
    assert (
        decide(list(SCANNER.scan(BytesSource(pdf), "pdf")), reject_active_content=True)
        is not Verdict.ACCEPT
    )


def test_object_stream_with_indirect_decode_parms_fails_closed() -> None:
    hidden = zlib.compress(b"5 0 << /A 1 >>")
    pdf = _raw_pdf(
        b"1 0 obj << /Type /Catalog >> endobj",
        b"4 0 obj << /Type /ObjStm /N 1 /First 4 /Filter /FlateDecode /DecodeParms 9 0 R /Length "
        + str(len(hidden)).encode()
        + b" >>\nstream\n"
        + hidden
        + b"\nendstream\nendobj",
    )
    assert codes(pdf, "pdf")["pdf_unscannable_stream"] == "high"


def test_objects_hidden_behind_a_short_length_are_scanned() -> None:
    pdf = _raw_pdf(
        b"1 0 obj << /Type /Catalog /OpenAction 5 0 R >> endobj",
        b"4 0 obj << /Length 5 >>\nstream\nAAAAA\n"
        b"5 0 obj << /Type /Action /S /JavaScript /JS (app.alert(1)) >> endobj\n"
        b"endstream\nendobj",
    )
    found = codes(pdf, "pdf")
    assert found["pdf_javascript"] == "critical"
    assert found["pdf_length_mismatch"] == "warning"


def test_correct_lengths_do_not_warn() -> None:
    pdf = _raw_pdf(
        b"1 0 obj << /Type /Catalog >> endobj",
        b"4 0 obj << /Length 5 >>\nstream\nAAAAA\nendstream\nendobj",
    )
    assert "pdf_length_mismatch" not in codes(pdf, "pdf")
