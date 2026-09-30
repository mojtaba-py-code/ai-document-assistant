"""Malware and active-content scanning for uploads.

Two independent layers run on every upload (after :mod:`~docassist.documents.validation`
has established what the file is):

* :class:`MalwareScanner` - a signature engine. :class:`ClamAvScanner` streams the bytes to
  ``clamd`` with the ``INSTREAM`` protocol and **fails closed**: any network error, timeout,
  size-limit or unexpected reply raises :class:`~docassist.core.errors.ServiceUnavailable`,
  so a configured scanner never lets a file through unscanned. :class:`NullScanner` is the
  development stand-in and reports that nothing was scanned.
* :class:`ActiveContentScanner` - deterministic heuristics that need no signatures: the EICAR
  test file, PDF actions/JavaScript/embedded files/encryption (with ``#xx`` name escapes
  decoded and compressed object streams inflated, so the classic obfuscations do not hide a
  ``/JavaScript`` key), OOXML macros/ActiveX/OLE objects/external relationships (remote
  template injection, UNC paths, protocol handlers)/DDE fields and formulas, and DDE
  formulas in CSV files.

:func:`decide` turns findings into a :class:`Verdict`: encrypted PDFs are rejected (they
cannot be inspected), malware is always quarantined, and ``high``/``critical`` active content
is quarantined while ``upload.reject_active_content`` is on. Quarantined files are stored
encrypted and never parsed.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import html
import io
import mmap
import re
import struct
import zipfile
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import BinaryIO, Final, Protocol
from urllib.parse import urlsplit

import defusedxml.ElementTree as SafeElementTree
from defusedxml.common import DefusedXmlException

from docassist.core.errors import ServiceUnavailable
from docassist.core.logging import get_logger
from docassist.core.text import clean_line_text

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Findings & verdicts
# --------------------------------------------------------------------------- #
class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: Final = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


@dataclass(frozen=True, slots=True)
class Finding:
    """One scanner observation. ``detail`` is generated here - never document text."""

    code: str
    severity: Severity
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity.value, "detail": self.detail}


class Verdict(StrEnum):
    ACCEPT = "accept"
    QUARANTINE = "quarantine"
    REJECT = "reject"


REJECT_CODES: Final = frozenset({"pdf_encrypted"})
"""Findings that make a file impossible to inspect: the upload is refused outright."""

MALWARE_CODES: Final = frozenset({"malware_detected", "eicar_test_signature"})
"""Findings that always quarantine, regardless of ``reject_active_content``."""


def decide(findings: Iterable[Finding], *, reject_active_content: bool) -> Verdict:
    """Upload policy: reject > quarantine > accept (see the module docstring)."""
    items = list(findings)
    if any(f.code in REJECT_CODES for f in items):
        return Verdict.REJECT
    if any(f.code in MALWARE_CODES for f in items):
        return Verdict.QUARANTINE
    if reject_active_content and any(f.severity.rank >= Severity.HIGH.rank for f in items):
        return Verdict.QUARANTINE
    return Verdict.ACCEPT


class _Findings:
    """Collects findings, one per code (repeats are counted into the detail)."""

    def __init__(self) -> None:
        self._items: dict[str, Finding] = {}
        self._counts: dict[str, int] = {}

    def add(self, code: str, severity: Severity, detail: str) -> None:
        self._counts[code] = self._counts.get(code, 0) + 1
        if code not in self._items:
            self._items[code] = Finding(code, severity, detail)

    def has(self, code: str) -> bool:
        return code in self._items

    def items(self) -> list[Finding]:
        out = []
        for code, finding in self._items.items():
            count = self._counts[code]
            detail = finding.detail if count == 1 else f"{finding.detail} ({count} occurrences)"
            out.append(Finding(finding.code, finding.severity, detail))
        return out


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
class ScanSource(Protocol):
    """Anything that can hand out independent binary readers (e.g. ``SpooledUpload``)."""

    @property
    def size(self) -> int: ...

    def open(self) -> BinaryIO: ...


_CHUNK = 256 * 1024


class _Readable(Protocol):
    def read(self, size: int = ..., /) -> bytes: ...


def _iter_handle(handle: _Readable, chunk_size: int = _CHUNK) -> Iterator[bytes]:
    while chunk := handle.read(chunk_size):
        yield chunk


def _stream_matches[T: (str, bytes)](
    chunks: Iterable[T], pattern: re.Pattern[T], max_match: int
) -> Iterator[re.Match[T]]:
    """``pattern.finditer`` over a chunked stream.

    Matches up to ``max_match`` long are found even when they straddle chunk boundaries;
    every match is yielded exactly once.
    """
    buffer: T | None = None
    for chunk in chunks:
        buffer = chunk if buffer is None else buffer + chunk
        safe_end = len(buffer) - max_match
        keep_from = max(safe_end, 0)
        for match in pattern.finditer(buffer):
            if match.end() > safe_end:
                keep_from = min(keep_from, match.start())
                break
            yield match
        buffer = buffer[keep_from:]
    if buffer is not None:
        yield from pattern.finditer(buffer)


# --------------------------------------------------------------------------- #
# EICAR
# --------------------------------------------------------------------------- #
# The EICAR anti-virus test string is stored XOR-encoded: endpoint anti-virus quarantines any
# file (this module, its .pyc, a checkout) that contains it verbatim. SHA-256 of the decoded
# value: 275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f.
_EICAR_KEY: Final = 0x5A
_EICAR_ENCODED: Final = bytes.fromhex(
    "026f157b0a7f1a1b0a016e060a00026f6e720a04736d1919736d277e1f13191b0877090e1b141e1b081e77"
    "1b140e130c13080f09770e1f090e771c13161f7b7e12711270"
)
EICAR_TEST_SIGNATURE: Final = bytes(b ^ _EICAR_KEY for b in _EICAR_ENCODED)


def contains_eicar(chunks: Iterable[bytes]) -> bool:
    """True if the EICAR test signature occurs anywhere in the stream."""
    tail = b""
    overlap = len(EICAR_TEST_SIGNATURE) - 1
    for chunk in chunks:
        window = tail + chunk
        if EICAR_TEST_SIGNATURE in window:
            return True
        tail = window[-overlap:]
    return False


# --------------------------------------------------------------------------- #
# Malware scanners
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ScanResult:
    clean: bool
    signature: str | None = None
    scanned: bool = True


class MalwareScanner(Protocol):
    @property
    def name(self) -> str: ...

    async def scan(self, source: ScanSource) -> ScanResult: ...


class NullScanner:
    """Development scanner: scans nothing and says so (``scanned=False``)."""

    name = "none"

    async def scan(self, source: ScanSource) -> ScanResult:
        return ScanResult(clean=True, signature=None, scanned=False)


_SIGNATURE_UNSAFE = re.compile(r"[^A-Za-z0-9._:/+\-]")
_MAX_REPLY = 4096


def parse_clamd_reply(raw: bytes) -> ScanResult:
    """Interpret a ``zINSTREAM`` reply; anything but OK/FOUND raises ServiceUnavailable."""
    text = raw.rstrip(b"\x00").decode("utf-8", "replace").strip()
    if text.endswith(" FOUND"):
        name = text.removesuffix(" FOUND")
        name = name.split(":", 1)[1] if ":" in name else name
        signature = _SIGNATURE_UNSAFE.sub("_", name.strip())[:128] or "unknown"
        return ScanResult(clean=False, signature=signature)
    if text.endswith(" OK") or text == "OK":
        return ScanResult(clean=True)
    raise ServiceUnavailable(
        "Malware scanning is temporarily unavailable. Please retry later.",
        internal_detail=f"unexpected clamd reply: {text[:120]!r}",
    )


class ClamAvScanner:
    """``clamd`` INSTREAM client over asyncio TCP.

    Framing: ``zINSTREAM\\0``, then chunks each prefixed with their length as a 4-byte
    big-endian integer, then a zero-length chunk; the null-terminated reply is
    ``stream: OK`` or ``stream: <signature> FOUND``. ``clamd``'s ``StreamMaxLength`` must be
    at least ``upload.max_upload_bytes`` - otherwise large files fail closed.
    """

    name = "clamav"

    def __init__(
        self, host: str, port: int, timeout_seconds: float, *, chunk_size: int = 64 * 1024
    ) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout_seconds
        self._chunk_size = chunk_size

    async def scan(self, source: ScanSource) -> ScanResult:
        try:
            async with asyncio.timeout(self._timeout):
                reply = await self._exchange(source)
        except TimeoutError as exc:
            raise self._unavailable("timeout") from exc
        except (OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError) as exc:
            raise self._unavailable(type(exc).__name__) from exc
        return parse_clamd_reply(reply)

    async def ping(self) -> bool:
        """Health check: ``True`` if clamd answers ``PONG`` within the timeout."""
        try:
            async with asyncio.timeout(self._timeout):
                reader, writer = await asyncio.open_connection(self._host, self._port)
                try:
                    writer.write(b"zPING\x00")
                    await writer.drain()
                    reply = await reader.readuntil(b"\x00")
                finally:
                    await self._close(writer)
        except (TimeoutError, OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            return False
        return reply.rstrip(b"\x00").strip() == b"PONG"

    async def _exchange(self, source: ScanSource) -> bytes:
        reader, writer = await asyncio.open_connection(self._host, self._port, limit=_MAX_REPLY)
        try:
            writer.write(b"zINSTREAM\x00")
            with source.open() as handle:
                while chunk := await asyncio.to_thread(handle.read, self._chunk_size):
                    writer.write(struct.pack("!I", len(chunk)) + chunk)
                    await writer.drain()
            writer.write(struct.pack("!I", 0))
            await writer.drain()
            return await reader.readuntil(b"\x00")
        finally:
            await self._close(writer)

    @staticmethod
    async def _close(writer: asyncio.StreamWriter) -> None:
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()

    def _unavailable(self, reason: str) -> ServiceUnavailable:
        log.warning("malware_scanner_unavailable", scanner=self.name, reason=reason)
        return ServiceUnavailable(
            "Malware scanning is temporarily unavailable. Please retry later.",
            internal_detail=f"clamd unavailable: {reason}",
        )


# --------------------------------------------------------------------------- #
# PDF heuristics
# --------------------------------------------------------------------------- #
_WS = rb"\x00\t\n\x0c\r "
_NAME_CHARS = rb"[^\x00\t\n\x0c\r ()<>\[\]{}/%]"
_NAME_END = rb"(?=[\x00\t\n\x0c\r ()<>\[\]{}/%]|\Z)"

_PDF_NAME_RULES: Final[dict[bytes, tuple[str, Severity, str]]] = {
    b"JavaScript": ("pdf_javascript", Severity.CRITICAL, "PDF contains JavaScript"),
    b"JS": ("pdf_javascript", Severity.CRITICAL, "PDF contains JavaScript"),
    b"Launch": ("pdf_launch_action", Severity.CRITICAL, "PDF contains a Launch action"),
    b"EmbeddedFile": ("pdf_embedded_file", Severity.HIGH, "PDF embeds a file"),
    b"EmbeddedFiles": ("pdf_embedded_file", Severity.HIGH, "PDF embeds a file"),
    b"RichMedia": ("pdf_rich_media", Severity.HIGH, "PDF contains rich media"),
    b"XFA": ("pdf_xfa_form", Severity.HIGH, "PDF contains an XFA form"),
    b"AA": ("pdf_additional_actions", Severity.HIGH, "PDF has automatic actions"),
    b"SubmitForm": ("pdf_submit_form", Severity.HIGH, "PDF can submit form data"),
    b"ImportData": ("pdf_import_data", Severity.HIGH, "PDF can import external data"),
    b"GoToR": ("pdf_remote_goto", Severity.WARNING, "PDF links to another file"),
    b"GoToE": ("pdf_embedded_goto", Severity.WARNING, "PDF links into an embedded file"),
    b"OpenAction": ("pdf_open_action", Severity.WARNING, "PDF runs an action when opened"),
    b"URI": ("pdf_uri", Severity.INFO, "PDF contains web links"),
    b"Encrypt": ("pdf_encrypted", Severity.CRITICAL, "PDF is encrypted and cannot be scanned"),
}
_PLAIN_DANGEROUS_NAME = re.compile(
    rb"/("
    + b"|".join(re.escape(n) for n in sorted(_PDF_NAME_RULES, key=len, reverse=True))
    + rb")"
    + _NAME_END
)
_ESCAPED_NAME = re.compile(rb"/" + _NAME_CHARS + rb"*#[0-9A-Fa-f]{2}" + _NAME_CHARS + rb"*")
_NAME_TOKEN = re.compile(rb"/" + _NAME_CHARS + rb"*")
_HEX_ESCAPE = re.compile(rb"#([0-9A-Fa-f]{2})")
_STREAM_START = re.compile(rb">>[" + _WS + rb"]*stream(?:\r\n|\n|\r)")
_ENDSTREAM_AT = re.compile(rb"[" + _WS + rb"]*endstream")
_DIRECT_LENGTH = re.compile(
    rb"/Length[" + _WS + rb"]+(\d{1,12})(?!\d)(?![" + _WS + rb"]+\d+[" + _WS + rb"]+R)"
)
_INDIRECT_LENGTH = re.compile(
    rb"/Length[" + _WS + rb"]+(\d{1,10})[" + _WS + rb"]+(\d{1,5})[" + _WS + rb"]+R"
)
_INTEGER_OBJECT = re.compile(
    rb"(?<!\d)(\d{1,10})[" + _WS + rb"]+(\d{1,5})[" + _WS + rb"]+obj[" + _WS + rb"]*"
    rb"(\d{1,12})[" + _WS + rb"]*endobj"
)
_OBJSTM_TYPE = re.compile(rb"/Type[" + _WS + rb"]*/ObjStm" + _NAME_END)
_FILTER = re.compile(rb"/Filter[" + _WS + rb"]*(\[[^\]]{0,512}\]|/" + _NAME_CHARS + rb"+)")
_FILTER_NAME = re.compile(rb"/(" + _NAME_CHARS + rb"+)")
_PREDICTOR = re.compile(rb"/Predictor[" + _WS + rb"]+(\d{1,3})")
_INDIRECT_PARMS = re.compile(rb"/DecodeParms[" + _WS + rb"]*\d")
_ENCRYPT_VALUE = re.compile(rb"[" + _WS + rb"]*(?:\d|<<)")
_FLATE_NAMES: Final = frozenset({b"FlateDecode", b"Fl"})
_HEADER_WINDOW = 16 * 1024


def _decode_name(token: bytes) -> bytes:
    return _HEX_ESCAPE.sub(lambda m: bytes([int(m.group(1), 16)]), token)


def _normalize_names(text: bytes) -> bytes:
    """Decode ``#xx`` escapes inside every name token (``/J#61vaScript`` -> ``/JavaScript``)."""
    if b"#" not in text:
        return text
    return _NAME_TOKEN.sub(lambda m: _decode_name(m.group()), text)


@dataclass(frozen=True, slots=True)
class _PdfStream:
    header: bytes
    body_start: int
    body_end: int


@dataclass(frozen=True, slots=True)
class ActiveContentLimits:
    max_pdf_stream_decoded_bytes: int = 16 * 1024 * 1024
    """Largest inflated object stream scanned (beyond it the file is flagged unscannable)."""
    max_pdf_total_decoded_bytes: int = 64 * 1024 * 1024
    """Budget of inflated object-stream bytes per file."""
    max_xml_part_bytes: int = 8 * 1024 * 1024
    """Largest relationship part parsed into memory."""


class _PdfScanner:
    def __init__(self, data: bytes | mmap.mmap, limits: ActiveContentLimits, out: _Findings):
        self._data = data
        self._limits = limits
        self._out = out
        self._objects: dict[tuple[int, int], int] | None = None
        self._decoded_budget = limits.max_pdf_total_decoded_bytes

    def run(self) -> None:
        regions, streams = self._layout()
        for start, end in regions:
            self._scan_names(self._data, start, end)
        for stream in streams:
            if _OBJSTM_TYPE.search(stream.header):
                decoded = self._decode_object_stream(stream)
                if decoded is not None:
                    self._scan_names(decoded, 0, len(decoded))

    # -- structure ------------------------------------------------------------
    def _layout(self) -> tuple[list[tuple[int, int]], list[_PdfStream]]:
        """Split the file into dictionary regions (scanned) and stream bodies (skipped).

        A body is skipped only where a PDF reader would also treat it as stream data: up to
        ``/Length`` when ``endstream`` follows exactly there, otherwise up to the first
        ``endstream`` keyword (the readers' recovery rule). Names hidden in stream data are
        therefore never interpreted by a reader either, while random binary data in images
        cannot produce false alarms. Object streams are inflated and scanned separately.
        """
        data = self._data
        size = len(data)
        regions: list[tuple[int, int]] = []
        streams: list[_PdfStream] = []
        pos = 0
        while pos < size:
            match = _STREAM_START.search(data, pos)
            if match is None:
                regions.append((pos, size))
                break
            body = match.end()
            header_floor = max(pos, match.start() - _HEADER_WINDOW)
            obj_at = data.rfind(b"obj", header_floor, match.start())
            header = _normalize_names(bytes(data[obj_at if obj_at >= 0 else header_floor : body]))
            regions.append((pos, body))
            end, gap_start = self._stream_end(body, header)
            if end is None:
                pos = body
                continue
            if gap_start is not None:
                # The file disagrees with its own /Length. Recovering readers treat the gap as
                # stream data, strict readers parse it as objects - so it is scanned as objects.
                regions.append((gap_start, end))
                self._out.add(
                    "pdf_length_mismatch",
                    Severity.WARNING,
                    "PDF stream length disagrees with its end marker",
                )
            streams.append(_PdfStream(header, body, gap_start if gap_start is not None else end))
            pos = end
        return regions, streams

    def _stream_end(self, body: int, header: bytes) -> tuple[int | None, int | None]:
        """``(end_of_stream_data, gap_start)``; ``gap_start`` is set when ``/Length`` ends the
        data *before* the first ``endstream`` (the bytes in between are scanned as objects)."""
        data = self._data
        length: int | None = None
        direct = _DIRECT_LENGTH.search(header)
        if direct is not None:
            length = int(direct.group(1))
        else:
            indirect = _INDIRECT_LENGTH.search(header)
            if indirect is not None:
                length = self._object_integer(int(indirect.group(1)), int(indirect.group(2)))
        if (
            length is not None
            and body + length <= len(data)
            and _ENDSTREAM_AT.match(data, body + length)
        ):
            return body + length, None
        found = data.find(b"endstream", body)
        if found < 0:
            return None, None
        if length is not None and body + length < found:
            return found, body + length
        return found, None

    def _object_integer(self, number: int, generation: int) -> int | None:
        if self._objects is None:
            self._objects = {}
            for match in _INTEGER_OBJECT.finditer(self._data):
                key = (int(match.group(1)), int(match.group(2)))
                self._objects.setdefault(key, int(match.group(3)))
        return self._objects.get((number, generation))

    # -- names ------------------------------------------------------------------
    def _scan_names(self, data: bytes | mmap.mmap, start: int, end: int) -> None:
        for match in _PLAIN_DANGEROUS_NAME.finditer(data, start, end):
            self._flag(match.group(1), data, match.end(), end)
        for match in _ESCAPED_NAME.finditer(data, start, end):
            decoded = _decode_name(match.group())[1:]
            if decoded in _PDF_NAME_RULES:
                self._out.add(
                    "pdf_obfuscated_name",
                    Severity.HIGH,
                    "PDF hides a sensitive name with #xx escapes",
                )
                self._flag(decoded, data, match.end(), end)

    def _flag(self, name: bytes, data: bytes | mmap.mmap, after: int, end: int) -> None:
        code, severity, detail = _PDF_NAME_RULES[name]
        if code == "pdf_encrypted" and not _ENCRYPT_VALUE.match(data, after, end):
            return  # "/Encrypt" not followed by a reference or dictionary is not a trailer key
        self._out.add(code, severity, detail)

    # -- object streams ------------------------------------------------------------
    def _decode_object_stream(self, stream: _PdfStream) -> bytes | None:
        header = stream.header
        filters: list[bytes] = []
        filter_match = _FILTER.search(header)
        if filter_match is not None:
            filters = _FILTER_NAME.findall(filter_match.group(1))
        # "/Filter 3 0 R" (an indirect reference) or any other value we could not parse must not
        # be mistaken for "no filter": the still-compressed bytes would be scanned as plaintext
        # and a hidden /JavaScript would pass. Fail closed instead.
        unresolved_filter = b"/Filter" in header and not filters
        predictor = _PREDICTOR.search(header)
        if (
            unresolved_filter
            or _INDIRECT_PARMS.search(header) is not None
            or len(filters) > 1
            or any(f not in _FLATE_NAMES for f in filters)
            or (predictor is not None and int(predictor.group(1)) > 1)
        ):
            self._out.add(
                "pdf_unscannable_stream",
                Severity.HIGH,
                "PDF object stream uses an encoding the scanner cannot inspect",
            )
            return None
        cap = min(self._limits.max_pdf_stream_decoded_bytes, self._decoded_budget)
        raw = bytes(self._data[stream.body_start : stream.body_end])
        if not filters:
            decoded, truncated, corrupt = raw[:cap], len(raw) > cap, False
        else:
            decoded, truncated, corrupt = _inflate(raw, cap)
        self._decoded_budget -= len(decoded)
        if truncated:
            self._out.add(
                "pdf_scan_limit",
                Severity.HIGH,
                "PDF object streams are too large to scan completely",
            )
        if corrupt:
            self._out.add("pdf_corrupt_stream", Severity.WARNING, "PDF has a damaged object stream")
        return decoded  # escaped names are decoded (and reported) by _scan_names


def _inflate(raw: bytes, cap: int) -> tuple[bytes, bool, bool]:
    """zlib-inflate at most ``cap`` bytes -> ``(data, truncated, corrupt)``.

    ``truncated``: more output than ``cap``; ``corrupt``: invalid data, or the input ended
    before the end of the compressed stream (``data`` then holds what could be recovered).
    """
    decompressor = zlib.decompressobj()
    out = bytearray()
    for index in range(0, len(raw), 64 * 1024):
        try:
            out += decompressor.decompress(raw[index : index + 64 * 1024], cap + 1 - len(out))
        except zlib.error:
            return bytes(out), False, True
        if len(out) > cap or decompressor.unconsumed_tail:
            return bytes(out[:cap]), True, False
        if decompressor.eof:
            break
    return bytes(out), False, not decompressor.eof  # input ended mid-stream: truncated


# --------------------------------------------------------------------------- #
# OOXML heuristics
# --------------------------------------------------------------------------- #
_WORD_STORY_PART = re.compile(
    r"word/(?:glossary/)?(?:document|header\d*|footer\d*|footnotes|endnotes|comments)\.xml"
)
_SHEET_PART = re.compile(r"xl/(?:worksheets|dialogsheets)/[^/]+\.xml")
_EXTERNAL_LINK_PART = re.compile(r"xl/externallinks/[^/]+\.xml")
_DDE_FIELD = re.compile(r"^\s*DDE(?:AUTO)?\b", re.IGNORECASE)
_FORMULA = re.compile(rb"<(?:[A-Za-z_][\w.\-]*:)?f(?:[\t\n\r ][^>]{0,2048})?>([^<]{0,16384})</")
_FORMULA_MAX_MATCH = 2048 + 16384 + 64
_FORMULA_STRING = re.compile(r'"[^"]*"')
_DDE_FORMULA = re.compile(r"[A-Za-z0-9_.$]+\s*\|\s*'?[^'!\r\n]{0,512}'?\s*!")
_CSV_DDE = re.compile(
    r"""(?:^|[,;\t"])[ \t]*[=+\-@][ \t]*[A-Za-z0-9_.$]+[ \t]*\|[ \t]*'?[^'\r\n!]{0,512}'?[ \t]*!""",
    re.MULTILINE,
)
_CSV_DDE_MAX_MATCH = 700
_FIELD_MARKER = re.compile(rb"instrText|fldSimple")
_APPLICATION_LINK = re.compile(rb"<(?:[A-Za-z_][\w.\-]*:)?(ddeLink|oleLink)\b")
_PROTOCOL_HANDLERS = ("mhtml:", "ms-", "search-ms:", "vbscript:", "javascript:", "jar:")

_RELATIONSHIP_RULES: Final[dict[str, tuple[str, Severity, str]]] = {
    "attachedtemplate": (
        "ooxml_remote_template",
        Severity.HIGH,
        "Document loads a remote template (template injection)",
    ),
    "oleobject": ("ooxml_external_ole_object", Severity.HIGH, "Document links an external object"),
    "frame": ("ooxml_external_frame", Severity.HIGH, "Document loads an external frame"),
    "subdocument": ("ooxml_external_frame", Severity.HIGH, "Document loads an external frame"),
    "hyperlink": ("ooxml_external_hyperlink", Severity.INFO, "Document contains web links"),
    "externallinkpath": (
        "ooxml_external_link",
        Severity.WARNING,
        "Workbook links to an external workbook",
    ),
    "image": ("ooxml_external_image", Severity.WARNING, "Document loads a remote image"),
}


def _local_name(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _attribute(element: object, name: str) -> str | None:
    attrib: dict[str, str] = getattr(element, "attrib", {})
    for key, value in attrib.items():
        if key == name or key.endswith("}" + name):
            return value
    return None


class _OoxmlScanner:
    def __init__(self, archive: zipfile.ZipFile, limits: ActiveContentLimits, out: _Findings):
        self._zip = archive
        self._limits = limits
        self._out = out

    def run(self) -> None:
        for info in self._zip.infolist():
            name = info.filename.lower()
            padded = "/" + name
            base = name.rsplit("/", 1)[-1]
            if base == "vbaproject.bin":
                self._out.add("ooxml_macro", Severity.CRITICAL, "Document contains a VBA macro")
            if "/macrosheets/" in padded:
                self._out.add(
                    "ooxml_xlm_macro", Severity.CRITICAL, "Workbook contains an Excel 4.0 macro"
                )
            if "/activex/" in padded:
                self._out.add("ooxml_activex", Severity.HIGH, "Document contains ActiveX controls")
            if "/embeddings/" in padded or base.startswith("oleobject"):
                if name.endswith(".bin"):
                    self._out.add(
                        "ooxml_ole_object", Severity.HIGH, "Document embeds an OLE object"
                    )
                elif not info.is_dir():
                    self._out.add(
                        "ooxml_embedded_object", Severity.WARNING, "Document embeds an object"
                    )
            try:
                if name.endswith(".rels"):
                    self._relationships(info)
                elif _WORD_STORY_PART.fullmatch(name):
                    self._word_fields(info)
                elif _SHEET_PART.fullmatch(name):
                    self._sheet_formulas(info)
                elif _EXTERNAL_LINK_PART.fullmatch(name):
                    self._external_link(info)
            except (zipfile.BadZipFile, OSError, EOFError, RuntimeError, NotImplementedError):
                self._out.add(
                    "ooxml_unreadable_part", Severity.HIGH, "A document part could not be scanned"
                )

    def _chunks(self, info: zipfile.ZipInfo) -> Iterator[bytes]:
        with self._zip.open(info) as part:
            yield from _iter_handle(part)

    # -- relationships ------------------------------------------------------------
    def _relationships(self, info: zipfile.ZipInfo) -> None:
        if info.file_size > self._limits.max_xml_part_bytes:
            self._out.add("ooxml_scan_limit", Severity.HIGH, "A relationship part is too large")
            return
        with self._zip.open(info) as part:
            data = part.read(self._limits.max_xml_part_bytes + 1)
        try:
            root = SafeElementTree.fromstring(data, forbid_dtd=True)
        except DefusedXmlException:
            self._out.add("ooxml_xml_dtd", Severity.HIGH, "Document XML declares a DTD or entities")
            return
        except (SafeElementTree.ParseError, ValueError):
            self._out.add("ooxml_malformed_xml", Severity.WARNING, "Document XML is malformed")
            return
        for element in root.iter():
            if _local_name(element.tag) != "Relationship":
                continue
            if (element.get("TargetMode") or "").strip().lower() != "external":
                continue
            kind = (element.get("Type") or "").rstrip("/").rsplit("/", 1)[-1].lower()
            target = (element.get("Target") or "").strip()
            self._external_relationship(kind, target)

    def _external_relationship(self, kind: str, target: str) -> None:
        code, severity, detail = _RELATIONSHIP_RULES.get(
            kind,
            ("ooxml_external_relationship", Severity.WARNING, "Document references external data"),
        )
        lowered = target.lower()
        if lowered.startswith(_PROTOCOL_HANDLERS):
            self._out.add(
                "ooxml_protocol_handler",
                Severity.HIGH,
                "Document links to an application protocol handler",
            )
        elif lowered.startswith(("\\\\", "file:")):
            self._out.add(
                "ooxml_unc_target",
                Severity.WARNING if kind == "hyperlink" else Severity.HIGH,
                "Document references a network file share",
            )
        host = urlsplit(target).hostname if lowered.startswith(("http:", "https:")) else None
        if host and severity.rank >= Severity.WARNING.rank:
            detail = f"{detail} (host {clean_line_text(host, 100)})"
        self._out.add(code, severity, detail)

    # -- Word fields ------------------------------------------------------------------
    def _word_fields(self, info: zipfile.ZipInfo) -> None:
        if next(_stream_matches(self._chunks(info), _FIELD_MARKER, 16), None) is None:
            return  # no field codes at all: skip the (slower) XML parse
        stack: list[list[str]] = []
        closed: list[bool] = []
        try:
            with self._zip.open(info) as part:
                for _event, element in SafeElementTree.iterparse(
                    part, events=("end",), forbid_dtd=True
                ):
                    tag = _local_name(element.tag)
                    if tag == "fldSimple":
                        self._check_field(_attribute(element, "instr") or "")
                    elif tag == "instrText" and stack and not closed[-1]:
                        stack[-1].append(element.text or "")
                    elif tag == "fldChar":
                        kind = (_attribute(element, "fldCharType") or "").lower()
                        if kind == "begin":
                            stack.append([])
                            closed.append(False)
                        elif kind == "separate" and stack and not closed[-1]:
                            self._check_field("".join(stack[-1]))
                            closed[-1] = True
                        elif kind == "end" and stack:
                            code_parts, was_closed = stack.pop(), closed.pop()
                            if not was_closed:
                                self._check_field("".join(code_parts))
                    element.clear()
        except DefusedXmlException:
            self._out.add("ooxml_xml_dtd", Severity.HIGH, "Document XML declares a DTD or entities")
        except (SafeElementTree.ParseError, ValueError):
            self._out.add("ooxml_malformed_xml", Severity.WARNING, "Document XML is malformed")

    def _check_field(self, code: str) -> None:
        if _DDE_FIELD.match(code):
            self._out.add("ooxml_dde_field", Severity.HIGH, "Document contains a DDE field")

    # -- Excel ------------------------------------------------------------------------
    def _sheet_formulas(self, info: zipfile.ZipInfo) -> None:
        for match in _stream_matches(self._chunks(info), _FORMULA, _FORMULA_MAX_MATCH):
            formula = html.unescape(match.group(1).decode("utf-8", "replace"))
            if "|" in formula and _DDE_FORMULA.search(_FORMULA_STRING.sub("", formula)):
                self._out.add("ooxml_dde_formula", Severity.HIGH, "Workbook contains DDE formulas")

    def _external_link(self, info: zipfile.ZipInfo) -> None:
        for match in _stream_matches(self._chunks(info), _APPLICATION_LINK, 64):
            kind = match.group(1).decode()
            self._out.add(
                "ooxml_dde_link" if kind == "ddeLink" else "ooxml_ole_link",
                Severity.HIGH,
                "Workbook links to another application (DDE/OLE)",
            )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
class ActiveContentScanner:
    """Deterministic active-content heuristics (blocking; run it in a worker thread)."""

    def __init__(self, limits: ActiveContentLimits | None = None) -> None:
        self._limits = limits or ActiveContentLimits()

    def scan(
        self,
        source: ScanSource,
        family: str,
        *,
        extension: str = "",
        text_encoding: str | None = None,
    ) -> list[Finding]:
        """Scan ``source`` of the given family (``pdf`` | ``ooxml`` | ``text``)."""
        out = _Findings()
        with source.open() as handle:
            if contains_eicar(_iter_handle(handle)):
                out.add(
                    "eicar_test_signature",
                    Severity.CRITICAL,
                    "File contains the EICAR anti-virus test signature",
                )
        if family == "pdf":
            self._scan_pdf(source, out)
        elif family == "ooxml":
            self._scan_ooxml(source, out)
        elif family == "text" and extension == "csv":
            self._scan_csv(source, text_encoding or "utf-8", out)
        return out.items()

    def _scan_pdf(self, source: ScanSource, out: _Findings) -> None:
        with source.open() as handle:
            mapped: mmap.mmap | None = None
            try:
                try:
                    mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                except (OSError, ValueError, io.UnsupportedOperation):
                    mapped = None
                data: bytes | mmap.mmap = mapped if mapped is not None else handle.read()
                _PdfScanner(data, self._limits, out).run()
            finally:
                if mapped is not None:
                    mapped.close()

    def _scan_ooxml(self, source: ScanSource, out: _Findings) -> None:
        with source.open() as handle:
            try:
                archive = zipfile.ZipFile(handle)
            except (zipfile.BadZipFile, OSError, EOFError, ValueError):
                out.add("ooxml_unreadable_part", Severity.HIGH, "The archive could not be scanned")
                return
            with archive:
                _OoxmlScanner(archive, self._limits, out).run()

    def _scan_csv(self, source: ScanSource, encoding: str, out: _Findings) -> None:
        decoder = codecs.getincrementaldecoder(encoding)(errors="replace")

        def texts() -> Iterator[str]:
            with source.open() as handle:
                for chunk in _iter_handle(handle):
                    yield decoder.decode(chunk)
                yield decoder.decode(b"", True)

        if any(True for _ in _stream_matches(texts(), _CSV_DDE, _CSV_DDE_MAX_MATCH)):
            out.add("csv_dde_formula", Severity.HIGH, "CSV contains DDE formulas")
