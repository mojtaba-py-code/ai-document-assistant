"""PDF parser (pypdf text layer).

* Each PDF page is one page; at most ``limits.max_pages`` pages are read (``pages_truncated``).
* Encrypted PDFs are ``unsupported`` (they cannot be scanned for active content either).
* A page whose text layer has fewer than :data:`SCANNED_MIN_CHARS` characters but carries
  image XObjects is flagged ``needs_ocr``. When asked (``ocr_images=True``), the page's
  images are handed back for OCR - JPEG/JPEG 2000 streams as-is, 8-bit gray/RGB and 1-bit
  gray raw samples converted to PNM - so no image decoder runs outside this sandbox.
* The text layer is segmented into headings (conservative heuristics), bullet lists and
  paragraphs; wrapped lines are re-joined and end-of-line hyphenation removed.
* A page whose extraction fails is skipped with ``page_extract_failed``; if every page
  fails the document is a ``parse_error``.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from typing import Any

from pypdf import PdfReader

from docassist.ingestion.model import DocumentMetadata, Limits, ParsedDocument
from docassist.ingestion.parsers.base import (
    BULLET_LINE,
    DocumentBuilder,
    ParseError,
    clip,
    heading_level,
    join_lines,
)

SCANNED_MIN_CHARS = 24
MAX_IMAGES_PER_PAGE = 4
MAX_XOBJECTS_PER_PAGE = 64
MAX_IMAGE_PIXELS = 40_000_000
_PARAGRAPH_END = (".", "!", "?", ":")
_PDF_DATE = re.compile(r"^D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?")
_RAW_FILTERS = frozenset(
    {"/FlateDecode", "/Fl", "/LZWDecode", "/LZW", "/ASCIIHexDecode", "/AHx", "/ASCII85Decode",
     "/A85", "/RunLengthDecode", "/RL"}
)  # fmt: skip


def _pdf_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = _PDF_DATE.match(value.strip())
    if not match:
        return None
    defaults = ("0000", "01", "01", "00", "00", "00")
    y, mo, d, h, mi, s = (
        part or default for part, default in zip(match.groups(), defaults, strict=True)
    )
    return f"{y}-{mo}-{d}T{h}:{mi}:{s}"


def _metadata(reader: PdfReader, page_count: int) -> DocumentMetadata:
    try:
        info = reader.metadata
        if info is None:
            return DocumentMetadata(page_count=page_count, page_basis="pages")
        return DocumentMetadata(
            title=clip(info.title),
            author=clip(info.author),
            subject=clip(info.subject),
            created=_pdf_date(info.get("/CreationDate")),
            modified=_pdf_date(info.get("/ModDate")),
            page_count=page_count,
            page_basis="pages",
        )
    except Exception:  # noqa: BLE001 - malformed info dictionaries are common; metadata is optional
        return DocumentMetadata(page_count=page_count, page_basis="pages")


# --------------------------------------------------------------------------- #
# Text layer -> blocks
# --------------------------------------------------------------------------- #
def add_text_layer(builder: DocumentBuilder, text: str) -> bool:
    """Segment one page of extracted text into heading / list / paragraph blocks."""
    lines = [line.rstrip() for line in text.splitlines()]
    width = max((len(line.strip()) for line in lines if line.strip()), default=0)
    paragraph: list[str] = []
    items: list[str] = []

    def flush() -> bool:
        ok = True
        if paragraph:
            ok = builder.add_text("paragraph", join_lines(paragraph))
            paragraph.clear()
        if ok and items:
            ok = builder.add_text("list", "\n".join(items))
            items.clear()
        return ok

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not flush():
                return False
            continue
        if not paragraph:
            level = heading_level(stripped)
            if level is not None:
                if not flush() or not builder.add_heading(stripped, level):
                    return False
                continue
        if BULLET_LINE.match(stripped):
            if paragraph and not flush():
                return False
            items.append(stripped)
            continue
        if items and not paragraph and not items[-1].endswith(_PARAGRAPH_END):
            items[-1] = join_lines([items[-1], stripped])
            continue
        if items and not flush():
            return False
        paragraph.append(stripped)
        if stripped.endswith(_PARAGRAPH_END) and len(stripped) < 0.7 * width and not flush():
            return False
    return flush()


# --------------------------------------------------------------------------- #
# Images for OCR
# --------------------------------------------------------------------------- #
def _names(value: Any) -> list[str]:
    if value is None:
        return []
    value = value.get_object() if hasattr(value, "get_object") else value
    if isinstance(value, list):
        return [str(item.get_object() if hasattr(item, "get_object") else item) for item in value]
    return [str(value)]


def _image_xobjects(page: Any) -> Iterator[Any]:
    resources = page.get("/Resources")
    if resources is None:
        return
    resources = resources.get_object()
    xobjects = resources.get("/XObject") if hasattr(resources, "get") else None
    if xobjects is None:
        return
    xobjects = xobjects.get_object()
    for count, name in enumerate(list(xobjects.keys())):
        if count >= MAX_XOBJECTS_PER_PAGE:
            return
        obj = xobjects[name].get_object()
        if hasattr(obj, "get") and obj.get("/Subtype") == "/Image":
            yield obj


def _components(colorspace: Any) -> int | None:
    names = _names(colorspace)
    if not names:
        return None
    if names[0] in ("/DeviceGray", "/CalGray", "/G"):
        return 1
    if names[0] in ("/DeviceRGB", "/CalRGB", "/RGB"):
        return 3
    if names[0] == "/ICCBased":
        cs = colorspace.get_object() if hasattr(colorspace, "get_object") else colorspace
        stream = cs[1].get_object()
        n = int(stream.get("/N", 0))
        return n if n in (1, 3) else None
    return None


def ocr_image(obj: Any) -> tuple[str, bytes] | None:
    """Convert one image XObject into a format the OCR engine accepts, or ``None``."""
    width = int(obj.get("/Width", 0))
    height = int(obj.get("/Height", 0))
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        return None
    filters = _names(obj.get("/Filter"))
    last = filters[-1] if filters else None
    if last in ("/DCTDecode", "/DCT"):
        return "jpeg", obj.get_data()
    if last == "/JPXDecode":
        return "jp2", obj.get_data()
    if last is not None and last not in _RAW_FILTERS:
        return None  # CCITT/JBIG2 and friends: not supported without an image library
    bits = int(obj.get("/BitsPerComponent", 8))
    components = _components(obj.get("/ColorSpace"))
    data = obj.get_data()
    if components == 1 and bits == 8:
        header, expected = f"P5\n{width} {height}\n255\n", width * height
    elif components == 3 and bits == 8:
        header, expected = f"P6\n{width} {height}\n255\n", width * height * 3
    elif components == 1 and bits == 1:
        header, expected = f"P4\n{width} {height}\n", (width + 7) // 8 * height
        data = bytes(b ^ 0xFF for b in data[:expected])  # PDF: 1 = white, PBM: 1 = black
    else:
        return None
    if len(data) < expected:
        return None
    return "pnm", header.encode("ascii") + bytes(data[:expected])


# --------------------------------------------------------------------------- #
def parse_pdf(data: bytes, limits: Limits, *, ocr_images: bool = False) -> ParsedDocument:
    builder = DocumentBuilder("pdf", limits)
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        encrypted = reader.is_encrypted
    except Exception as exc:
        raise ParseError("parse_error", "unreadable pdf") from exc
    if encrypted:
        raise ParseError("unsupported", "encrypted pdf")
    try:
        total = len(reader.pages)
    except Exception as exc:
        raise ParseError("parse_error", "unreadable page tree") from exc
    if total == 0:
        raise ParseError("empty_document")
    if total > limits.max_pages:
        builder.warn("pages_truncated")
    processed = failures = 0
    for index in range(min(total, limits.max_pages)):
        builder.new_page(index + 1)
        processed += 1
        try:
            page = reader.pages[index]
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one broken page must not sink the document
            failures += 1
            builder.warn("page_extract_failed")
            continue
        if len(text.strip()) < SCANNED_MIN_CHARS:
            try:
                images = list(_image_xobjects(page))
            except Exception:  # noqa: BLE001 - unreadable resources only cost OCR coverage
                images = []
                builder.warn("page_images_unreadable")
            if images:
                builder.mark_needs_ocr()
                if ocr_images:
                    _collect_images(builder, images)
        if not add_text_layer(builder, text):
            break
    if failures == processed:
        raise ParseError("parse_error", "no page could be read")
    document = builder.build(_metadata(reader, total))
    if document.char_count == 0 and not document.needs_ocr:
        raise ParseError("empty_document")
    return document


def _collect_images(builder: DocumentBuilder, images: list[Any]) -> None:
    for obj in images[:MAX_IMAGES_PER_PAGE]:
        try:
            converted = ocr_image(obj)
        except Exception:  # noqa: BLE001 - an undecodable image only costs OCR coverage
            builder.warn("ocr_image_unreadable")
            continue
        if converted is None:
            builder.warn("ocr_image_unsupported")
            continue
        if not builder.add_ocr_image(*converted):
            return
