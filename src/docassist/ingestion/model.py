"""Parser output model shared by the sandbox child and the parent process.

This module is imported by the sandbox child, so it depends on the standard library only.

The child serialises a :class:`ParsedDocument` to JSON; the parent **never trusts** that
JSON - :func:`parsed_document_from_json` re-validates every type, size and count against
the same :class:`Limits` the child was given before any value is used. A compromised or
confused child can therefore produce a parse failure, but not an oversized, malformed or
unexpected object inside the worker.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field, fields
from typing import Any

FORMATS: tuple[str, ...] = ("pdf", "docx", "xlsx", "csv", "txt", "md")
BLOCK_KINDS: tuple[str, ...] = ("heading", "paragraph", "table", "list")
OCR_IMAGE_FORMATS: tuple[str, ...] = ("jpeg", "jp2", "pnm")
PAGE_BASES: tuple[str, ...] = (
    "pages",
    "rendered_breaks",
    "page_breaks",
    "sections",
    "sheets",
    "single",
)
ERROR_CODES: tuple[str, ...] = ("parse_error", "unsupported", "empty_document", "too_large")
MAX_HEADING_LEVEL = 9
MAX_METADATA_CHARS = 500
MAX_SHEET_NAME_CHARS = 200
MAX_SHEET_NAMES = 1_000
_WARNING_RE = re.compile(r"^[a-z0-9_]{1,48}$")
_METADATA_TEXT_KEYS = ("title", "author", "subject", "created", "modified")


class ModelValidationError(ValueError):
    """The sandbox output does not match the expected shape or limits."""


@dataclass(frozen=True, slots=True)
class Limits:
    """Budgets shared by the parsers (enforced while parsing) and the parent (re-checked)."""

    max_pages: int = 2_000
    max_blocks: int = 100_000
    max_block_chars: int = 100_000
    max_total_chars: int = 8_000_000
    max_hidden_chars: int = 20_000
    max_sheet_rows: int = 100_000
    max_sheet_cols: int = 200
    max_cells: int = 2_000_000
    max_csv_field_chars: int = 131_072
    max_warnings: int = 64
    max_ocr_images: int = 64
    max_ocr_bytes: int = 16_777_216

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"limit {item.name} must be a positive integer")

    def to_arg(self) -> str:
        """Compact ``key=value,...`` form for the child's command line (no quoting needed)."""
        return ",".join(f"{item.name}={getattr(self, item.name)}" for item in fields(self))

    @classmethod
    def from_arg(cls, raw: str) -> Limits:
        known = {item.name for item in fields(cls)}
        values: dict[str, int] = {}
        for part in raw.split(","):
            if not part:
                continue
            key, sep, value = part.partition("=")
            if not sep or key not in known or not value.isdigit() or len(value) > 12:
                raise ValueError("malformed limits argument")
            values[key] = int(value)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class Block:
    """One structural unit of text. ``level`` is set for headings only.

    ``hidden`` holds text a human reader would not see in the rendered document (hidden or
    white runs, hidden sheets). It is also part of ``text``; it is kept separately so the
    injection scanner can treat it as a hidden channel.
    """

    kind: str
    text: str
    level: int | None = None
    hidden: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "level": self.level, "hidden": self.hidden}


@dataclass(frozen=True, slots=True)
class ParsedPage:
    number: int
    blocks: tuple[Block, ...] = ()
    needs_ocr: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "blocks": [block.to_json() for block in self.blocks],
            "needs_ocr": self.needs_ocr,
        }


@dataclass(frozen=True, slots=True)
class OcrImage:
    """An image extracted (inside the sandbox) from a page that needs OCR."""

    page: int
    format: str
    data: bytes

    def to_json(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "format": self.format,
            "data": base64.b64encode(self.data).decode("ascii"),
        }


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    created: str | None = None
    modified: str | None = None
    page_count: int | None = None
    sheet_names: tuple[str, ...] = ()
    page_basis: str = "pages"

    def to_json(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "created": self.created,
            "modified": self.modified,
            "page_count": self.page_count,
            "sheet_names": list(self.sheet_names),
            "page_basis": self.page_basis,
        }


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    format: str
    pages: tuple[ParsedPage, ...]
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    warnings: tuple[str, ...] = ()
    needs_ocr: bool = False
    network_isolated: bool = False
    rlimits_applied: bool = False
    ocr_images: tuple[OcrImage, ...] = ()

    @property
    def char_count(self) -> int:
        return sum(len(block.text) for page in self.pages for block in page.blocks)

    def with_sandbox_facts(
        self, *, network_isolated: bool, rlimits_applied: bool
    ) -> ParsedDocument:
        return ParsedDocument(
            format=self.format,
            pages=self.pages,
            metadata=self.metadata,
            warnings=self.warnings,
            needs_ocr=self.needs_ocr,
            network_isolated=network_isolated,
            rlimits_applied=rlimits_applied,
            ocr_images=self.ocr_images,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "pages": [page.to_json() for page in self.pages],
            "metadata": self.metadata.to_json(),
            "warnings": list(self.warnings),
            "needs_ocr": self.needs_ocr,
            "network_isolated": self.network_isolated,
            "rlimits_applied": self.rlimits_applied,
            "ocr_images": [image.to_json() for image in self.ocr_images],
        }


# --------------------------------------------------------------------------- #
# Strict validation of untrusted JSON (parent side)
# --------------------------------------------------------------------------- #
def _fail(message: str) -> ModelValidationError:
    return ModelValidationError(message)


def _exact_keys(obj: object, keys: tuple[str, ...], where: str) -> dict[str, Any]:
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise _fail(f"{where}: unexpected shape")
    return obj


def _int(value: object, where: str, *, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise _fail(f"{where}: integer out of range")
    return value


def _bool(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(f"{where}: expected a boolean")
    return value


def _str(value: object, where: str, *, max_chars: int) -> str:
    if not isinstance(value, str) or len(value) > max_chars:
        raise _fail(f"{where}: expected a bounded string")
    return value


def _opt_str(value: object, where: str, *, max_chars: int) -> str | None:
    return None if value is None else _str(value, where, max_chars=max_chars)


def _metadata(obj: object, limits: Limits) -> DocumentMetadata:
    keys = (*_METADATA_TEXT_KEYS, "page_count", "sheet_names", "page_basis")
    data = _exact_keys(obj, keys, "metadata")
    texts = {
        key: _opt_str(data[key], key, max_chars=MAX_METADATA_CHARS) for key in _METADATA_TEXT_KEYS
    }
    page_count = data["page_count"]
    if page_count is not None:
        page_count = _int(page_count, "page_count", minimum=0, maximum=10_000_000)
    names = data["sheet_names"]
    if not isinstance(names, list) or len(names) > MAX_SHEET_NAMES:
        raise _fail("sheet_names: expected a bounded list")
    sheet_names = tuple(_str(n, "sheet_name", max_chars=MAX_SHEET_NAME_CHARS) for n in names)
    basis = data["page_basis"]
    if basis not in PAGE_BASES:
        raise _fail("page_basis: unknown value")
    return DocumentMetadata(
        title=texts["title"],
        author=texts["author"],
        subject=texts["subject"],
        created=texts["created"],
        modified=texts["modified"],
        page_count=page_count,
        sheet_names=sheet_names,
        page_basis=basis,
    )


def parsed_document_from_json(
    obj: object, *, expected_format: str, limits: Limits
) -> ParsedDocument:
    """Validate untrusted sandbox output and build a :class:`ParsedDocument` from it."""
    keys = (
        "format", "pages", "metadata", "warnings", "needs_ocr",
        "network_isolated", "rlimits_applied", "ocr_images",
    )  # fmt: skip
    data = _exact_keys(obj, keys, "document")
    if data["format"] != expected_format or expected_format not in FORMATS:
        raise _fail("document: unexpected format")
    raw_pages = data["pages"]
    if not isinstance(raw_pages, list) or len(raw_pages) > limits.max_pages:
        raise _fail("pages: expected a bounded list")
    pages: list[ParsedPage] = []
    total_chars = 0
    hidden_chars = 0
    block_count = 0
    last_number = 0
    for raw_page in raw_pages:
        page = _exact_keys(raw_page, ("number", "blocks", "needs_ocr"), "page")
        number = _int(page["number"], "page.number", minimum=last_number + 1, maximum=10_000_000)
        last_number = number
        raw_blocks = page["blocks"]
        if not isinstance(raw_blocks, list):
            raise _fail("page.blocks: expected a list")
        blocks: list[Block] = []
        for raw_block in raw_blocks:
            block_count += 1
            if block_count > limits.max_blocks:
                raise _fail("too many blocks")
            item = _exact_keys(raw_block, ("kind", "text", "level", "hidden"), "block")
            kind = item["kind"]
            if kind not in BLOCK_KINDS:
                raise _fail("block.kind: unknown value")
            text = _str(item["text"], "block.text", max_chars=limits.max_block_chars)
            hidden = _str(item["hidden"], "block.hidden", max_chars=limits.max_hidden_chars)
            level = item["level"]
            if kind == "heading":
                level = _int(level, "block.level", minimum=1, maximum=MAX_HEADING_LEVEL)
            elif level is not None:
                raise _fail("block.level: only headings have a level")
            total_chars += len(text)
            hidden_chars += len(hidden)
            if total_chars > limits.max_total_chars or hidden_chars > limits.max_hidden_chars:
                raise _fail("text budget exceeded")
            blocks.append(Block(kind=kind, text=text, level=level, hidden=hidden))
        pages.append(
            ParsedPage(
                number=number,
                blocks=tuple(blocks),
                needs_ocr=_bool(page["needs_ocr"], "page.needs_ocr"),
            )
        )
    warnings = data["warnings"]
    if not isinstance(warnings, list) or len(warnings) > limits.max_warnings:
        raise _fail("warnings: expected a bounded list")
    for warning in warnings:
        if not isinstance(warning, str) or not _WARNING_RE.fullmatch(warning):
            raise _fail("warnings: invalid code")
    images = _ocr_images(data["ocr_images"], limits, {page.number for page in pages})
    return ParsedDocument(
        format=expected_format,
        pages=tuple(pages),
        metadata=_metadata(data["metadata"], limits),
        warnings=tuple(dict.fromkeys(warnings)),
        needs_ocr=_bool(data["needs_ocr"], "needs_ocr"),
        network_isolated=_bool(data["network_isolated"], "network_isolated"),
        rlimits_applied=_bool(data["rlimits_applied"], "rlimits_applied"),
        ocr_images=images,
    )


def _ocr_images(obj: object, limits: Limits, page_numbers: set[int]) -> tuple[OcrImage, ...]:
    if not isinstance(obj, list) or len(obj) > limits.max_ocr_images:
        raise _fail("ocr_images: expected a bounded list")
    images: list[OcrImage] = []
    total = 0
    for raw in obj:
        item = _exact_keys(raw, ("page", "format", "data"), "ocr_image")
        page = _int(item["page"], "ocr_image.page", minimum=1, maximum=10_000_000)
        if page not in page_numbers:
            raise _fail("ocr_image.page: unknown page")
        if item["format"] not in OCR_IMAGE_FORMATS:
            raise _fail("ocr_image.format: unknown value")
        encoded = _str(item["data"], "ocr_image.data", max_chars=limits.max_ocr_bytes * 2)
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise _fail("ocr_image.data: invalid base64") from exc
        total += len(data)
        if not data or total > limits.max_ocr_bytes:
            raise _fail("ocr_images: byte budget exceeded")
        images.append(OcrImage(page=page, format=item["format"], data=data))
    return tuple(images)
