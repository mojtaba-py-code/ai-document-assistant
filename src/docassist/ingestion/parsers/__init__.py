"""Format parsers. Imported by the sandbox child: stdlib, pypdf, docx, openpyxl only.

:func:`parse_document` is the single entry point: it dispatches on the (already validated)
format and turns every failure into a :class:`ParseError` with a small, fixed error code, so
no exception text - which could quote document content - ever leaves the parser.

Parser modules are imported lazily: a CSV never loads pypdf or openpyxl into the child,
which keeps start-up fast and the attack surface per format minimal.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import cast

from docassist.ingestion.model import FORMATS, Limits, ParsedDocument
from docassist.ingestion.parsers.base import ParseError

Parser = Callable[[bytes, Limits], ParsedDocument]

PARSERS: dict[str, tuple[str, str]] = {
    "pdf": ("docassist.ingestion.parsers.pdf", "parse_pdf"),
    "docx": ("docassist.ingestion.parsers.docx", "parse_docx"),
    "xlsx": ("docassist.ingestion.parsers.xlsx", "parse_xlsx"),
    "csv": ("docassist.ingestion.parsers.csv_", "parse_csv"),
    "txt": ("docassist.ingestion.parsers.text", "parse_txt"),
    "md": ("docassist.ingestion.parsers.text", "parse_md"),
}
if set(PARSERS) != set(FORMATS):  # pragma: no cover - import-time consistency check
    raise RuntimeError("every supported format needs exactly one parser")


def get_parser(fmt: str) -> Parser:
    module_name, attribute = PARSERS[fmt]
    return cast(Parser, getattr(importlib.import_module(module_name), attribute))


def parse_document(
    fmt: str, data: bytes, limits: Limits, *, ocr_images: bool = False
) -> ParsedDocument:
    """Parse ``data`` as ``fmt``. Raises :class:`ParseError` (and nothing else) on failure."""
    if fmt not in PARSERS:
        raise ParseError("unsupported", "unknown format")
    if not data:
        raise ParseError("empty_document")
    try:
        if fmt == "pdf":
            from docassist.ingestion.parsers.pdf import parse_pdf

            return parse_pdf(data, limits, ocr_images=ocr_images)
        return get_parser(fmt)(data, limits)
    except ParseError:
        raise
    except MemoryError as exc:
        raise ParseError("too_large", "memory limit reached") from exc
    except RecursionError as exc:
        raise ParseError("parse_error", "structure nested too deeply") from exc
    except Exception as exc:
        raise ParseError("parse_error", type(exc).__name__) from exc


__all__ = ["PARSERS", "ParseError", "get_parser", "parse_document"]
