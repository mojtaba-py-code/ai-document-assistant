"""CSV parser (stdlib ``csv``): sniffed dialect limited to ``, ; TAB |``, bounded fields/rows.

The whole file becomes one "page" of table blocks; the header row is repeated at the top
of every block so each block (and later each chunk) is self-describing.
"""

from __future__ import annotations

import csv
import io

from docassist.ingestion.model import DocumentMetadata, Limits, ParsedDocument
from docassist.ingestion.parsers.base import DocumentBuilder, ParseError, decode_text

DELIMITERS = ",;\t|"
_SNIFF_BYTES = 16_384


def _dialect(sample: str) -> type[csv.Dialect] | csv.Dialect:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=DELIMITERS)
    except csv.Error:
        return csv.excel
    if dialect.delimiter not in DELIMITERS:
        return csv.excel
    return dialect


def parse_csv(data: bytes, limits: Limits) -> ParsedDocument:
    builder = DocumentBuilder("csv", limits)
    text = decode_text(data, builder)
    if not text.strip():
        raise ParseError("empty_document")
    csv.field_size_limit(limits.max_csv_field_chars)
    reader = csv.reader(io.StringIO(text, newline=""), _dialect(text[:_SNIFF_BYTES]))
    rows: list[list[str]] = []
    cells = 0
    try:
        for record in reader:
            if not any(cell.strip() for cell in record):
                continue
            row = record[: limits.max_sheet_cols]
            if len(record) > limits.max_sheet_cols:
                builder.warn("columns_truncated")
            rows.append(row)
            cells += len(row)
            if len(rows) >= limits.max_sheet_rows or cells >= limits.max_cells:
                builder.warn("rows_truncated")
                break
    except csv.Error as exc:
        if not rows:
            raise ParseError("parse_error", "malformed csv") from exc
        builder.warn("csv_error_truncated")
    if not rows:
        raise ParseError("empty_document")
    builder.new_page(1)
    builder.add_table(rows)
    if builder.char_count == 0:
        raise ParseError("empty_document")
    return builder.build(DocumentMetadata(page_count=1, page_basis="single"))
