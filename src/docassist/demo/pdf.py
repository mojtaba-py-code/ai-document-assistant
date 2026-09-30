"""A tiny, dependency-free PDF writer for synthetic demo documents.

It emits a plain PDF 1.4 file: one standard Type 1 font (Helvetica, WinAnsi encoding), text
drawn with ``Tj``/``T*`` operators, automatic line wrapping and pagination, an ``/Info``
dictionary and a correct cross-reference table. No JavaScript, actions, annotations,
embedded files or encryption - so the upload scanners accept it and ordinary text
extraction (``pypdf``) recovers every line.
"""

from __future__ import annotations

import textwrap
from datetime import datetime

PAGE_BREAK = "\f"
_WIDTH, _HEIGHT = 612, 792  # US Letter, points
_MARGIN_X, _TOP = 56, 736
_FONT_SIZE, _LEADING = 11, 15
_WRAP = 92
_LINES_PER_PAGE = 44


def _escape(text: str) -> bytes:
    """PDF literal-string escaping in WinAnsi (cp1252); unsupported characters become ``?``."""
    raw = text.encode("cp1252", "replace")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _paginate(lines: list[str]) -> list[list[str]]:
    pages: list[list[str]] = [[]]
    for line in lines:
        if line == PAGE_BREAK:
            if pages[-1]:
                pages.append([])
            continue
        wrapped = textwrap.wrap(line, _WRAP, break_long_words=True) or [""]
        for part in wrapped:
            if len(pages[-1]) >= _LINES_PER_PAGE:
                pages.append([])
            pages[-1].append(part)
    if not pages[-1] and len(pages) > 1:
        pages.pop()
    return pages


def _content_stream(lines: list[str]) -> bytes:
    ops = [b"BT", f"/F1 {_FONT_SIZE} Tf {_LEADING} TL {_MARGIN_X} {_TOP} Td".encode()]
    for index, line in enumerate(lines):
        if index:
            ops.append(b"T*")
        ops.append(b"(" + _escape(line) + b") Tj")
    ops.append(b"ET")
    return b"\n".join(ops)


def build_pdf(lines: list[str], *, title: str, created: datetime) -> bytes:
    """Render ``lines`` (``PAGE_BREAK`` forces a new page) into PDF bytes."""
    pages = _paginate(lines)
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")  # placeholder, filled once the page tree id is known
    tree = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    stamp = created.strftime("D:%Y%m%d%H%M%S")
    info = add(
        b"<< /Title ("
        + _escape(title)
        + b") /Producer (docassist demo generator) /CreationDate ("
        + stamp.encode()
        + b") >>"
    )
    kids: list[int] = []
    for page_lines in pages:
        stream = _content_stream(page_lines)
        content = add(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
        kids.append(
            add(
                f"<< /Type /Page /Parent {tree} 0 R /MediaBox [0 0 {_WIDTH} {_HEIGHT}] "
                f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>".encode()
            )
        )
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {tree} 0 R >>".encode()
    kid_refs = " ".join(f"{k} 0 R" for k in kids)
    objects[tree - 1] = f"<< /Type /Pages /Kids [{kid_refs}] /Count {len(kids)} >>".encode()

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R /Info {info} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)
