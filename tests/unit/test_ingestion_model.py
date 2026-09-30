"""Strict validation of sandbox output (the parent never trusts the child's JSON)."""

from __future__ import annotations

import base64
import copy
from typing import Any

import pytest

from docassist.ingestion.model import (
    Block,
    DocumentMetadata,
    Limits,
    ModelValidationError,
    OcrImage,
    ParsedDocument,
    ParsedPage,
    parsed_document_from_json,
)


def valid() -> dict[str, Any]:
    document = ParsedDocument(
        format="pdf",
        pages=(
            ParsedPage(
                1, (Block("heading", "Title", 1), Block("paragraph", "Body", hidden="h")), False
            ),
            ParsedPage(2, (Block("table", "a | b\n1 | 2"),), True),
        ),
        metadata=DocumentMetadata(title="T", page_count=2, sheet_names=("S",), page_basis="pages"),
        warnings=("pages_truncated",),
        needs_ocr=True,
        network_isolated=True,
        rlimits_applied=False,
        ocr_images=(OcrImage(2, "jpeg", b"\xff\xd8data"),),
    )
    return document.to_json()


def test_limits_argument_round_trip() -> None:
    limits = Limits(max_pages=12, max_total_chars=999)
    assert Limits.from_arg(limits.to_arg()) == limits
    assert Limits.from_arg("") == Limits()


@pytest.mark.parametrize(
    "raw", ["max_pages", "bogus=1", "max_pages=-1", "max_pages=1e3", "max_pages=" + "9" * 13]
)
def test_limits_argument_rejects_garbage(raw: str) -> None:
    with pytest.raises(ValueError):
        Limits.from_arg(raw)


def test_limits_must_be_positive_integers() -> None:
    with pytest.raises(ValueError):
        Limits(max_pages=0)
    with pytest.raises(ValueError):
        Limits(max_blocks=True)


def test_valid_document_round_trips() -> None:
    parsed = parsed_document_from_json(valid(), expected_format="pdf", limits=Limits())
    assert parsed.to_json() == valid()
    assert parsed.char_count == len("Title") + len("Body") + len("a | b\n1 | 2")
    assert parsed.ocr_images[0].data == b"\xff\xd8data"


def mutate(path: list[Any], value: Any) -> dict[str, Any]:
    doc = copy.deepcopy(valid())
    target = doc
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return doc


BAD: list[tuple[str, dict[str, Any]]] = [
    ("not a dict", []),  # type: ignore[list-item]
    ("format mismatch", mutate(["format"], "docx")),
    ("extra key", {**valid(), "extra": 1}),
    ("missing key", mutate(["warnings"], KeyError)),
    ("pages not list", mutate(["pages"], {})),
    ("page number bool", mutate(["pages", 0, "number"], True)),
    ("page numbers not increasing", mutate(["pages", 1, "number"], 1)),
    ("unknown block kind", mutate(["pages", 0, "blocks", 0, "kind"], "script")),
    ("heading without level", mutate(["pages", 0, "blocks", 0, "level"], None)),
    ("heading level too deep", mutate(["pages", 0, "blocks", 0, "level"], 10)),
    ("level on paragraph", mutate(["pages", 0, "blocks", 1, "level"], 2)),
    ("text not string", mutate(["pages", 0, "blocks", 1, "text"], 42)),
    ("extra block key", mutate(["pages", 0, "blocks", 1, "html"], "<b>")),
    ("needs_ocr not bool", mutate(["needs_ocr"], "yes")),
    ("warning code", mutate(["warnings"], ["Bad Code!"])),
    ("warning type", mutate(["warnings"], [1])),
    ("metadata extra", mutate(["metadata", "script"], "x")),
    ("metadata too long", mutate(["metadata", "title"], "x" * 501)),
    ("page basis", mutate(["metadata", "page_basis"], "magic")),
    ("sheet names type", mutate(["metadata", "sheet_names"], "S")),
    ("ocr image page", mutate(["ocr_images", 0, "page"], 9)),
    ("ocr image format", mutate(["ocr_images", 0, "format"], "exe")),
    ("ocr image base64", mutate(["ocr_images", 0, "data"], "***")),
    ("ocr image empty", mutate(["ocr_images", 0, "data"], "")),
]


@pytest.mark.parametrize(("name", "payload"), BAD, ids=[b[0] for b in BAD])
def test_invalid_output_is_rejected(name: str, payload: Any) -> None:
    with pytest.raises(ModelValidationError):
        parsed_document_from_json(payload, expected_format="pdf", limits=Limits())


@pytest.mark.parametrize(
    "limits",
    [
        Limits(max_pages=1),
        Limits(max_blocks=2),
        Limits(max_block_chars=4),
        Limits(max_total_chars=10),
        Limits(max_hidden_chars=1),
        Limits(max_ocr_images=1, max_ocr_bytes=3),
    ],
    ids=["pages", "blocks", "block-chars", "total-chars", "hidden-chars", "ocr-bytes"],
)
def test_limits_are_enforced_on_untrusted_output(limits: Limits) -> None:
    doc = valid()
    doc["pages"][0]["blocks"][1]["hidden"] = "hh"
    with pytest.raises(ModelValidationError):
        parsed_document_from_json(doc, expected_format="pdf", limits=limits)


def test_duplicate_warnings_are_collapsed() -> None:
    doc = mutate(["warnings"], ["a_b", "a_b"])
    assert parsed_document_from_json(doc, expected_format="pdf", limits=Limits()).warnings == (
        "a_b",
    )


def test_base64_images_are_decoded() -> None:
    doc = mutate(["ocr_images", 0, "data"], base64.b64encode(b"\x89PNG").decode())
    assert (
        parsed_document_from_json(doc, expected_format="pdf", limits=Limits()).ocr_images[0].data
        == b"\x89PNG"
    )
