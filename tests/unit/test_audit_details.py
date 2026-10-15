"""Audit details keep identifiers intact and still redact personal data."""

from __future__ import annotations

from docassist.audit.service import _sanitize_details

# The digit-only middle groups look like a phone number to the PII patterns.
DIGIT_UUID = "01a137c3-6012-7345-8123-3c51a221b759"


def test_uuid_with_digit_groups_is_kept() -> None:
    details = {"document_id": DIGIT_UUID, "document_ids": [DIGIT_UUID, DIGIT_UUID.upper()]}
    assert _sanitize_details(details) == details


def test_personal_data_next_to_ids_is_still_redacted() -> None:
    clean = _sanitize_details(
        {"query": "call +1 415 555 0100", "ids": [DIGIT_UUID, "alice@example.com"]}
    )
    assert clean["query"] == "call [REDACTED:PHONE]"
    assert clean["ids"] == [DIGIT_UUID, "[REDACTED:EMAIL]"]


def test_text_that_only_contains_a_uuid_is_redacted_as_text() -> None:
    clean = _sanitize_details({"note": f"id {DIGIT_UUID}"})
    assert DIGIT_UUID not in clean["note"]
