"""Rules-based field extraction on realistic contract and invoice text."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from docassist.ingestion.extraction_rules import (
    MAX_EVIDENCE_CHARS,
    MAX_FIELDS,
    MAX_UNLABELLED_AMOUNTS,
    FieldCandidate,
    evidence,
    extract_fields,
    parse_amount,
)
from tests.helpers_ingestion import CONTRACT_TEXT, INVOICE_TEXT


def by_field(fields: list[FieldCandidate], name: str) -> list[FieldCandidate]:
    return [f for f in fields if f.field == name]


def test_contract_fields() -> None:
    fields = extract_fields(CONTRACT_TEXT)
    [effective] = by_field(fields, "effective_date")
    assert effective.value_date == date(2026, 1, 15) and effective.confidence >= 0.8
    assert date(2027, 12, 31) in {f.value_date for f in by_field(fields, "expiration_date")}
    [renewal] = by_field(fields, "renewal_date")
    assert (
        renewal.value_date is None
        and renewal.value_text == "01/12/2027"
        and renewal.confidence < 0.5
    )
    assert {f.value_text for f in by_field(fields, "party")} == {
        "Acme Corporation",
        "Beta Supplies Ltd.",
    }
    terms = {f.value_text: f.value_number for f in by_field(fields, "payment_terms")}
    assert terms["Net 45"] == Decimal(45) and terms["Within 45 days"] == Decimal(45)
    [value] = by_field(fields, "contract_value")
    assert (value.value_number, value.currency) == (Decimal("250000.00"), "USD")


def test_invoice_fields() -> None:
    fields = extract_fields(INVOICE_TEXT)
    assert by_field(fields, "invoice_number")[0].value_text == "INV-2026-0042"
    assert by_field(fields, "invoice_date")[0].value_date == date(2026, 3, 3)
    due = by_field(fields, "due_date")[0]
    assert due.value_date is None and due.value_text == "02/04/2026"  # 2 April or 4 February?
    labelled = {
        f.field: f.value_number
        for f in fields
        if f.field in {"subtotal", "tax_amount", "amount_due"}
    }
    assert labelled == {
        "subtotal": Decimal("1820.50"),
        "tax_amount": Decimal("364.10"),
        "amount_due": Decimal("2184.60"),
    }
    assert all(
        f.currency == "USD" and f.confidence < 0.85 for f in fields if f.field == "amount_due"
    )  # "$" is ambiguous
    assert by_field(fields, "payment_terms")[0].value_text == "Within 30 days"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Effective date: 2026-10-14.", date(2026, 10, 14)),
        ("effective as of 14 October 2026", date(2026, 10, 14)),
        ("Effective on the 3rd day of March, 2026", date(2026, 3, 3)),
        ("Commencement date: October 14, 2026", date(2026, 10, 14)),
        ("Effective Date: Oct. 14, 2026", date(2026, 10, 14)),
        ("Effective Date: 14/10/2026", date(2026, 10, 14)),
        ("Effective Date: 10/14/2026", date(2026, 10, 14)),
        ("Effective Date: 05.05.2026", date(2026, 5, 5)),
    ],
)
def test_date_formats(text: str, expected: date) -> None:
    [field] = by_field(extract_fields(text), "effective_date")
    assert field.value_date == expected


def test_impossible_and_unlabelled_dates_are_ignored() -> None:
    assert extract_fields("Effective Date: 31/02/2026 and 2026-13-45.") == []
    assert extract_fields("The meeting on 2026-05-01 went well.") == []


def test_nearest_label_wins() -> None:
    fields = extract_fields("The agreement is effective 2026-01-01 and expires on 2027-01-01.")
    assert {(f.field, f.value_date) for f in fields} == {
        ("effective_date", date(2026, 1, 1)),
        ("expiration_date", date(2027, 1, 1)),
    }


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("1,234.56", Decimal("1234.56")),
        ("1.234,56", Decimal("1234.56")),
        ("1'234.50", Decimal("1234.50")),
        ("1234,5", Decimal("1234.5")),
        ("1,234", Decimal("1234")),
        ("1.234", Decimal("1234")),
        ("12.5", Decimal("12.5")),
        ("9" * 16, None),
    ],
)
def test_parse_amount(raw: str, value: Decimal | None) -> None:
    assert parse_amount(raw) == value


@pytest.mark.parametrize(
    ("text", "currency", "amount"),
    [
        ("Total: " + chr(0x20AC) + "1.200,50", "EUR", Decimal("1200.50")),
        ("Total: 1.200,50 " + chr(0x20AC), "EUR", Decimal("1200.50")),
        ("Total: " + chr(0x00A3) + "99", "GBP", Decimal("99")),
        ("Total: CHF 5'000.00", "CHF", Decimal("5000.00")),
        ("Total: 300 EUR", "EUR", Decimal("300")),
        ("Total: C$ 45.10", "CAD", Decimal("45.10")),
    ],
)
def test_currencies(text: str, currency: str, amount: Decimal) -> None:
    [field] = by_field(extract_fields(text), "total_amount")
    assert (field.currency, field.value_number) == (currency, amount)


def test_lowercase_words_are_not_currency_codes() -> None:
    assert extract_fields("we will try 5 times and eur 3 is not a code") == []


def test_unlabelled_amounts_are_capped() -> None:
    text = "\n".join(f"Line {i}: USD {i}.00" for i in range(1, 60))
    assert len(by_field(extract_fields(text), "amount")) == MAX_UNLABELLED_AMOUNTS


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Payment terms: Net 30.", "Net 30"),
        ("Terms 2/10 net30", "Net 30"),
        ("Invoices are payable within 60 calendar days of receipt.", "Within 60 days"),
        ("Payment is due upon receipt.", "Due on receipt"),
    ],
)
def test_payment_terms(text: str, label: str) -> None:
    assert label in {f.value_text for f in by_field(extract_fields(text), "payment_terms")}


def test_net_income_is_not_payment_terms() -> None:
    assert (
        by_field(
            extract_fields("Net income was 5,000 and net 12% growth; net 3 million."),
            "payment_terms",
        )
        == []
    )


def test_parties_need_proper_names() -> None:
    fields = extract_fields(
        "This NDA is made between the Company (Globex Inc.) and Jane Doe, an individual."
    )
    assert [f.value_text for f in by_field(fields, "party")] == ["Company", "Jane Doe"]
    assert by_field(extract_fields("Calls are between 9am and 5pm on weekdays."), "party") == []


def test_invoice_numbers_need_a_digit() -> None:
    assert by_field(extract_fields("Invoice number: ABCDEF"), "invoice_number") == []
    assert (
        by_field(extract_fields("Invoice #: 2026/77"), "invoice_number")[0].value_text == "2026/77"
    )


def test_evidence_is_bounded_and_contains_the_match() -> None:
    text = "x " * 400 + "Due Date: 2026-05-01 " + "y " * 400
    [field] = extract_fields(text)
    assert len(field.evidence) <= MAX_EVIDENCE_CHARS and "2026-05-01" in field.evidence
    assert text[field.start : field.end] == "2026-05-01"
    assert evidence("abc", 0, 3) == "abc"


def test_hostile_input_is_fast_enough() -> None:
    import time

    started = time.perf_counter()
    extract_fields(("between " * 2_000) + ("1," * 5_000) + ("$" * 5_000) + ("due by " * 2_000))
    many = extract_fields("Due date: 2026-01-01 USD 1.00\n" * 40_000)  # 80k mentions
    assert len(many) <= MAX_FIELDS
    assert time.perf_counter() - started < 60  # linear-ish; generous for slow CI machines
