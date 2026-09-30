"""Structured extraction: schemas, value parsing, evidence verification, merging."""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from docassist.intelligence.access import ChunkView, FieldRow
from docassist.intelligence.extraction import (
    CONTRACT_FIELDS,
    INVOICE_FIELDS,
    candidate_value,
    confidence_for,
    currency_in,
    extraction_schema,
    locate_evidence,
    merge_candidates,
    merge_with_rules,
    normalize_currency,
    parse_value,
    select_chunks,
    value_support,
    verify_output,
)

C1 = ChunkView(
    id=uuid.uuid4(),
    index=0,
    content=(
        "This Master Services Agreement is made between Alpha Ltd and Beta Inc. "
        "It is effective from 1 January 2026 and expires on 31 December 2027."
    ),
    page_start=1,
    page_end=1,
    section="Parties",
)
C2 = ChunkView(
    id=uuid.uuid4(),
    index=1,
    content=(
        "Payment terms: net 30 days. The total contract value is USD 120,000.00. "
        "Late payments incur a fee of 1.5% per month. This Agreement is governed by the "
        "laws of England."
    ),
    page_start=2,
    page_end=2,
    section="Payment",
)
BATCH = {"C1": C1, "C2": C2}


def value(
    found: bool = True, v: str = "", evidence: str = "", source: str = "C1"
) -> dict[str, Any]:
    return {"found": found, "value": v, "evidence": evidence, "source": source}


def absent() -> dict[str, Any]:
    return value(found=False, source="")


def contract_output(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {spec.name: absent() for spec in CONTRACT_FIELDS}
    data["parties"] = []
    data.update(overrides)
    return data


def _walk(schema: dict[str, Any]) -> list[dict[str, Any]]:
    found = [schema]
    for sub in schema.get("properties", {}).values():
        found += _walk(sub)
    if "items" in schema:
        found += _walk(schema["items"])
    return found


@pytest.mark.parametrize("kind", ["contract", "invoice"])
def test_extraction_schema_is_strict_everywhere(kind: str) -> None:
    schema = extraction_schema(kind)  # type: ignore[arg-type]
    objects = [s for s in _walk(schema) if s.get("type") == "object"]
    assert objects
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    names = {spec.name for spec in (CONTRACT_FIELDS if kind == "contract" else INVOICE_FIELDS)}
    assert names <= set(schema["properties"])
    if kind == "invoice":
        assert schema["properties"]["line_items"]["maxItems"] == 50


def test_contract_and_invoice_cover_the_specified_fields() -> None:
    assert [s.name for s in CONTRACT_FIELDS] == [
        "parties", "effective_date", "expiration_date", "renewal_terms", "payment_terms",
        "payment_deadline_days", "late_penalty", "currency", "total_value", "governing_law",
        "termination_notice_days",
    ]  # fmt: skip
    assert [s.name for s in INVOICE_FIELDS] == [
        "invoice_number", "vendor", "customer", "issue_date", "due_date", "currency",
        "subtotal", "tax", "total",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("kind", "raw", "display"),
    [
        ("date", "2027-12-31", "2027-12-31"),
        ("date", "31 December 2027", "2027-12-31"),
        ("number", "USD 120,000.00", "120000"),
        ("integer", "30", "30"),
        ("currency", "usd", "USD"),
        ("currency", chr(0x20AC), "EUR"),
        ("text", "  Net 30\u0000 days ", "Net 30 days"),
    ],
)
def test_parse_value_valid(kind: str, raw: str, display: str) -> None:
    parsed = parse_value(kind, raw)  # type: ignore[arg-type]
    assert parsed is not None and parsed.display == display


@pytest.mark.parametrize(
    ("kind", "raw"),
    [
        ("date", "next year"),
        ("date", "03/04/2026"),  # ambiguous numeric date is never guessed
        ("number", "a lot"),
        ("integer", "30.5"),
        ("integer", "-3"),
        ("integer", "99999"),
        ("currency", "dollars"),
        ("text", "   "),
    ],
)
def test_parse_value_invalid(kind: str, raw: str) -> None:
    assert parse_value(kind, raw) is None  # type: ignore[arg-type]


def test_value_support_by_kind() -> None:
    date_value = parse_value("date", "2027-12-31")
    assert date_value is not None
    assert value_support("date", date_value, "expires on 31 December 2027") == (True, False)
    assert value_support("date", date_value, "expires two years later") == (False, False)
    ambiguous = parse_value("date", "2026-04-03")
    assert ambiguous is not None
    assert value_support("date", ambiguous, "signed 03/04/2026") == (True, True)
    number = parse_value("number", "120000")
    assert number is not None
    assert value_support("number", number, "value is USD 120,000.00")[0]
    assert not value_support("number", number, "value is USD 12,000.00")[0]
    currency = parse_value("currency", "USD")
    assert currency is not None
    assert value_support("currency", currency, "USD 120,000")[0]
    assert value_support("currency", currency, "$120,000")[0]
    assert not value_support("currency", currency, "EUR 120,000")[0]
    text = parse_value("text", "laws of England")
    assert text is not None
    assert value_support("text", text, "governed by the laws of England")[0]
    assert not value_support("text", text, "governed by French law")[0]


def test_locate_evidence_prefers_cited_chunk_then_others() -> None:
    assert locate_evidence("expires on 31 December 2027", "C1", BATCH) == (C1, 1.0, False)
    chunk, score, moved = locate_evidence("net 30 days", "c1", BATCH)
    assert chunk is C2 and score == 1.0 and moved
    assert locate_evidence("a sentence that is not there at all", "C1", BATCH) == (None, 0.0, False)
    assert locate_evidence("expires on 31 December 2027", "C404", BATCH)[0] is C1


def test_confidence_rules() -> None:
    assert confidence_for(1.0, moved=False, supported=True, ambiguous=False) == 0.9
    assert confidence_for(0.9, moved=False, supported=True, ambiguous=False) == 0.8
    assert confidence_for(1.0, moved=True, supported=True, ambiguous=False) == 0.8
    assert confidence_for(1.0, moved=False, supported=True, ambiguous=True) == 0.6
    assert confidence_for(1.0, moved=False, supported=False, ambiguous=False) == 0.5


def test_verify_output_keeps_verified_values_and_counts_the_rest() -> None:
    data = contract_output(
        parties=[
            value(v="Alpha Ltd", evidence="made between Alpha Ltd and Beta Inc", source="C1"),
            value(v="Beta Inc", evidence="made between Alpha Ltd and Beta Inc", source="C1"),
        ],
        expiration_date=value(v="2027-12-31", evidence="expires on 31 December 2027", source="C1"),
        payment_terms=value(v="net 30 days", evidence="Payment terms: net 30 days", source="C1"),
        total_value=value(
            v="120000", evidence="total contract value is USD 120,000.00", source="C2"
        ),
        governing_law=value(
            v="England", evidence="Governed by Martian law since 2090", source="C2"
        ),
        payment_deadline_days=value(v="thirty", evidence="net 30 days", source="C2"),
        currency=value(v="USD", evidence="USD 120,000.00", source="C2"),
    )
    outcome = verify_output("contract", data, BATCH)
    assert outcome is not None
    got = {(c.field, c.parsed.display): c for c in outcome.candidates}
    assert set(got) == {
        ("parties", "Alpha Ltd"),
        ("parties", "Beta Inc"),
        ("expiration_date", "2027-12-31"),
        ("payment_terms", "net 30 days"),
        ("total_value", "120000"),
        ("currency", "USD"),
    }
    assert outcome.unverified == 1  # the invented governing-law quote
    assert outcome.invalid == 1  # "thirty" is not an integer
    moved = got[("payment_terms", "net 30 days")]
    assert moved.chunk is C2 and moved.confidence == 0.8  # cited C1, found in C2
    assert got[("expiration_date", "2027-12-31")].confidence == 0.9


def test_verify_output_rejects_non_objects_and_bad_entries() -> None:
    assert verify_output("contract", None, BATCH) is None
    assert verify_output("contract", ["not", "a", "dict"], BATCH) is None  # type: ignore[arg-type]
    data = contract_output(expiration_date={"found": True, "value": "2027-12-31"})
    outcome = verify_output("contract", data, BATCH)
    assert outcome is not None and outcome.invalid == 1 and outcome.candidates == []


def test_verify_output_flags_values_not_written_in_the_quote() -> None:
    data = contract_output(
        expiration_date=value(v="2028-01-01", evidence="expires on 31 December 2027", source="C1")
    )
    outcome = verify_output("contract", data, BATCH)
    assert outcome is not None
    (candidate,) = outcome.candidates
    assert not candidate.supported and candidate.confidence == 0.5


def test_invoice_line_items_are_verified() -> None:
    chunk = ChunkView(
        id=uuid.uuid4(),
        index=0,
        content="Consulting services | 10 | 150.00 | 1,500.00\nTravel | 1 | 300.00 | 300.00",
        page_start=1,
        page_end=1,
        section=None,
    )
    batch = {"C1": chunk}
    data: dict[str, Any] = {spec.name: absent() for spec in INVOICE_FIELDS}
    data["line_items"] = [
        {
            "description": "Consulting services",
            "quantity": "10",
            "unit_price": "150.00",
            "amount": "1500.00",
            "evidence": "Consulting services | 10 | 150.00 | 1,500.00",
            "source": "C1",
        },
        {
            "description": "Hidden fee",
            "quantity": "1",
            "unit_price": "999",
            "amount": "999",
            "evidence": "Hidden fee | 1 | 999 | 999",
            "source": "C1",
        },
        {
            "description": "Travel",
            "quantity": "one",
            "unit_price": "300",
            "amount": "300",
            "evidence": "Travel | 1 | 300.00 | 300.00",
            "source": "C1",
        },
    ]
    outcome = verify_output("invoice", data, batch)
    assert outcome is not None
    assert [(li.description, li.amount, li.confidence) for li in outcome.line_items] == [
        ("Consulting services", "1500", 0.9)
    ]
    assert outcome.unverified == 1 and outcome.invalid == 1


def test_merge_candidates_best_value_and_unique_parties() -> None:
    first = verify_output(
        "contract",
        contract_output(
            parties=[value(v="Alpha Ltd", evidence="between Alpha Ltd and Beta Inc", source="C1")],
            expiration_date=value(
                v="2027-12-31", evidence="expires on 31 December 2027", source="C2"
            ),
        ),
        BATCH,
    )
    second = verify_output(
        "contract",
        contract_output(
            parties=[
                value(v="ALPHA LTD", evidence="between Alpha Ltd and Beta Inc", source="C1"),
                value(v="Beta Inc", evidence="between Alpha Ltd and Beta Inc", source="C1"),
            ],
            expiration_date=value(
                v="2027-12-31", evidence="expires on 31 December 2027", source="C1"
            ),
        ),
        BATCH,
    )
    assert first is not None and second is not None
    candidates, lines = merge_candidates([first, second])
    assert [c.parsed.display for c in candidates if c.field == "parties"] == [
        "Alpha Ltd",
        "Beta Inc",
    ]
    (expiry,) = [c for c in candidates if c.field == "expiration_date"]
    assert expiry.confidence == 0.9 and expiry.chunk is C1  # the un-moved, better candidate
    assert lines == []


def rules_row(field: str, **kw: Any) -> FieldRow:
    base: dict[str, Any] = {
        "id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "version_id": uuid.uuid4(),
        "field": field,
        "value_text": None,
        "value_date": None,
        "value_number": None,
        "currency": None,
        "confidence": 0.7,
        "method": "rules",
        "chunk_id": None,
        "page": 1,
        "evidence": "e",
    }
    base.update(kw)
    return FieldRow(**base)


def test_merge_with_rules_fills_gaps_and_reports_conflicts() -> None:
    outcome = verify_output(
        "contract",
        contract_output(
            expiration_date=value(
                v="2027-12-31", evidence="expires on 31 December 2027", source="C1"
            ),
            total_value=value(
                v="120000", evidence="total contract value is USD 120,000.00", source="C2"
            ),
        ),
        BATCH,
    )
    assert outcome is not None
    candidates, _ = merge_candidates([outcome])
    llm_values = [candidate_value(c, None) for c in candidates]
    rules = [
        rules_row("expiration_date", value_date=date(2027, 6, 30)),
        rules_row("effective_date", value_date=date(2026, 1, 1)),
        rules_row("total_value", value_number=Decimal("120000.0000"), currency="USD"),
        rules_row("due_date", value_date=date(2026, 2, 1)),  # not a contract field
        rules_row("party", value_text="Alpha Ltd"),  # alias of "parties"
    ]
    merged, conflicts = merge_with_rules("contract", llm_values, rules)
    fields = [(v.field, v.method) for v in merged]
    assert fields == [
        ("parties", "rules"),
        ("effective_date", "rules"),
        ("expiration_date", "llm"),
        ("total_value", "llm"),
    ]
    assert [(c.field, c.llm_value, c.rules_value) for c in conflicts] == [
        ("expiration_date", "2027-12-31", "2027-06-30")
    ]
    total = next(v for v in merged if v.field == "total_value")
    assert total.currency == "USD" and total.value == "120000 USD"


def test_currency_helpers() -> None:
    assert normalize_currency(" gbp ") == "GBP"
    assert normalize_currency(chr(0xA3)) == "GBP"
    assert normalize_currency("pounds") is None
    assert currency_in("Total: EUR 5") == "EUR"
    assert currency_in("Total " + chr(0x20AC) + "5") == "EUR"
    assert currency_in("ABC 5 and XYZ") is None


def test_select_chunks_keeps_everything_that_fits_else_the_relevant_ones() -> None:
    filler = [
        (f"C{i}", ChunkView(uuid.uuid4(), i, "lorem ipsum " * 60, i, i, None)) for i in range(1, 11)
    ]
    relevant = (
        "C11",
        ChunkView(
            uuid.uuid4(), 11, "Payment is due 30 days after the invoice date " * 3, 11, 11, None
        ),
    )
    eligible = [*filler[:5], relevant, *filler[5:]]
    assert select_chunks(eligible, "contract", 1_000_000) == eligible
    chosen = select_chunks(eligible, "contract", 800)
    ids = [sid for sid, _ in chosen]
    assert "C1" in ids and "C11" in ids and "C10" in ids  # first, relevant, last
    order = [sid for sid, _ in eligible]
    assert ids == sorted(ids, key=order.index)  # document order is preserved


def test_every_output_schema_passes_the_gateway_strictness_check() -> None:
    from docassist.intelligence.compare import CHANGE_SUMMARY_SCHEMA
    from docassist.intelligence.summarize import summary_schema
    from docassist.llm import schema as gateway_schema

    for schema in (
        extraction_schema("contract"),
        extraction_schema("invoice"),
        summary_schema(12),
        CHANGE_SUMMARY_SCHEMA,
    ):
        gateway_schema.check_strict(schema)
