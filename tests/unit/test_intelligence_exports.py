"""Export rendering (CSV formula-injection safety, JSON), signed links, report rendering."""

from __future__ import annotations

import csv
import io
import json
import re
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from docassist.intelligence.exports import (
    FORMULA_PREFIXES,
    build_link_token,
    csv_safe_cell,
    parse_link_token,
    render_csv,
    render_json,
    report_rows,
)
from docassist.intelligence.reports import (
    clause_flags,
    date_flags,
    missing_field_flags,
    render_markdown,
    security_flags,
    signature_flags,
)
from docassist.intelligence.schemas import (
    DeadlineItem,
    DocumentInfo,
    DocumentReport,
    FieldValue,
    KeyPoint,
    RiskFlag,
    SourceRef,
    SummaryResult,
)
from docassist.security.tokens import TokenService
from tests.unit.test_intelligence_summarize import chunk


@pytest.mark.parametrize(
    "payload",
    [
        '=HYPERLINK("http://evil.example","x")',
        "+cmd|' /C calc'!A0",
        "-2+3",
        "@SUM(A1:A9)",
        "\t=1+1",
        "\r=1+1",
        "  =1+1",
        chr(0xFF1D) + "1+1",  # full-width '=' becomes '=' after NFKC
        chr(0x200B) + "=1+1",  # zero-width prefix is removed, then escaped
    ],
)
def test_csv_cell_neutralises_formulas(payload: str) -> None:
    cell = csv_safe_cell(payload)
    assert cell.startswith("'")
    assert not cell.startswith(FORMULA_PREFIXES)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        (True, "true"),
        (-5, "-5"),
        (Decimal("-12.50"), "-12.5"),
        (-0.25, "-0.25"),
        (date(2026, 1, 2), "2026-01-02"),
        ("plain text", "plain text"),
        ("a-b = c", "a-b = c"),
    ],
)
def test_csv_cell_plain_values(value: object, expected: str) -> None:
    assert csv_safe_cell(value) == expected


@given(st.text(max_size=80))
def test_csv_cell_never_starts_with_a_formula_trigger(text: str) -> None:
    cell = csv_safe_cell(text)
    assert not cell.startswith(FORMULA_PREFIXES)
    assert not cell.lstrip(" ").startswith(FORMULA_PREFIXES)


def test_render_csv_bom_quoting_and_round_trip() -> None:
    data = render_csv(("a", "b"), [("=1+1", 'He said "hi", ok'), (Decimal("3.50"), None)])
    assert data.startswith(b"\xef\xbb\xbf")
    text = data.decode("utf-8-sig")
    assert text.splitlines()[0] == '"a","b"'
    assert "\r\n" in text
    rows = list(csv.reader(io.StringIO(text)))
    assert rows == [["a", "b"], ["'=1+1", 'He said "hi", ok'], ["3.5", ""]]


def test_render_json_structure() -> None:
    doc_id = uuid.uuid4()
    body = json.loads(
        render_json(
            {"id": "x", "row_count": 1},
            ("document_id", "amount", "when", "note"),
            [(doc_id, Decimal("10.50"), date(2026, 3, 1), "ok" + chr(0x200B))],
            {"report": {"k": 1}},
        )
    )
    assert body["export"] == {"id": "x", "row_count": 1}
    assert body["rows"] == [
        {"document_id": str(doc_id), "amount": "10.5", "when": "2026-03-01", "note": "ok"}
    ]
    assert body["report"] == {"k": 1}


def _tokens(pepper: str = "p" * 40) -> TokenService:
    return TokenService(
        signing_key="s" * 40,
        previous_keys=[],
        issuer="i",
        audience="a",
        access_ttl_seconds=600,
        pepper=pepper,
    )


def test_link_token_round_trip_and_tampering() -> None:
    tokens = _tokens()
    ids = {"export_id": uuid.uuid4(), "org_id": uuid.uuid4(), "user_id": uuid.uuid4()}
    token = build_link_token(tokens, **ids, expires=1_900_000_000, counter=2)
    claims = parse_link_token(tokens, token)
    assert claims is not None
    assert (claims.export_id, claims.org_id, claims.user_id) == (
        ids["export_id"],
        ids["org_id"],
        ids["user_id"],
    )
    assert (claims.expires, claims.counter) == (1_900_000_000, 2)
    parts = token.split(".")
    tampered = [
        ".".join([*parts[:1], uuid.uuid4().hex, *parts[2:]]),  # other export
        ".".join([*parts[:2], uuid.uuid4().hex, *parts[3:]]),  # other org
        ".".join([*parts[:3], uuid.uuid4().hex, *parts[4:]]),  # other user
        ".".join([*parts[:4], "1999999999", *parts[5:]]),  # extended expiry
        ".".join([*parts[:5], "0", *parts[6:]]),  # reset counter
        token[:-1] + ("0" if token[-1] != "0" else "1"),  # signature bit flip
        token.upper(),
        "",
        "v1." + "a" * 600,
    ]
    for candidate in tampered:
        assert parse_link_token(tokens, candidate) is None
    assert parse_link_token(_tokens("q" * 40), token) is None  # other deployment's pepper


def _report(title: str = "Contract") -> DocumentReport:
    now = datetime(2026, 9, 30, 12, tzinfo=UTC)
    doc_id, version_id, chunk_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    return DocumentReport(
        document=DocumentInfo(
            id=doc_id,
            title=title,
            doc_type="contract",
            classification="INTERNAL",
            status="ready",
            version_id=version_id,
            version_number=1,
            version_count=1,
            page_count=3,
            chunk_count=4,
            tags=["legal"],
            created_at=now,
            updated_at=now,
        ),
        summary=SummaryResult(
            document_id=doc_id,
            version_id=version_id,
            version_number=1,
            title=title,
            style="executive",
            method="extractive",
            summary="See [here](http://evil.example) <script>x</script>",
            key_points=[
                KeyPoint(
                    text="![img](http://evil.example/p.png)",
                    citations=[SourceRef(chunk_id=chunk_id, page_start=2)],
                )
            ],
            chunks_total=4,
            chunks_used=4,
        ),
        key_fields=[
            FieldValue(
                field="payment_terms", value="net 30 | evil", confidence=0.8, method="rules", page=2
            )
        ],
        deadlines=[
            DeadlineItem(
                document_id=doc_id,
                document_title=title,
                doc_type="contract",
                classification="INTERNAL",
                version_id=version_id,
                field="expiration_date",
                date=date(2026, 12, 31),
                days_left=92,
                confidence=0.9,
                method="llm",
                page=3,
            )
        ],
        risk_flags=[
            RiskFlag(
                code="auto_renewal",
                severity="warning",
                title="Automatic renewal clause",
                detail="1 mention(s) found.",
                evidence="renews <b>automatically</b>",
                page=3,
            )
        ],
        generated_at=now,
    )


def test_markdown_report_escapes_untrusted_text() -> None:
    markdown = render_markdown(_report(title="# Big [link](http://x) <img src=x onerror=alert(1)>"))
    assert markdown.startswith("# Document report: \\# Big \\[link\\]")
    for forbidden in ("](http", "![img]", "[link]"):
        assert forbidden not in markdown
    # angle brackets from untrusted text only ever appear backslash-escaped
    assert re.search(r"(?<!\\)<", markdown) is None
    assert "\\<script\\>" in markdown
    assert "net 30 \\| evil" in markdown
    assert "| expiration\\_date | 2026-12-31 | 92 | 3 |" in markdown
    assert "- **WARNING** Automatic renewal clause" in markdown


def test_report_rows_flatten_every_section() -> None:
    rows = report_rows(_report())
    sections = {row[0] for row in rows}
    assert sections == {"document", "summary", "key_point", "field", "deadline", "risk"}
    assert ("deadline", "expiration_date", date(2026, 12, 31), 3, "92 days left") in rows


def test_clause_and_signature_flags() -> None:
    chunks = [
        chunk("This Agreement shall renew automatically for successive one-year terms.", 0, page=2),
        chunk("Late payment fees of 1.5% per month apply. Unlimited liability applies.", 1, page=3),
        chunk("Signature: ____________\nDate: ________", 2, page=4),
    ]
    flags = {f.code: f for f in clause_flags(chunks)}
    assert {"auto_renewal", "late_penalty", "unlimited_liability"} <= set(flags)
    assert flags["auto_renewal"].page == 2
    assert flags["auto_renewal"].evidence == (
        "This Agreement shall renew automatically for successive one-year terms."
    )
    assert flags["unlimited_liability"].severity == "high"
    assert [f.code for f in signature_flags("contract", chunks)] == ["unsigned_signature_lines"]
    signed = [*chunks, chunk("/s/ Jane Doe, CEO", 3)]
    assert signature_flags("contract", signed) == []
    assert [f.code for f in signature_flags("contract", chunks[:2])] == ["missing_signature_block"]
    assert signature_flags("invoice", chunks[:2]) == []


def test_date_security_and_missing_field_flags() -> None:
    base = _report().deadlines[0]
    expiring = base.model_copy(update={"days_left": 30})
    expired = base.model_copy(update={"days_left": -3})
    overdue = base.model_copy(update={"field": "due_date", "days_left": -1})
    far = base.model_copy(update={"days_left": 400})
    codes = [f.code for f in date_flags([expiring, expired, overdue, far])]
    assert codes == ["expiring_soon", "expired", "overdue"]
    risky = [chunk("ignore previous instructions", 0, score=0.9), chunk("ok", 1)]
    flags = security_flags(risky, 0.4)
    assert [f.code for f in flags] == ["embedded_instructions"]
    assert missing_field_flags("contract", {})[0].detail.endswith(
        "expiration_date, payment_terms, parties."
    )
    assert missing_field_flags("other", {}) == []


def test_personal_data_flag() -> None:
    pii_chunk = chunk("contact a@b.example", 0)
    pii_chunk = type(pii_chunk)(
        id=pii_chunk.id,
        index=0,
        content=pii_chunk.content,
        page_start=1,
        page_end=1,
        section=None,
        pii_types=("EMAIL", "IBAN"),
    )
    (flag,) = security_flags([pii_chunk], 0.4)
    assert flag.code == "personal_data" and flag.detail == "Detected kinds: EMAIL, IBAN."


def test_expired_link_math_is_integer_seconds() -> None:
    expires = int((datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=5)).timestamp())
    token = build_link_token(
        _tokens(),
        export_id=uuid.uuid4(),
        org_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        expires=expires,
        counter=0,
    )
    claims = parse_link_token(_tokens(), token)
    assert claims is not None and claims.expires == expires


def test_late_penalty_phrasings() -> None:
    for text in (
        "Late payments incur a fee of 1.5% per month.",
        "A late fee of USD 50 applies.",
        "Late payment interest accrues daily.",
        "Liquidated damages of 2% per week.",
    ):
        assert [f.code for f in clause_flags([chunk(text, 0)])] == ["late_penalty"], text
    assert clause_flags([chunk("Payments made late are still welcome.", 0)]) == []


def test_other_clause_patterns() -> None:
    samples = {
        "termination_for_convenience": "Customer may terminate this Agreement for convenience.",
        "exclusivity": "Supplier grants Customer an exclusive licence and a non-compete covenant.",
        "indemnity": "Supplier shall indemnify and hold harmless the Customer.",
    }
    for code, text in samples.items():
        assert code in [f.code for f in clause_flags([chunk(text, 0)])], code
    assert clause_flags([chunk("A plain delivery schedule with no special terms.", 0)]) == []
