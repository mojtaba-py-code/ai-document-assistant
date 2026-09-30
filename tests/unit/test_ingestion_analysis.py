"""Document-type classifier, sensitivity signals and language guess."""

from __future__ import annotations

import pytest

from docassist.core.enums import Classification, DocumentType
from docassist.ingestion.classify import classify_document
from docassist.ingestion.language import guess_language
from docassist.ingestion.sensitivity import (
    CONTACT_LIST_THRESHOLD,
    SensitivityAccumulator,
    assess,
    assess_texts,
)
from tests.helpers_ingestion import CONTRACT_TEXT, INVOICE_TEXT

SAMPLES = {
    DocumentType.CONTRACT: CONTRACT_TEXT,
    DocumentType.INVOICE: INVOICE_TEXT,
    DocumentType.POLICY: (
        "Information Security Policy. Purpose and scope: this policy applies to all employees and "
        "contractors. Employees must lock their screens. Violations of this policy may result in "
        "disciplinary action. Policy owner: CISO. Next review date: 2027-01-01. Acceptable use of "
        "company devices is described in the related procedure."
    ),
    DocumentType.HR: (
        "Offer letter. We are pleased to offer you employment as Data Analyst. Your annual salary "
        "is 60,000 EUR with standard benefits and 25 days of annual leave. A probationary period "
        "of six months applies. Human Resources will contact you about onboarding and payroll."
    ),
    DocumentType.TECHNICAL: (
        "Authentication architecture. The API gateway validates OAuth tokens before requests reach "
        "the microservices. Each endpoint runs in Kubernetes; deployment uses Docker images. "
        "Database configuration and latency budgets are listed in the system requirements."
    ),
    DocumentType.FINANCIAL: (
        "Consolidated income statement and balance sheet for fiscal year 2025. Revenue rose 8% while "
        "gross margin held at 41%. EBITDA reached 12.4m; net income 7.1m. Operating expenses and "
        "capex follow the budget; the cash flow forecast for next quarter is attached."
    ),
    DocumentType.LEGAL: (
        "In the District Court. Plaintiff Acme Corp. v. Defendant Beta LLC. Counsel for the plaintiff "
        "filed a motion to compel; the court issued a subpoena and entered judgment pursuant to the "
        "statute. The attorney for the defendant will appeal. Litigation continues."
    ),
    DocumentType.REPORT: (
        "Quarterly report. Executive summary: key metrics improved. Methodology: survey of 300 "
        "customers. Findings: satisfaction up 6 points. Recommendations: expand support hours. "
        "Conclusion: results support the strategy. Analysis of KPIs follows."
    ),
}


@pytest.mark.parametrize("expected", list(SAMPLES), ids=[t.value for t in SAMPLES])
def test_classifier_recognises_each_type(expected: DocumentType) -> None:
    result = classify_document(SAMPLES[expected])
    assert result.doc_type is expected, result
    assert 0 < result.confidence <= 1


def test_classifier_falls_back_to_other() -> None:
    result = classify_document("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    assert result.doc_type is DocumentType.OTHER and result.confidence == 0.0


def test_title_tips_an_ambiguous_document() -> None:
    body = (
        "Please find the details below regarding the payment of the amount due and the agreement."
    )
    assert classify_document(body, title="Invoice 2026-001").doc_type is DocumentType.INVOICE


def test_classifier_is_deterministic() -> None:
    assert classify_document(CONTRACT_TEXT) == classify_document(CONTRACT_TEXT)


# --------------------------------------------------------------------------- #
# Sensitivity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "signal", "suggested"),
    [
        ("api_key = sk-ant-0123456789abcdefghijklmnop", "SECRET", Classification.RESTRICTED),
        (
            "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----",
            "SECRET",
            Classification.RESTRICTED,
        ),
        ("Employee SSN: 123-45-6789", "US_SSN", Classification.CONFIDENTIAL),
        ("Pay to IBAN DE89 3704 0044 0532 0130 00 please", "IBAN", Classification.CONFIDENTIAL),
        ("Card 4111 1111 1111 1111 on file", "CREDIT_CARD", Classification.CONFIDENTIAL),
        ("Contact jane@example.com for details", "EMAIL", None),
    ],
)
def test_sensitivity_rules(text: str, signal: str, suggested: Classification | None) -> None:
    result = assess_texts([text])
    assert signal in result.signals and result.suggested is suggested


def test_contact_lists_suggest_internal() -> None:
    emails = [f"person{i}@example.com" for i in range(CONTACT_LIST_THRESHOLD)]
    assert assess_texts([" ".join(emails)]).suggested is Classification.INTERNAL
    assert assess_texts([" ".join(emails[:-1])]).suggested is None


def test_escalation_never_lowers() -> None:
    result = assess({"US_SSN": 1})
    assert result.escalation_over(Classification.INTERNAL) is Classification.CONFIDENTIAL
    assert result.escalation_over(Classification.CONFIDENTIAL) is None
    assert result.escalation_over(Classification.RESTRICTED) is None
    assert assess({}).escalation_over(Classification.PUBLIC) is None


def test_accumulator_deduplicates_values_across_chunks() -> None:
    accumulator = SensitivityAccumulator()
    assert accumulator.add("mail a@example.com and 123-45-6789") == ["EMAIL", "US_SSN"]
    assert accumulator.add("again a@example.com") == ["EMAIL"]
    assert accumulator.counts == {"EMAIL": 1, "US_SSN": 1}
    assert accumulator.result().suggested is Classification.CONFIDENTIAL


def test_ip_addresses_are_not_personal_signals() -> None:
    assert assess_texts(["server at 10.0.0.1"]).signals == ()


# --------------------------------------------------------------------------- #
# Language
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "code"),
    [
        (
            "The contract is valid for two years and the supplier shall deliver the goods on time. "
            * 3,
            "en",
        ),
        (
            "Der Vertrag ist gültig und die Lieferung erfolgt mit der Rechnung von dem Lieferanten. "
            * 3,
            "de",
        ),
        (
            "Le contrat est valable pour deux ans et le fournisseur doit livrer les produits dans les délais. "
            * 3,
            "fr",
        ),
        (
            "El contrato es válido por dos años y el proveedor debe entregar los productos con la factura. "
            * 3,
            "es",
        ),
        (
            "Этот договор действует два года, и поставщик обязан доставить товары вовремя. " * 3,
            "ru",
        ),
        (
            "این قرارداد به مدت دو سال معتبر است و تامین کننده باید کالا را به موقع تحویل دهد. "
            * 3,
            "fa",
        ),
        ("هذا العقد صالح لمدة عامين ويجب على المورد تسليم البضائع في الوقت المحدد. " * 3, "ar"),
        ("本合同有效期为两年供应商必须按时交付货物。" * 5, "zh"),
        ("この契約は二年間有効であり、供給者は期限内に商品を納品しなければなりません。" * 3, "ja"),
    ],
)
def test_language_guess(text: str, code: str) -> None:
    assert guess_language(text) == code


def test_language_guess_abstains() -> None:
    assert guess_language("Too short.") is None
    assert guess_language("1234 5678 " * 50) is None
