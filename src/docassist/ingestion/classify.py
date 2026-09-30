"""Rules-based document-type classifier (deterministic, explainable, offline).

Each :class:`~docassist.core.enums.DocumentType` has weighted phrases. A phrase contributes
``weight * (1 + ln(occurrences))`` (occurrences capped at :data:`MAX_OCCURRENCES`), plus
``weight * (LEAD_BONUS - 1)`` when it also appears in the title or the opening
:data:`LEAD_CHARS` characters - titles and first paragraphs say what a document *is*. The best
type wins when its score reaches :data:`MIN_SCORE`; otherwise the document is ``other`` with
confidence 0.

``confidence = best / (best + runner_up) * min(1, best / SATURATION)``: high when one type
clearly dominates *and* there is enough evidence, in ``[0, 1]``.

The pipeline applies the result only when the uploader did not choose a type
(``documents.doc_type_source != "user"``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from docassist.core.enums import DocumentType

MAX_SCAN_CHARS = 200_000
LEAD_CHARS = 1_500
LEAD_BONUS = 1.5
MAX_OCCURRENCES = 20
MIN_SCORE = 4.0
SATURATION = 12.0

_RULES: dict[DocumentType, tuple[tuple[str, float], ...]] = {
    DocumentType.CONTRACT: (
        (r"agreement", 1.5), (r"this agreement", 2.5), (r"by and between", 4.0),
        (r"hereinafter", 2.0), (r"whereas", 2.0), (r"in witness whereof", 4.0),
        (r"governing law", 2.0), (r"term and termination", 3.0), (r"indemnif(?:y|ication)", 2.0),
        (r"counterparts", 2.0), (r"effective date", 1.5), (r"the parties", 1.5),
        (r"services agreement|supply agreement|non-disclosure agreement|nda", 3.0),
        (r"contract", 1.0), (r"obligations of", 1.0),
    ),
    DocumentType.INVOICE: (
        (r"invoice", 3.0), (r"invoice (?:number|no\.?|#)", 4.0), (r"bill to", 3.0),
        (r"amount due", 3.0), (r"total due|balance due", 3.0), (r"subtotal", 3.0),
        (r"due date", 1.5), (r"unit price", 3.0), (r"qty|quantity", 1.5),
        (r"remit(?:tance)? to", 3.0), (r"vat|sales tax", 1.0), (r"purchase order|po number", 1.5),
    ),
    DocumentType.POLICY: (
        (r"policy", 2.0), (r"this policy (?:applies|sets out|describes)", 4.0), (r"scope", 1.0),
        (r"purpose", 0.8), (r"compliance", 1.0), (r"employees (?:must|shall|should)", 2.0),
        (r"procedures?", 1.0), (r"violations? of this policy", 4.0), (r"policy owner", 3.0),
        (r"review date|next review", 2.0), (r"acceptable use", 3.0), (r"must not", 0.5),
    ),
    DocumentType.HR: (
        (r"employee", 1.0), (r"employment", 2.0), (r"salary|salaries", 2.0), (r"compensation", 1.0),
        (r"benefits", 1.0), (r"annual leave|vacation|paid time off", 2.5),
        (r"performance review", 3.0), (r"onboarding", 3.0), (r"job description", 3.0),
        (r"probation(?:ary)? period", 3.0), (r"human resources", 3.0), (r"payroll", 2.0),
        (r"offer letter|letter of offer", 4.0), (r"headcount", 2.0),
    ),
    DocumentType.TECHNICAL: (
        (r"api", 1.0), (r"architecture", 2.0), (r"database", 1.0), (r"server", 1.0),
        (r"deployment", 2.0), (r"configuration", 1.0), (r"endpoint", 2.0),
        (r"install(?:ation)?", 1.0), (r"latency", 2.0), (r"kubernetes|docker", 3.0),
        (r"source code", 2.0),
        (r"algorithm", 2.0), (r"system requirements", 3.0), (r"authentication|oauth|tls", 1.5),
        (r"microservices?", 2.5),
    ),
    DocumentType.FINANCIAL: (
        (r"revenue", 2.0), (r"balance sheet", 4.0), (r"income statement|profit and loss", 4.0),
        (r"cash flow", 3.0), (r"ebitda", 4.0), (r"fiscal year|financial year", 3.0),
        (r"quarter(?:ly)?", 1.0), (r"gross margin", 3.0), (r"net income", 3.0),
        (r"operating expenses|opex|capex", 3.0), (r"budget", 2.0), (r"forecast", 2.0),
        (r"liabilities", 2.0), (r"assets", 1.0),
    ),
    DocumentType.LEGAL: (
        (r"court", 3.0), (r"plaintiff", 4.0), (r"defendant", 4.0), (r"pursuant to", 1.0),
        (r"statute", 2.0), (r"jurisdiction", 1.0), (r"litigation", 3.0), (r"judg(?:e)?ment", 3.0),
        (r"counsel", 2.0), (r"subpoena", 4.0), (r"legal notice", 3.0), (r"attorney", 2.0),
        (r"motion to", 3.0), (r"cease and desist", 4.0), (r"hereby", 0.8),
    ),
    DocumentType.REPORT: (
        (r"report", 2.0), (r"executive summary", 4.0), (r"findings", 2.0),
        (r"recommendations?", 2.0), (r"conclusions?", 1.0), (r"methodology", 3.0),
        (r"analysis", 1.0), (r"results", 1.0),
        (r"key metrics|kpis?", 2.0), (r"(?:quarterly|annual|monthly) report", 4.0),
        (r"table of contents", 1.0),
    ),
}  # fmt: skip

_COMPILED: dict[DocumentType, tuple[tuple[re.Pattern[str], float], ...]] = {
    doc_type: tuple((re.compile(r"\b(?:" + phrase + r")s?\b"), weight) for phrase, weight in rules)
    for doc_type, rules in _RULES.items()
}


@dataclass(frozen=True, slots=True)
class Classification:
    doc_type: DocumentType
    confidence: float
    scores: dict[str, float]


def _score(text: str, lead: str) -> dict[DocumentType, float]:
    scores: dict[DocumentType, float] = {}
    for doc_type, rules in _COMPILED.items():
        total = 0.0
        for pattern, weight in rules:
            count = 0
            for _ in pattern.finditer(text):
                count += 1
                if count >= MAX_OCCURRENCES:
                    break
            if count:
                total += weight * (1.0 + math.log(count))
                if pattern.search(lead):
                    total += weight * (LEAD_BONUS - 1.0)
        scores[doc_type] = round(total, 4)
    return scores


def classify_document(text: str, title: str = "") -> Classification:
    """Best-matching document type for ``text`` (and its ``title``)."""
    body = text[:MAX_SCAN_CHARS].casefold()
    lead = (title.casefold() + "\n" + body[:LEAD_CHARS]).strip()
    scores = _score(body + "\n" + title.casefold(), lead)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0].value))
    (best_type, best), (_, runner_up) = ranked[0], ranked[1]
    summary = {doc_type.value: value for doc_type, value in ranked if value > 0}
    if best < MIN_SCORE:
        return Classification(DocumentType.OTHER, 0.0, summary)
    confidence = best / (best + runner_up) * min(1.0, best / SATURATION)
    return Classification(best_type, round(min(1.0, confidence), 3), summary)
