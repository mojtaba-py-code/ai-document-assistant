"""Quote support: order-aware matching that rejects meaning-changing edits."""

from __future__ import annotations

import pytest

from docassist.core.quotes import find_supported_span, normalize_for_match

SOURCE = (
    "Acme pays Beta a termination fee of EUR 5,000 if the agreement ends early. "
    "The Supplier is liable for all damages caused by gross negligence (see clause 9)."
)


def test_exact_match_returns_the_source_text_with_punctuation() -> None:
    match = find_supported_span("the supplier is LIABLE for all damages", SOURCE)
    assert match is not None and match.exact and match.score == 1.0
    assert match.span == "The Supplier is liable for all damages"
    tail = find_supported_span("caused by gross negligence (see clause 9)", SOURCE)
    assert tail is not None and tail.span.endswith("(see clause 9).")


@pytest.mark.parametrize(
    "quote",
    [
        "Beta pays Acme a termination fee of EUR 5,000",  # swapped parties
        "The Supplier is not liable for damages caused by gross negligence",  # invented negation
        "Acme pays Beta a termination fee of EUR 6,000",  # changed number
        "The Supplier is never liable for all damages caused by negligence",  # never
        "The Customer may terminate at any time without cause",  # invented sentence
        "fee",  # too short to prove anything
    ],
)
def test_meaning_changing_edits_are_rejected(quote: str) -> None:
    assert find_supported_span(quote, SOURCE) is None


@pytest.mark.parametrize(
    "quote",
    [
        "The Supplier is liable for damages caused by gross negligence",  # dropped a word
        "The Suplier is liable for all damages caused by gross negligence",  # typo
    ],
)
def test_harmless_differences_are_tolerated_and_source_wording_is_returned(quote: str) -> None:
    match = find_supported_span(quote, SOURCE)
    assert match is not None and not match.exact
    assert match.span.startswith("The Supplier is liable for all damages")


def test_normalization() -> None:
    assert normalize_for_match("  Net-30,  PAYMENT!! ") == "net 30 payment"
