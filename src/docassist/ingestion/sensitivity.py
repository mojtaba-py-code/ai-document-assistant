"""Sensitivity signals: which kinds of personal data / secrets a document contains.

Detection uses :func:`docassist.core.redaction.find_pii` (validators such as Luhn and IBAN
mod-97 keep false positives low). The result *suggests* a minimum classification:

* any ``SECRET`` (private key, API key, ``password: ...``)            -> RESTRICTED
* any ``US_SSN``, ``IBAN`` or ``CREDIT_CARD``                          -> CONFIDENTIAL
* at least :data:`CONTACT_LIST_THRESHOLD` distinct e-mail addresses and
  phone numbers combined (a contact list or roster)                  -> INTERNAL

A suggestion **never lowers** a classification and is never applied automatically: the
pipeline stores it in ``documents.suggested_classification`` (only when it is above the
current one) and audits ``document.sensitivity_escalation_suggested`` for a human to act on.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from docassist.core.enums import Classification
from docassist.core.redaction import PERSONAL_KINDS, PiiKind, find_pii

CONTACT_LIST_THRESHOLD = 20
_MAX_DISTINCT_VALUES = 1_000
_RESTRICTED_KINDS = frozenset({PiiKind.SECRET})
_CONFIDENTIAL_KINDS = frozenset({PiiKind.US_SSN, PiiKind.IBAN, PiiKind.CREDIT_CARD})
_CONTACT_KINDS = frozenset({PiiKind.EMAIL, PiiKind.PHONE})


@dataclass(slots=True)
class SensitivityAccumulator:
    """Collects PII kinds over many texts (e.g. chunks), de-duplicating repeated values."""

    _values: dict[PiiKind, set[str]] = field(default_factory=dict, init=False)

    def add(self, text: str) -> list[str]:
        """Record the PII in ``text``; returns the personal kinds found in it (sorted)."""
        kinds: set[str] = set()
        for match in find_pii(text, PERSONAL_KINDS):
            kinds.add(match.kind.value)
            bucket = self._values.setdefault(match.kind, set())
            if len(bucket) < _MAX_DISTINCT_VALUES:
                bucket.add(match.value)
        return sorted(kinds)

    @property
    def counts(self) -> dict[str, int]:
        return {kind.value: len(values) for kind, values in sorted(self._values.items())}

    def result(self) -> SensitivityResult:
        return assess(self.counts)


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    signals: tuple[str, ...]
    counts: dict[str, int]
    suggested: Classification | None

    def escalation_over(self, current: Classification) -> Classification | None:
        """The suggestion if it is strictly above ``current``; never a lower level."""
        if self.suggested is not None and self.suggested.rank > current.rank:
            return self.suggested
        return None


def assess(counts: dict[str, int]) -> SensitivityResult:
    present = {PiiKind(kind) for kind, count in counts.items() if count > 0}
    suggested: Classification | None = None
    if present & _RESTRICTED_KINDS:
        suggested = Classification.RESTRICTED
    elif present & _CONFIDENTIAL_KINDS:
        suggested = Classification.CONFIDENTIAL
    elif sum(counts.get(kind.value, 0) for kind in _CONTACT_KINDS) >= CONTACT_LIST_THRESHOLD:
        suggested = Classification.INTERNAL
    return SensitivityResult(
        signals=tuple(sorted(kind.value for kind in present)),
        counts=dict(counts),
        suggested=suggested,
    )


def assess_texts(texts: Iterable[str]) -> SensitivityResult:
    accumulator = SensitivityAccumulator()
    for text in texts:
        accumulator.add(text)
    return accumulator.result()
