"""Detection, redaction and reversible pseudonymisation of PII and secrets.

Used in four places:

* logging      - secrets never reach a log sink (``redact_secrets``);
* audit        - search queries are stored redacted (``redact``);
* LLM gateway  - PII is pseudonymised before text leaves for an external provider and
                 restored in the answer (``Pseudonymizer``);
* output guard - model output is scanned for secrets before it is returned.

Validators (Luhn, IBAN mod-97) keep false positives low so ordinary numbers survive.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum


class PiiKind(StrEnum):
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    IBAN = "IBAN"
    US_SSN = "US_SSN"
    IP_ADDRESS = "IP_ADDRESS"
    # a category label, not a credential
    SECRET = "SECRET"  # noqa: S105  # nosec B105


@dataclass(frozen=True, slots=True)
class PiiMatch:
    kind: PiiKind
    start: int
    end: int
    value: str


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _iban_ok(raw: str) -> bool:
    iban = raw.replace(" ", "").upper()
    if not 15 <= len(iban) <= 34:
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def _card_ok(raw: str) -> bool:
    digits = re.sub(r"\D", "", raw)
    return 13 <= len(digits) <= 19 and _luhn_ok(digits)


def _ssn_ok(raw: str) -> bool:
    area, group, serial = raw.split("-")
    return (
        area not in {"000", "666"}
        and not area.startswith("9")
        and group != "00"
        and serial != "0000"
    )


def _phone_ok(raw: str) -> bool:
    digits = re.sub(r"\D", "", raw)
    return 9 <= len(digits) <= 15


_PATTERNS: list[tuple[PiiKind, re.Pattern[str], Callable[[str], bool] | None]] = [
    (
        PiiKind.SECRET,
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
            r"|\bsk-ant-[A-Za-z0-9_\-]{20,}"
            r"|\bsk-[A-Za-z0-9_\-]{20,}"
            r"|\bAKIA[0-9A-Z]{16}\b"
            r"|\bgh[pousr]_[A-Za-z0-9]{36,}\b"
            r"|\bxox[abprs]-[A-Za-z0-9-]{10,}"
            r"|\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"
            r"|(?i:\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*\S{6,})"
        ),
        None,
    ),
    (
        PiiKind.EMAIL,
        re.compile(r"\b[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,253}\.[A-Za-z]{2,24}\b"),
        None,
    ),
    (
        PiiKind.IBAN,
        re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"),
        _iban_ok,
    ),
    (PiiKind.CREDIT_CARD, re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), _card_ok),
    (PiiKind.US_SSN, re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), _ssn_ok),
    (
        PiiKind.PHONE,
        re.compile(r"(?<![\w.])\+?\(?\d{1,4}\)?(?:[ .\-]?\(?\d{2,4}\)?){2,5}(?![\w.])"),
        _phone_ok,
    ),
    (
        PiiKind.IP_ADDRESS,
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
        None,
    ),
]

ALL_KINDS = frozenset(PiiKind)
PERSONAL_KINDS = frozenset(ALL_KINDS - {PiiKind.IP_ADDRESS})


def find_pii(text: str, kinds: Iterable[PiiKind] = ALL_KINDS) -> list[PiiMatch]:
    """Return non-overlapping matches, earlier patterns (secrets first) winning ties."""
    wanted = set(kinds)
    taken: list[tuple[int, int]] = []
    matches: list[PiiMatch] = []
    for kind, pattern, validator in _PATTERNS:
        if kind not in wanted:
            continue
        for m in pattern.finditer(text):
            start, end = m.span()
            value = m.group(0)
            if validator is not None and not validator(value):
                continue
            if any(start < t_end and end > t_start for t_start, t_end in taken):
                continue
            taken.append((start, end))
            matches.append(PiiMatch(kind, start, end, value))
    matches.sort(key=lambda match: match.start)
    return matches


def pii_kinds_in(text: str) -> list[str]:
    return sorted({m.kind.value for m in find_pii(text, PERSONAL_KINDS)})


def redact(text: str, kinds: Iterable[PiiKind] = ALL_KINDS) -> str:
    out: list[str] = []
    cursor = 0
    for match in find_pii(text, kinds):
        out.append(text[cursor : match.start])
        out.append(f"[REDACTED:{match.kind.value}]")
        cursor = match.end
    out.append(text[cursor:])
    return "".join(out)


def redact_secrets(text: str) -> str:
    return redact(text, {PiiKind.SECRET})


def contains_secret(text: str) -> bool:
    return bool(find_pii(text, {PiiKind.SECRET}))


class Pseudonymizer:
    """Reversible, per-request substitution of PII with opaque placeholders.

    Placeholders carry a random per-request salt so text inside a document cannot forge a
    placeholder that would be "restored" into some other value. The mapping lives only in
    memory for the duration of one LLM call.
    """

    def __init__(self, kinds: Iterable[PiiKind] = PERSONAL_KINDS) -> None:
        self._kinds = frozenset(kinds)
        self._salt = secrets.token_hex(3)
        self._forward: dict[tuple[PiiKind, str], str] = {}
        self._reverse: dict[str, str] = {}
        self._counters: dict[PiiKind, int] = {}

    @property
    def replacements(self) -> int:
        return len(self._reverse)

    def _token_for(self, kind: PiiKind, value: str) -> str:
        key = (kind, value)
        token = self._forward.get(key)
        if token is None:
            self._counters[kind] = self._counters.get(kind, 0) + 1
            token = f"[{kind.value}_{self._counters[kind]}_{self._salt}]"
            self._forward[key] = token
            self._reverse[token] = value
        return token

    def pseudonymize(self, text: str) -> str:
        out: list[str] = []
        cursor = 0
        for match in find_pii(text, self._kinds):
            out.append(text[cursor : match.start])
            out.append(self._token_for(match.kind, match.value))
            cursor = match.end
        out.append(text[cursor:])
        return "".join(out)

    def restore(self, text: str) -> str:
        if not self._reverse:
            return text
        pattern = re.compile("|".join(re.escape(token) for token in self._reverse))
        return pattern.sub(lambda m: self._reverse[m.group(0)], text)
