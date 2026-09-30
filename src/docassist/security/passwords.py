"""Password hashing (Argon2id) and policy (NIST SP 800-63B style).

* Argon2id with a configurable memory cost - memory-hard, so GPU cracking is expensive.
* ``verify`` runs a dummy hash for unknown users so response time does not reveal whether an
  account exists (user enumeration by timing).
* Policy = length + blocklist + "not derived from the email", no arbitrary composition rules
  (which push users toward predictable patterns).
"""

from __future__ import annotations

import hashlib
import secrets
import unicodedata
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

# A compact blocklist of the most common passwords/patterns; production deployments can
# layer a k-anonymity breach check (e.g. HIBP range API) on top via egress-allowed hosts.
_COMMON = frozenset(
    {
        "password",
        "password1",
        "password123",
        "123456789012",
        "qwertyuiop",
        "letmein",
        "welcome",
        "welcome1",
        "admin",
        "administrator",
        "iloveyou",
        "monkey",
        "dragon",
        "sunshine",
        "princess",
        "football",
        "baseball",
        "passw0rd",
        "p@ssw0rd",
        "p@ssword",
        "changeme",
        "secret",
        "trustno1",
        "qwerty123",
        "1q2w3e4r5t",
        "abc123456789",
        "zaq12wsx",
        "superman",
        "master",
        "hello123",
        "whatever",
        "freedom",
        "shadow",
        "correcthorsebatterystaple",
        "company123",
        "summer2026",
        "winter2026",
        "spring2026",
        "autumn2026",
        "january2026",
        "123qweasdzxc",
        "qwertyuiop123",
        "1234567890",
    }
)


class PasswordPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PasswordPolicy:
    min_length: int = 12
    max_length: int = 256

    def validate(self, password: str, *, email: str | None = None, name: str | None = None) -> None:
        normalized = unicodedata.normalize("NFKC", password)
        if len(normalized) < self.min_length:
            raise PasswordPolicyError(f"Password must be at least {self.min_length} characters.")
        if len(normalized) > self.max_length:
            raise PasswordPolicyError(f"Password must be at most {self.max_length} characters.")
        lowered = normalized.lower()
        if lowered in _COMMON or lowered.rstrip("!.0123456789") in _COMMON:
            raise PasswordPolicyError("This password is too common.")
        if len(set(normalized)) < 5:
            raise PasswordPolicyError("Password is too repetitive.")
        if email:
            local = email.split("@", 1)[0].lower()
            if len(local) >= 4 and local in lowered:
                raise PasswordPolicyError("Password must not contain your email address.")
        if name:
            for part in name.lower().split():
                if len(part) >= 4 and part in lowered:
                    raise PasswordPolicyError("Password must not contain your name.")


class PasswordService:
    def __init__(self, *, time_cost: int, memory_kib: int, parallelism: int) -> None:
        self._hasher = PasswordHasher(
            time_cost=time_cost, memory_cost=memory_kib, parallelism=parallelism
        )
        # Pre-computed hash of random data used to equalise timing for unknown accounts.
        self._dummy_hash = self._hasher.hash(secrets.token_urlsafe(24))

    @staticmethod
    def _prepare(password: str) -> str:
        return unicodedata.normalize("NFKC", password)

    def hash(self, password: str) -> str:
        return self._hasher.hash(self._prepare(password))

    def verify(self, password_hash: str | None, password: str) -> bool:
        """Constant-shape verification; ``password_hash=None`` burns equivalent time."""
        target = password_hash or self._dummy_hash
        try:
            ok = self._hasher.verify(target, self._prepare(password))
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False
        return ok and password_hash is not None

    def needs_rehash(self, password_hash: str) -> bool:
        return self._hasher.check_needs_rehash(password_hash)


def fingerprint(value: str) -> str:
    """Short non-reversible fingerprint for logs/metrics (never for authentication)."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]
