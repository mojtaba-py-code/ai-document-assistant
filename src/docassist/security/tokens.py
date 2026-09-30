"""Access tokens (JWT) and opaque secrets (refresh / reset / MFA challenge tokens).

JWT hardening:

* algorithm pinned to HS256 on *verification* (``alg=none`` and RS/HS confusion rejected);
* ``iss``, ``aud``, ``exp``, ``iat``, ``nbf``, ``jti``, ``sid`` all required;
* key rotation: tokens carry ``kid``; previous keys verify, only the active key signs;
* small clock skew leeway only.

Opaque tokens are 256-bit random values. Only an HMAC-SHA256 of the token keyed with a
server-side *pepper* is stored, so a database dump cannot be replayed as live tokens.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import jwt

from docassist.core.context import utcnow

_ALGORITHM = "HS256"
_REQUIRED_CLAIMS = ["iss", "aud", "exp", "iat", "nbf", "sub", "jti", "sid", "typ"]


class TokenError(Exception):
    """Any token that fails verification. Never tell the caller *why*."""


@dataclass(frozen=True, slots=True)
class AccessClaims:
    user_id: uuid.UUID
    org_id: uuid.UUID | None
    role: str
    session_id: uuid.UUID
    token_version: int
    jti: str
    expires_at: datetime


class TokenService:
    def __init__(
        self,
        *,
        signing_key: str,
        previous_keys: list[str],
        issuer: str,
        audience: str,
        access_ttl_seconds: int,
        pepper: str,
    ) -> None:
        self._active_kid = self._kid(signing_key)
        self._keys = {self._active_kid: signing_key}
        for key in previous_keys:
            self._keys.setdefault(self._kid(key), key)
        self._issuer = issuer
        self._audience = audience
        self._ttl = timedelta(seconds=access_ttl_seconds)
        self._pepper = pepper.encode()

    @staticmethod
    def _kid(key: str) -> str:
        return hashlib.sha256(b"kid:" + key.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------ JWT
    def issue_access_token(
        self,
        *,
        user_id: uuid.UUID,
        org_id: uuid.UUID | None,
        role: str,
        session_id: uuid.UUID,
        token_version: int,
    ) -> tuple[str, datetime]:
        now = utcnow()
        expires = now + self._ttl
        claims = {
            "iss": self._issuer,
            "aud": self._audience,
            "sub": str(user_id),
            "org": str(org_id) if org_id else None,
            "role": role,
            "sid": str(session_id),
            "tv": token_version,
            "typ": "access",
            "jti": secrets.token_urlsafe(16),
            "iat": int(now.timestamp()),
            "nbf": int(now.timestamp()),
            "exp": int(expires.timestamp()),
        }
        token = jwt.encode(
            claims,
            self._keys[self._active_kid],
            algorithm=_ALGORITHM,
            headers={"kid": self._active_kid},
        )
        return token, expires

    def verify_access_token(self, token: str) -> AccessClaims:
        if not token or len(token) > 4096:
            raise TokenError("malformed")
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise TokenError("malformed") from exc
        if header.get("alg") != _ALGORITHM:
            raise TokenError("algorithm")
        key = self._keys.get(str(header.get("kid", "")))
        if key is None:
            raise TokenError("unknown key")
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=[_ALGORITHM],
                audience=self._audience,
                issuer=self._issuer,
                leeway=10,
                options={"require": _REQUIRED_CLAIMS, "strict_aud": True},
            )
        except jwt.PyJWTError as exc:
            raise TokenError("invalid") from exc
        if claims.get("typ") != "access":
            raise TokenError("type")
        try:
            org = claims.get("org")
            return AccessClaims(
                user_id=uuid.UUID(claims["sub"]),
                org_id=uuid.UUID(org) if org else None,
                role=str(claims["role"]),
                session_id=uuid.UUID(claims["sid"]),
                token_version=int(claims["tv"]),
                jti=str(claims["jti"]),
                expires_at=datetime.fromtimestamp(int(claims["exp"]), tz=utcnow().tzinfo),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise TokenError("claims") from exc

    # ------------------------------------------------------------ opaque
    @staticmethod
    def new_opaque_token() -> str:
        return secrets.token_urlsafe(32)

    def hash_opaque(self, token: str) -> bytes:
        return hmac.new(self._pepper, token.encode(), hashlib.sha256).digest()

    def sign(self, purpose: str, payload: str) -> str:
        """Compact HMAC signature for short-lived links (e.g. export downloads)."""
        mac = hmac.new(self._pepper, f"{purpose}|{payload}".encode(), hashlib.sha256)
        return mac.hexdigest()

    def verify_signature(self, purpose: str, payload: str, signature: str) -> bool:
        return hmac.compare_digest(self.sign(purpose, payload), signature)
