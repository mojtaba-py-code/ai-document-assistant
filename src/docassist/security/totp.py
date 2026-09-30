"""RFC 6238 TOTP (SHA-1, 30 s, 6 digits - what authenticator apps support).

Replay protection: the caller persists the last accepted time step and passes it back;
a code for a step <= ``last_used_step`` is rejected even if still inside the window.
"""

from __future__ import annotations

import base64
import hmac
import secrets
import struct
import time
from urllib.parse import quote, urlencode

PERIOD = 30
DIGITS = 6


def new_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def _decode(secret_b32: str) -> bytes:
    padded = secret_b32.upper() + "=" * (-len(secret_b32) % 8)
    return base64.b32decode(padded, casefold=True)


def code_at(secret_b32: str, step: int) -> str:
    digest = hmac.new(_decode(secret_b32), struct.pack(">Q", step), "sha1").digest()
    offset = digest[-1] & 0x0F
    value = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFF_FFFF
    return str(value % 10**DIGITS).zfill(DIGITS)


def current_step(now: float | None = None) -> int:
    return int((time.time() if now is None else now) // PERIOD)


def verify(
    secret_b32: str,
    code: str,
    *,
    last_used_step: int | None,
    window: int = 1,
    now: float | None = None,
) -> int | None:
    """Return the matched step (to persist) or ``None``. Constant-time comparison."""
    candidate = code.strip().replace(" ", "")
    if len(candidate) != DIGITS or not candidate.isdigit():
        return None
    step = current_step(now)
    matched: int | None = None
    for offset in range(-window, window + 1):
        test_step = step + offset
        if hmac.compare_digest(code_at(secret_b32, test_step), candidate):
            matched = test_step
    if matched is None or (last_used_step is not None and matched <= last_used_step):
        return None
    return matched


def provisioning_uri(secret_b32: str, account: str, issuer: str) -> str:
    label = quote(f"{issuer}:{account}")
    query = urlencode({"secret": secret_b32, "issuer": issuer, "digits": DIGITS, "period": PERIOD})
    return f"otpauth://totp/{label}?{query}"
