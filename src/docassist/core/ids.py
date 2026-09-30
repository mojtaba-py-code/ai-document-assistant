"""Identifier helpers.

UUIDv7 (RFC 9562) is time-ordered, so primary-key inserts land at the right edge of the
B-tree instead of scattering page writes - it also keeps IDs unguessable (74 random bits).
"""

from __future__ import annotations

import os
import time
import uuid

_UUID_RE_LEN = 36


def uuid7() -> uuid.UUID:
    unix_ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    rand_a = rand >> 62 & 0xFFF
    rand_b = rand & 0x3FFF_FFFF_FFFF_FFFF
    value = (unix_ms & 0xFFFF_FFFF_FFFF) << 80
    value |= 0x7 << 76
    value |= rand_a << 64
    value |= 0b10 << 62
    value |= rand_b
    return uuid.UUID(int=value)


def parse_uuid(value: str) -> uuid.UUID | None:
    """Strictly parse a canonical UUID string; ``None`` instead of raising."""
    if len(value) != _UUID_RE_LEN:
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None
