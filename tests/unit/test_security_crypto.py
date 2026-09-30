"""Envelope encryption: round trips, binding to context, tamper/truncation detection, rotation."""

from __future__ import annotations

import io
import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from docassist.security.crypto import (
    SEGMENT_SIZE,
    DecryptionError,
    KeyRing,
    StreamEncryptor,
    decrypt_bytes,
    decrypt_stream,
    encrypt_bytes,
)


def ring(active: str = "k1", **extra: bytes) -> KeyRing:
    keys = {"k1": b"\x01" * 32, **extra}
    return KeyRing(keys=keys, active_kid=active)


@given(st.binary(max_size=3 * SEGMENT_SIZE + 17))
@settings(max_examples=40, deadline=None)
def test_round_trip_any_size(data: bytes) -> None:
    r = ring()
    assert decrypt_bytes(r, encrypt_bytes(r, data, b"ctx"), b"ctx") == data


@pytest.mark.parametrize(
    "size", [0, 1, SEGMENT_SIZE - 1, SEGMENT_SIZE, SEGMENT_SIZE + 1, 2 * SEGMENT_SIZE]
)
def test_segment_boundaries(size: int) -> None:
    r = ring()
    data = os.urandom(size)
    assert decrypt_bytes(r, encrypt_bytes(r, data, b"c"), b"c") == data


def test_streaming_encryptor_matches_one_shot_decrypt() -> None:
    r = ring()
    enc = StreamEncryptor(r, b"obj")
    parts = [enc.header]
    data = os.urandom(200_000)
    for i in range(0, len(data), 7_777):
        parts.append(enc.update(data[i : i + 7_777]))
    parts.append(enc.finalize())
    assert b"".join(decrypt_stream(r, io.BytesIO(b"".join(parts)), b"obj")) == data


def test_ciphertext_is_bound_to_context() -> None:
    r = ring()
    blob = encrypt_bytes(r, b"tenant A secret", b"org:A|version:1")
    with pytest.raises(DecryptionError):
        decrypt_bytes(r, blob, b"org:B|version:1")


def test_bit_flip_detected() -> None:
    r = ring()
    blob = bytearray(encrypt_bytes(r, b"x" * 1000, b"c"))
    blob[-20] ^= 0x01
    with pytest.raises(DecryptionError):
        decrypt_bytes(r, bytes(blob), b"c")


def test_truncation_at_segment_boundary_detected() -> None:
    r = ring()
    data = os.urandom(SEGMENT_SIZE * 3)
    blob = encrypt_bytes(r, data, b"c")
    header_len = len(blob) - (3 * (SEGMENT_SIZE + 16) + 16)
    truncated = blob[: header_len + 2 * (SEGMENT_SIZE + 16)]
    with pytest.raises(DecryptionError):
        decrypt_bytes(r, truncated, b"c")


def test_segment_reordering_detected() -> None:
    r = ring()
    data = os.urandom(SEGMENT_SIZE * 2 + 10)
    blob = encrypt_bytes(r, data, b"c")
    seg = SEGMENT_SIZE + 16
    header_len = len(blob) - (2 * seg + 10 + 16)
    header, s0, s1, tail = (
        blob[:header_len],
        blob[header_len : header_len + seg],
        blob[header_len + seg : header_len + 2 * seg],
        blob[header_len + 2 * seg :],
    )
    with pytest.raises(DecryptionError):
        decrypt_bytes(r, header + s1 + s0 + tail, b"c")


def test_key_rotation_old_objects_still_readable() -> None:
    old = ring()
    blob = encrypt_bytes(old, b"legacy", b"c")
    rotated = KeyRing(keys={"k1": b"\x01" * 32, "k2": b"\x02" * 32}, active_kid="k2")
    assert decrypt_bytes(rotated, blob, b"c") == b"legacy"
    fresh = encrypt_bytes(rotated, b"new", b"c")
    with pytest.raises(DecryptionError):
        decrypt_bytes(old, fresh, b"c")  # old ring does not know k2


def test_value_encryption_round_trip_and_binding() -> None:
    r = ring()
    blob = r.encrypt_value(b"JBSWY3DPEHPK3PXP", b"mfa:user-1")
    assert r.decrypt_value(blob, b"mfa:user-1") == b"JBSWY3DPEHPK3PXP"
    with pytest.raises(DecryptionError):
        r.decrypt_value(blob, b"mfa:user-2")
    with pytest.raises(DecryptionError):
        r.decrypt_value(b"garbage", b"mfa:user-1")


def test_keyring_validates_keys() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        KeyRing(keys={"k": b"short"}, active_kid="k")
    with pytest.raises(ValueError, match="active"):
        KeyRing(keys={"k": b"\x00" * 32}, active_kid="missing")


def test_not_an_encrypted_object() -> None:
    with pytest.raises(DecryptionError):
        decrypt_bytes(ring(), b"%PDF-1.7 plain text", b"c")
