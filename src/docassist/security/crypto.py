"""Envelope encryption for data at rest (documents, exports, MFA secrets, cache values).

File format ``DAENC1`` (streaming, chunked AES-256-GCM)::

    magic(6) | version(1) | kid_len(1) | kid | wrapped_dek_len(2) | wrapped_dek | nonce_prefix(8)
    segment_0 ... segment_n          each = AES-GCM(DEK, nonce_prefix||counter, plaintext<=64KiB)

* A fresh random 256-bit **data key (DEK)** per object, wrapped by a **key-encryption key
  (KEK)** identified by ``kid`` - KEKs can be rotated without re-encrypting file bodies.
* Every segment's AAD binds: the object's *context* (e.g. ``org/<id>/version/<id>``), the
  segment counter and a *final* flag. Swapping a blob between tenants/documents, reordering,
  truncating or appending segments all fail authentication.
* The DEK wrap also binds ``kid`` and the object context.
"""

from __future__ import annotations

import os
import struct
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import BinaryIO

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"DAENC1"
VERSION = 1
SEGMENT_SIZE = 64 * 1024
_TAG = 16
_NONCE_PREFIX = 8
_MAX_SEGMENTS = 2**32 - 1


class DecryptionError(Exception):
    """Ciphertext is corrupt, truncated, tampered with, or bound to another object."""


@dataclass(frozen=True, slots=True)
class KeyRing:
    keys: dict[str, bytes]
    active_kid: str

    def __post_init__(self) -> None:
        if self.active_kid not in self.keys:
            raise ValueError("active key id missing from key ring")
        for kid, key in self.keys.items():
            if len(key) != 32:
                raise ValueError(f"key {kid!r} must be 32 bytes")
            if len(kid.encode()) > 64:
                raise ValueError("key id too long")

    # ---------------------------------------------------------------- DEK wrap
    def wrap(self, dek: bytes, context: bytes) -> tuple[str, bytes]:
        kid = self.active_kid
        nonce = os.urandom(12)
        wrapped = nonce + AESGCM(self.keys[kid]).encrypt(nonce, dek, kid.encode() + b"|" + context)
        return kid, wrapped

    def unwrap(self, kid: str, wrapped: bytes, context: bytes) -> bytes:
        key = self.keys.get(kid)
        if key is None:
            raise DecryptionError("unknown key id")
        try:
            return AESGCM(key).decrypt(wrapped[:12], wrapped[12:], kid.encode() + b"|" + context)
        except InvalidTag as exc:
            raise DecryptionError("key unwrap failed") from exc

    # ---------------------------------------------------------- small values
    def encrypt_value(self, plaintext: bytes, context: bytes) -> bytes:
        """Encrypt a small value (MFA secret, cache entry) into one self-describing blob."""
        dek = AESGCM.generate_key(bit_length=256)
        kid, wrapped = self.wrap(dek, context)
        nonce = os.urandom(12)
        body = AESGCM(dek).encrypt(nonce, plaintext, context)
        kid_b = kid.encode()
        return b"".join(
            [
                b"DAV1",
                bytes([len(kid_b)]),
                kid_b,
                struct.pack(">H", len(wrapped)),
                wrapped,
                nonce,
                body,
            ]
        )

    def decrypt_value(self, blob: bytes, context: bytes) -> bytes:
        try:
            if blob[:4] != b"DAV1":
                raise DecryptionError("bad value header")
            kid_len = blob[4]
            kid = blob[5 : 5 + kid_len].decode()
            pos = 5 + kid_len
            (wrapped_len,) = struct.unpack(">H", blob[pos : pos + 2])
            pos += 2
            wrapped = blob[pos : pos + wrapped_len]
            pos += wrapped_len
            nonce, body = blob[pos : pos + 12], blob[pos + 12 :]
            dek = self.unwrap(kid, wrapped, context)
            return AESGCM(dek).decrypt(nonce, body, context)
        except (IndexError, struct.error, UnicodeDecodeError, InvalidTag) as exc:
            raise DecryptionError("value decryption failed") from exc


def _segment_aad(context: bytes, counter: int, final: bool) -> bytes:
    return MAGIC + context + struct.pack(">I?", counter, final)


def _nonce(prefix: bytes, counter: int) -> bytes:
    return prefix + struct.pack(">I", counter)


class StreamEncryptor:
    """Incrementally encrypt a stream; call ``update`` then ``finalize`` once."""

    def __init__(self, ring: KeyRing, context: bytes) -> None:
        self._context = context
        self._dek = AESGCM.generate_key(bit_length=256)
        self._aead = AESGCM(self._dek)
        self._prefix = os.urandom(_NONCE_PREFIX)
        self._counter = 0
        self._buffer = bytearray()
        kid, wrapped = ring.wrap(self._dek, context)
        kid_b = kid.encode()
        self.header = b"".join(
            [
                MAGIC,
                bytes([VERSION, len(kid_b)]),
                kid_b,
                struct.pack(">H", len(wrapped)),
                wrapped,
                self._prefix,
            ]
        )
        self._finalized = False

    def _seal(self, chunk: bytes, final: bool) -> bytes:
        if self._counter >= _MAX_SEGMENTS:
            raise ValueError("object too large for one encryption stream")
        out = self._aead.encrypt(
            _nonce(self._prefix, self._counter),
            chunk,
            _segment_aad(self._context, self._counter, final),
        )
        self._counter += 1
        return out

    def update(self, data: bytes) -> bytes:
        if self._finalized:
            raise RuntimeError("encryptor already finalized")
        self._buffer.extend(data)
        out = bytearray()
        # Keep at least one byte buffered so the final segment is never empty-by-surprise.
        while len(self._buffer) > SEGMENT_SIZE:
            out += self._seal(bytes(self._buffer[:SEGMENT_SIZE]), final=False)
            del self._buffer[:SEGMENT_SIZE]
        return bytes(out)

    def finalize(self) -> bytes:
        if self._finalized:
            raise RuntimeError("encryptor already finalized")
        self._finalized = True
        return self._seal(bytes(self._buffer), final=True)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise DecryptionError("truncated header")
    return data


def decrypt_stream(ring: KeyRing, stream: BinaryIO, context: bytes) -> Iterator[bytes]:
    """Yield authenticated plaintext segments. Raises :class:`DecryptionError` on tampering."""
    if _read_exact(stream, len(MAGIC)) != MAGIC:
        raise DecryptionError("not an encrypted object")
    version, kid_len = _read_exact(stream, 2)
    if version != VERSION:
        raise DecryptionError("unsupported version")
    kid = _read_exact(stream, kid_len).decode(errors="strict")
    (wrapped_len,) = struct.unpack(">H", _read_exact(stream, 2))
    wrapped = _read_exact(stream, wrapped_len)
    prefix = _read_exact(stream, _NONCE_PREFIX)
    aead = AESGCM(ring.unwrap(kid, wrapped, context))

    counter = 0
    segment_ct = SEGMENT_SIZE + _TAG
    current = stream.read(segment_ct)
    if not current:
        raise DecryptionError("missing final segment")
    while True:
        following = stream.read(segment_ct)
        final = not following
        try:
            plain = aead.decrypt(
                _nonce(prefix, counter), current, _segment_aad(context, counter, final)
            )
        except InvalidTag as exc:
            raise DecryptionError("segment authentication failed") from exc
        yield plain
        if final:
            return
        counter += 1
        current = following


def encrypt_bytes(ring: KeyRing, data: bytes, context: bytes) -> bytes:
    enc = StreamEncryptor(ring, context)
    return enc.header + enc.update(data) + enc.finalize()


def decrypt_bytes(ring: KeyRing, blob: bytes, context: bytes) -> bytes:
    import io

    return b"".join(decrypt_stream(ring, io.BytesIO(blob), context))


async def aiter_bytes(chunks: Iterator[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk
