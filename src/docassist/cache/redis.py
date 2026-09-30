"""Redis access: namespaced keys, short timeouts, encrypted values, graceful degradation.

* Keys are always ``<prefix>:<purpose>:<org>:<digest>`` - user-supplied text never becomes a
  key verbatim (it is hashed), so keys cannot collide across tenants or be crafted.
* Values that contain document-derived text are encrypted with the application key ring
  before they reach Redis (a Redis snapshot on disk does not leak content).
* Every operation has a socket timeout; a Redis outage degrades features (cache misses,
  in-process rate limiting) instead of failing requests.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from docassist.core.logging import get_logger
from docassist.security.crypto import DecryptionError, KeyRing

log = get_logger(__name__)


def digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:40]


class Cache:
    def __init__(self, client: Redis | None, prefix: str, ring: KeyRing) -> None:
        self.client = client
        self._prefix = prefix
        self._ring = ring

    @property
    def available(self) -> bool:
        return self.client is not None

    def key(self, purpose: str, scope: str, *parts: str) -> str:
        return f"{self._prefix}:{purpose}:{scope}:{digest(*parts)}"

    async def get_json(self, key: str, *, encrypted: bool = True) -> Any | None:
        if self.client is None:
            return None
        try:
            raw = await self.client.get(key)
        except RedisError:
            log.warning("cache_get_failed")
            return None
        if raw is None:
            return None
        try:
            blob = raw.encode() if isinstance(raw, str) else bytes(raw)
            data = self._ring.decrypt_value(blob, key.encode()) if encrypted else blob
            return json.loads(data)
        except (DecryptionError, ValueError):
            log.warning("cache_value_invalid")
            return None

    async def set_json(
        self, key: str, value: Any, ttl_seconds: int, *, encrypted: bool = True
    ) -> None:
        if self.client is None or ttl_seconds <= 0:
            return
        data = json.dumps(value, separators=(",", ":"), default=str).encode()
        if encrypted:
            data = self._ring.encrypt_value(data, key.encode())
        try:
            await self.client.set(key, data, ex=ttl_seconds)
        except RedisError:
            log.warning("cache_set_failed")

    async def delete(self, key: str) -> None:
        if self.client is None:
            return
        try:
            await self.client.delete(key)
        except RedisError:
            log.warning("cache_delete_failed")

    async def ping(self) -> bool:
        if self.client is None:
            return False
        try:
            return bool(await self.client.ping())
        except RedisError:
            return False


def create_redis(url: str, *, socket_timeout: float, max_connections: int) -> Redis:
    return Redis.from_url(
        url,
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_timeout,
        max_connections=max_connections,
        health_check_interval=30,
        decode_responses=False,
    )
