"""Rate limiting with GCRA (generic cell rate algorithm).

GCRA keeps one timestamp per key (the *theoretical arrival time*) and is exact, burst-aware
and O(1). The Redis implementation is a single atomic Lua script using the server clock, so
many API replicas share one limit. If Redis is unavailable the limiter falls back to an
in-process GCRA - limits become per-replica (looser) but never disappear (no fail-open).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

from docassist.cache.redis import digest
from docassist.core.config import RateRule
from docassist.core.errors import RateLimited
from docassist.core.logging import get_logger
from docassist.observability import metrics

log = get_logger(__name__)

_GCRA_LUA = """
local key = KEYS[1]
local emission = tonumber(ARGV[1])
local tolerance = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000000 + tonumber(t[2])
local tat = tonumber(redis.call('GET', key) or now)
if tat < now then tat = now end
local new_tat = tat + emission
local allow_at = new_tat - tolerance
if allow_at > now then
  return {0, allow_at - now}
end
redis.call('SET', key, new_tat, 'PX', ttl)
return {1, 0}
"""


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    retry_after: float


def _params(rule: RateRule) -> tuple[int, int]:
    emission_us = int(rule.per_seconds * 1_000_000 / rule.requests)
    tolerance_us = emission_us * rule.requests  # burst of `requests`
    return emission_us, tolerance_us


class _LocalGcra:
    def __init__(self, max_keys: int = 50_000) -> None:
        self._tat: dict[str, int] = {}
        self._max_keys = max_keys
        self._lock = asyncio.Lock()

    async def hit(self, key: str, rule: RateRule) -> Decision:
        emission_us, tolerance_us = _params(rule)
        # Integer microseconds, as in the Lua script: with a float clock, a new key's
        # `now + emission - tolerance` can round above `now` and reject its first request.
        now = time.monotonic_ns() // 1_000
        async with self._lock:
            if len(self._tat) > self._max_keys:
                cutoff = now
                self._tat = {k: v for k, v in self._tat.items() if v > cutoff}
            tat = max(self._tat.get(key, now), now)
            new_tat = tat + emission_us
            allow_at = new_tat - tolerance_us
            if allow_at > now:
                return Decision(False, (allow_at - now) / 1_000_000)
            self._tat[key] = new_tat
            return Decision(True, 0.0)


class RateLimiter:
    def __init__(self, redis: Redis | None, prefix: str, *, enabled: bool = True) -> None:
        self._redis = redis
        self._prefix = prefix
        self._enabled = enabled
        self._local = _LocalGcra()
        self._script = redis.register_script(_GCRA_LUA) if redis is not None else None

    def _key(self, bucket: str, identity: str) -> str:
        return f"{self._prefix}:rl:{bucket}:{digest(identity)}"

    async def check(self, bucket: str, identity: str, rule: RateRule) -> Decision:
        if not self._enabled:
            return Decision(True, 0.0)
        key = self._key(bucket, identity)
        if self._script is not None:
            emission_us, tolerance_us = _params(rule)
            ttl_ms = max(1_000, (tolerance_us + emission_us) // 1_000)
            try:
                allowed, wait_us = await self._script(
                    keys=[key], args=[emission_us, tolerance_us, ttl_ms]
                )
                return Decision(bool(int(allowed)), int(wait_us) / 1_000_000)
            except RedisError:
                log.warning("rate_limit_redis_unavailable", bucket=bucket)
                metrics.DEGRADED_MODE.labels(component="rate_limiter").inc()
        return await self._local.hit(key, rule)

    async def enforce(self, bucket: str, identity: str, rule: RateRule) -> None:
        decision = await self.check(bucket, identity, rule)
        if not decision.allowed:
            metrics.RATE_LIMITED.labels(bucket=bucket).inc()
            raise RateLimited(decision.retry_after)
