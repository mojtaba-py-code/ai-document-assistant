"""Per-provider circuit breaker (closed -> open -> half-open -> closed).

* **closed**: calls flow; ``failure_threshold`` *consecutive* transient failures open it.
* **open**: calls are rejected immediately (no request leaves the process) until
  ``reset_seconds`` have passed.
* **half-open**: exactly one probe call is let through; success closes the circuit,
  failure re-opens it for another ``reset_seconds``.

The breaker is process-local and meant for one asyncio event loop (no locking needed:
state changes happen between awaits). Only *transient* failures (timeouts, 5xx, 429,
connection errors) should be recorded - a request the provider rejected as invalid says
nothing about the provider's health.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int,
        reset_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1 or reset_seconds <= 0:
            raise ValueError("failure_threshold must be >= 1 and reset_seconds > 0")
        self.name = name
        self._threshold = failure_threshold
        self._reset = reset_seconds
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        if self._state is CircuitState.OPEN and self._clock() - self._opened_at >= self._reset:
            return CircuitState.HALF_OPEN
        return self._state

    def allow(self) -> bool:
        """Ask for permission to make one call (reserves the probe slot when half-open)."""
        state = self.state
        if state is CircuitState.CLOSED:
            return True
        if state is CircuitState.OPEN:
            return False
        if self._probe_in_flight:
            return False
        self._state = CircuitState.HALF_OPEN
        self._probe_in_flight = True
        return True

    def record_success(self) -> None:
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._probe_in_flight = False

    def record_failure(self) -> None:
        self._probe_in_flight = False
        if self._state is CircuitState.HALF_OPEN:
            self._trip()
            return
        self._failures += 1
        if self._failures >= self._threshold:
            self._trip()

    def release(self) -> None:
        """The permitted call ended without a health verdict (cancelled, invalid request)."""
        self._probe_in_flight = False

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._failures = 0
