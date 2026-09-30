"""Job handler registry.

Each feature area declares its background handlers in ``<area>/jobs.py`` with the
``@job_handler("kind")`` decorator; the worker imports :data:`HANDLER_MODULES` at start-up.
A job kind nobody registered is dead-lettered instead of being retried forever.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docassist.api.container import Container
    from docassist.db.session import Database
    from docassist.jobs.queue import ClaimedJob


@dataclass(frozen=True, slots=True)
class JobContext:
    container: Container
    worker_db: Database


Handler = Callable[["JobContext", "ClaimedJob"], Awaitable[dict[str, Any] | None]]

HANDLERS: dict[str, Handler] = {}

HANDLER_MODULES = (
    "docassist.ingestion.jobs",
    "docassist.search.jobs",
    "docassist.intelligence.jobs",
)


def job_handler(kind: str) -> Callable[[Handler], Handler]:
    def decorator(fn: Handler) -> Handler:
        if kind in HANDLERS and HANDLERS[kind] is not fn:
            raise RuntimeError(f"duplicate job handler for {kind!r}")
        HANDLERS[kind] = fn
        return fn

    return decorator


def load_handlers() -> dict[str, Handler]:
    import importlib

    for module in HANDLER_MODULES:
        importlib.import_module(module)
    return HANDLERS
