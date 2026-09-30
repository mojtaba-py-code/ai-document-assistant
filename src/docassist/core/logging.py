"""Structured JSON logging with secret redaction.

Every log line passes through ``_redact_processor`` which

* replaces values of sensitive keys (password, token, authorization, cookie, api_key ...),
* masks secret-looking substrings (JWTs, API keys, private keys) inside any string value,
* truncates very long values so a log line can never carry a whole document.

Standard-library loggers (uvicorn, sqlalchemy, httpx) are routed through the same pipeline.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

import structlog

from docassist.core.redaction import redact_secrets

_SENSITIVE_KEYS = (
    "password",
    "passwd",
    "secret",
    "token",
    "authorization",
    "cookie",
    "api_key",
    "apikey",
    "private_key",
    "set-cookie",
    "mfa_code",
    "otp",
    "credential",
)
_MAX_VALUE_CHARS = 2_000


_COUNTER_SUFFIXES = ("_tokens", "tokens", "token_count", "token_version", "token_type")


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    if lowered.endswith(_COUNTER_SUFFIXES):
        return False  # usage counters such as input_tokens are not credentials
    return any(marker in lowered for marker in _SENSITIVE_KEYS)


def _scrub(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "[depth-limit]"
    if isinstance(value, str):
        cleaned = redact_secrets(value)
        if len(cleaned) > _MAX_VALUE_CHARS:
            cleaned = cleaned[:_MAX_VALUE_CHARS] + "...[truncated]"
        return cleaned
    if isinstance(value, Mapping):
        return {
            k: "[REDACTED]" if isinstance(k, str) and _is_sensitive_key(k) else _scrub(v, depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [_scrub(v, depth + 1) for v in value[:50]]
    return value


def _redact_processor(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key in list(event_dict.keys()):
        if key in {"timestamp", "level", "logger"}:
            continue
        if _is_sensitive_key(key):
            event_dict[key] = "[REDACTED]"
        else:
            event_dict[key] = _scrub(event_dict[key])
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "json", service: str = "docassist") -> None:
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _redact_processor,
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # Libraries that would otherwise log request bodies / SQL parameters stay quiet.
    for noisy in ("httpx", "httpcore", "httpx2", "anthropic", "sqlalchemy.engine", "asyncio"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLevelName(level)))
    logging.getLogger("uvicorn.access").disabled = True  # replaced by our access log middleware
    structlog.contextvars.bind_contextvars(service=service)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(name)
