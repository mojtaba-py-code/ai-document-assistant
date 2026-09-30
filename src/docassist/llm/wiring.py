"""Service wiring for the LLM area: providers per settings + the gateway (``container.llm``).

Provider roles
--------------
* ``main``  - ``settings.llm.provider``: ``anthropic`` (external), ``openai_compatible``
  (external unless its host is on ``outbound.private_network_allowlist``, i.e. an
  explicitly allowlisted on-premises service) or ``local_extractive`` (offline).
* ``local`` - ``settings.llm.local_provider``: ``openai_compatible`` (declared local) or
  ``local_extractive``; ``none`` disables it. It serves data above the external ceiling.

``container.overrides["llm_providers"]`` (a mapping ``{"main": provider, "local": provider}``)
replaces the configured providers - tests use it to inject fakes.

The DB-backed usage recorder, budget guard and organisation policy live in
:mod:`docassist.audit.usage`. They are resolved by module name here, at composition time, so
the ``docassist.llm`` package keeps no static dependency on persistence (import-linter
contract "The LLM layer never touches the database or storage directly").
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol
from urllib.parse import urlsplit

from docassist.cache.ratelimit import RateLimiter
from docassist.cache.redis import Cache
from docassist.core.config import Settings
from docassist.llm.anthropic_provider import AnthropicProvider
from docassist.llm.base import LLMProvider
from docassist.llm.gateway import LOCAL, MAIN, LLMGateway
from docassist.llm.local_extractive import LocalExtractiveProvider
from docassist.llm.openai_compat import OpenAICompatibleChatProvider
from docassist.security.ssrf import EgressPolicy, validate_url

ANTHROPIC_DEFAULT_URL = "https://api.anthropic.com"
USAGE_MODULE = "docassist.audit.usage"


class LlmWiringTarget(Protocol):
    """The parts of the application container this area reads and writes."""

    settings: Settings
    egress: EgressPolicy
    limiter: RateLimiter
    cache: Cache
    db: Any
    overrides: dict[str, Any]
    llm: LLMGateway


def _secret(value: Any) -> str | None:
    return value.get_secret_value() if value is not None else None


def build_providers(settings: Settings, egress: EgressPolicy) -> dict[str, LLMProvider]:
    cfg = settings.llm
    providers: dict[str, LLMProvider] = {}
    if cfg.provider == "anthropic":
        api_key = _secret(cfg.anthropic_api_key)
        if not api_key:
            raise ValueError("llm.anthropic_api_key is required for the anthropic provider")
        validate_url(cfg.anthropic_base_url or ANTHROPIC_DEFAULT_URL, egress)
        providers[MAIN] = AnthropicProvider(
            api_key=api_key,
            base_url=cfg.anthropic_base_url,
            timeout_seconds=cfg.request_timeout_seconds,
            max_retries=cfg.max_retries,
            effort=cfg.main_effort,
            enable_server_fallbacks=cfg.enable_server_fallbacks,
            default_max_output_tokens=cfg.max_output_tokens,
        )
    elif cfg.provider == "openai_compatible":
        if not cfg.local_base_url:
            raise ValueError("llm.local_base_url is required for the openai_compatible provider")
        host = (urlsplit(cfg.local_base_url).hostname or "").lower().rstrip(".")
        providers[MAIN] = OpenAICompatibleChatProvider(
            base_url=cfg.local_base_url,
            api_key=_secret(cfg.local_api_key),
            egress=egress,
            is_external=host not in egress.private_allowlist,
            name="openai_compatible",
            timeout_seconds=cfg.request_timeout_seconds,
            max_retries=cfg.max_retries,
            default_max_output_tokens=cfg.max_output_tokens,
        )
    else:
        providers[MAIN] = LocalExtractiveProvider()

    if cfg.local_provider == "openai_compatible":
        if not cfg.local_base_url:
            raise ValueError(
                "llm.local_base_url is required for the local openai_compatible provider"
            )
        providers[LOCAL] = OpenAICompatibleChatProvider(
            base_url=cfg.local_base_url,
            api_key=_secret(cfg.local_api_key),
            egress=egress,
            is_external=False,
            name="local_openai_compatible",
            timeout_seconds=cfg.request_timeout_seconds,
            max_retries=cfg.max_retries,
            default_max_output_tokens=cfg.max_output_tokens,
        )
    elif cfg.local_provider == "local_extractive":
        providers[LOCAL] = LocalExtractiveProvider()
    return providers


def wire(container: LlmWiringTarget) -> None:
    """Attach ``container.llm``."""
    settings = container.settings
    override = container.overrides.get("llm_providers")
    providers: dict[str, LLMProvider] = (
        dict(override) if override is not None else build_providers(settings, container.egress)
    )
    accounting = importlib.import_module(USAGE_MODULE)
    container.llm = LLMGateway(
        settings,
        providers,
        usage=accounting.DbUsageRecorder(container.db),
        budget=accounting.DbBudgetGuard(container.db, settings, container.cache),
        limiter=container.limiter,
        org_policy=accounting.DbOrgLlmPolicy(container.db, container.cache),
    )
