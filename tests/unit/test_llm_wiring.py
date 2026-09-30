"""Provider construction from settings and the test override hook."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from docassist.cache.ratelimit import RateLimiter
from docassist.core.enums import Classification
from docassist.llm.anthropic_provider import AnthropicProvider
from docassist.llm.gateway import LOCAL, MAIN, LLMGateway
from docassist.llm.local_extractive import LocalExtractiveProvider
from docassist.llm.openai_compat import OpenAICompatibleChatProvider
from docassist.llm.wiring import build_providers, wire
from docassist.security.ssrf import EgressDenied, EgressPolicy
from tests.conftest import make_settings
from tests.helpers_rag import ScriptedProvider

KEY = "sk-ant-api03-" + "w" * 40


def egress(*private: str) -> EgressPolicy:
    return EgressPolicy.build(["api.anthropic.com"], private)


async def test_defaults_are_offline() -> None:
    providers = build_providers(make_settings(), egress())
    assert isinstance(providers[MAIN], LocalExtractiveProvider)
    assert isinstance(providers[LOCAL], LocalExtractiveProvider)


async def test_anthropic_main_with_local_fallback() -> None:
    settings = make_settings(llm={"provider": "anthropic", "anthropic_api_key": KEY})
    providers = build_providers(settings, egress())
    assert isinstance(providers[MAIN], AnthropicProvider) and providers[MAIN].is_external
    assert isinstance(providers[LOCAL], LocalExtractiveProvider)
    await providers[MAIN].aclose()


def test_anthropic_requires_a_key_and_an_allowlisted_endpoint() -> None:
    with pytest.raises(ValueError, match="anthropic_api_key"):
        build_providers(make_settings(llm={"provider": "anthropic"}), egress())
    settings = make_settings(
        llm={
            "provider": "anthropic",
            "anthropic_api_key": KEY,
            "anthropic_base_url": "https://proxy.evil.example",
        }
    )
    with pytest.raises(EgressDenied):
        build_providers(settings, egress())


async def test_openai_compatible_is_local_only_when_its_host_is_a_private_allowlisted_service() -> (
    None
):
    settings = make_settings(
        llm={
            "provider": "openai_compatible",
            "local_base_url": "http://vllm.internal:8000/v1",
            "local_provider": "none",
        }
    )
    providers = build_providers(settings, egress("vllm.internal"))
    main = providers[MAIN]
    assert isinstance(main, OpenAICompatibleChatProvider) and main.is_external is False
    assert LOCAL not in providers
    await main.aclose()
    hosted = make_settings(
        llm={
            "provider": "openai_compatible",
            "local_base_url": "https://llm.vendor.example/v1",
            "local_provider": "none",
        }
    )
    policy = EgressPolicy.build(["llm.vendor.example"])
    external = build_providers(hosted, policy)[MAIN]
    assert external.is_external is True
    await external.aclose()


async def test_local_openai_compatible_provider() -> None:
    settings = make_settings(
        llm={
            "provider": "anthropic",
            "anthropic_api_key": KEY,
            "local_provider": "openai_compatible",
            "local_base_url": "http://ollama.internal:11434/v1",
        }
    )
    providers = build_providers(settings, egress("ollama.internal"))
    local = providers[LOCAL]
    assert isinstance(local, OpenAICompatibleChatProvider)
    assert local.is_external is False and local.name == "local_openai_compatible"
    for provider in providers.values():
        await provider.aclose()


def test_wire_honours_the_provider_override(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, Any] = {}

    class _Accounting:
        @staticmethod
        def DbUsageRecorder(db: Any) -> Any:
            created["usage"] = db
            return SimpleNamespace(record=None)

        @staticmethod
        def DbBudgetGuard(db: Any, settings: Any, cache: Any) -> Any:
            created["budget"] = (db, cache)
            return SimpleNamespace(check=None)

        @staticmethod
        def DbOrgLlmPolicy(db: Any, cache: Any) -> Any:
            return None

    monkeypatch.setattr("docassist.llm.wiring.importlib.import_module", lambda name: _Accounting)
    fake = ScriptedProvider(name="fake-main")
    container = SimpleNamespace(
        settings=make_settings(),
        egress=egress(),
        limiter=RateLimiter(None, "t"),
        cache=object(),
        db=object(),
        overrides={"llm_providers": {MAIN: fake}},
    )
    wire(container)  # type: ignore[arg-type]
    gateway = container.llm
    assert isinstance(gateway, LLMGateway)
    assert gateway.route(Classification.INTERNAL) == "fake-main"
    assert gateway.route(Classification.RESTRICTED) is None  # no local provider in the override
    assert created["usage"] is container.db
