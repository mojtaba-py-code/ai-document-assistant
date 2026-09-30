"""LLMGateway: routing/data policy, pseudonymisation, budgets, circuit breaker, schema repair."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from docassist.cache.ratelimit import RateLimiter
from docassist.core.enums import Classification
from docassist.core.errors import QuotaExceeded, RateLimited
from docassist.llm.base import (
    ChatMessage,
    LLMError,
    LLMOutputInvalid,
    LLMPolicyDenied,
    LLMRefused,
    LLMRequest,
    LLMTask,
    LLMUnavailable,
    ToolCall,
    ToolSpec,
)
from docassist.llm.circuit import CircuitBreaker, CircuitState
from docassist.llm.gateway import LOCAL, MAIN, LLMGateway
from docassist.llm.local_extractive import LOCAL_EXTRACTIVE_MODEL, LocalExtractiveProvider
from tests.conftest import make_settings
from tests.helpers_rag import (
    FixedOrgPolicy,
    MemoryBudget,
    MemoryUsage,
    ScriptedProvider,
    make_gateway,
    result,
)

ORG = uuid.uuid4()
USER = uuid.uuid4()
EMAIL = "jane.doe@example.com"
IBAN = "DE89 3704 0044 0532 0130 00"
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string", "maxLength": 40}},
}


@pytest.fixture(scope="module")
def settings() -> Any:
    return make_settings(
        llm={
            "provider": "anthropic",
            "anthropic_api_key": "sk-ant-api03-" + "k" * 40,
            "circuit_failure_threshold": 2,
            "circuit_reset_seconds": 30,
            "max_context_tokens": 1000,
        }
    )


def req(**kwargs: Any) -> LLMRequest:
    base: dict[str, Any] = {
        "task": LLMTask.ANSWER,
        "system": "system prompt",
        "messages": [ChatMessage(role="user", content="question")],
    }
    base.update(kwargs)
    return LLMRequest(**base)


# --------------------------------------------------------------------------- #
# Routing and data governance
# --------------------------------------------------------------------------- #
def test_routing_matrix_with_local_provider(settings: Any) -> None:
    gateway = make_gateway(settings, ScriptedProvider(name="anthropic"), LocalExtractiveProvider())
    assert gateway.route(Classification.PUBLIC) == "anthropic"
    assert gateway.route(Classification.CONFIDENTIAL) == "anthropic"
    assert gateway.route(Classification.RESTRICTED) == "local_extractive"


def test_restricted_is_refused_without_a_local_provider(settings: Any) -> None:
    gateway = make_gateway(settings, ScriptedProvider(name="anthropic"))
    assert gateway.route(Classification.CONFIDENTIAL) == "anthropic"
    assert gateway.route(Classification.RESTRICTED) is None


def test_a_local_main_provider_serves_everything(settings: Any) -> None:
    gateway = make_gateway(settings, ScriptedProvider(name="onprem", is_external=False))
    assert gateway.route(Classification.RESTRICTED) == "onprem"


def test_lower_deployment_ceiling() -> None:
    strict = make_settings(llm={"external_max_classification": "INTERNAL"})
    gateway = make_gateway(strict, ScriptedProvider(name="anthropic"))
    assert gateway.route(Classification.INTERNAL) == "anthropic"
    assert gateway.route(Classification.CONFIDENTIAL) is None


async def test_restricted_never_reaches_the_external_provider(settings: Any) -> None:
    external = ScriptedProvider(name="anthropic")
    gateway = make_gateway(settings, external)
    with pytest.raises(LLMPolicyDenied):
        await gateway.complete(req(data_classification=Classification.RESTRICTED), org_id=ORG)
    assert external.requests == []


async def test_restricted_goes_to_the_local_provider(settings: Any) -> None:
    external = ScriptedProvider(name="anthropic")
    local = ScriptedProvider([result(text="local answer")], name="local", is_external=False)
    gateway = make_gateway(settings, external, local)
    out = await gateway.complete(req(data_classification=Classification.RESTRICTED), org_id=ORG)
    assert out.text == "local answer"
    assert external.requests == [] and len(local.requests) == 1


async def test_org_policy_can_lower_but_not_raise_the_ceiling(settings: Any) -> None:
    lowered = make_gateway(
        settings,
        ScriptedProvider(name="anthropic"),
        org_policy=FixedOrgPolicy(Classification.PUBLIC),
    )
    assert await lowered.route_for_org(Classification.INTERNAL, ORG) is None
    assert await lowered.route_for_org(Classification.PUBLIC, ORG) == "anthropic"
    with pytest.raises(LLMPolicyDenied):
        await lowered.complete(req(data_classification=Classification.INTERNAL), org_id=ORG)
    raised = make_gateway(
        settings,
        ScriptedProvider(name="anthropic"),
        org_policy=FixedOrgPolicy(Classification.RESTRICTED),
    )
    assert await raised.route_for_org(Classification.RESTRICTED, ORG) is None
    assert await raised.max_routable(ORG) is Classification.CONFIDENTIAL


async def test_policy_table_for_the_ui(settings: Any) -> None:
    gateway = make_gateway(settings, ScriptedProvider(name="anthropic"), LocalExtractiveProvider())
    rows = {row.classification: row for row in await gateway.policy()}
    assert (
        rows[Classification.INTERNAL].model == "claude-opus-5-5"
        and rows[Classification.INTERNAL].external
    )
    assert rows[Classification.RESTRICTED].provider == "local_extractive"
    assert rows[Classification.RESTRICTED].model == LOCAL_EXTRACTIVE_MODEL
    assert not rows[Classification.RESTRICTED].external


async def test_fast_tier_for_classify_extract_rerank(settings: Any) -> None:
    provider = ScriptedProvider([result(text="x")] * 4, name="anthropic")
    gateway = make_gateway(settings, provider)
    for task in (LLMTask.CLASSIFY, LLMTask.EXTRACT, LLMTask.RERANK, LLMTask.ANSWER):
        await gateway.complete(req(task=task), org_id=ORG)
    assert provider.models == [
        "claude-haiku-4-5",
        "claude-haiku-4-5",
        "claude-haiku-4-5",
        "claude-opus-5-5",
    ]


# --------------------------------------------------------------------------- #
# Request shape checks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "messages",
    [
        [],
        [ChatMessage(role="assistant", content="hi")],
        [ChatMessage(role="user", content="q"), ChatMessage(role="assistant", content="prefill {")],
    ],
)
async def test_invalid_message_shapes_are_rejected(
    settings: Any, messages: list[ChatMessage]
) -> None:
    provider = ScriptedProvider(name="anthropic")
    with pytest.raises(LLMError):
        await make_gateway(settings, provider).complete(req(messages=messages), org_id=ORG)
    assert provider.requests == []


async def test_non_strict_schema_is_rejected(settings: Any) -> None:
    loose = {"type": "object", "properties": {"a": {"type": "string"}}}
    with pytest.raises(LLMError):
        await make_gateway(settings, ScriptedProvider(name="anthropic")).complete(
            req(output_schema=loose), org_id=ORG
        )


async def test_oversized_requests_are_rejected_before_the_call(settings: Any) -> None:
    provider = ScriptedProvider(name="anthropic")
    huge = "word " * 40_000
    with pytest.raises(LLMError) as info:
        await make_gateway(settings, provider).complete(
            req(messages=[ChatMessage(role="user", content=huge)]), org_id=ORG
        )
    assert "too large" in info.value.public_message
    assert provider.requests == []


async def test_output_tokens_are_capped_by_settings(settings: Any) -> None:
    provider = ScriptedProvider([result(text="x")], name="anthropic")
    await make_gateway(settings, provider).complete(req(max_output_tokens=10**6), org_id=ORG)
    assert provider.requests[0].max_output_tokens == settings.llm.max_output_tokens


async def test_tools_require_a_tool_capable_provider(settings: Any) -> None:
    tool = ToolSpec(
        name="t",
        description="d",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": [],
            "properties": {},
        },
    )
    gateway = make_gateway(settings, ScriptedProvider(name="anthropic", supports_tools=False))
    with pytest.raises(LLMError):
        await gateway.complete(req(task=LLMTask.AGENT, tools=[tool]), org_id=ORG)


# --------------------------------------------------------------------------- #
# Pseudonymisation
# --------------------------------------------------------------------------- #
async def test_external_provider_never_sees_raw_pii_and_output_is_restored(settings: Any) -> None:
    def respond(request: LLMRequest, _model: str) -> Any:
        prompt = request.messages[0].content
        assert isinstance(prompt, str)
        email_token = next(tok for tok in prompt.split() if tok.startswith("[EMAIL_"))
        return result(
            data={"answer": f"Write to {email_token}"},
            tool_calls=[
                ToolCall(id="1", name="t", arguments={"to": email_token, "nested": [email_token]})
            ],
        )

    external = ScriptedProvider(name="anthropic", responder=respond)
    usage = MemoryUsage()
    gateway = make_gateway(settings, external, usage=usage)
    out = await gateway.complete(
        req(
            system=f"Contact {EMAIL} for help.",
            messages=[ChatMessage(role="user", content=f"Pay {IBAN} and email {EMAIL} today")],
            output_schema=SCHEMA,
        ),
        org_id=ORG,
        user_id=USER,
    )
    seen = external.seen_text()
    assert EMAIL not in seen and IBAN not in seen and "DE89" not in seen
    assert "[EMAIL_1_" in seen and "[IBAN_1_" in seen
    assert out.data == {"answer": f"Write to {EMAIL}"}
    assert out.text.endswith(f'{EMAIL}"}}')
    assert out.tool_calls[0].arguments == {"to": EMAIL, "nested": [EMAIL]}
    assert out.pseudonymized == 2
    assert usage.rows[0]["status"] == "ok" and usage.rows[0]["user_id"] == USER


async def test_tool_results_in_block_content_are_pseudonymized(settings: Any) -> None:
    external = ScriptedProvider([result(text="done")], name="anthropic")
    messages = [
        ChatMessage(role="user", content=[{"type": "text", "text": f"mail {EMAIL}"}]),
        ChatMessage(
            role="assistant",
            content=[
                {"type": "tool_use", "id": "t1", "name": "x", "input": {"q": "[EMAIL_1_abc123]"}}
            ],
        ),
        ChatMessage(
            role="user",
            content=[{"type": "tool_result", "tool_use_id": "t1", "content": f"owner {EMAIL}"}],
        ),
    ]
    await make_gateway(settings, external).complete(req(messages=messages), org_id=ORG)
    seen = external.seen_text()
    assert EMAIL not in seen
    # provider-originated assistant content is forwarded byte-for-byte
    assert external.requests[0].messages[1].content == messages[1].content


async def test_session_keeps_placeholders_stable_across_turns(settings: Any) -> None:
    external = ScriptedProvider([result(text="a"), result(text="b")], name="anthropic")
    gateway = make_gateway(settings, external)
    session = gateway.new_session()
    first = [ChatMessage(role="user", content=f"about {EMAIL}")]
    await gateway.complete(req(messages=first), org_id=ORG, session=session)
    second = [
        *first,
        ChatMessage(role="assistant", content=[{"type": "text", "text": "a"}]),
        ChatMessage(role="user", content=f"and again {EMAIL}"),
    ]
    await gateway.complete(req(messages=second), org_id=ORG, session=session)
    turn_one = external.requests[0].messages[0].content
    turn_two = external.requests[1].messages[0].content
    assert turn_one == turn_two  # identical prefix -> prompt cache and thinking replay stay valid


async def test_session_cannot_switch_providers(settings: Any) -> None:
    external = ScriptedProvider([result(text="a")], name="anthropic")
    local = ScriptedProvider([result(text="b")], name="local", is_external=False)
    gateway = make_gateway(settings, external, local)
    session = gateway.new_session()
    await gateway.complete(req(), org_id=ORG, session=session)
    with pytest.raises(LLMPolicyDenied):
        await gateway.complete(
            req(data_classification=Classification.RESTRICTED), org_id=ORG, session=session
        )


async def test_pseudonymization_can_be_disabled_per_request(settings: Any) -> None:
    external = ScriptedProvider([result(text="x")], name="anthropic")
    await make_gateway(settings, external).complete(
        req(messages=[ChatMessage(role="user", content=EMAIL)], pseudonymize=False), org_id=ORG
    )
    assert EMAIL in external.seen_text()


async def test_local_providers_receive_the_original_text(settings: Any) -> None:
    local = ScriptedProvider([result(text="x")], name="onprem", is_external=False)
    await make_gateway(settings, local).complete(
        req(messages=[ChatMessage(role="user", content=EMAIL)]), org_id=ORG
    )
    assert EMAIL in local.seen_text()


# --------------------------------------------------------------------------- #
# Budgets, rate limits, circuit breaker
# --------------------------------------------------------------------------- #
async def test_budget_exceeded_is_a_429(settings: Any) -> None:
    provider = ScriptedProvider(name="anthropic")
    gateway = make_gateway(settings, provider, budget=MemoryBudget(exceeded=True))
    with pytest.raises(QuotaExceeded) as info:
        await gateway.complete(req(), org_id=ORG)
    assert info.value.status_code == 429
    assert provider.requests == []


async def test_per_org_rate_limit() -> None:
    tight = make_settings(rate_limit={"llm_per_org": {"requests": 1, "per_seconds": 3600}})
    provider = ScriptedProvider([result(text="x"), result(text="y")], name="anthropic")
    gateway = make_gateway(tight, provider, limiter=RateLimiter(None, "t"))
    await gateway.complete(req(), org_id=ORG)
    with pytest.raises(RateLimited):
        await gateway.complete(req(), org_id=ORG)
    await gateway.complete(req(), org_id=uuid.uuid4())  # other organisations are unaffected


async def test_circuit_opens_after_failures_and_blocks_calls(settings: Any) -> None:
    provider = ScriptedProvider(
        [LLMUnavailable(internal_detail="down"), LLMUnavailable(internal_detail="down")],
        name="anthropic",
    )
    usage = MemoryUsage()
    gateway = make_gateway(settings, provider, usage=usage)
    for _ in range(2):
        with pytest.raises(LLMUnavailable):
            await gateway.complete(req(), org_id=ORG)
    assert gateway.circuit_states() == {"main:anthropic": "open"}
    with pytest.raises(LLMUnavailable):
        await gateway.complete(req(), org_id=ORG)
    assert len(provider.requests) == 2  # the third call never left the process
    assert [row["status"] for row in usage.rows] == ["unavailable", "unavailable"]


def test_circuit_breaker_state_machine() -> None:
    now = [0.0]
    breaker = CircuitBreaker("p", failure_threshold=2, reset_seconds=10, clock=lambda: now[0])
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN and not breaker.allow()
    now[0] = 10.0
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow()  # the single probe
    assert not breaker.allow()  # no second concurrent probe
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    now[0] = 25.0
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED and breaker.allow()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED  # failures must be consecutive


def test_circuit_breaker_release_frees_the_probe() -> None:
    now = [0.0]
    breaker = CircuitBreaker("p", failure_threshold=1, reset_seconds=1, clock=lambda: now[0])
    breaker.record_failure()
    now[0] = 2.0
    assert breaker.allow()
    breaker.release()
    assert breaker.allow()


async def test_refusals_and_client_errors_do_not_trip_the_circuit(settings: Any) -> None:
    provider = ScriptedProvider(
        [
            LLMRefused().with_usage(10, 2, "claude-opus-5-5"),
            LLMError(internal_detail="400"),
            LLMRefused(),
        ],
        name="anthropic",
    )
    usage = MemoryUsage()
    gateway = make_gateway(settings, provider, usage=usage)
    for expected in (LLMRefused, LLMError, LLMRefused):
        with pytest.raises(expected):
            await gateway.complete(req(), org_id=ORG)
    assert gateway.circuit_states() == {"main:anthropic": "closed"}
    assert usage.rows[0]["status"] == "refused" and usage.rows[0]["input_tokens"] == 10


# --------------------------------------------------------------------------- #
# Schema validation and repair
# --------------------------------------------------------------------------- #
async def test_invalid_output_is_repaired_once(settings: Any) -> None:
    provider = ScriptedProvider(
        [result(data={"answer": "x" * 100}), result(data={"answer": "short"})], name="anthropic"
    )
    out = await make_gateway(settings, provider).complete(req(output_schema=SCHEMA), org_id=ORG)
    assert out.data == {"answer": "short"}
    repair = provider.requests[1]
    assert repair.messages[-1].role == "user"
    assert "maxLength" in str(repair.messages[-1].content)
    assert out.input_tokens == 200  # both calls are accounted


async def test_output_still_invalid_after_repair_raises(settings: Any) -> None:
    provider = ScriptedProvider([result(text="nope"), result(data={"wrong": 1})], name="anthropic")
    usage = MemoryUsage()
    with pytest.raises(LLMOutputInvalid):
        await make_gateway(settings, provider, usage=usage).complete(
            req(output_schema=SCHEMA), org_id=ORG
        )
    assert len(provider.requests) == 2
    assert usage.rows[-1]["status"] == "invalid_output"


async def test_usage_and_cost_are_recorded(settings: Any) -> None:
    provider = ScriptedProvider(
        [result(text="x", model="claude-opus-5-5", input_tokens=1_000_000, output_tokens=0)],
        name="anthropic",
    )
    usage = MemoryUsage()
    await make_gateway(settings, provider, usage=usage).complete(req(), org_id=ORG, user_id=USER)
    row = usage.rows[0]
    assert row["cost_usd"] == pytest.approx(4.0)
    assert row["task"] == "answer" and row["provider"] == "anthropic" and row["org_id"] == ORG


async def test_closing_the_gateway_closes_providers(settings: Any) -> None:
    main, local = ScriptedProvider(name="a"), ScriptedProvider(name="b", is_external=False)
    await LLMGateway(
        settings, {MAIN: main, LOCAL: local}, MemoryUsage(), MemoryBudget(), RateLimiter(None, "t")
    ).aclose()
    assert main.closed and local.closed


def test_gateway_requires_a_main_provider(settings: Any) -> None:
    with pytest.raises(ValueError, match="main"):
        LLMGateway(
            settings,
            {LOCAL: ScriptedProvider()},
            MemoryUsage(),
            MemoryBudget(),
            RateLimiter(None, "t"),
        )
    with pytest.raises(ValueError, match="unknown"):
        LLMGateway(
            settings,
            {MAIN: ScriptedProvider(), "backup": ScriptedProvider()},
            MemoryUsage(),
            MemoryBudget(),
            RateLimiter(None, "t"),
        )
