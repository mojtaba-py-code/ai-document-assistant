"""The single entry point for every model call in the platform.

For each :class:`LLMRequest` the gateway, in order:

1. validates the request shape (non-empty, starts and ends with a user turn - assistant
   prefill is rejected by current models);
2. **routes by data classification**: text of classification C may go to the external
   (main) provider only if ``C.rank <= ceiling.rank`` where the ceiling is
   ``settings.llm.external_max_classification``, optionally *lowered* by the organisation's
   own policy; otherwise it goes to the local provider, or is refused with
   :class:`LLMPolicyDenied`. A local main provider serves every classification;
3. picks the model tier (fast for classify/rerank/extract, main otherwise);
4. enforces the per-organisation rate limit (``rate_limit.llm_per_org``) and the monthly
   token budget (:class:`BudgetGuard`);
5. rejects the call early when the provider's circuit is open;
6. **pseudonymises PII** (emails, IBANs, cards, phone numbers, SSNs, secrets) in the system
   prompt and messages when the provider is external, ``llm.pseudonymize_pii_for_external``
   is on and the request allows it - so raw PII never leaves the process - and restores the
   placeholders in ``text``, ``data`` and tool-call arguments afterwards;
7. estimates the input size and rejects requests above the context budget;
8. calls the provider, validates ``data`` against ``output_schema`` (one repair round-trip,
   then :class:`LLMOutputInvalid`);
9. records usage and estimated cost (:class:`UsageRecorder`, metrics) - log lines carry
   counts and identifiers only, never prompt or completion text.

Agent loops pass an :class:`LLMSession` so that (a) the same PII value maps to the same
placeholder on every turn (the conversation prefix stays byte-identical, which prompt
caching and thinking-block replay require) and (b) every turn stays on the provider that
produced the earlier turns.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from docassist.cache.ratelimit import RateLimiter
from docassist.core.config import Settings
from docassist.core.enums import Classification
from docassist.core.logging import get_logger
from docassist.core.redaction import Pseudonymizer
from docassist.core.text import estimate_tokens
from docassist.llm import schema as jsonschema
from docassist.llm.base import (
    FAST_TASKS,
    BudgetGuard,
    ChatMessage,
    LLMError,
    LLMOutputInvalid,
    LLMPolicyDenied,
    LLMProvider,
    LLMRefused,
    LLMRequest,
    LLMResult,
    LLMUnavailable,
    OrgLlmPolicy,
    ToolCall,
    UsageRecorder,
)
from docassist.llm.circuit import CircuitBreaker
from docassist.llm.local_extractive import LOCAL_EXTRACTIVE_MODEL
from docassist.llm.pricing import estimate_cost
from docassist.observability import metrics

log = get_logger(__name__)

MAIN = "main"
LOCAL = "local"
REQUEST_OVERHEAD_TOKENS = 8_000
"""Input allowance on top of ``llm.max_context_tokens`` for the system prompt, the question,
conversation/tool-call bookkeeping and the output schema."""
MAX_MESSAGES = 64


@dataclass(frozen=True, slots=True)
class ProviderSlot:
    role: str  # "main" | "local"
    provider: LLMProvider
    main_model: str
    fast_model: str

    def model_for(self, request: LLMRequest) -> str:
        return self.fast_model if request.task in FAST_TASKS else self.main_model


@dataclass(frozen=True, slots=True)
class RouteInfo:
    classification: Classification
    allowed: bool
    provider: str | None
    model: str | None
    external: bool


@dataclass(frozen=True, slots=True)
class _Call:
    slot: ProviderSlot
    model: str
    request: LLMRequest
    org_id: uuid.UUID
    user_id: uuid.UUID | None
    started: float


@dataclass(slots=True)
class LLMSession:
    """State shared by the turns of one multi-turn exchange (agent loop)."""

    pseudonymizer: Pseudonymizer = field(default_factory=Pseudonymizer)
    slot_role: str | None = None


def _models_for(settings: Settings, role: str, provider: LLMProvider) -> tuple[str, str]:
    cfg = settings.llm
    kind = cfg.provider if role == MAIN else cfg.local_provider
    if kind == "anthropic":
        return cfg.main_model, cfg.fast_model
    if kind == "openai_compatible":
        return cfg.local_model, cfg.local_model
    if kind == "local_extractive" or provider.name == "local_extractive":
        return LOCAL_EXTRACTIVE_MODEL, LOCAL_EXTRACTIVE_MODEL
    return cfg.main_model, cfg.fast_model


class LLMGateway:
    def __init__(
        self,
        settings: Settings,
        providers: Mapping[str, LLMProvider],
        usage: UsageRecorder,
        budget: BudgetGuard,
        limiter: RateLimiter,
        *,
        org_policy: OrgLlmPolicy | None = None,
        breakers: Mapping[str, CircuitBreaker] | None = None,
    ) -> None:
        if MAIN not in providers:
            raise ValueError("providers must contain a 'main' provider")
        unknown = set(providers) - {MAIN, LOCAL}
        if unknown:
            raise ValueError(f"unknown provider roles: {sorted(unknown)}")
        self.settings = settings
        self.usage = usage
        self.budget = budget
        self.org_policy = org_policy
        self._limiter = limiter
        self._slots: dict[str, ProviderSlot] = {}
        for role, provider in providers.items():
            main_model, fast_model = _models_for(settings, role, provider)
            self._slots[role] = ProviderSlot(role, provider, main_model, fast_model)
        cfg = settings.llm
        self._breakers: dict[str, CircuitBreaker] = dict(breakers or {})
        for role in self._slots:
            self._breakers.setdefault(
                role,
                CircuitBreaker(
                    role,
                    failure_threshold=cfg.circuit_failure_threshold,
                    reset_seconds=cfg.circuit_reset_seconds,
                ),
            )

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    @property
    def deployment_ceiling(self) -> Classification:
        return self.settings.llm.external_max_classification

    def _ceiling(self, org_ceiling: Classification | None) -> Classification:
        ceiling = self.deployment_ceiling
        if org_ceiling is not None and org_ceiling.rank < ceiling.rank:
            return org_ceiling  # organisations may only LOWER the deployment ceiling
        return ceiling

    def _slot_for(
        self, classification: Classification, org_ceiling: Classification | None = None
    ) -> ProviderSlot | None:
        main = self._slots[MAIN]
        if not main.provider.is_external:
            return main
        if classification.rank <= self._ceiling(org_ceiling).rank:
            return main
        local = self._slots.get(LOCAL)
        if local is not None and not local.provider.is_external:
            return local
        return None

    def route(self, classification: Classification) -> str | None:
        """Provider name for ``classification`` under the deployment policy (None = refused)."""
        slot = self._slot_for(classification)
        return slot.provider.name if slot else None

    async def org_ceiling(self, org_id: uuid.UUID) -> Classification | None:
        if self.org_policy is None:
            return None
        return await self.org_policy.external_ceiling(org_id)

    async def route_for_org(self, classification: Classification, org_id: uuid.UUID) -> str | None:
        slot = self._slot_for(classification, await self.org_ceiling(org_id))
        return slot.provider.name if slot else None

    async def max_routable(self, org_id: uuid.UUID) -> Classification | None:
        """The highest classification this organisation may send to *some* provider."""
        org_ceiling = await self.org_ceiling(org_id)
        allowed = [c for c in Classification if self._slot_for(c, org_ceiling) is not None]
        return Classification.highest(allowed) if allowed else None

    async def policy(self, org_id: uuid.UUID | None = None) -> list[RouteInfo]:
        """Which provider/model serves each classification (for the UI)."""
        org_ceiling = await self.org_ceiling(org_id) if org_id is not None else None
        rows: list[RouteInfo] = []
        for classification in Classification:
            slot = self._slot_for(classification, org_ceiling)
            rows.append(
                RouteInfo(
                    classification=classification,
                    allowed=slot is not None,
                    provider=slot.provider.name if slot else None,
                    model=slot.main_model if slot else None,
                    external=bool(slot and slot.provider.is_external),
                )
            )
        return rows

    async def tools_supported(self, classification: Classification, org_id: uuid.UUID) -> bool:
        slot = self._slot_for(classification, await self.org_ceiling(org_id))
        return bool(slot and slot.provider.supports_tools)

    def new_session(self) -> LLMSession:
        return LLMSession()

    def circuit_states(self) -> dict[str, str]:
        """``{"<role>:<provider>": "closed"|"open"|"half_open"}`` for health checks."""
        return {
            f"{role}:{self._slots[role].provider.name}": breaker.state.value
            for role, breaker in self._breakers.items()
        }

    # ------------------------------------------------------------------ #
    # Completion
    # ------------------------------------------------------------------ #
    async def complete(
        self,
        request: LLMRequest,
        *,
        org_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
        session: LLMSession | None = None,
    ) -> LLMResult:
        self._check_shape(request)
        slot = self._slot_for(request.data_classification, await self.org_ceiling(org_id))
        if slot is None:
            metrics.LLM_REQUESTS.labels(
                provider="none", model="none", task=request.task.value, status="policy_denied"
            ).inc()
            raise LLMPolicyDenied(
                internal_detail=f"no provider may process {request.data_classification.value} data"
            )
        if session is not None:
            if session.slot_role is None:
                session.slot_role = slot.role
            elif session.slot_role != slot.role:
                raise LLMPolicyDenied(
                    internal_detail="session would switch providers mid-conversation"
                )
        if request.tools and not slot.provider.supports_tools:
            raise LLMError(internal_detail=f"{slot.provider.name} does not support tools")
        model = slot.model_for(request)
        cfg = self.settings.llm
        await self._limiter.enforce("llm_org", str(org_id), self.settings.rate_limit.llm_per_org)
        await self.budget.check(org_id)

        breaker = self._breakers[slot.role]
        if not breaker.allow():
            self._count(slot, model, request, "circuit_open")
            raise LLMUnavailable(internal_detail=f"circuit open for {slot.provider.name}")

        pseudonymizer: Pseudonymizer | None = None
        outbound = request
        if slot.provider.is_external and cfg.pseudonymize_pii_for_external and request.pseudonymize:
            pseudonymizer = session.pseudonymizer if session is not None else Pseudonymizer()
            outbound = _pseudonymize_request(request, pseudonymizer)
        max_tokens = min(request.max_output_tokens or cfg.max_output_tokens, cfg.max_output_tokens)
        outbound = replace(outbound, max_output_tokens=max_tokens)

        estimated = _estimate_input(outbound)
        limit = cfg.max_context_tokens + REQUEST_OVERHEAD_TOKENS
        if estimated > limit:
            breaker.release()
            raise LLMError(
                "The request is too large for the AI service.",
                internal_detail=f"estimated {estimated} input tokens > limit {limit}",
            )

        started = time.perf_counter()
        call = _Call(slot, model, request, org_id, user_id, started)
        try:
            result = await slot.provider.complete(outbound, model=model)
            if request.output_schema is not None:
                result = await self._ensure_schema(slot, outbound, result, model, pseudonymizer)
        except LLMUnavailable as exc:
            breaker.record_failure()
            await self._account_failure(call, exc, "unavailable")
            raise
        except LLMRefused as exc:
            breaker.record_success()
            await self._account_failure(call, exc, "refused")
            raise
        except LLMOutputInvalid as exc:
            breaker.record_success()
            await self._account_failure(call, exc, "invalid_output")
            raise
        except LLMError as exc:
            breaker.release()
            await self._account_failure(call, exc, "error")
            raise
        except BaseException:
            breaker.release()
            raise
        breaker.record_success()

        if pseudonymizer is not None:
            result = _restore_result(result, pseudonymizer)
        latency_ms = int((time.perf_counter() - started) * 1000)
        result = replace(result, latency_ms=latency_ms)
        await self._account_success(slot, request, result, org_id, user_id)
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_shape(request: LLMRequest) -> None:
        if not request.messages:
            raise LLMError(internal_detail="request has no messages")
        if len(request.messages) > MAX_MESSAGES:
            raise LLMError(internal_detail="too many messages")
        if request.messages[0].role != "user":
            raise LLMError(internal_detail="the first message must be a user turn")
        if request.messages[-1].role != "user":
            raise LLMError(internal_detail="assistant prefill is not supported")
        if request.output_schema is not None:
            try:
                jsonschema.check_strict(request.output_schema)
            except jsonschema.SchemaDefinitionError as exc:
                raise LLMError(internal_detail=f"output schema rejected: {exc}") from exc
        for tool in request.tools or []:
            try:
                jsonschema.check_strict(tool.input_schema)
            except jsonschema.SchemaDefinitionError as exc:
                raise LLMError(internal_detail=f"tool {tool.name} schema rejected: {exc}") from exc

    async def _ensure_schema(
        self,
        slot: ProviderSlot,
        outbound: LLMRequest,
        result: LLMResult,
        model: str,
        pseudonymizer: Pseudonymizer | None,
    ) -> LLMResult:
        schema = outbound.output_schema or {}
        errors = _schema_errors(result, schema, pseudonymizer)
        if not errors:
            return result
        log.info("llm_schema_repair", provider=slot.provider.name, model=model, errors=len(errors))
        repair = replace(
            outbound,
            messages=[
                *outbound.messages,
                ChatMessage(role="assistant", content=result.text or "{}"),
                ChatMessage(
                    role="user",
                    content=(
                        "Your previous reply did not match the required JSON schema. Problems: "
                        + "; ".join(errors[:10])
                        + ". Reply again with only a JSON object that satisfies the schema."
                    ),
                ),
            ],
        )
        second = await slot.provider.complete(repair, model=model)
        second = replace(
            second,
            input_tokens=second.input_tokens + result.input_tokens,
            output_tokens=second.output_tokens + result.output_tokens,
        )
        if _schema_errors(second, schema, pseudonymizer):
            raise LLMOutputInvalid(
                internal_detail="output failed schema validation after repair"
            ).with_usage(second.input_tokens, second.output_tokens, second.model)
        return second

    def _count(self, slot: ProviderSlot, model: str, request: LLMRequest, status: str) -> None:
        metrics.LLM_REQUESTS.labels(
            provider=slot.provider.name, model=model, task=request.task.value, status=status
        ).inc()

    async def _account_success(
        self,
        slot: ProviderSlot,
        request: LLMRequest,
        result: LLMResult,
        org_id: uuid.UUID,
        user_id: uuid.UUID | None,
    ) -> None:
        cost = estimate_cost(
            self.settings.llm.pricing,
            result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_read_input_tokens=result.cache_read_input_tokens,
            cache_creation_input_tokens=result.cache_creation_input_tokens,
        )
        provider = slot.provider.name
        self._count(slot, result.model, request, "ok")
        metrics.LLM_TOKENS.labels(provider=provider, model=result.model, direction="input").inc(
            result.input_tokens
            + result.cache_read_input_tokens
            + result.cache_creation_input_tokens
        )
        metrics.LLM_TOKENS.labels(provider=provider, model=result.model, direction="output").inc(
            result.output_tokens
        )
        metrics.LLM_COST.labels(provider=provider, model=result.model).inc(float(cost))
        metrics.LLM_LATENCY.labels(provider=provider, task=request.task.value).observe(
            result.latency_ms / 1000
        )
        log.info(
            "llm_call",
            provider=provider,
            model=result.model,
            task=request.task.value,
            status="ok",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            pseudonymized=result.pseudonymized,
        )
        await self.usage.record(
            org_id=org_id,
            user_id=user_id,
            task=request.task.value,
            provider=provider,
            model=result.model,
            input_tokens=result.input_tokens
            + result.cache_read_input_tokens
            + result.cache_creation_input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=float(cost),
            latency_ms=result.latency_ms,
            status="ok",
        )

    async def _account_failure(self, call: _Call, exc: LLMError, status: str) -> None:
        slot, model, request = call.slot, call.model, call.request
        org_id, user_id, started = call.org_id, call.user_id, call.started
        latency_ms = int((time.perf_counter() - started) * 1000)
        served = exc.model or model
        self._count(slot, served, request, status)
        cost = estimate_cost(
            self.settings.llm.pricing,
            served,
            input_tokens=exc.input_tokens,
            output_tokens=exc.output_tokens,
        )
        log.warning(
            "llm_call",
            provider=slot.provider.name,
            model=served,
            task=request.task.value,
            status=status,
            error=exc.code,
            detail=exc.internal_detail,
            latency_ms=latency_ms,
        )
        await self.usage.record(
            org_id=org_id,
            user_id=user_id,
            task=request.task.value,
            provider=slot.provider.name,
            model=served,
            input_tokens=exc.input_tokens,
            output_tokens=exc.output_tokens,
            cost_usd=float(cost),
            latency_ms=latency_ms,
            status=status,
        )

    async def aclose(self) -> None:
        for slot in self._slots.values():
            await slot.provider.aclose()


# --------------------------------------------------------------------------- #
# Pseudonymisation helpers
# --------------------------------------------------------------------------- #
def _pseudonymize_block(block: dict[str, Any], pz: Pseudonymizer) -> dict[str, Any]:
    kind = block.get("type")
    if kind == "text" and isinstance(block.get("text"), str):
        return {**block, "text": pz.pseudonymize(block["text"])}
    if kind == "tool_result":
        content = block.get("content")
        if isinstance(content, str):
            return {**block, "content": pz.pseudonymize(content)}
        if isinstance(content, list):
            return {
                **block,
                "content": [_pseudonymize_block(b, pz) for b in content if isinstance(b, dict)],
            }
    return dict(block)


def _pseudonymize_message(message: ChatMessage, pz: Pseudonymizer) -> ChatMessage:
    if isinstance(message.content, str):
        return ChatMessage(role=message.role, content=pz.pseudonymize(message.content))
    if message.role == "assistant":
        # Provider-originated raw content (already in the provider's pseudonymised domain):
        # forwarded byte-for-byte so thinking blocks and cache prefixes stay valid.
        return message
    return ChatMessage(
        role=message.role, content=[_pseudonymize_block(b, pz) for b in message.content]
    )


def _pseudonymize_request(request: LLMRequest, pz: Pseudonymizer) -> LLMRequest:
    return replace(
        request,
        system=pz.pseudonymize(request.system),
        messages=[_pseudonymize_message(m, pz) for m in request.messages],
    )


def _restore_value(value: Any, pz: Pseudonymizer) -> Any:
    if isinstance(value, str):
        return pz.restore(value)
    if isinstance(value, list):
        return [_restore_value(v, pz) for v in value]
    if isinstance(value, dict):
        return {k: _restore_value(v, pz) for k, v in value.items()}
    return value


def _restore_result(result: LLMResult, pz: Pseudonymizer) -> LLMResult:
    return replace(
        result,
        text=pz.restore(result.text),
        data=_restore_value(result.data, pz) if result.data is not None else None,
        tool_calls=[
            ToolCall(id=c.id, name=c.name, arguments=_restore_value(c.arguments, pz))
            for c in result.tool_calls
        ],
        pseudonymized=pz.replacements,
    )


def _schema_errors(
    result: LLMResult, schema: dict[str, Any], pseudonymizer: Pseudonymizer | None
) -> list[str]:
    if result.data is None:
        return ["the reply is not a JSON object"]
    data = _restore_value(result.data, pseudonymizer) if pseudonymizer is not None else result.data
    return jsonschema.validate(data, schema)


def _estimate_input(request: LLMRequest) -> int:
    total = estimate_tokens(request.system)
    for message in request.messages:
        if isinstance(message.content, str):
            total += estimate_tokens(message.content)
        else:
            for block in message.content:
                total += estimate_tokens(
                    str(block.get("text") or block.get("content") or block.get("input") or "")
                )
    if request.output_schema is not None:
        total += estimate_tokens(str(request.output_schema))
    for tool in request.tools or []:
        total += estimate_tokens(tool.description) + estimate_tokens(str(tool.input_schema))
    return total
