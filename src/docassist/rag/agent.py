"""Bounded, read-only tool-use agent.

``AgentService.run(principal, task=...)`` (the router checks ``assistant:agent``):

* refuses deterministically when the task itself is an abuse attempt (same analysis as
  questions);
* fixes a **data-governance ceiling** before the first call: the highest classification,
  at most the caller's clearance, whose route supports tool use. Tools never return
  content above it, and every model request is sent with that classification, so the
  gateway's routing covers everything the model can ever see. No tool-capable route at
  all -> :class:`FeatureDisabled` (e.g. only the offline extractive provider is configured);
* loops at most ``llm.agent_max_iterations`` model turns and ``llm.agent_max_tool_calls``
  tool executions; each executed tool call is bounded by ``llm.tool_timeout_seconds`` and
  counted against ``rate_limit.tool_calls_per_user``, and every call - rejected ones
  included - is audited as ``assistant.tool_call`` (tool name, outcome, argument size -
  never arguments or results);
* unknown tools, oversized or invalid arguments, permission failures, missing/unreadable
  documents and timeouts come back to the model as error results - they never raise into
  the loop and never reveal whether a foreign document exists;
* tool output is truncated and wrapped as ``<tool_output nonce=...>`` untrusted data with
  angle brackets escaped;
* the final answer (tool ``submit_answer``) is validated like a RAG answer: each citation
  must name a document a tool returned in this run and quote text a tool returned for it;
  the output guard runs on the answer; an "answered" result without a verified citation is
  downgraded to ``insufficient_context``.

The whole run shares one :class:`docassist.llm.gateway.LLMSession`, so pseudonymised values
stay stable across turns and the conversation stays on one provider.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from docassist.audit.service import Actor
from docassist.authz.permissions import Permission
from docassist.authz.principal import Principal
from docassist.core.context import utcnow
from docassist.core.enums import AuditOutcome, Classification
from docassist.core.errors import FeatureDisabled, RateLimited, ValidationFailed
from docassist.core.logging import get_logger
from docassist.core.text import clean_line_text, sanitize_text
from docassist.llm.base import (
    ChatMessage,
    LLMOutputInvalid,
    LLMRefused,
    LLMRequest,
    LLMResult,
    LLMTask,
    ToolCall,
    ToolSpec,
)
from docassist.rag.citations import validate_citations
from docassist.rag.context import escape_untrusted, new_nonce
from docassist.rag.guard import REFUSAL_TEXT, OutputGuard
from docassist.rag.ports import RagDependencies
from docassist.rag.prompts import (
    AGENT_ANSWER_SCHEMA,
    AGENT_ANSWER_TOOL,
    agent_system_prompt,
    canary_token,
)
from docassist.rag.query import analyze_query
from docassist.rag.service import REFUSALS, UNVERIFIED_TEXT
from docassist.rag.tools import (
    MAX_ARGUMENT_BYTES,
    SUBMIT_ANSWER_SPEC_DESCRIPTION,
    TOOLS,
    SubmitAnswerInput,
    ToolContext,
    ToolDefinition,
    ToolFailure,
    specs,
)
from docassist.rag.types import AgentCitation, AgentResult, AgentStep, AnswerStatus, Usage

log = get_logger(__name__)

MAX_TOOL_OUTPUT_CHARS = 6_000
WARN_STEP_LIMIT = "The agent stopped after reaching its step limit."
WARN_CEILING = (
    "Documents classified above {c} are not available to the agent under the "
    "data-governance policy."
)
WARN_TOOL_BUDGET = "The agent used its whole tool-call budget."
WARN_GUARD = "Parts of the response were removed by the output safety filter."
INCOMPLETE_TEXT = "The agent could not complete the task."


@dataclass(slots=True)
class _Run:
    """Accumulated state of one agent run."""

    started: float
    warnings: list[str]
    steps: list[AgentStep] = field(default_factory=list)
    model: str | None = None
    provider: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def finish(
        self, status: AnswerStatus, answer: str, citations: list[AgentCitation] | None = None
    ) -> AgentResult:
        return AgentResult(
            status=status,
            answer=answer,
            citations=citations or [],
            steps=self.steps,
            warnings=list(dict.fromkeys(self.warnings)),
            model=self.model,
            provider=self.provider,
            usage=Usage(self.input_tokens, self.output_tokens),
            latency_ms=int((time.perf_counter() - self.started) * 1000),
            iterations=self.iterations,
        )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class AgentService:
    def __init__(
        self,
        deps: RagDependencies,
        *,
        tools: tuple[ToolDefinition, ...] = TOOLS,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._deps = deps
        self._tools = {tool.name: tool for tool in tools}
        self._clock = clock

    @property
    def canary(self) -> str:
        return canary_token(self._deps.settings.security.token_pepper.get_secret_value())

    async def ceiling_for(self, principal: Principal) -> Classification | None:
        """Highest classification <= clearance whose route supports tools (None: no agent)."""
        org_id = principal.require_org()
        for classification in sorted(
            Classification.at_most(principal.clearance), key=lambda c: c.rank, reverse=True
        ):
            if await self._deps.llm.tools_supported(classification, org_id):
                return classification
        return None

    async def run(
        self, principal: Principal, *, task: str, max_iterations: int | None = None
    ) -> AgentResult:
        started = time.perf_counter()
        principal.require_org()
        principal.require(Permission.ASSISTANT_AGENT)
        settings = self._deps.settings
        text = " ".join(sanitize_text(task)[0].split())
        if not text:
            raise ValidationFailed("The task is empty.")
        if len(text) > settings.llm.max_question_chars:
            raise ValidationFailed(
                f"The task is too long (at most {settings.llm.max_question_chars} characters)."
            )
        ceiling = await self.ceiling_for(principal)
        if ceiling is None:
            raise FeatureDisabled(
                "The agent needs an AI model with tool support; none is configured.",
                internal_detail="no tool-capable route",
            )
        await self._deps.limiter.enforce(
            "llm_user", str(principal.user_id), settings.rate_limit.llm_per_user
        )
        now = self._clock()
        analysis = analyze_query(
            text,
            today=now.date(),
            warn_threshold=settings.retrieval.injection_warn_threshold,
            block_threshold=settings.retrieval.injection_exclude_threshold,
        )
        warnings: list[str] = []
        if ceiling.rank < principal.clearance.rank:
            warnings.append(WARN_CEILING.format(c=ceiling.value))
        if analysis.blocked_reason is not None:
            result = AgentResult(
                status="refused",
                answer=REFUSALS.get(analysis.blocked_reason, REFUSAL_TEXT),
                citations=[],
                steps=[],
                warnings=warnings,
                model=None,
                provider=None,
                usage=Usage(),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            await self._audit_run(principal, text, result, analysis.flags)
            return result

        ctx = ToolContext(principal=principal, deps=self._deps, now=now, ceiling=ceiling)
        result = await self._loop(
            ctx,
            text=text,
            max_iterations=min(
                max_iterations or settings.llm.agent_max_iterations,
                settings.llm.agent_max_iterations,
            ),
            warnings=warnings,
            started=started,
        )
        await self._audit_run(principal, text, result, analysis.flags)
        return result

    # ------------------------------------------------------------------ #
    def _tool_specs(self) -> list[ToolSpec]:
        return [
            *specs(list(self._tools.values())),
            ToolSpec(
                name=AGENT_ANSWER_TOOL,
                description=SUBMIT_ANSWER_SPEC_DESCRIPTION,
                input_schema=AGENT_ANSWER_SCHEMA,
            ),
        ]

    async def _loop(
        self,
        ctx: ToolContext,
        *,
        text: str,
        max_iterations: int,
        warnings: list[str],
        started: float,
    ) -> AgentResult:
        settings = self._deps.settings
        llm = self._deps.llm
        org_id = ctx.principal.require_org()
        session = llm.new_session()
        nonce = new_nonce()
        system = agent_system_prompt(self.canary, max_tool_calls=settings.llm.agent_max_tool_calls)
        tool_specs = self._tool_specs()
        task_block = f'<task nonce="{nonce}">\n{escape_untrusted(text)}\n</task>'
        messages: list[ChatMessage] = [ChatMessage(role="user", content=task_block)]
        run = _Run(started=started, warnings=warnings)
        final: ToolCall | None = None
        final_text = ""
        tool_calls = 0
        stop = False
        for iteration in range(1, max_iterations + 1):
            run.iterations = iteration
            request = LLMRequest(
                task=LLMTask.AGENT,
                system=system,
                messages=messages,
                tools=tool_specs,
                data_classification=ctx.ceiling,
            )
            try:
                result: LLMResult = await llm.complete(
                    request, org_id=org_id, user_id=ctx.principal.user_id, session=session
                )
            except LLMRefused as exc:
                run.model = exc.model or run.model
                run.add_usage(exc.input_tokens, exc.output_tokens)
                return run.finish("refused", "The AI model declined this task.")
            except LLMOutputInvalid as exc:
                run.add_usage(exc.input_tokens, exc.output_tokens)
                run.warnings.append("The AI response could not be validated.")
                break
            run.add_usage(
                result.input_tokens
                + result.cache_read_input_tokens
                + result.cache_creation_input_tokens,
                result.output_tokens,
            )
            run.model, run.provider = result.model, result.provider
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=result.raw_content or [{"type": "text", "text": result.text}],
                )
            )
            if not result.tool_calls:
                final_text = result.text
                break
            results: list[dict[str, Any]] = []
            for call in result.tool_calls:
                if call.name == AGENT_ANSWER_TOOL and final is None:
                    final = call
                    results.append(_tool_result(call.id, "Answer received.", nonce, call.name))
                    continue
                if tool_calls >= settings.llm.agent_max_tool_calls:
                    if WARN_TOOL_BUDGET not in run.warnings:
                        run.warnings.append(WARN_TOOL_BUDGET)
                    message = f"Tool-call budget exhausted. Call {AGENT_ANSWER_TOOL} now."
                    results.append(_tool_result(call.id, message, nonce, call.name, error=True))
                    continue
                tool_calls += 1
                step, payload, is_error = await self._execute(ctx, call)
                run.steps.append(step)
                stop = stop or step.detail == "rate_limited"
                results.append(_tool_result(call.id, payload, nonce, call.name, error=is_error))
            if final is not None or stop:
                break
            messages.append(ChatMessage(role="user", content=results))
        else:
            run.warnings.append(WARN_STEP_LIMIT)

        if stop and final is None:
            run.warnings.append("The tool-call rate limit was reached.")
        if final is None:
            if final_text:
                return run.finish("insufficient_context", UNVERIFIED_TEXT)
            return run.finish("insufficient_context", INCOMPLETE_TEXT)
        return self._final_answer(ctx, final, run)

    async def _execute(self, ctx: ToolContext, call: ToolCall) -> tuple[AgentStep, str, bool]:
        """Run one tool call; returns (step, payload for the model, is_error)."""
        began = time.perf_counter()

        def step(outcome: str, ok: bool) -> AgentStep:
            return AgentStep(
                tool=clean_line_text(call.name, 64),
                ok=ok,
                detail=outcome,
                duration_ms=int((time.perf_counter() - began) * 1000),
            )

        principal = ctx.principal
        tool = self._tools.get(call.name)
        arg_bytes = len(json.dumps(call.arguments, default=str))
        if tool is None:
            outcome, payload = "unknown_tool", "Unknown tool. Use only the tools provided."
        elif arg_bytes > MAX_ARGUMENT_BYTES:
            outcome, payload = "arguments_too_large", "Arguments are too large."
        elif not principal.has(tool.permission):
            outcome, payload = "denied", "You are not permitted to use this tool."
        else:
            try:
                await self._deps.limiter.enforce(
                    "tool_calls_user",
                    str(principal.user_id),
                    self._deps.settings.rate_limit.tool_calls_per_user,
                )
            except RateLimited:
                result = step("rate_limited", False)
                await self._audit_tool(principal, call.name, "rate_limited", arg_bytes, 0)
                return result, "Rate limit reached. Stop and submit your answer.", True
            try:
                arguments = tool.input_model.model_validate(call.arguments)
            except ValidationError as exc:
                outcome = "invalid_arguments"
                problems = "; ".join(
                    f"{'.'.join(str(p) for p in err['loc'])}: {err['type']}"
                    for err in exc.errors()[:5]
                )
                payload = f"Invalid arguments ({problems})."
            else:
                before = set(ctx.seen)
                try:
                    async with asyncio.timeout(self._deps.settings.llm.tool_timeout_seconds):
                        data = await tool.handler(ctx, arguments)
                except ToolFailure as exc:
                    outcome, payload = "not_found", str(exc)
                except TimeoutError:
                    outcome, payload = "timeout", "The tool timed out."
                except (ValidationFailed, ValueError):
                    outcome, payload = "invalid_arguments", "Invalid arguments."
                else:
                    text = json.dumps(data, ensure_ascii=False, default=str)
                    if len(text) > MAX_TOOL_OUTPUT_CHARS:
                        text = text[:MAX_TOOL_OUTPUT_CHARS] + " ...[truncated]"
                    result = step("ok", True)
                    disclosed = sorted(set(ctx.seen) - before) or sorted(
                        doc_id for doc_id in ctx.seen if doc_id in text
                    )
                    await self._audit_tool(
                        principal, call.name, "ok", arg_bytes, len(text), documents=disclosed
                    )
                    return result, text, False
        result = step(outcome, False)
        await self._audit_tool(principal, call.name, outcome, arg_bytes, 0)
        return result, payload, True

    def _final_answer(self, ctx: ToolContext, call: ToolCall, run: _Run) -> AgentResult:
        try:
            submitted = SubmitAnswerInput.model_validate(call.arguments)
        except ValidationError:
            run.warnings.append("The final answer of the agent was malformed.")
            return run.finish("insufficient_context", INCOMPLETE_TEXT)
        guard = OutputGuard(self.canary)
        sources = [doc.joined for doc in ctx.seen.values()]
        checked = guard.check(submitted.answer, sources=sources)
        if checked.blocked:
            run.warnings.append(WARN_GUARD)
            return run.finish("refused", REFUSAL_TEXT)
        if checked.actions:
            run.warnings.append(WARN_GUARD)
        report = validate_citations(
            [c.model_dump() for c in submitted.citations],
            ctx.seen,
            key_field="document_id",
            text_of=lambda doc: doc.joined,
        )
        if report.invalid:
            run.warnings.append(
                f"{report.invalid} citation(s) could not be verified and were removed."
            )
        citations = [
            AgentCitation(
                n=n,
                document_id=uuid.UUID(valid.key),
                document_title=valid.target.title,
                quote=guard.check(valid.quote, sources=sources, max_chars=500).text,
            )
            for n, valid in enumerate(report.valid, start=1)
        ]
        status: AnswerStatus = submitted.status
        answer = checked.text
        if status == "answered" and not citations:
            status, answer = "insufficient_context", UNVERIFIED_TEXT
        return run.finish(status, answer, citations)

    async def _audit_tool(
        self,
        principal: Principal,
        tool: str,
        outcome: str,
        arg_bytes: int,
        result_chars: int,
        *,
        documents: list[str] | None = None,
    ) -> None:
        """Audit one tool call, including which documents' content it disclosed."""
        details: dict[str, object] = {
            "outcome": outcome,
            "argument_bytes": arg_bytes,
            "result_chars": result_chars,
        }
        if documents:
            details["document_ids"] = documents[:50]
        await self._deps.audit.record_detached(
            Actor.of(principal),
            "assistant.tool_call",
            outcome=AuditOutcome.SUCCESS if outcome == "ok" else AuditOutcome.FAILURE,
            resource_type="tool",
            resource_id=clean_line_text(tool, 64),
            details=details,
        )

    async def _audit_run(
        self, principal: Principal, task: str, result: AgentResult, flags: tuple[str, ...]
    ) -> None:
        await self._deps.audit.record_detached(
            Actor.of(principal),
            "assistant.agent_run",
            resource_type="agent",
            details={
                "task_chars": len(task),
                "task_sha256": _sha(task)[:16],
                "status": result.status,
                "iterations": result.iterations,
                "tool_calls": len(result.steps),
                "failed_tool_calls": sum(1 for s in result.steps if not s.ok),
                "model": result.model,
                "provider": result.provider,
                "flags": list(flags)[:10],
                "cited_document_ids": sorted({str(c.document_id) for c in result.citations})[:10],
            },
        )


def _tool_result(
    tool_use_id: str, payload: str, nonce: str, tool: str, *, error: bool = False
) -> dict[str, Any]:
    name = escape_untrusted(clean_line_text(tool, 64)).replace('"', "")
    wrapped = (
        f'<tool_output nonce="{nonce}" tool="{name}">\n{escape_untrusted(payload)}\n</tool_output>'
    )
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": wrapped}
    if error:
        block["is_error"] = True
    return block


__all__ = ["AgentService"]
