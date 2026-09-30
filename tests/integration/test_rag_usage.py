"""DB-backed usage accounting, monthly token budget and organisation AI policy."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import select, update

from docassist.audit.usage import DbBudgetGuard, DbOrgLlmPolicy, DbUsageRecorder, month_start
from docassist.core.enums import Classification
from docassist.core.errors import QuotaExceeded
from docassist.db.models import LlmUsage, Organization
from docassist.db.session import DbContext
from tests.conftest import login, make_settings
from tests.helpers_rag import ChunkSpec, KeywordRetriever, seed_document

pytestmark = [pytest.mark.db]


async def _set_org_llm(container: Any, org_id: uuid.UUID, llm: dict[str, Any]) -> None:
    async with container.db.transaction(DbContext(org_id=org_id)) as session:
        await session.execute(
            update(Organization).where(Organization.id == org_id).values(settings={"llm": llm})
        )


async def test_usage_rows_are_written_under_the_org_context(container: Any, factory: Any) -> None:
    org = await factory.org()
    user = await factory.user(org)
    recorder = DbUsageRecorder(container.db)
    await recorder.record(
        org_id=org, user_id=user.id, task="answer", provider="anthropic", model="claude-opus-5-5",
        input_tokens=1200, output_tokens=300, cost_usd=0.0108, latency_ms=850, status="ok",
    )  # fmt: skip
    async with container.db.session(DbContext(org_id=org)) as session:
        rows = (
            (await session.execute(select(LlmUsage).where(LlmUsage.organization_id == org)))
            .scalars()
            .all()
        )
    assert len(rows) == 1
    row = rows[0]
    assert (row.task, row.provider, row.model, row.input_tokens, row.output_tokens) == (
        "answer", "anthropic", "claude-opus-5-5", 1200, 300,
    )  # fmt: skip
    assert row.cost_usd == Decimal("0.010800") and row.user_id == user.id
    other = await factory.org()
    async with container.db.session(DbContext(org_id=other)) as session:
        assert (
            await session.execute(select(LlmUsage).where(LlmUsage.organization_id == org))
        ).all() == []


async def test_usage_write_failures_never_raise(container: Any) -> None:
    recorder = DbUsageRecorder(container.db)
    await recorder.record(  # unknown organisation: FK violation is logged and swallowed
        org_id=uuid.uuid4(), user_id=None, task="answer", provider="x", model="y",
        input_tokens=1, output_tokens=1, cost_usd=0.0, latency_ms=1, status="ok",
    )  # fmt: skip


async def test_budget_guard_with_org_override(container: Any, factory: Any) -> None:
    org = await factory.org()
    recorder = DbUsageRecorder(container.db)
    for _ in range(2):
        await recorder.record(
            org_id=org, user_id=None, task="answer", provider="p", model="m",
            input_tokens=400, output_tokens=100, cost_usd=0.0, latency_ms=1, status="ok",
        )  # fmt: skip
    guard = DbBudgetGuard(
        container.db, make_settings(llm={"monthly_token_budget_per_org": 5000}), container.cache
    )
    status = await guard.status(org)
    assert (status.limit, status.used, status.remaining) == (5000, 1000, 4000)
    await guard.check(org)

    await _set_org_llm(container, org, {"monthly_token_budget": 800})
    fresh = DbBudgetGuard(
        container.db, make_settings(llm={"monthly_token_budget_per_org": 5000}), container.cache
    )
    next_month = datetime(2099, 1, 15, tzinfo=UTC)  # different cache bucket
    assert (await fresh.status(org, now=next_month)).used == 0
    await container.cache.delete(
        container.cache.key("llmbudget", str(org), month_start(datetime.now(UTC)).strftime("%Y-%m"))
    )
    with pytest.raises(QuotaExceeded) as info:
        await fresh.check(org)
    assert info.value.status_code == 429


async def test_org_override_cannot_raise_the_budget_and_zero_means_unlimited(
    container: Any, factory: Any
) -> None:
    org = await factory.org()
    await _set_org_llm(container, org, {"monthly_token_budget": 10**9})
    guard = DbBudgetGuard(
        container.db, make_settings(llm={"monthly_token_budget_per_org": 1000}), container.cache
    )
    assert (await guard.status(org)).limit == 1000
    unlimited = DbBudgetGuard(
        container.db, make_settings(llm={"monthly_token_budget_per_org": 0}), container.cache
    )
    other = await factory.org()
    assert (await unlimited.status(other)).limit is None
    await unlimited.check(other)


async def test_org_policy_ceiling(container: Any, factory: Any) -> None:
    org = await factory.org()
    policy = DbOrgLlmPolicy(container.db, container.cache)
    assert await policy.external_ceiling(org) is None
    other = await factory.org()
    await _set_org_llm(container, other, {"external_max_classification": "PUBLIC"})
    assert await policy.external_ceiling(other) is Classification.PUBLIC
    bogus = await factory.org()
    await _set_org_llm(container, bogus, {"external_max_classification": "TOP_SECRET"})
    assert await policy.external_ceiling(bogus) is None


async def test_budget_exhaustion_through_the_api_is_429(
    client: Any, container: Any, factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(container, "search", KeywordRetriever(container), raising=False)
    org = await factory.org()
    user = await factory.user(org)
    await seed_document(
        container, org_id=org, owner_id=user.id, title="Travel Policy",
        chunks=[ChunkSpec("Employees book economy class for flights under six hours.", section="Flights")],
    )  # fmt: skip
    await _set_org_llm(container, org, {"monthly_token_budget": 10})
    await DbUsageRecorder(container.db).record(
        org_id=org, user_id=user.id, task="answer", provider="p", model="m",
        input_tokens=50, output_tokens=0, cost_usd=0.0, latency_ms=1, status="ok",
    )  # fmt: skip
    headers = await login(client, user)
    response = await client.post(
        "/api/v1/assistant/ask",
        json={"question": "Which class do employees book for flights?"},
        headers=headers,
    )
    assert response.status_code == 429 and response.json()["code"] == "quota_exceeded"


async def test_gateway_records_usage_for_real_calls(
    client: Any, container: Any, factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(container, "search", KeywordRetriever(container), raising=False)
    org = await factory.org()
    user = await factory.user(org)
    await seed_document(
        container, org_id=org, owner_id=user.id, title="Travel Policy",
        chunks=[ChunkSpec("Employees book economy class for flights under six hours.", section="Flights")],
    )  # fmt: skip
    headers = await login(client, user)
    response = await client.post(
        "/api/v1/assistant/ask",
        json={"question": "Which class do employees book for flights?"},
        headers=headers,
    )
    assert response.status_code == 200 and response.json()["status"] == "answered"
    async with container.db.session(DbContext(org_id=org)) as session:
        rows = (
            (await session.execute(select(LlmUsage).where(LlmUsage.organization_id == org)))
            .scalars()
            .all()
        )
    assert len(rows) == 1 and rows[0].task == "answer" and rows[0].provider == "local_extractive"
    assert rows[0].user_id == user.id and rows[0].status == "ok" and rows[0].input_tokens > 0
