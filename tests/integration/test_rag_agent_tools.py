"""The real read-only agent tools against PostgreSQL, driven by a scripted malicious model."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import select

from docassist.core.context import utcnow
from docassist.db.models import AuditEvent
from docassist.db.session import DbContext
from docassist.llm.base import LLMRequest, ToolCall
from docassist.rag.agent import AgentService
from docassist.rag.prompts import AGENT_ANSWER_TOOL
from tests.helpers_rag import (
    ChunkSpec,
    FieldSpec,
    KeywordRetriever,
    ScriptedProvider,
    make_gateway,
    result,
    seed_document,
)

pytestmark = [pytest.mark.db]

EXPIRY = "This Agreement expires on 31 December 2027 unless renewed in writing by both parties."
SALARY = "Senior engineer salary band: 95,000 to 120,000 EUR."


@pytest.fixture
async def world(container: Any, factory: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(container, "search", KeywordRetriever(container), raising=False)
    org = await factory.org()
    other = await factory.org()
    hr = await factory.department(org)
    manager = await factory.user(org, "department_manager", managed=[hr], clearance="RESTRICTED")
    colleague = await factory.user(org, "department_manager", clearance="RESTRICTED")
    outsider = await factory.user(other, "employee")
    contract = await seed_document(
        container, org_id=org, owner_id=manager.id, title="Acme Supply Agreement", doc_type="contract",
        chunks=[ChunkSpec(EXPIRY, page=9, section="Term")],
        fields=[FieldSpec("expiration_date", value_date=(utcnow() + timedelta(days=30)).date(), evidence=EXPIRY, page=9)],
    )  # fmt: skip
    own_restricted = await seed_document(
        container, org_id=org, owner_id=manager.id, title="Salary Bands", classification="RESTRICTED",
        department_id=hr, chunks=[ChunkSpec(SALARY)],
    )  # fmt: skip
    colleagues_restricted = await seed_document(
        container, org_id=org, owner_id=colleague.id, title="Board Minutes", classification="RESTRICTED",
        chunks=["The board approved the acquisition of Initech for 40 million EUR."],
    )  # fmt: skip
    foreign = await seed_document(
        container, org_id=other, owner_id=outsider.id, title="Globex Secrets", classification="INTERNAL",
        chunks=["Globex merger codename BLUEBIRD."],
    )  # fmt: skip
    return {
        "manager": await factory.principal(manager),
        "contract": contract,
        "own_restricted": own_restricted,
        "colleagues_restricted": colleagues_restricted,
        "foreign": foreign,
        "org": org,
    }


def _call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"call_{name}_{len(json.dumps(arguments))}", name=name, arguments=arguments)


def _results(request: LLMRequest) -> list[dict[str, Any]]:
    content = request.messages[-1].content
    assert isinstance(content, list)
    return content


async def test_tools_are_principal_scoped_and_governed(container: Any, world: Any) -> None:
    excerpt = {"max_chars": 2000, "page": None}
    turn_one = result(
        tool_calls=[
            _call("get_document_excerpt", document_id=str(world["foreign"].id), **excerpt),
            _call(
                "get_document_excerpt",
                document_id=str(world["colleagues_restricted"].id),
                **excerpt,
            ),
            _call("get_document_excerpt", document_id=str(world["own_restricted"].id), **excerpt),
            _call("get_document_excerpt", document_id=str(world["contract"].id), **excerpt),
            _call(
                "search_documents", query="salary band agreement expires", doc_types=None, limit=10
            ),
            _call("find_deadlines", within_days=90, doc_type="contract"),
            _call("list_extracted_fields", document_id=str(world["contract"].id)),
            _call("get_document_metadata", document_id=str(world["contract"].id)),
        ]
    )
    final = result(
        tool_calls=[
            ToolCall(
                id="final",
                name=AGENT_ANSWER_TOOL,
                arguments={
                    "status": "answered",
                    "answer": "The Acme agreement expires on 31 December 2027.",
                    "citations": [
                        {"document_id": str(world["contract"].id), "quote": EXPIRY},
                        {
                            "document_id": str(world["foreign"].id),
                            "quote": "Globex merger codename BLUEBIRD.",
                        },
                    ],
                },
            )
        ]
    )
    provider = ScriptedProvider([turn_one, final], name="anthropic", is_external=True)
    gateway = make_gateway(container.settings, provider)
    deps = _Deps(container, gateway)
    service = AgentService(deps)  # type: ignore[arg-type]
    out = await service.run(world["manager"], task="When does the Acme agreement expire?")

    assert [s.detail for s in out.steps] == [
        "not_found",
        "not_found",
        "not_found",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
    ]
    results = _results(provider.requests[1])
    # foreign-tenant, unreadable and above-ceiling documents are indistinguishable
    assert results[0]["content"] == results[1]["content"] == results[2]["content"]
    assert "not found or not accessible" in results[0]["content"]
    seen = provider.seen_text()
    for secret in ("BLUEBIRD", "Initech", "95,000"):
        assert secret not in seen
    search_payload = results[4]["content"]
    assert "Acme Supply Agreement" in search_payload and "Salary Bands" not in search_payload
    assert "2027" in results[3]["content"] and "expiration_date" in results[6]["content"]
    assert "contract" in results[7]["content"]
    assert out.status == "answered"
    assert [c.document_id for c in out.citations] == [world["contract"].id]
    assert any("above CONFIDENTIAL" in w for w in out.warnings)  # the agent's governance ceiling

    async with container.db.session(DbContext(org_id=world["org"])) as session:
        events = (
            (
                await session.execute(
                    select(AuditEvent).where(
                        AuditEvent.organization_id == world["org"],
                        AuditEvent.action.in_(["assistant.tool_call", "assistant.agent_run"]),
                    )
                )
            )
            .scalars()
            .all()
        )
    tool_events = [e for e in events if e.action == "assistant.tool_call"]
    assert len(tool_events) == 8
    assert {e.details["outcome"] for e in tool_events} == {"ok", "not_found"}
    assert not any("2027" in json.dumps(e.details) for e in events)


class _Deps:
    """The container with a swapped gateway (other attributes pass through)."""

    def __init__(self, container: Any, llm: Any) -> None:
        self._container = container
        self.llm = llm

    def __getattr__(self, name: str) -> Any:
        return getattr(self._container, name)
