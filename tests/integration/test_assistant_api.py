"""/api/v1/assistant through the real HTTP stack (auth, RBAC, problem responses)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import login
from tests.helpers_rag import ChunkSpec, KeywordRetriever, seed_document

pytestmark = [pytest.mark.db]

PAYMENT = (
    "The Customer shall pay each undisputed invoice within thirty (30) days of receipt (net 30)."
)


@pytest.fixture
async def setup(container: Any, factory: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(container, "search", KeywordRetriever(container), raising=False)
    org = await factory.org()
    finance = await factory.department(org)
    employee = await factory.user(org, "employee", departments=[finance])
    colleague = await factory.user(org, "employee", departments=[finance])
    manager = await factory.user(org, "department_manager", managed=[finance])
    auditor = await factory.user(org, "auditor")
    await seed_document(
        container,
        org_id=org,
        owner_id=manager.id,
        title="Acme Supply Agreement",
        doc_type="contract",
        chunks=[ChunkSpec(PAYMENT, page=4, section="Payment Terms")],
    )
    return {"employee": employee, "colleague": colleague, "manager": manager, "auditor": auditor}


async def test_ask_and_conversation_lifecycle(client: Any, setup: Any) -> None:
    headers = await login(client, setup["employee"])
    response = await client.post(
        "/api/v1/assistant/ask",
        json={"question": "What are the payment terms for invoices?"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["status"] == "answered"
    assert body["citations"][0]["quote"] == PAYMENT
    assert body["citations"][0]["document_title"] == "Acme Supply Agreement"
    assert set(body) >= {
        "answer", "confidence", "confidence_label", "citations", "evidence", "warnings", "model",
        "provider", "usage", "conversation_id", "message_id", "latency_ms",
    }  # fmt: skip
    conversation_id = body["conversation_id"]

    follow = await client.post(
        "/api/v1/assistant/ask",
        json={"question": "and late payments?", "conversation_id": conversation_id},
        headers=headers,
    )
    assert follow.status_code == 200 and follow.json()["conversation_id"] == conversation_id

    listing = await client.get("/api/v1/assistant/conversations", headers=headers)
    assert listing.status_code == 200
    assert listing.json()["items"][0]["id"] == conversation_id
    assert listing.json()["items"][0]["message_count"] == 4
    detail = await client.get(f"/api/v1/assistant/conversations/{conversation_id}", headers=headers)
    assert [m["role"] for m in detail.json()["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]

    other = await login(client, setup["colleague"])
    for method in ("get", "delete"):
        denied = await getattr(client, method)(
            f"/api/v1/assistant/conversations/{conversation_id}", headers=other
        )
        assert denied.status_code == 404 and denied.json()["code"] == "not_found"
    hijack = await client.post(
        "/api/v1/assistant/ask",
        json={"question": "continue", "conversation_id": conversation_id},
        headers=other,
    )
    assert hijack.status_code == 404

    deleted = await client.delete(
        f"/api/v1/assistant/conversations/{conversation_id}", headers=headers
    )
    assert deleted.status_code == 204
    gone = await client.get(f"/api/v1/assistant/conversations/{conversation_id}", headers=headers)
    assert gone.status_code == 404


async def test_rbac(client: Any, setup: Any) -> None:
    auditor = await login(client, setup["auditor"])
    assert (
        await client.post("/api/v1/assistant/ask", json={"question": "q?"}, headers=auditor)
    ).status_code == 403
    assert (await client.get("/api/v1/assistant/conversations", headers=auditor)).status_code == 403
    employee = await login(client, setup["employee"])
    agent = await client.post(
        "/api/v1/assistant/agent", json={"task": "find deadlines"}, headers=employee
    )
    assert agent.status_code == 403
    assert (await client.post("/api/v1/assistant/ask", json={"question": "q"})).status_code == 401


async def test_agent_is_disabled_with_the_offline_provider(client: Any, setup: Any) -> None:
    manager = await login(client, setup["manager"])
    response = await client.post(
        "/api/v1/assistant/agent", json={"task": "find deadlines"}, headers=manager
    )
    assert response.status_code == 404
    assert response.json()["code"] == "feature_disabled"
    assert "tool support" in response.json()["detail"]


async def test_policy_endpoint(client: Any, setup: Any) -> None:
    headers = await login(client, setup["employee"])
    response = await client.get("/api/v1/assistant/policy", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert [r["classification"] for r in body["routes"]] == [
        "PUBLIC",
        "INTERNAL",
        "CONFIDENTIAL",
        "RESTRICTED",
    ]
    assert all(
        r["allowed"] and r["provider"] == "local_extractive" and not r["external"]
        for r in body["routes"]
    )
    assert body["agent_available"] is False
    assert body["external_max_classification"] == "CONFIDENTIAL"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "ok", "unexpected": 1},
        {"question": "ok", "filters": {"doc_types": ["spaceship"]}},
        {"question": "ok", "conversation_id": "not-a-uuid"},
        {"question": "x" * 20_001},
    ],
)
async def test_invalid_requests_are_422(client: Any, setup: Any, payload: dict[str, Any]) -> None:
    headers = await login(client, setup["employee"])
    response = await client.post("/api/v1/assistant/ask", json=payload, headers=headers)
    assert response.status_code == 422
    assert "x" * 100 not in response.text  # rejected input is not echoed back


async def test_filters_and_question_limit(client: Any, setup: Any, container: Any) -> None:
    headers = await login(client, setup["employee"])
    response = await client.post(
        "/api/v1/assistant/ask",
        json={"question": "What are the payment terms?", "filters": {"doc_types": ["invoice"]}},
        headers=headers,
    )
    assert response.status_code == 200 and response.json()["status"] == "insufficient_context"
    too_long = "word " * (container.settings.llm.max_question_chars // 5 + 10)
    response = await client.post(
        "/api/v1/assistant/ask", json={"question": too_long}, headers=headers
    )
    assert response.status_code == 422 and response.json()["code"] == "validation_failed"
    naive = await client.post(
        "/api/v1/assistant/ask",
        json={"question": "terms?", "filters": {"created_after": "2026-01-01T00:00:00"}},
        headers=headers,
    )
    assert naive.status_code == 422
