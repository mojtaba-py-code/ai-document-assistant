"""``POST /api/v1/search`` through the real HTTP stack (auth, RBAC, validation, headers)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tests.conftest import login
from tests.helpers_search import ChunkSpec, seed_document

pytestmark = [pytest.mark.db]

RESULT_KEYS = {
    "chunk_id",
    "document_id",
    "document_title",
    "version_number",
    "is_current",
    "classification",
    "doc_type",
    "page_start",
    "page_end",
    "section",
    "snippet",
    "score",
    "keyword_score",
    "semantic_score",
    "flagged",
}


async def _employee_with_doc(container: Any, factory: Any) -> tuple[Any, Any]:
    org = await factory.org()
    dept = await factory.department(org)
    user = await factory.user(org, "employee", departments=[dept])
    doc = await seed_document(
        container,
        org_id=org,
        owner_id=user.id,
        title="Supplier agreement",
        doc_type="contract",
        tags=["legal"],
        chunks=[ChunkSpec("The supplier agreement renews on 1 March.", section="Term", page=2)],
    )
    return user, doc


async def test_search_endpoint_contract(client, container, factory) -> None:
    user, doc = await _employee_with_doc(container, factory)
    headers = await login(client, user)
    response = await client.post(
        "/api/v1/search", json={"query": "supplier agreement renews"}, headers=headers
    )
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert set(body) == {"results", "mode_used", "degraded", "took_ms"}
    assert body["mode_used"] == "hybrid" and body["degraded"] is False
    [hit] = body["results"]
    assert set(hit) == RESULT_KEYS
    assert hit["document_id"] == str(doc.id) and hit["document_title"] == "Supplier agreement"
    assert hit["section"] == "Term" and hit["page_start"] == 2 and hit["version_number"] == 1
    assert hit["classification"] == "INTERNAL" and hit["doc_type"] == "contract"
    assert hit["flagged"] is False and 0 < hit["score"] <= 1
    assert "renews" in hit["snippet"]


async def test_filters_round_trip(client, container, factory) -> None:
    user, doc = await _employee_with_doc(container, factory)
    headers = await login(client, user)
    now = datetime.now(UTC)
    for filters, expected in (
        ({"doc_types": ["contract"], "tags": ["legal"]}, 1),
        ({"doc_types": ["invoice"]}, 0),
        ({"document_ids": [str(doc.id)], "classifications": ["INTERNAL"]}, 1),
        ({"created_after": (now - timedelta(hours=1)).isoformat()}, 1),
        ({"created_before": (now - timedelta(hours=1)).isoformat()}, 0),
        ({"include_old_versions": True}, 1),
    ):
        response = await client.post(
            "/api/v1/search",
            json={"query": "supplier", "mode": "keyword", "filters": filters, "limit": 5},
            headers=headers,
        )
        assert response.status_code == 200, (filters, response.text)
        assert len(response.json()["results"]) == expected, filters


async def test_authentication_and_permission_are_required(client, container, factory) -> None:
    user, _ = await _employee_with_doc(container, factory)
    assert (await client.post("/api/v1/search", json={"query": "x"})).status_code == 401
    auditor = await factory.user(user.org_id, "auditor")
    response = await client.post(
        "/api/v1/search", json={"query": "supplier"}, headers=await login(client, auditor)
    )
    assert response.status_code == 403
    assert response.json()["code"] == "permission_denied"


@pytest.mark.parametrize(
    "body",
    [
        {"query": "x", "unexpected": True},
        {"query": "x", "mode": "fuzzy"},
        {"query": "x", "limit": 0},
        {"query": "x", "limit": 51},
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 10_001},
        {"query": "x" * 1_001},
        {"query": "x", "filters": {"document_ids": ["not-a-uuid"]}},
        {"query": "x", "filters": {"doc_types": ["contract'--"]}},
        {"query": "x", "filters": {"classifications": ["TOP_SECRET"]}},
        {"query": "x", "filters": {"created_after": "2026-01-01T00:00:00"}},
        {"query": "x", "filters": {"tags": ["bad\u0000tag"]}},
        {"query": "x", "filters": {"tags": ["t"] * 51}},
        {"query": "x", "filters": {"tags": ["x" * 65]}},
        {"query": "x", "filters": {"surprise": 1}},
        {
            "query": "x",
            "filters": {
                "created_after": "2026-02-01T00:00:00Z",
                "created_before": "2026-01-01T00:00:00Z",
            },
        },
        {"query": ["list"]},
    ],
)
async def test_invalid_requests_get_422_without_echo(
    client, container, factory, body: dict
) -> None:
    user, _ = await _employee_with_doc(container, factory)
    response = await client.post("/api/v1/search", json=body, headers=await login(client, user))
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "validation_failed"
    for value in _strings(body):
        if len(value) >= 4:  # the rejected input is never echoed back
            assert value not in response.text
            assert json.dumps(value)[1:-1] not in response.text


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


async def test_unreadable_documents_never_leak_through_the_api(client, container, factory) -> None:
    org = await factory.org()
    finance, sales = await factory.department(org), await factory.department(org)
    cfo = await factory.user(org, "department_manager", departments=[finance])
    hidden = await seed_document(
        container,
        org_id=org,
        owner_id=cfo.id,
        title="Project Nightingale layoffs",
        classification="CONFIDENTIAL",
        department_id=finance,
        chunks=["Project Nightingale layoffs are planned for the second quarter."],
    )
    seller = await factory.user(org, "employee", departments=[sales])
    headers = await login(client, seller)
    for body in (
        {"query": "Nightingale layoffs"},
        {"query": "Nightingale layoffs", "mode": "semantic"},
        {"query": "Nightingale", "filters": {"document_ids": [str(hidden.id)]}},
        {"query": "Nightingale", "filters": {"department_ids": [str(finance)]}},
        {"query": "Nightin", "mode": "keyword"},
    ):
        response = await client.post("/api/v1/search", json=body, headers=headers)
        assert response.status_code == 200
        assert response.json()["results"] == []
        assert "Nightingale" not in response.text
        assert str(hidden.id) not in response.text
    # the finance manager does find it
    response = await client.post(
        "/api/v1/search", json={"query": "Nightingale layoffs"}, headers=await login(client, cfo)
    )
    assert [r["document_id"] for r in response.json()["results"]] == [str(hidden.id)]
