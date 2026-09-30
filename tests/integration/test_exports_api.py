"""Exports end-to-end: creation, worker generation, downloads, signed links, expiry, caps."""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from sqlalchemy import select, update

from docassist.db.models import AuditEvent, Export, Organization, User
from docassist.db.session import DbContext
from docassist.intelligence import exports as exports_module
from docassist.intelligence.exports import build_link_token
from docassist.jobs.queue import PermanentJobError
from docassist.search.types import SearchResponse, SearchResult
from tests.conftest import login
from tests.helpers_intelligence import (
    CONTRACT,
    FieldSpec,
    export_job,
    installed_storage,
    replaced_attribute,
    run_export_job,
    seed_document,
    tenant,
)

pytestmark = [pytest.mark.db]


@pytest.fixture(autouse=True)
def storage(container: Any) -> Any:
    with installed_storage(container) as installed:
        yield installed


def fields(day: date) -> list[FieldSpec]:
    return [
        FieldSpec("expiration_date", value_date=day, evidence="expires soon", chunk=0),
        FieldSpec(
            "payment_terms",
            value_text="net 30 days",
            evidence="Payment terms: net 30 days",
            chunk=1,
        ),
    ]


async def create(
    client: Any, headers: dict[str, str], kind: str, fmt: str = "csv", **params: Any
) -> Any:
    return await client.post(
        "/api/v1/exports", json={"kind": kind, "format": fmt, "params": params}, headers=headers
    )


async def ready_export(
    client: Any, container: Any, headers: dict[str, str], kind: str, fmt: str = "csv", **params: Any
) -> str:
    response = await create(client, headers, kind, fmt, **params)
    assert response.status_code == 202, response.text
    export_id = response.json()["id"]
    outcome = await run_export_job(container, uuid.UUID(export_id))
    assert isinstance(outcome, dict) and "row_count" in outcome, outcome
    return export_id


async def audit(container: Any, org: uuid.UUID, action: str) -> list[AuditEvent]:
    async with container.db.session(DbContext(org_id=org)) as session:
        rows = await session.execute(
            select(AuditEvent).where(AuditEvent.organization_id == org, AuditEvent.action == action)
        )
        return list(rows.scalars())


async def export_row(container: Any, org: uuid.UUID, export_id: str) -> Export:
    async with container.worker_db.session(DbContext(org_id=org)) as session:
        return (
            await session.execute(select(Export).where(Export.id == uuid.UUID(export_id)))
        ).scalar_one()


async def set_export(container: Any, org: uuid.UUID, export_id: str, **values: Any) -> None:
    async with container.worker_db.transaction(DbContext(org_id=org)) as session:
        await session.execute(
            update(Export).where(Export.id == uuid.UUID(export_id)).values(**values)
        )


def read_csv(content: bytes) -> list[list[str]]:
    assert content.startswith(b"\xef\xbb\xbf")
    return list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))


async def test_csv_export_lifecycle(client, factory, container) -> None:
    t = await tenant(factory)
    soon = date.today() + timedelta(days=40)
    readable = await seed_document(
        container, t.org, t.manager.id, title="Readable contract", versions=[CONTRACT, CONTRACT],
        fields={0: [FieldSpec("old_version_only", value_text="stale", chunk=0)], 1: fields(soon)},
    )  # fmt: skip
    await seed_document(
        container, t.org, t.admin.id, title="Hidden contract", classification="CONFIDENTIAL",
        department_id=t.other_dept, fields={0: fields(soon)},
    )  # fmt: skip
    headers = await login(client, t.manager)
    response = await create(client, headers, "extracted_fields")
    assert response.status_code == 202, response.text
    created = response.json()
    assert created["status"] == "pending" and created["kind"] == "extracted_fields"
    job = await export_job(container, uuid.UUID(created["id"]))
    assert job is not None and job.kind == "export_generate" and job.organization_id == t.org
    assert job.payload == {"export_id": created["id"]}  # identifiers only
    outcome = await run_export_job(container, uuid.UUID(created["id"]))
    assert outcome == {"export_id": created["id"], "row_count": 2, "truncated": False}
    info = (await client.get(f"/api/v1/exports/{created['id']}", headers=headers)).json()
    assert info["status"] == "ready" and info["row_count"] == 2 and not info["truncated"]
    ready_at = datetime.fromisoformat(info["ready_at"])
    expires_at = datetime.fromisoformat(info["expires_at"])
    assert expires_at - ready_at == timedelta(hours=container.settings.retention.export_ttl_hours)
    download = await client.get(f"/api/v1/exports/{created['id']}/download", headers=headers)
    assert download.status_code == 200
    assert download.headers["content-type"] == "text/csv; charset=utf-8"
    assert download.headers["content-disposition"].startswith(
        'attachment; filename="export-extracted-fields-'
    )
    assert download.headers["cache-control"] == "no-store"
    assert download.headers["x-content-type-options"] == "nosniff"
    rows = read_csv(download.content)
    assert rows[0] == list(exports_module.COLUMNS["extracted_fields"])
    assert {(r[1], r[4], r[5]) for r in rows[1:]} == {
        ("Readable contract", "expiration_date", soon.isoformat()),
        ("Readable contract", "payment_terms", "net 30 days"),
    }
    assert all(r[0] == str(readable.id) for r in rows[1:])
    assert (await client.get(f"/api/v1/exports/{created['id']}", headers=headers)).json()[
        "download_count"
    ] == 1
    generated = await audit(container, t.org, "export.generated")
    assert any(e.resource_id == created["id"] and e.details["row_count"] == 2 for e in generated)
    downloaded = await audit(container, t.org, "export.downloaded")
    assert any(
        e.resource_id == created["id"]
        and e.details["row_count"] == 2
        and e.details["via"] == "session"
        for e in downloaded
    )
    assert any(
        e.resource_id == created["id"] for e in await audit(container, t.org, "export.created")
    )
    listing = (await client.get("/api/v1/exports", headers=headers)).json()["items"]
    assert [item["id"] for item in listing] == [created["id"]]


async def test_json_deadline_export(client, factory, container) -> None:
    t = await tenant(factory)
    soon = date.today() + timedelta(days=10)
    await seed_document(container, t.org, t.manager.id, title="Lease", fields={0: fields(soon)})
    headers = await login(client, t.manager)
    export_id = await ready_export(
        client, container, headers, "deadlines", "json", within_days=30, timezone="+02:00"
    )
    response = await client.get(f"/api/v1/exports/{export_id}/download", headers=headers)
    assert response.headers["content-type"] == "application/json"
    body = json.loads(response.content)
    assert body["export"]["kind"] == "deadlines" and body["export"]["row_count"] == 1
    (row,) = body["rows"]
    assert row["document_title"] == "Lease" and row["field"] == "expiration_date"
    assert row["date"] == soon.isoformat() and isinstance(row["days_left"], int)


async def test_exports_are_private_to_their_creator(client, factory, container) -> None:
    t = await tenant(factory)
    other = await tenant(factory)
    await seed_document(container, t.org, t.manager.id, fields={0: fields(date.today())})
    owner = await login(client, t.manager)
    export_id = await ready_export(client, container, owner, "extracted_fields")
    colleague = await factory.user(t.org, "department_manager", managed=[t.dept])
    for user in (colleague, t.admin, other.admin):
        headers = await login(client, user)
        assert (
            await client.get(f"/api/v1/exports/{export_id}", headers=headers)
        ).status_code == 404
        assert (
            await client.get(f"/api/v1/exports/{export_id}/download", headers=headers)
        ).status_code == 404
        assert (
            await client.post(f"/api/v1/exports/{export_id}/link", headers=headers)
        ).status_code == 404
        assert (await client.get("/api/v1/exports", headers=headers)).json()["items"] == []
    employee = await login(client, t.employee)
    assert (
        await create(client, employee, "extracted_fields")
    ).status_code == 403  # no export:create
    assert (await client.get("/api/v1/exports", headers=employee)).status_code == 403
    assert (await client.get(f"/api/v1/exports/{export_id}")).status_code == 401


async def test_signed_link_is_single_use(client, factory, container) -> None:
    t = await tenant(factory)
    await seed_document(container, t.org, t.manager.id, fields={0: fields(date.today())})
    headers = await login(client, t.manager)
    export_id = await ready_export(client, container, headers, "extracted_fields")
    link = (await client.post(f"/api/v1/exports/{export_id}/link", headers=headers)).json()
    parts = urlsplit(link["url"])
    assert link["url"].startswith(container.settings.public_base_url)
    assert parts.path == "/api/v1/exports/download"
    expires = datetime.fromisoformat(link["expires_at"])
    assert timedelta(0) < expires - datetime.now(UTC) <= timedelta(seconds=300)
    token = parse_qs(parts.query)["token"][0]
    first = await client.get("/api/v1/exports/download", params={"token": token})  # no bearer
    assert first.status_code == 200 and read_csv(first.content)[0][0] == "document_id"
    second = await client.get("/api/v1/exports/download", params={"token": token})
    assert second.status_code == 410
    tampered = token[:-4] + ("0000" if not token.endswith("0000") else "1111")
    assert (
        await client.get("/api/v1/exports/download", params={"token": tampered})
    ).status_code == 404
    assert (
        await client.get("/api/v1/exports/download", params={"token": "garbage"})
    ).status_code == 404
    # a link is consumed by any download of the export, including a session download
    unused = (await client.post(f"/api/v1/exports/{export_id}/link", headers=headers)).json()
    assert (
        await client.get(f"/api/v1/exports/{export_id}/download", headers=headers)
    ).status_code == 200
    stale = parse_qs(urlsplit(unused["url"]).query)["token"][0]
    assert (
        await client.get("/api/v1/exports/download", params={"token": stale})
    ).status_code == 410
    via = [
        e.details["via"]
        for e in await audit(container, t.org, "export.downloaded")
        if e.resource_id == export_id
    ]
    assert sorted(via) == ["link", "session"]
    reasons = {e.details["reason"] for e in await audit(container, t.org, "export.link_rejected")}
    assert {"reused"} <= reasons
    assert await audit(container, t.org, "export.link_created")


async def test_expired_links_and_exports(client, factory, container) -> None:
    t = await tenant(factory)
    await seed_document(container, t.org, t.manager.id, fields={0: fields(date.today())})
    headers = await login(client, t.manager)
    export_id = await ready_export(client, container, headers, "extracted_fields")
    past = int((datetime.now(UTC) - timedelta(seconds=1)).timestamp())
    expired_token = build_link_token(
        container.tokens,
        export_id=uuid.UUID(export_id),
        org_id=t.org,
        user_id=t.manager.id,
        expires=past,
        counter=0,
    )
    assert (
        await client.get("/api/v1/exports/download", params={"token": expired_token})
    ).status_code == 410
    far = int((datetime.now(UTC) + timedelta(days=1)).timestamp())
    forged_lifetime = build_link_token(
        container.tokens,
        export_id=uuid.UUID(export_id),
        org_id=t.org,
        user_id=t.manager.id,
        expires=far,
        counter=0,
    )
    assert (
        await client.get("/api/v1/exports/download", params={"token": forged_lifetime})
    ).status_code == 404
    link = (await client.post(f"/api/v1/exports/{export_id}/link", headers=headers)).json()
    token = parse_qs(urlsplit(link["url"]).query)["token"][0]
    await set_export(
        container, t.org, export_id, expires_at=datetime.now(UTC) - timedelta(minutes=1)
    )
    assert (await client.get(f"/api/v1/exports/{export_id}", headers=headers)).json()[
        "status"
    ] == "expired"
    assert (
        await client.get(f"/api/v1/exports/{export_id}/download", headers=headers)
    ).status_code == 410
    assert (
        await client.post(f"/api/v1/exports/{export_id}/link", headers=headers)
    ).status_code == 410
    assert (
        await client.get("/api/v1/exports/download", params={"token": token})
    ).status_code == 410


async def test_pending_and_failed_exports(client, factory, container) -> None:
    t = await tenant(factory)
    await seed_document(container, t.org, t.manager.id, fields={0: fields(date.today())})
    headers = await login(client, t.manager)
    pending = (await create(client, headers, "extracted_fields")).json()
    assert (
        await client.get(f"/api/v1/exports/{pending['id']}/download", headers=headers)
    ).status_code == 409
    assert (
        await client.post(f"/api/v1/exports/{pending['id']}/link", headers=headers)
    ).status_code == 409
    async with container.db.transaction(DbContext(org_id=t.org)) as session:
        await session.execute(update(User).where(User.id == t.manager.id).values(status="disabled"))
    outcome = await run_export_job(container, uuid.UUID(pending["id"]))
    assert isinstance(outcome, PermanentJobError) and outcome.code == "user_inactive"
    row = await export_row(container, t.org, pending["id"])
    assert (row.status, row.error_code, row.storage_key) == ("failed", "user_inactive", None)
    assert await audit(container, t.org, "export.failed")
    rerun = await run_export_job(container, uuid.UUID(pending["id"]))
    assert rerun == {"export_id": pending["id"], "skipped": "failed"}  # idempotent


async def test_permission_revoked_before_generation(client, factory, container) -> None:
    t = await tenant(factory)
    headers = await login(client, t.manager)
    created = (await create(client, headers, "extracted_fields")).json()
    async with container.db.transaction(DbContext(org_id=t.org)) as session:
        await session.execute(update(User).where(User.id == t.manager.id).values(role="employee"))
    outcome = await run_export_job(container, uuid.UUID(created["id"]))
    assert isinstance(outcome, PermanentJobError) and outcome.code == "permission_revoked"


async def test_row_caps(client, factory, container, monkeypatch) -> None:
    t = await tenant(factory)
    for index in range(3):
        await seed_document(
            container, t.org, t.manager.id, title=f"Doc {index}", fields={0: fields(date.today())}
        )
    headers = await login(client, t.manager)
    async with container.db.transaction(DbContext(org_id=None, platform=True)) as session:
        await session.execute(
            update(Organization)
            .where(Organization.id == t.org)
            .values(settings={"exports": {"max_rows": 4}})
        )
    capped = await ready_export(client, container, headers, "extracted_fields", "json")
    info = (await client.get(f"/api/v1/exports/{capped}", headers=headers)).json()
    assert info["row_count"] == 4 and info["truncated"] and "_meta" not in info["params"]
    body = json.loads(
        (await client.get(f"/api/v1/exports/{capped}/download", headers=headers)).content
    )
    assert body["export"]["truncated"] and len(body["rows"]) == 4
    monkeypatch.setattr(exports_module, "EXPORT_MAX_ROWS", 2)  # the global cap wins when lower
    smaller = await ready_export(client, container, headers, "extracted_fields")
    assert (await client.get(f"/api/v1/exports/{smaller}", headers=headers)).json()[
        "row_count"
    ] == 2
    monkeypatch.setattr(exports_module, "EXPORT_MAX_ROWS", 10_000)
    async with container.db.transaction(DbContext(org_id=None, platform=True)) as session:
        await session.execute(
            update(Organization)
            .where(Organization.id == t.org)
            .values(settings={"exports": {"max_rows": True}})
        )
    full = await ready_export(client, container, headers, "extracted_fields")
    info = (await client.get(f"/api/v1/exports/{full}", headers=headers)).json()
    assert info["row_count"] == 6 and not info["truncated"]


async def test_parameter_validation(client, factory, container) -> None:
    t = await tenant(factory)
    hidden = await seed_document(
        container, t.org, t.admin.id, classification="CONFIDENTIAL", department_id=t.other_dept
    )
    headers = await login(client, t.manager)
    cases: list[tuple[dict[str, Any], int]] = [
        ({"kind": "everything", "format": "csv", "params": {}}, 422),
        ({"kind": "deadlines", "format": "xlsx", "params": {}}, 422),
        ({"kind": "deadlines", "format": "csv", "params": {"within_days": 90, "sql": "1"}}, 422),
        ({"kind": "deadlines", "format": "csv", "params": {"timezone": "Mars/Base"}}, 422),
        ({"kind": "deadlines", "format": "csv", "params": {"fields": ["password_hash"]}}, 422),
        ({"kind": "extracted_fields", "format": "csv", "params": {"fields": ["DROP TABLE"]}}, 422),
        ({"kind": "search_results", "format": "csv", "params": {}}, 422),
        (
            {
                "kind": "document_report",
                "format": "json",
                "params": {"document_id": str(hidden.id)},
            },
            404,
        ),
        ({"kind": "document_report", "format": "json", "params": {"document_id": "nope"}}, 422),
        ({"kind": "deadlines", "format": "csv", "params": {}, "user_id": str(uuid.uuid4())}, 422),
    ]
    for payload, expected in cases:
        response = await client.post("/api/v1/exports", json=payload, headers=headers)
        assert response.status_code == expected, (payload, response.text)
    assert (await client.get("/api/v1/exports", headers=headers)).json()["items"] == []


async def test_document_report_export(client, factory, container) -> None:
    t = await tenant(factory)
    doc = await seed_document(
        container,
        t.org,
        t.manager.id,
        title="Report me",
        fields={0: fields(date.today() + timedelta(days=5))},
    )
    headers = await login(client, t.manager)
    json_id = await ready_export(
        client, container, headers, "document_report", "json", document_id=str(doc.id)
    )
    body = json.loads(
        (await client.get(f"/api/v1/exports/{json_id}/download", headers=headers)).content
    )
    assert body["report"]["document"]["title"] == "Report me"
    assert {row["section"] for row in body["rows"]} >= {
        "document",
        "summary",
        "field",
        "deadline",
        "risk",
    }
    csv_id = await ready_export(
        client, container, headers, "document_report", "csv", document_id=str(doc.id)
    )
    rows = read_csv(
        (await client.get(f"/api/v1/exports/{csv_id}/download", headers=headers)).content
    )
    assert rows[0] == ["section", "item", "value", "page", "detail"]


async def test_search_results_export_runs_as_the_creator(client, factory, container) -> None:
    t = await tenant(factory)
    headers = await login(client, t.manager)
    doc_id = uuid.uuid4()
    calls: list[dict[str, Any]] = []

    class FakeSearch:
        async def search(
            self, principal: Any, *, query: str, mode: str, filters: Any, limit: int
        ) -> SearchResponse:
            calls.append(
                {
                    "user": principal.user_id,
                    "org": principal.org_id,
                    "query": query,
                    "mode": mode,
                    "filters": filters,
                    "limit": limit,
                }
            )
            hit = SearchResult(
                chunk_id=uuid.uuid4(), document_id=doc_id, version_id=uuid.uuid4(), document_title="=cmd|' /C calc'!A0",
                version_number=1, is_current=True, classification="INTERNAL", doc_type="contract", page_start=2,
                page_end=2, section="Payment", snippet="net 30 days", score=0.87654, keyword_score=None,
                semantic_score=None, flagged=False,
            )  # fmt: skip
            return SearchResponse(results=[hit], mode_used="keyword", degraded=False)

    with replaced_attribute(container, "search", FakeSearch()):
        export_id = await ready_export(
            client,
            container,
            headers,
            "search_results",
            query="payment terms",
            mode="keyword",
            doc_types=["contract"],
            limit=5,
        )
    (call,) = calls
    assert (call["user"], call["org"], call["query"], call["mode"], call["limit"]) == (
        t.manager.id,
        t.org,
        "payment terms",
        "keyword",
        5,
    )
    assert call["filters"].doc_types == ("contract",)
    rows = read_csv(
        (await client.get(f"/api/v1/exports/{export_id}/download", headers=headers)).content
    )
    assert rows[1][:3] == ["1", str(doc_id), "'=cmd|' /C calc'!A0"]
    assert rows[1][7] == "0.8765"
    with replaced_attribute(container, "search", None):
        created = (await create(client, headers, "search_results", query="x")).json()
        outcome = await run_export_job(container, uuid.UUID(created["id"]))
    assert isinstance(outcome, PermanentJobError) and outcome.code == "search_unavailable"


async def test_export_creation_is_rate_limited(client, factory, container) -> None:
    t = await tenant(factory)
    headers = await login(client, t.manager)
    tight = container.settings.rate_limit.model_copy(
        update={
            "export_per_user": container.settings.rate_limit.export_per_user.model_copy(
                update={"requests": 1, "per_seconds": 3600}
            )
        }
    )
    with replaced_attribute(
        container, "settings", container.settings.model_copy(update={"rate_limit": tight})
    ):
        assert (await create(client, headers, "extracted_fields")).status_code == 202
        limited = await create(client, headers, "extracted_fields")
    assert limited.status_code == 429 and "retry-after" in limited.headers


async def test_storage_failure_is_retried_then_marked_failed(
    client, factory, container, monkeypatch
) -> None:
    from docassist.documents.storage import StorageError
    from docassist.jobs.queue import JobError

    t = await tenant(factory)
    headers = await login(client, t.manager)
    created = (await create(client, headers, "extracted_fields")).json()
    export_id = uuid.UUID(created["id"])

    async def broken(*_args: Any, **_kwargs: Any) -> int:
        raise StorageError("disk full")

    monkeypatch.setattr(container.storage, "put_bytes", broken)
    first = await run_export_job(container, export_id, attempts=1, max_attempts=3)
    assert isinstance(first, JobError) and not isinstance(first, PermanentJobError)
    assert first.code == "export_retry"
    assert (await export_row(container, t.org, created["id"])).status == "pending"
    last = await run_export_job(container, export_id, attempts=3, max_attempts=3)
    assert isinstance(last, JobError)
    row = await export_row(container, t.org, created["id"])
    assert (row.status, row.error_code, row.storage_key) == ("failed", "storage_failed", None)
