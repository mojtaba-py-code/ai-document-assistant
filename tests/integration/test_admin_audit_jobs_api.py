"""Audit trail, job, usage and admin-health endpoints."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import asyncpg
import pytest
from sqlalchemy import update

from docassist.db.models import Job, LlmUsage
from docassist.db.session import DbContext
from docassist.jobs.queue import enqueue
from tests.conftest import APP_PASSWORD, TEST_DB_URL, login
from tests.helpers_admin import audit_events, make_tenant, seal_all

pytestmark = [pytest.mark.db]


async def _superuser(container) -> asyncpg.Connection:
    assert TEST_DB_URL
    db_name = container.settings.database.url.get_secret_value().rsplit("/", 1)[1]
    return await asyncpg.connect(TEST_DB_URL.rsplit("/", 1)[0] + "/" + db_name)


async def _all_pages(client, headers, params: dict[str, object]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    cursor = None
    while True:
        query = {**params, **({"cursor": cursor} if cursor else {})}
        response = await client.get("/api/v1/audit/events", headers=headers, params=query)
        assert response.status_code == 200, response.text
        items += response.json()["items"]
        cursor = response.json()["next_cursor"]
        if not cursor:
            return items


# --------------------------------------------------------------------------- #
# Audit events
# --------------------------------------------------------------------------- #
async def test_audit_event_filters(client, factory, container) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    user = await client.post(
        "/api/v1/users",
        headers=admin,
        json={
            "email": f"aud-{uuid.uuid4().hex[:8]}@example.test",
            "full_name": "A",
            "role": "employee",
        },
    )
    dept = await client.post("/api/v1/departments", headers=admin, json={"name": "Legal"})
    denied = await client.post(
        "/api/v1/users",
        headers=admin,
        json={"email": "pa@example.test", "full_name": "P", "role": "platform_admin"},
    )
    assert (user.status_code, dept.status_code, denied.status_code) == (201, 201, 403)
    other = await make_tenant(factory)
    await client.post(
        "/api/v1/departments",
        headers=await login(client, other.admin),
        json={"name": "Other Legal"},
    )

    auditor = await login(client, await factory.user(t.org_id, "auditor"))
    exact = await _all_pages(client, auditor, {"action": "admin.user_created"})
    assert [e["resource_id"] for e in exact] == [user.json()["id"]]
    assert exact[0]["actor_user_id"] == str(t.admin.id) and exact[0]["outcome"] == "success"

    prefix = await _all_pages(client, auditor, {"action": "admin.*"})
    assert {e["action"] for e in prefix} >= {
        "admin.user_created",
        "admin.department_created",
        "admin.denied",
    }
    assert all(e["action"].startswith("admin.") for e in prefix)

    denials = await _all_pages(client, auditor, {"outcome": "denied"})
    assert {e["action"] for e in denials} == {"admin.denied"}

    by_resource = await _all_pages(
        client, auditor, {"resource_type": "department", "resource_id": dept.json()["id"]}
    )
    assert [e["action"] for e in by_resource] == ["admin.department_created"]

    by_actor = await _all_pages(client, auditor, {"actor": str(t.admin.id)})
    assert by_actor and all(e["actor_user_id"] == str(t.admin.id) for e in by_actor)

    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    assert await _all_pages(client, auditor, {"from": future}) == []
    assert await _all_pages(client, auditor, {"to": past}) == []

    # keyset paging returns every event exactly once, newest first
    paged = await _all_pages(client, auditor, {"limit": 1, "action": "admin.*"})
    ids = [e["id"] for e in paged]
    assert ids == sorted(ids, reverse=True) and len(ids) == len(set(ids)) == len(prefix)

    # the other tenant's events never appear
    everything = await _all_pages(client, auditor, {"limit": 200})
    assert "admin.department_created" in {e["action"] for e in everything}
    other_ids = {str(e.id) for e in await audit_events(container, other.org_id)}
    assert not other_ids & {str(e["id"]) for e in everything}

    # reading the trail is audited
    viewed = await audit_events(container, t.org_id, action="audit.events_viewed")
    assert viewed and viewed[-1].details["returned"] >= 1

    for bad in (
        {"action": "DROP TABLE"},
        {"action": "admin.*.*"},
        {"cursor": "xyz"},
        {"outcome": "maybe"},
    ):
        response = await client.get("/api/v1/audit/events", headers=auditor, params=bad)
        assert response.status_code == 422, bad
    employee = await login(client, await factory.user(t.org_id))
    assert (await client.get("/api/v1/audit/events", headers=employee)).status_code == 403


async def test_platform_admin_sees_only_the_platform_chain(client, factory) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    created = await client.post(
        "/api/v1/users",
        headers=admin,
        json={
            "email": f"pc-{uuid.uuid4().hex[:8]}@example.test",
            "full_name": "B",
            "role": "employee",
        },
    )
    operator = await login(client, await factory.user(None, "platform_admin"))
    org = await client.post(
        "/api/v1/platform/organizations",
        headers=operator,
        json={
            "slug": f"pc-{uuid.uuid4().hex[:8]}",
            "name": "Platform Chain Co",
            "admin_email": f"pc-admin-{uuid.uuid4().hex[:8]}@example.test",
            "admin_name": "Admin",
        },
    )
    platform_events = await _all_pages(client, operator, {"action": "platform.*", "limit": 200})
    assert org.json()["organization"]["id"] in {e["resource_id"] for e in platform_events}
    tenant_view = await _all_pages(client, operator, {"action": "admin.user_created", "limit": 200})
    assert created.json()["id"] not in {e["resource_id"] for e in tenant_view}
    assert org.json()["admin_user_id"] not in {e["resource_id"] for e in tenant_view}
    verify = await client.post("/api/v1/audit/verify", headers=operator)
    assert verify.status_code == 200
    assert verify.json()["chain"] == "platform" and verify.json()["organization_id"] is None


async def test_verify_detects_modified_and_deleted_events(client, factory, container) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    for name in ("Alpha", "Beta", "Gamma", "Delta"):
        assert (
            await client.post("/api/v1/departments", headers=admin, json={"name": name})
        ).status_code == 201
    await seal_all(container)
    auditor = await login(client, await factory.user(t.org_id, "auditor"))
    clean = await client.post("/api/v1/audit/verify", headers=auditor)
    assert clean.status_code == 200, clean.text
    report = clean.json()
    assert report["valid"] is True and report["chain"] == "organization"
    assert report["organization_id"] == str(t.org_id) and report["checked"] >= 4

    conn = await _superuser(container)
    try:
        async with conn.transaction():
            await conn.execute("SET LOCAL session_replication_role = replica")
            await conn.execute(
                'UPDATE audit_events SET details = \'{"slug": "forged"}\'::jsonb'
                " WHERE organization_id = $1 AND action = 'admin.department_created'"
                " AND seal_seq = (SELECT min(seal_seq) FROM audit_events"
                "   WHERE organization_id = $1 AND action = 'admin.department_created')",
                t.org_id,
            )
    finally:
        await conn.close()
    tampered = (await client.post("/api/v1/audit/verify", headers=auditor)).json()
    assert tampered["valid"] is False and tampered["reason"] == "hash mismatch"
    failures = await audit_events(container, t.org_id, action="audit.chain_verified")
    assert failures[-1].outcome == "failure"

    second = await make_tenant(factory)
    second_admin = await login(client, second.admin)
    for name in ("One", "Two", "Three"):
        await client.post("/api/v1/departments", headers=second_admin, json={"name": name})
    await seal_all(container)
    conn = await _superuser(container)
    try:
        async with conn.transaction():
            await conn.execute("SET LOCAL session_replication_role = replica")
            await conn.execute(
                "DELETE FROM audit_events WHERE organization_id = $1 AND seal_seq = 2",
                second.org_id,
            )
    finally:
        await conn.close()
    deleted = (await client.post("/api/v1/audit/verify", headers=second_admin)).json()
    assert deleted["valid"] is False and deleted["reason"] == "gap or reordering"
    assert deleted["first_bad_seq"] == 2


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
async def _job(
    container, org_id, *, created_by, status="queued", kind="ingest_version", **payload
) -> uuid.UUID:
    async with container.db.transaction(DbContext(org_id=org_id)) as session:
        job_id = await enqueue(
            session,
            kind=kind,
            organization_id=org_id,
            payload=payload or {"version_id": str(uuid.uuid4())},
            created_by=created_by,
        )
        assert job_id is not None
        if status != "queued":
            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    status=status,
                    attempts=5,
                    last_error_code="parse_error",
                    last_error="Traceback: secret internal detail at /srv/app.py",
                )
            )
    return job_id


async def test_jobs_listing_is_scoped_and_minimal(client, factory, container) -> None:
    t = await make_tenant(factory)
    employee = await factory.user(t.org_id)
    colleague = await factory.user(t.org_id)
    document_id = str(uuid.uuid4())
    mine = await _job(
        container,
        t.org_id,
        created_by=employee.id,
        status="dead",
        document_id=document_id,
        note="free text",
        other_id="not-a-uuid",
    )
    theirs = await _job(container, t.org_id, created_by=colleague.id)
    foreign = await make_tenant(factory)
    await _job(container, foreign.org_id, created_by=foreign.admin.id)

    own = (await client.get("/api/v1/jobs", headers=await login(client, employee))).json()
    assert [j["id"] for j in own["items"]] == [str(mine)]
    job = own["items"][0]
    assert job["error_code"] == "parse_error" and job["status"] == "dead"
    assert job["resource_ids"] == {"document_id": document_id}
    listing_text = str(own)
    assert "Traceback" not in listing_text and "free text" not in listing_text

    admin = await login(client, t.admin)
    everything = (await client.get("/api/v1/jobs", headers=admin)).json()
    assert {j["id"] for j in everything["items"]} == {str(mine), str(theirs)}
    dead = (await client.get("/api/v1/jobs", headers=admin, params={"status": "dead"})).json()
    assert [j["id"] for j in dead["items"]] == [str(mine)]
    by_kind = (
        await client.get("/api/v1/jobs", headers=admin, params={"kind": "export_generate"})
    ).json()
    assert by_kind["items"] == []
    assert (
        await client.get("/api/v1/jobs", headers=admin, params={"kind": "Bad Kind"})
    ).status_code == 422
    page = (await client.get("/api/v1/jobs", headers=admin, params={"limit": 1})).json()
    assert len(page["items"]) == 1 and page["next_cursor"]
    auditor = await login(client, await factory.user(t.org_id, "auditor"))
    assert (await client.get("/api/v1/jobs", headers=auditor)).status_code == 403


async def test_retry_and_cancel_rules(client, factory, container) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    dead = await _job(container, t.org_id, created_by=t.admin.id, status="dead")
    queued = await _job(container, t.org_id, created_by=t.admin.id)
    running = await _job(container, t.org_id, created_by=t.admin.id, status="running")

    retried = await client.post(f"/api/v1/jobs/{dead}/retry", headers=admin)
    assert retried.status_code == 200
    assert retried.json()["status"] == "queued" and retried.json()["attempts"] == 0
    assert (await client.post(f"/api/v1/jobs/{queued}/retry", headers=admin)).status_code == 409

    cancelled = await client.post(f"/api/v1/jobs/{queued}/cancel", headers=admin)
    assert cancelled.json()["status"] == "cancelled" and cancelled.json()["finished_at"]
    assert (await client.post(f"/api/v1/jobs/{running}/cancel", headers=admin)).status_code == 409
    assert (await client.post(f"/api/v1/jobs/{queued}/cancel", headers=admin)).status_code == 409

    other = await make_tenant(factory)
    foreign_dead = await _job(container, other.org_id, created_by=other.admin.id, status="dead")
    assert (
        await client.post(f"/api/v1/jobs/{foreign_dead}/retry", headers=admin)
    ).status_code == 404
    manager = await login(client, await factory.user(t.org_id, "department_manager"))
    assert (await client.post(f"/api/v1/jobs/{running}/cancel", headers=manager)).status_code == 403

    actions = [e.action for e in await audit_events(container, t.org_id)]
    assert "jobs.retried" in actions and "jobs.cancelled" in actions


# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #
async def test_usage_report_and_budget(client, factory, container) -> None:
    t = await make_tenant(factory)
    now = datetime.now(UTC)
    last_month = now.replace(day=1) - timedelta(days=1)
    async with container.db.transaction(DbContext(org_id=t.org_id)) as session:
        for model, task, tin, tout, cost, at in (
            ("claude-opus-5-5", "answer", 1000, 200, "0.008", now),
            ("claude-opus-5-5", "answer", 500, 100, "0.004", now),
            ("claude-haiku-4-5", "classify", 300, 10, "0.00035", now),
            ("claude-opus-5-5", "answer", 9999, 9999, "1", last_month),
        ):
            session.add(
                LlmUsage(
                    organization_id=t.org_id,
                    task=task,
                    provider="anthropic",
                    model=model,
                    input_tokens=tin,
                    output_tokens=tout,
                    cost_usd=Decimal(cost),
                    status="ok",
                    created_at=at,
                )
            )
    admin = await login(client, t.admin)
    await client.patch(
        "/api/v1/organization/settings", headers=admin, json={"llm": {"monthly_token_budget": 2000}}
    )
    report = (await client.get("/api/v1/usage", headers=admin)).json()
    assert report["month"] == f"{now.year:04d}-{now.month:02d}"
    rows = {(r["model"], r["task"]): r for r in report["rows"]}
    assert rows[("claude-opus-5-5", "answer")]["requests"] == 2
    assert rows[("claude-opus-5-5", "answer")]["input_tokens"] == 1500
    assert Decimal(report["totals"]["cost_usd"]) == Decimal("0.01235")
    assert report["totals"]["total_tokens"] == 2110
    assert report["budget"] == {
        "deployment_limit": container.settings.llm.monthly_token_budget_per_org,
        "organization_limit": 2000,
        "effective_limit": 2000,
        "used_tokens": 2110,
        "remaining_tokens": 0,
        "exhausted": True,
    }
    previous = (
        await client.get(
            "/api/v1/usage", headers=admin, params={"month": last_month.strftime("%Y-%m")}
        )
    ).json()
    assert previous["totals"]["total_tokens"] == 19998
    assert (
        await client.get("/api/v1/usage", headers=admin, params={"month": "2026-13"})
    ).status_code == 422
    assert (
        await client.get("/api/v1/usage", headers=admin, params={"month": "May"})
    ).status_code == 422
    auditor = await login(client, await factory.user(t.org_id, "auditor"))
    assert (await client.get("/api/v1/usage", headers=auditor)).status_code == 200
    employee = await login(client, await factory.user(t.org_id))
    assert (await client.get("/api/v1/usage", headers=employee)).status_code == 403


# --------------------------------------------------------------------------- #
# Admin health
# --------------------------------------------------------------------------- #
async def test_admin_health_reveals_states_not_secrets(client, factory, container) -> None:
    t = await make_tenant(factory)
    await _job(container, t.org_id, created_by=t.admin.id)
    response = await client.get("/api/v1/admin/health", headers=await login(client, t.admin))
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["components"]["database"]["status"] == "ok"
    assert body["components"]["redis"]["status"] == "ok"
    assert body["status"] in {"ok", "degraded"}
    assert body["queue"]["scope"] == "organization"
    assert body["queue"]["depth"]["queued"] == 1
    assert set(body["queue"]["depth"]) == {
        "queued",
        "running",
        "succeeded",
        "failed",
        "dead",
        "cancelled",
    }
    text = response.text
    for secret in (
        APP_PASSWORD,
        container.settings.security.jwt_signing_key.get_secret_value(),
        "postgresql",
        "127.0.0.1",
    ):
        assert secret not in text

    operator = await login(client, await factory.user(None, "platform_admin"))
    platform = (await client.get("/api/v1/admin/health", headers=operator)).json()
    assert platform["queue"]["scope"] == "platform"
    for role in ("auditor", "employee"):
        headers = await login(client, await factory.user(t.org_id, role))
        assert (await client.get("/api/v1/admin/health", headers=headers)).status_code == 403
