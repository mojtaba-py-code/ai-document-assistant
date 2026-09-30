"""Privilege-escalation and tenant-isolation rules of the administration API.

Each test names the rule it proves. Refusals must change nothing and must be audited.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid

import pytest

from docassist.authz.permissions import ROLE_PERMISSIONS, Permission
from docassist.core.enums import Classification, Role, UserStatus
from docassist.core.errors import Conflict, PermissionDenied
from docassist.identity.admin import AdminService
from docassist.identity.schemas import OrgSettingsPatch, UserUpdate
from tests.conftest import login
from tests.helpers_admin import audit_events, load_user, make_tenant

pytestmark = [pytest.mark.db]


def _new_user(role: str, **extra: object) -> dict[str, object]:
    return {
        "email": f"esc-{uuid.uuid4().hex[:10]}@example.test",
        "full_name": "Escalation Test",
        "role": role,
        **extra,
    }


async def _denials(container, org_id: uuid.UUID) -> list[dict[str, object]]:
    return [e.details for e in await audit_events(container, org_id, action="admin.denied")]


# --------------------------------------------------------------------------- #
# Only ASSIGNABLE_ROLES; platform admins never through tenant APIs
# --------------------------------------------------------------------------- #
async def test_platform_admin_cannot_be_created_or_assigned_via_tenant_api(
    client, factory, container
) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    created = await client.post("/api/v1/users", headers=admin, json=_new_user("platform_admin"))
    assert created.status_code == 403
    listing = await client.get("/api/v1/users", headers=admin)
    assert {u["role"] for u in listing.json()["items"]} == {"organization_admin"}

    employee = await factory.user(t.org_id)
    promoted = await client.patch(
        f"/api/v1/users/{employee.id}", headers=admin, json={"role": "platform_admin"}
    )
    assert promoted.status_code == 403
    assert (await load_user(container, t.org_id, employee.id)).role == "employee"
    reasons = [d["reason"] for d in await _denials(container, t.org_id)]
    assert any("platform_admin" in str(r) for r in reasons)


# --------------------------------------------------------------------------- #
# Nobody assigns a clearance above their own
# --------------------------------------------------------------------------- #
async def test_clearance_ceiling_is_the_actors_own(client, factory, container) -> None:
    t = await make_tenant(factory, admin_clearance="CONFIDENTIAL")
    admin = await login(client, t.admin)
    too_high = await client.post(
        "/api/v1/users", headers=admin, json=_new_user("employee", clearance="RESTRICTED")
    )
    assert too_high.status_code == 403
    # without an explicit clearance the role default is capped at the actor's clearance
    capped = await client.post("/api/v1/users", headers=admin, json=_new_user("organization_admin"))
    assert capped.status_code == 201
    assert capped.json()["clearance"] == "CONFIDENTIAL"

    employee = await factory.user(t.org_id, clearance="INTERNAL")
    raised = await client.patch(
        f"/api/v1/users/{employee.id}", headers=admin, json={"clearance": "RESTRICTED"}
    )
    assert raised.status_code == 403
    assert (await load_user(container, t.org_id, employee.id)).clearance == "INTERNAL"
    ok = await client.patch(
        f"/api/v1/users/{employee.id}", headers=admin, json={"clearance": "CONFIDENTIAL"}
    )
    assert ok.status_code == 200


async def test_cannot_manage_a_user_with_higher_clearance(client, factory, container) -> None:
    t = await make_tenant(factory, admin_clearance="CONFIDENTIAL")
    senior = await factory.user(t.org_id, "organization_admin", clearance="RESTRICTED")
    admin = await login(client, t.admin)
    for path, payload in (
        (f"/api/v1/users/{senior.id}", {"status": "disabled"}),
        (f"/api/v1/users/{senior.id}", {"role": "employee"}),
        (f"/api/v1/users/{senior.id}", {"departments": [{"department_id": str(t.hr)}]}),
    ):
        assert (await client.patch(path, headers=admin, json=payload)).status_code == 403
    assert (
        await client.post(f"/api/v1/users/{senior.id}/revoke-sessions", headers=admin)
    ).status_code == 403
    assert (
        await client.post(f"/api/v1/users/{senior.id}/send-reset", headers=admin)
    ).status_code == 403
    row = await load_user(container, t.org_id, senior.id)
    assert (row.role, row.status) == ("organization_admin", "active")
    # a harmless change (display name) is still allowed
    renamed = await client.patch(
        f"/api/v1/users/{senior.id}", headers=admin, json={"full_name": "Senior Admin"}
    )
    assert renamed.status_code == 200


# --------------------------------------------------------------------------- #
# No self role / clearance / status change
# --------------------------------------------------------------------------- #
async def test_no_self_privilege_changes(client, factory, container) -> None:
    t = await make_tenant(factory)
    await factory.user(t.org_id, "organization_admin")  # a second admin exists
    admin = await login(client, t.admin)
    for payload in ({"role": "employee"}, {"clearance": "PUBLIC"}, {"status": "disabled"}):
        response = await client.patch(f"/api/v1/users/{t.admin.id}", headers=admin, json=payload)
        assert response.status_code == 403, payload
    row = await load_user(container, t.org_id, t.admin.id)
    assert (row.role, row.clearance, row.status) == ("organization_admin", "RESTRICTED", "active")
    assert (await client.get("/api/v1/auth/me", headers=admin)).status_code == 200
    reasons = [d["reason"] for d in await _denials(container, t.org_id)]
    assert reasons.count("self privilege change") == 3


# --------------------------------------------------------------------------- #
# Last active organisation admin
# --------------------------------------------------------------------------- #
async def test_admins_demoting_each_other_concurrently_leave_one_admin(factory, container) -> None:
    t = await make_tenant(factory)
    other = await factory.user(t.org_id, "organization_admin")
    p_a = await factory.principal(t.admin)
    p_b = await factory.principal(other)
    demote = UserUpdate(role=Role.EMPLOYEE)
    results = await asyncio.gather(
        container.admin.update_user(p_a, other.id, demote),
        container.admin.update_user(p_b, t.admin.id, demote),
        return_exceptions=True,
    )
    failures = [r for r in results if isinstance(r, BaseException)]
    assert len(failures) == 1 and isinstance(failures[0], PermissionDenied)
    roles = {(await load_user(container, t.org_id, uid)).role for uid in (t.admin.id, other.id)}
    assert roles == {"organization_admin", "employee"}


async def test_last_admin_guard_holds_even_if_the_actor_check_is_bypassed(
    factory, container, monkeypatch
) -> None:
    """Defence in depth: the guard does not rely on the actor being an admin themselves."""
    t = await make_tenant(factory)
    employee = await factory.user(t.org_id)
    principal = await factory.principal(employee)
    impostor = dataclasses.replace(
        principal, role=Role.ORGANIZATION_ADMIN, clearance=Classification.RESTRICTED
    )
    monkeypatch.setattr(
        AdminService, "_acting_clearance", staticmethod(lambda *_a: Classification.RESTRICTED)
    )
    with pytest.raises(Conflict):
        await container.admin.update_user(
            impostor, t.admin.id, UserUpdate(status=UserStatus.DISABLED)
        )
    assert (await load_user(container, t.org_id, t.admin.id)).status == "active"
    reasons = [d["reason"] for d in await _denials(container, t.org_id)]
    assert "last active organization admin" in reasons


async def test_a_disabled_admin_does_not_count_as_an_actor(factory, container) -> None:
    t = await make_tenant(factory)
    other = await factory.user(t.org_id, "organization_admin", status="disabled")
    principal = await factory.principal(other)  # e.g. a still-valid token of a disabled admin
    with pytest.raises(PermissionDenied):
        await container.admin.update_user(principal, t.admin.id, UserUpdate(role=Role.EMPLOYEE))
    assert (await load_user(container, t.org_id, t.admin.id)).role == "organization_admin"


# --------------------------------------------------------------------------- #
# Read-only and forbidden roles
# --------------------------------------------------------------------------- #
async def test_auditor_reads_but_changes_nothing(client, factory) -> None:
    t = await make_tenant(factory)
    auditor = await login(client, await factory.user(t.org_id, "auditor"))
    employee = await factory.user(t.org_id)
    assert (await client.get("/api/v1/users", headers=auditor)).status_code == 200
    assert (await client.get(f"/api/v1/users/{employee.id}", headers=auditor)).status_code == 200
    assert (await client.get("/api/v1/departments", headers=auditor)).status_code == 200
    assert (await client.get("/api/v1/organization", headers=auditor)).status_code == 200
    assert (await client.get("/api/v1/usage", headers=auditor)).status_code == 200
    writes = [
        ("post", "/api/v1/users", _new_user("employee")),
        ("patch", f"/api/v1/users/{employee.id}", {"status": "disabled"}),
        ("post", f"/api/v1/users/{employee.id}/revoke-sessions", None),
        ("post", f"/api/v1/users/{employee.id}/send-reset", None),
        ("post", "/api/v1/departments", {"name": "Audit Dept"}),
        ("patch", f"/api/v1/departments/{t.finance}", {"name": "Renamed"}),
        ("delete", f"/api/v1/departments/{t.finance}", None),
        ("patch", "/api/v1/organization/settings", {"llm": {"monthly_token_budget": 10}}),
        ("post", f"/api/v1/jobs/{uuid.uuid4()}/retry", None),
        ("post", f"/api/v1/jobs/{uuid.uuid4()}/cancel", None),
    ]
    for method, path, body in writes:
        kwargs = {"json": body} if body is not None else {}
        response = await client.request(method.upper(), path, headers=auditor, **kwargs)
        assert response.status_code == 403, (method, path)


@pytest.mark.parametrize("role", ["employee", "department_manager"])
async def test_non_admins_cannot_administer_users(client, factory, role) -> None:
    t = await make_tenant(factory)
    actor = await factory.user(
        t.org_id, role, managed=[t.finance] if role == "department_manager" else None
    )
    headers = await login(client, actor)
    victim = await factory.user(t.org_id, departments=[t.finance])
    assert (
        await client.post("/api/v1/users", headers=headers, json=_new_user("employee"))
    ).status_code == 403
    assert (
        await client.patch(
            f"/api/v1/users/{victim.id}", headers=headers, json={"status": "disabled"}
        )
    ).status_code == 403
    assert (
        await client.post("/api/v1/departments", headers=headers, json={"name": "X"})
    ).status_code == 403
    assert (await client.get("/api/v1/admin/health", headers=headers)).status_code == 403


async def test_platform_admin_cannot_use_tenant_content_apis(client, factory) -> None:
    t = await make_tenant(factory)
    operator = await factory.user(None, "platform_admin")
    headers = await login(client, operator)
    for path in (
        "/api/v1/users",
        f"/api/v1/users/{t.admin.id}",
        "/api/v1/departments",
        "/api/v1/organization",
        "/api/v1/usage",
        "/api/v1/jobs",
    ):
        assert (await client.get(path, headers=headers)).status_code == 403, path
    documents = await client.get("/api/v1/documents", headers=headers)
    assert documents.status_code in {403, 404}  # 404 only while the documents router is absent
    created = await client.post("/api/v1/users", headers=headers, json=_new_user("employee"))
    assert created.status_code == 403


async def test_tenant_admin_cannot_use_platform_apis(client, factory) -> None:
    t = await make_tenant(factory)
    headers = await login(client, t.admin)
    assert (await client.get("/api/v1/platform/organizations", headers=headers)).status_code == 403
    assert (
        await client.get(f"/api/v1/platform/organizations/{t.org_id}", headers=headers)
    ).status_code == 403
    create = await client.post(
        "/api/v1/platform/organizations",
        headers=headers,
        json={"slug": "evil", "name": "Evil", "admin_email": "e@example.test", "admin_name": "E"},
    )
    assert create.status_code == 403
    suspend = await client.patch(
        f"/api/v1/platform/organizations/{t.org_id}", headers=headers, json={"status": "suspended"}
    )
    assert suspend.status_code == 403


# --------------------------------------------------------------------------- #
# Cross-tenant administration -> 404, nothing changes
# --------------------------------------------------------------------------- #
async def test_cross_tenant_administration_is_invisible(client, factory, container) -> None:
    a = await make_tenant(factory)
    b = await make_tenant(factory)
    victim = await factory.user(b.org_id)
    headers = await login(client, a.admin)
    attempts = [
        ("GET", f"/api/v1/users/{victim.id}", None),
        ("PATCH", f"/api/v1/users/{victim.id}", {"status": "disabled"}),
        ("PATCH", f"/api/v1/users/{b.admin.id}", {"role": "employee"}),
        ("POST", f"/api/v1/users/{victim.id}/revoke-sessions", None),
        ("POST", f"/api/v1/users/{victim.id}/send-reset", None),
        ("PATCH", f"/api/v1/departments/{b.finance}", {"name": "Pwned"}),
        ("DELETE", f"/api/v1/departments/{b.hr}", None),
    ]
    for method, path, body in attempts:
        kwargs = {"json": body} if body is not None else {}
        response = await client.request(method, path, headers=headers, **kwargs)
        assert response.status_code == 404, (method, path)
    listing = await client.get("/api/v1/users", headers=headers, params={"q": victim.email})
    assert listing.json()["items"] == []
    row = await load_user(container, b.org_id, victim.id)
    assert row.status == "active"
    departments = await client.get("/api/v1/departments", headers=await login(client, b.admin))
    assert {d["name"] for d in departments.json()["items"]} == {"Finance", "Human Resources"}


# --------------------------------------------------------------------------- #
# Organisation settings cannot relax deployment limits
# --------------------------------------------------------------------------- #
async def test_org_settings_cannot_raise_deployment_ceilings(client, factory, container) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    deployment_budget = container.settings.llm.monthly_token_budget_per_org
    for payload in (
        {"llm": {"external_max_classification": "RESTRICTED"}},
        {"llm": {"monthly_token_budget": deployment_budget + 1}},
        {"retention": {"audit_days": container.settings.retention.audit_days - 1}},
        {"llm": {"monthly_token_budget": 0}},
        {"llm": {"provider": "anthropic"}},
        {"exports": {"max_rows": 10_001}},
    ):
        response = await client.patch("/api/v1/organization/settings", headers=admin, json=payload)
        assert response.status_code == 422, payload
    stricter = await client.patch(
        "/api/v1/organization/settings",
        headers=admin,
        json={
            "llm": {"external_max_classification": "INTERNAL", "monthly_token_budget": 1000},
            "retention": {"conversation_days": 7},
            "exports": {"max_rows": 500},
        },
    )
    assert stricter.status_code == 200, stricter.text
    body = stricter.json()
    assert body["effective"]["external_max_classification"] == "INTERNAL"
    assert body["effective"]["monthly_token_budget"] == 1000
    assert body["effective"]["retention"]["conversation_days"] == 7
    assert body["effective"]["export_max_rows"] == 500
    assert body["deployment"]["external_max_classification"] == "CONFIDENTIAL"
    # null clears one override and leaves the others
    cleared = await client.patch(
        "/api/v1/organization/settings", headers=admin, json={"llm": {"monthly_token_budget": None}}
    )
    assert cleared.json()["settings"]["llm"] == {
        "external_max_classification": "INTERNAL",
        "monthly_token_budget": None,
    }
    assert cleared.json()["effective"]["monthly_token_budget"] == deployment_budget
    changes = await audit_events(container, t.org_id, action="admin.org_settings_changed")
    assert changes[-1].details["before"] == {"llm.monthly_token_budget": 1000}
    employee = await login(client, await factory.user(t.org_id))
    assert (
        await client.patch(
            "/api/v1/organization/settings",
            headers=employee,
            json={"llm": {"monthly_token_budget": 5}},
        )
    ).status_code == 403


async def test_ai_settings_require_llm_configure(factory, container, monkeypatch) -> None:
    t = await make_tenant(factory)
    principal = await factory.principal(t.admin)
    monkeypatch.setitem(
        ROLE_PERMISSIONS,
        Role.ORGANIZATION_ADMIN,
        ROLE_PERMISSIONS[Role.ORGANIZATION_ADMIN] - {Permission.LLM_CONFIGURE},
    )
    with pytest.raises(PermissionDenied):
        await container.admin.update_organization_settings(
            principal, OrgSettingsPatch.model_validate({"llm": {"monthly_token_budget": 10}})
        )
    ok = await container.admin.update_organization_settings(
        principal, OrgSettingsPatch.model_validate({"retention": {"job_days": 5}})
    )
    assert ok.settings.retention.job_days == 5
