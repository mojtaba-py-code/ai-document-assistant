"""Platform operator flows: organisation provisioning, listing, renaming and suspension."""

from __future__ import annotations

import uuid

import pytest

from tests.conftest import login
from tests.helpers_admin import (
    audit_events,
    emails_to,
    invited_user,
    make_tenant,
    set_password_via_link,
)

pytestmark = [pytest.mark.db]


def _org_payload(**overrides: str) -> dict[str, str]:
    suffix = uuid.uuid4().hex[:8]
    return {
        "slug": f"org-{suffix}",
        "name": f"Org {suffix}",
        "admin_email": f"founder-{suffix}@example.test",
        "admin_name": "Founding Admin",
        **overrides,
    }


async def _operator_headers(client, factory) -> dict[str, str]:
    return await login(client, await factory.user(None, "platform_admin"))


async def test_create_organization_invites_first_admin(client, factory, container) -> None:
    headers = await _operator_headers(client, factory)
    payload = _org_payload()
    response = await client.post("/api/v1/platform/organizations", headers=headers, json=payload)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["organization"]["slug"] == payload["slug"]
    assert body["organization"]["status"] == "active"
    assert body["invitation_sent"] is True
    assert "password" not in response.text.lower()
    org_id = uuid.UUID(body["organization"]["id"])

    # the invited admin chooses a password through the emailed link, then administers the org
    assert len(emails_to(container, payload["admin_email"])) == 1
    password = await set_password_via_link(client, container, payload["admin_email"])
    admin = invited_user(
        {
            "id": body["admin_user_id"],
            "email": payload["admin_email"],
            "role": "organization_admin",
        },
        org_id,
        password,
    )
    admin_headers = await login(client, admin)
    me = (await client.get("/api/v1/auth/me", headers=admin_headers)).json()
    assert me["organization_id"] == str(org_id) and me["role"] == "organization_admin"
    users = await client.get("/api/v1/users", headers=admin_headers)
    assert [u["id"] for u in users.json()["items"]] == [body["admin_user_id"]]
    assert users.json()["items"][0]["clearance"] == "RESTRICTED"

    platform_events = await audit_events(container, None, action="platform.organization_created")
    assert str(org_id) in {e.resource_id for e in platform_events}
    tenant_events = await audit_events(container, org_id, action="admin.user_created")
    assert tenant_events[0].details["first_admin"] is True


async def test_create_organization_validation_and_conflicts(client, factory) -> None:
    headers = await _operator_headers(client, factory)
    first = _org_payload()
    assert (
        await client.post("/api/v1/platform/organizations", headers=headers, json=first)
    ).status_code == 201
    same_slug = _org_payload(slug=first["slug"])
    assert (
        await client.post("/api/v1/platform/organizations", headers=headers, json=same_slug)
    ).status_code == 409
    same_email = _org_payload(admin_email=first["admin_email"])
    conflict = await client.post("/api/v1/platform/organizations", headers=headers, json=same_email)
    assert conflict.status_code == 409
    # the failed attempt left no half-created organisation behind
    listing = await client.get(
        "/api/v1/platform/organizations", headers=headers, params={"q": same_email["slug"]}
    )
    assert listing.json()["items"] == []
    for bad in (
        _org_payload(slug="Bad Slug"),
        _org_payload(slug="-leading"),
        _org_payload(admin_email="nope"),
        _org_payload(name="   "),
        {**_org_payload(), "status": "active"},
    ):
        response = await client.post("/api/v1/platform/organizations", headers=headers, json=bad)
        assert response.status_code == 422, bad


async def test_list_get_and_rename(client, factory) -> None:
    headers = await _operator_headers(client, factory)
    marker = uuid.uuid4().hex[:6]
    created = []
    for i in range(3):
        payload = _org_payload(slug=f"list-{marker}-{i}", name=f"Listing {marker} {i}")
        created.append(
            (
                await client.post("/api/v1/platform/organizations", headers=headers, json=payload)
            ).json()
        )
    seen: list[str] = []
    cursor = None
    while True:
        params: dict[str, object] = {"q": f"list-{marker}", "limit": 2}
        if cursor:
            params["cursor"] = cursor
        page = (
            await client.get("/api/v1/platform/organizations", headers=headers, params=params)
        ).json()
        seen += [o["id"] for o in page["items"]]
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert sorted(seen) == sorted(c["organization"]["id"] for c in created)

    org_id = created[0]["organization"]["id"]
    got = await client.get(f"/api/v1/platform/organizations/{org_id}", headers=headers)
    assert got.status_code == 200 and got.json()["slug"] == f"list-{marker}-0"
    renamed = await client.patch(
        f"/api/v1/platform/organizations/{org_id}", headers=headers, json={"name": "Renamed Co"}
    )
    assert renamed.json()["name"] == "Renamed Co"
    assert (
        await client.get(f"/api/v1/platform/organizations/{uuid.uuid4()}", headers=headers)
    ).status_code == 404
    assert (
        await client.patch(f"/api/v1/platform/organizations/{org_id}", headers=headers, json={})
    ).status_code == 422


async def test_suspension_revokes_sessions_and_blocks_sign_in(client, factory, container) -> None:
    headers = await _operator_headers(client, factory)
    t = await make_tenant(factory)
    employee = await factory.user(t.org_id)
    tenant_headers = [await login(client, t.admin), await login(client, employee)]

    suspended = await client.patch(
        f"/api/v1/platform/organizations/{t.org_id}", headers=headers, json={"status": "suspended"}
    )
    assert suspended.status_code == 200 and suspended.json()["status"] == "suspended"
    for h in tenant_headers:
        assert (await client.get("/api/v1/auth/me", headers=h)).status_code == 401
    refused = await client.post(
        "/api/v1/auth/login", json={"email": employee.email, "password": employee.password}
    )
    assert refused.status_code == 401
    filtered = await client.get(
        "/api/v1/platform/organizations", headers=headers, params={"status": "suspended"}
    )
    assert str(t.org_id) in {o["id"] for o in filtered.json()["items"]}

    events = await audit_events(container, t.org_id, action="organization.suspended")
    assert events[0].details["revoked_sessions"] == 2

    reactivated = await client.patch(
        f"/api/v1/platform/organizations/{t.org_id}", headers=headers, json={"status": "active"}
    )
    assert reactivated.json()["status"] == "active"
    assert (
        await client.get("/api/v1/auth/me", headers=await login(client, employee))
    ).status_code == 200
    # old sessions stay dead after reactivation
    assert (await client.get("/api/v1/auth/me", headers=tenant_headers[1])).status_code == 401


async def test_openapi_documents_the_admin_routes(client) -> None:
    schema = (await client.get("/openapi.json")).json()
    paths = set(schema["paths"])
    for path in (
        "/api/v1/platform/organizations",
        "/api/v1/platform/organizations/{organization_id}",
        "/api/v1/organization",
        "/api/v1/organization/settings",
        "/api/v1/departments",
        "/api/v1/departments/{department_id}",
        "/api/v1/users",
        "/api/v1/users/{user_id}",
        "/api/v1/users/{user_id}/revoke-sessions",
        "/api/v1/users/{user_id}/send-reset",
        "/api/v1/usage",
        "/api/v1/admin/health",
        "/api/v1/audit/events",
        "/api/v1/audit/verify",
        "/api/v1/jobs",
        "/api/v1/jobs/{job_id}/retry",
        "/api/v1/jobs/{job_id}/cancel",
    ):
        assert path in paths, path
    list_params = {p["name"] for p in schema["paths"]["/api/v1/audit/events"]["get"]["parameters"]}
    assert {
        "action",
        "actor",
        "resource_type",
        "resource_id",
        "outcome",
        "from",
        "to",
        "cursor",
    } <= list_params
