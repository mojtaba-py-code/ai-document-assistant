"""User and department administration through the real HTTP API and database."""

from __future__ import annotations

import uuid

import pytest

from docassist.db.models import Document
from docassist.db.session import DbContext
from tests.conftest import login
from tests.helpers_admin import (
    audit_events,
    emails_to,
    invited_user,
    load_user,
    login_pair,
    make_tenant,
    set_password_via_link,
)

pytestmark = [pytest.mark.db]


def _email() -> str:
    return f"new-{uuid.uuid4().hex[:10]}@example.test"


# --------------------------------------------------------------------------- #
# Create / invite
# --------------------------------------------------------------------------- #
async def test_create_user_invites_by_email_without_any_password(
    client, factory, container
) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    email = _email()
    response = await client.post(
        "/api/v1/users",
        headers=admin,
        json={
            "email": email.upper(),
            "full_name": "  New" + chr(0x200B) + "   Person ",
            "role": "employee",
            "departments": [{"department_id": str(t.finance)}],
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["email"] == email  # normalised
    assert body["full_name"] == "New Person"  # invisible characters removed, whitespace collapsed
    assert body["clearance"] == "CONFIDENTIAL"  # role default
    assert body["departments"][0]["department_id"] == str(t.finance)
    assert "password" not in response.text.lower()
    assert len(emails_to(container, email)) == 1

    # before choosing a password the account behaves exactly like an unknown one
    invited = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "Guess-1234-abcd"}
    )
    unknown = await client.post(
        "/api/v1/auth/login", json={"email": _email(), "password": "Guess-1234-abcd"}
    )
    assert invited.status_code == unknown.status_code == 401
    assert invited.json()["detail"] == unknown.json()["detail"]
    password = await set_password_via_link(client, container, email)
    user = invited_user(body, t.org_id, password)
    me = await client.get("/api/v1/auth/me", headers=await login(client, user))
    assert me.json()["department_ids"] == [str(t.finance)]

    created = await audit_events(container, t.org_id, action="admin.user_created")
    assert [e.resource_id for e in created] == [body["id"]]
    assert created[0].actor_user_id == t.admin.id


async def test_create_user_rejects_duplicates_and_bad_input(client, factory) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    existing = await factory.user(t.org_id)
    duplicate = await client.post(
        "/api/v1/users",
        headers=admin,
        json={"email": existing.email, "full_name": "Dup", "role": "employee"},
    )
    assert duplicate.status_code == 409
    other_org = await factory.org()
    foreign = await factory.user(other_org)  # emails are globally unique
    assert (
        await client.post(
            "/api/v1/users",
            headers=admin,
            json={"email": foreign.email, "full_name": "X", "role": "employee"},
        )
    ).status_code == 409
    for payload in (
        {"email": "not-an-email", "full_name": "X", "role": "employee"},
        {"email": _email(), "full_name": "   ", "role": "employee"},
        {"email": _email(), "full_name": "X", "role": "superuser"},
        {"email": _email(), "full_name": "X", "role": "employee", "password": "hunter2hunter2"},
        {
            "email": _email(),
            "full_name": "X",
            "role": "employee",
            "departments": [{"department_id": str(t.hr)}, {"department_id": str(t.hr)}],
        },
    ):
        response = await client.post("/api/v1/users", headers=admin, json=payload)
        assert response.status_code == 422, payload
        assert "hunter2" not in response.text


async def test_department_memberships_must_belong_to_the_org(client, factory) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    other = await make_tenant(factory)
    for dept in (other.finance, uuid.uuid4()):
        response = await client.post(
            "/api/v1/users",
            headers=admin,
            json={
                "email": _email(),
                "full_name": "X",
                "role": "department_manager",
                "departments": [{"department_id": str(dept), "is_manager": True}],
            },
        )
        # foreign and non-existent departments are indistinguishable
        assert response.status_code == 422
        assert response.json()["detail"] == "One or more departments do not exist."


async def test_only_department_managers_can_manage_a_department(client, factory) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    bad = await client.post(
        "/api/v1/users",
        headers=admin,
        json={
            "email": _email(),
            "full_name": "X",
            "role": "employee",
            "departments": [{"department_id": str(t.finance), "is_manager": True}],
        },
    )
    assert bad.status_code == 422
    ok = await client.post(
        "/api/v1/users",
        headers=admin,
        json={
            "email": _email(),
            "full_name": "Mgr",
            "role": "department_manager",
            "departments": [{"department_id": str(t.finance), "is_manager": True}],
        },
    )
    assert ok.status_code == 201
    manager_id = ok.json()["id"]
    # demoting a manager without sending departments ends their management
    demoted = await client.patch(
        f"/api/v1/users/{manager_id}", headers=admin, json={"role": "employee"}
    )
    assert demoted.status_code == 200
    assert demoted.json()["departments"] == [
        {"department_id": str(t.finance), "department_name": "Finance", "is_manager": False}
    ]


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
async def test_list_filters_search_and_cursor(client, factory) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    percent = await factory.user(t.org_id, email=f"a%b-{uuid.uuid4().hex[:6]}@example.test")
    finance_people = [await factory.user(t.org_id, departments=[t.finance]) for _ in range(3)]
    auditor = await factory.user(t.org_id, "auditor")

    by_role = await client.get("/api/v1/users", headers=admin, params={"role": "auditor"})
    assert [u["id"] for u in by_role.json()["items"]] == [str(auditor.id)]

    by_dept = await client.get(
        "/api/v1/users", headers=admin, params={"department_id": str(t.finance)}
    )
    assert {u["id"] for u in by_dept.json()["items"]} == {str(u.id) for u in finance_people}

    # LIKE wildcards in the search term are literal
    search = await client.get("/api/v1/users", headers=admin, params={"q": "a%b"})
    assert [u["id"] for u in search.json()["items"]] == [str(percent.id)]

    seen: list[str] = []
    cursor = None
    while True:
        params = {"limit": 2, **({"cursor": cursor} if cursor else {})}
        page = (await client.get("/api/v1/users", headers=admin, params=params)).json()
        seen.extend(u["id"] for u in page["items"])
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert len(seen) == len(set(seen)) == 6  # admin + percent + 3 finance + auditor

    bad_cursor = await client.get("/api/v1/users", headers=admin, params={"cursor": "!!nope"})
    assert bad_cursor.status_code == 422


async def test_get_user_and_readers(client, factory) -> None:
    t = await make_tenant(factory)
    target = await factory.user(t.org_id, departments=[t.hr])
    for role in ("organization_admin", "department_manager", "auditor"):
        reader = await factory.user(t.org_id, role)
        response = await client.get(
            f"/api/v1/users/{target.id}", headers=await login(client, reader)
        )
        assert response.status_code == 200, role
        assert response.json()["departments"][0]["department_name"] == "Human Resources"
    employee = await factory.user(t.org_id)
    assert (
        await client.get(f"/api/v1/users/{target.id}", headers=await login(client, employee))
    ).status_code == 403
    missing = await client.get(
        f"/api/v1/users/{uuid.uuid4()}", headers=await login(client, t.admin)
    )
    assert missing.status_code == 404


# --------------------------------------------------------------------------- #
# Update, disable, sessions, reset
# --------------------------------------------------------------------------- #
async def test_role_change_bumps_token_version_and_forces_relogin(
    client, factory, container
) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    employee = await factory.user(t.org_id, departments=[t.finance])
    old_headers, old_refresh = await login_pair(client, employee)
    before = await load_user(container, t.org_id, employee.id)

    response = await client.patch(
        f"/api/v1/users/{employee.id}",
        headers=admin,
        json={
            "role": "department_manager",
            "departments": [{"department_id": str(t.finance), "is_manager": True}],
        },
    )
    assert response.status_code == 200, response.text
    after = await load_user(container, t.org_id, employee.id)
    assert after.token_version == before.token_version + 1

    assert (await client.get("/api/v1/auth/me", headers=old_headers)).status_code == 401
    refresh = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": old_refresh, "token_transport": "body"}
    )
    assert refresh.status_code == 401
    me = await client.get("/api/v1/auth/me", headers=await login(client, employee))
    assert me.json()["role"] == "department_manager"
    assert me.json()["managed_department_ids"] == [str(t.finance)]

    actions = [e.action for e in await audit_events(container, t.org_id)]
    assert "admin.role_changed" in actions
    assert "admin.user_departments_changed" in actions
    assert "admin.sessions_revoked" in actions
    changed = await audit_events(container, t.org_id, action="admin.role_changed")
    assert changed[-1].details == {"before": "employee", "after": "department_manager"}


async def test_name_change_does_not_force_relogin(client, factory, container) -> None:
    t = await make_tenant(factory)
    employee = await factory.user(t.org_id)
    user_headers = await login(client, employee)
    response = await client.patch(
        f"/api/v1/users/{employee.id}",
        headers=await login(client, t.admin),
        json={"full_name": "Renamed Person"},
    )
    assert response.json()["full_name"] == "Renamed Person"
    assert (await client.get("/api/v1/auth/me", headers=user_headers)).status_code == 200


async def test_disable_and_enable(client, factory) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    employee = await factory.user(t.org_id)
    user_headers = await login(client, employee)
    disabled = await client.patch(
        f"/api/v1/users/{employee.id}", headers=admin, json={"status": "disabled"}
    )
    assert disabled.json()["status"] == "disabled"
    assert (await client.get("/api/v1/auth/me", headers=user_headers)).status_code == 401
    refused = await client.post(
        "/api/v1/auth/login", json={"email": employee.email, "password": employee.password}
    )
    assert refused.status_code == 401
    reset = await client.post(f"/api/v1/users/{employee.id}/send-reset", headers=admin)
    assert reset.status_code == 409
    enabled = await client.patch(
        f"/api/v1/users/{employee.id}", headers=admin, json={"status": "active"}
    )
    assert enabled.json()["status"] == "active"
    assert (
        await client.get("/api/v1/auth/me", headers=await login(client, employee))
    ).status_code == 200


async def test_revoke_sessions_and_send_reset(client, factory, container) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    employee = await factory.user(t.org_id)
    first = await login(client, employee)
    second = await login(client, employee)
    revoked = await client.post(f"/api/v1/users/{employee.id}/revoke-sessions", headers=admin)
    assert revoked.status_code == 200
    assert revoked.json()["revoked_sessions"] == 2
    for h in (first, second):
        assert (await client.get("/api/v1/auth/me", headers=h)).status_code == 401

    sent = await client.post(f"/api/v1/users/{employee.id}/send-reset", headers=admin)
    assert sent.status_code == 202
    assert sent.json() == {"email_sent": True}
    employee.password = await set_password_via_link(client, container, employee.email)
    assert (
        await client.get("/api/v1/auth/me", headers=await login(client, employee))
    ).status_code == 200
    assert await audit_events(container, t.org_id, action="admin.password_reset_sent")


async def test_update_rejects_empty_and_unknown_fields(client, factory) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    employee = await factory.user(t.org_id)
    for payload in ({}, {"email": "x@example.test"}, {"token_version": 99}):
        response = await client.patch(f"/api/v1/users/{employee.id}", headers=admin, json=payload)
        assert response.status_code == 422, payload


# --------------------------------------------------------------------------- #
# Departments
# --------------------------------------------------------------------------- #
async def test_department_crud(client, factory, container) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    created = await client.post(
        "/api/v1/departments",
        headers=admin,
        json={"name": "Research & D" + chr(0xE9) + "veloppement", "description": "R&D"},
    )
    assert created.status_code == 201, created.text
    dept = created.json()
    assert dept["slug"] == "research-developpement"
    duplicate = await client.post(
        "/api/v1/departments", headers=admin, json={"name": "Other", "slug": dept["slug"]}
    )
    assert duplicate.status_code == 409
    bad_slug = await client.post(
        "/api/v1/departments", headers=admin, json={"name": "Bad", "slug": "Not A Slug!"}
    )
    assert bad_slug.status_code == 422

    await factory.user(t.org_id, departments=[uuid.UUID(dept["id"])])
    updated = await client.patch(
        f"/api/v1/departments/{dept['id']}",
        headers=admin,
        json={"name": "Research", "description": None},
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "Research"
    assert updated.json()["description"] is None
    assert updated.json()["member_count"] == 1

    listing = await client.get("/api/v1/departments", headers=admin)
    names = {d["name"]: d for d in listing.json()["items"]}
    assert set(names) == {"Finance", "Human Resources", "Research"}
    assert names["Research"]["member_count"] == 1

    deleted = await client.delete(f"/api/v1/departments/{dept['id']}", headers=admin)
    assert deleted.status_code == 204
    assert (
        await client.delete(f"/api/v1/departments/{dept['id']}", headers=admin)
    ).status_code == 404
    actions = [e.action for e in await audit_events(container, t.org_id)]
    for action in (
        "admin.department_created",
        "admin.department_updated",
        "admin.department_deleted",
    ):
        assert action in actions


async def test_department_with_documents_cannot_be_deleted(client, factory, container) -> None:
    t = await make_tenant(factory)
    admin = await login(client, t.admin)
    async with container.db.transaction(DbContext(org_id=t.org_id)) as session:
        session.add(
            Document(
                organization_id=t.org_id,
                department_id=t.finance,
                owner_id=t.admin.id,
                title="Budget",
                classification="INTERNAL",
                doc_type="financial",
                status="deleted",  # even soft-deleted documents keep the department
            )
        )
    response = await client.delete(f"/api/v1/departments/{t.finance}", headers=admin)
    assert response.status_code == 409
    assert response.json()["document_count"] == 1
