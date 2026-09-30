"""Authorization of the documents API.

Tenant isolation (IDOR), departmental confidentiality, read vs manage, clearance ceilings,
the organisation-admin RESTRICTED rule, ``allowed_roles``, quarantine visibility and the RBAC
gate for auditors and platform administrators.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from docassist.documents.scanning import EICAR_TEST_SIGNATURE
from tests.conftest import login
from tests.helpers_documents import audit_actions, make_ready, make_tenant, unique_text, upload_file

pytestmark = [pytest.mark.db]

URL = "/api/v1/documents"


async def _statuses(client: Any, headers: dict[str, str], doc_id: str) -> dict[str, int]:
    """Status code of every per-document endpoint for this caller (delete last)."""
    grant = {"grantee_type": "role", "role": "employee"}
    file = {"file": ("v2.txt", unique_text(), "text/plain")}
    out = {
        "get": await client.get(f"{URL}/{doc_id}", headers=headers),
        "patch": await client.patch(f"{URL}/{doc_id}", headers=headers, json={"title": "pwned"}),
        "versions": await client.post(f"{URL}/{doc_id}/versions", headers=headers, files=file),
        "download": await client.get(f"{URL}/{doc_id}/download", headers=headers),
        "download_v1": await client.get(f"{URL}/{doc_id}/versions/1/download", headers=headers),
        "content": await client.get(f"{URL}/{doc_id}/content", headers=headers),
        "grants": await client.get(f"{URL}/{doc_id}/grants", headers=headers),
        "grant_add": await client.post(f"{URL}/{doc_id}/grants", headers=headers, json=grant),
        "grant_revoke": await client.delete(
            f"{URL}/{doc_id}/grants/{uuid.uuid4()}", headers=headers
        ),
        "delete": await client.delete(f"{URL}/{doc_id}", headers=headers),
    }
    return {name: response.status_code for name, response in out.items()}


async def _listed(client: Any, headers: dict[str, str]) -> set[str]:
    return {d["id"] for d in (await client.get(URL, headers=headers)).json()["items"]}


async def _ready_doc(client: Any, container: Any, tenant: Any, headers: Any, **fields: Any) -> str:
    response = await upload_file(client, headers, unique_text(), "doc.txt", **fields)
    assert response.status_code == 201, response.text
    doc_id: str = response.json()["document_id"]
    await make_ready(container, tenant.org, uuid.UUID(doc_id))
    return doc_id


async def test_other_organisation_sees_nothing(client, factory, container) -> None:
    tenant = await make_tenant(factory)
    doc_id = await _ready_doc(client, container, tenant, await login(client, tenant.alice))
    intruder_org = await make_tenant(factory)
    for user in (intruder_org.admin, intruder_org.manager, intruder_org.alice):
        headers = await login(client, user)
        assert set((await _statuses(client, headers, doc_id)).values()) == {404}
        assert doc_id not in await _listed(client, headers)


async def test_confidential_document_of_another_department_is_invisible(
    client, factory, container
) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    doc_id = await _ready_doc(
        client,
        container,
        tenant,
        alice,
        classification="CONFIDENTIAL",
        department_id=tenant.finance,
    )
    bob = await login(client, tenant.bob)
    assert set((await _statuses(client, bob, doc_id)).values()) == {404}
    assert doc_id not in await _listed(client, bob)
    carol = await login(client, tenant.carol)  # same department: readable
    assert (await client.get(f"{URL}/{doc_id}/download", headers=carol)).status_code == 200


async def test_employee_can_read_but_not_manage_a_colleagues_document(
    client, factory, container
) -> None:
    tenant = await make_tenant(factory)
    doc_id = await _ready_doc(client, container, tenant, await login(client, tenant.alice))
    statuses = await _statuses(client, await login(client, tenant.carol), doc_id)
    assert statuses == {
        "get": 200,
        "patch": 403,
        "versions": 403,
        "download": 200,
        "download_v1": 200,
        "content": 200,
        "grants": 403,
        "grant_add": 403,
        "grant_revoke": 403,
        "delete": 403,
    }
    detail = (await client.get(f"{URL}/{doc_id}", headers=await login(client, tenant.alice))).json()
    assert detail["title"] != "pwned"
    denials = [
        d["operation"]
        for a, o, d in await audit_actions(container, tenant.org)
        if a == "document.access_denied"
    ]
    assert {"update", "add_version", "delete", "add_grant"} <= set(denials)


async def test_department_manager_manages_only_their_departments(
    client, factory, container
) -> None:
    tenant = await make_tenant(factory)
    manager = await login(client, tenant.manager)
    finance_doc = await _ready_doc(
        client, container, tenant, await login(client, tenant.alice), department_id=tenant.finance
    )
    hr_doc = await _ready_doc(
        client, container, tenant, await login(client, tenant.bob), department_id=tenant.hr
    )
    ok = await client.patch(f"{URL}/{finance_doc}", headers=manager, json={"tags": ["reviewed"]})
    assert ok.status_code == 200 and ok.json()["can_manage"]
    assert (
        await client.patch(f"{URL}/{hr_doc}", headers=manager, json={"tags": ["x"]})
    ).status_code == 403
    moved = await client.patch(
        f"{URL}/{finance_doc}", headers=manager, json={"department_id": str(tenant.hr)}
    )
    assert moved.status_code == 403  # cannot move documents into a department they don't manage


async def test_clearance_and_department_rules_on_upload_and_update(
    client, factory, container
) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    above = await upload_file(
        client, alice, unique_text(), "secret.txt", classification="RESTRICTED"
    )
    assert above.status_code == 403
    foreign = await upload_file(client, alice, unique_text(), "x.txt", department_id=tenant.hr)
    assert foreign.status_code == 403
    admin = await login(client, tenant.admin)
    ghost = await upload_file(client, admin, unique_text(), "x.txt", department_id=uuid.uuid4())
    assert ghost.status_code == 422
    bad_role = await upload_file(
        client, alice, unique_text(), "x.txt", allowed_roles="platform_admin"
    )
    assert bad_role.status_code == 422
    assert await _listed(client, alice) == set()  # nothing was created

    doc_id = await _ready_doc(client, container, tenant, alice)
    raised = await client.patch(
        f"{URL}/{doc_id}", headers=alice, json={"classification": "RESTRICTED"}
    )
    assert raised.status_code == 403
    by_admin = await client.patch(
        f"{URL}/{doc_id}", headers=admin, json={"classification": "RESTRICTED"}
    )
    assert by_admin.status_code == 200
    # the owner is now above their own clearance: the document disappears for them
    assert (await client.get(f"{URL}/{doc_id}", headers=alice)).status_code == 404


async def test_org_admin_manages_restricted_documents_but_needs_a_grant_to_read(
    client, factory, container
) -> None:
    tenant = await make_tenant(factory)
    manager = await login(client, tenant.manager)
    admin = await login(client, tenant.admin)
    doc_id = await _ready_doc(
        client,
        container,
        tenant,
        manager,
        classification="RESTRICTED",
        department_id=tenant.finance,
    )
    detail = (await client.get(f"{URL}/{doc_id}", headers=admin)).json()
    assert detail["title"] and detail["can_manage"] and not detail["can_read"]
    assert detail["metadata"] == {}  # file-derived metadata is content: readers only
    assert (await client.get(f"{URL}/{doc_id}/download", headers=admin)).status_code == 403
    assert (await client.get(f"{URL}/{doc_id}/content", headers=admin)).status_code == 403
    alice = await login(client, tenant.alice)
    assert (
        await client.get(f"{URL}/{doc_id}", headers=alice)
    ).status_code == 404  # above clearance

    grant = {"grantee_type": "user", "user_id": str(tenant.admin.id)}
    assert (
        await client.post(f"{URL}/{doc_id}/grants", headers=admin, json=grant)
    ).status_code == 201
    assert (await client.get(f"{URL}/{doc_id}/download", headers=admin)).status_code == 200
    [added] = [
        d for a, _o, d in await audit_actions(container, tenant.org) if a == "document.grant_added"
    ]
    assert added["self_grant"] is True


async def test_allowed_roles_are_a_hard_filter(client, factory, container) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    doc_id = await _ready_doc(
        client,
        container,
        tenant,
        alice,
        allowed_roles="department_manager",
        department_id=tenant.finance,
    )
    assert (
        await client.get(f"{URL}/{doc_id}", headers=await login(client, tenant.carol))
    ).status_code == 404
    owner_view = (await client.get(f"{URL}/{doc_id}", headers=alice)).json()
    assert owner_view["can_manage"] and not owner_view["can_read"]  # even the owner
    manager = await login(client, tenant.manager)
    assert (await client.get(f"{URL}/{doc_id}/download", headers=manager)).status_code == 200


async def test_auditors_and_platform_admins_have_no_document_access(client, factory) -> None:
    tenant = await make_tenant(factory)
    auditor = await login(client, tenant.auditor)
    assert (await client.get(URL, headers=auditor)).status_code == 403
    assert (await upload_file(client, auditor, unique_text(), "a.txt")).status_code == 403
    platform = await factory.user(None, "platform_admin")
    platform_headers = await login(client, platform)
    assert (await client.get(URL, headers=platform_headers)).status_code == 403
    assert (
        await client.post(URL, files={"file": ("a.txt", b"x", "text/plain")})
    ).status_code == 401


async def test_quarantined_documents_are_visible_only_to_managers(
    client, factory, container
) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    carol = await login(client, tenant.carol)
    admin = await login(client, tenant.admin)
    upload = await upload_file(client, alice, b"note " + EICAR_TEST_SIGNATURE, "note.txt")
    assert upload.status_code == 201 and upload.json()["status"] == "quarantined"
    doc_id = upload.json()["document_id"]

    # an INTERNAL document is normally visible to every employee - but not while quarantined
    assert doc_id not in await _listed(client, carol)
    assert (await client.get(f"{URL}/{doc_id}", headers=carol)).status_code == 404
    assert (await client.get(f"{URL}/{doc_id}/download", headers=carol)).status_code == 404

    owner_view = (await client.get(f"{URL}/{doc_id}", headers=alice)).json()
    assert owner_view["status"] == "quarantined" and owner_view["can_manage"]
    codes = {f["code"] for f in owner_view["versions"][0]["findings"]}
    assert "eicar_test_signature" in codes

    refused = await client.get(f"{URL}/{doc_id}/download", headers=alice)
    assert refused.status_code == 403 and refused.json()["reason"] == "acknowledge_risk_required"
    for headers in (alice, admin):
        risky = await client.get(
            f"{URL}/{doc_id}/download", headers=headers, params={"acknowledge_risk": "true"}
        )
        assert risky.status_code == 200
        assert risky.content == b"note " + EICAR_TEST_SIGNATURE
        assert risky.headers["x-docassist-quarantined"] == "true"
        assert risky.headers["content-disposition"].startswith("attachment;")
    downloads = [
        d for a, _o, d in await audit_actions(container, tenant.org) if a == "document.download"
    ]
    assert [d["acknowledged_risk"] for d in downloads] == [True, True]


async def test_quarantined_restricted_file_needs_content_entitlement(client, factory) -> None:
    tenant = await make_tenant(factory)
    manager = await login(client, tenant.manager)
    admin = await login(client, tenant.admin)
    upload = await upload_file(
        client,
        manager,
        EICAR_TEST_SIGNATURE,
        "x.txt",
        classification="RESTRICTED",
        department_id=tenant.finance,
    )
    doc_id = upload.json()["document_id"]
    risky = await client.get(
        f"{URL}/{doc_id}/download", headers=admin, params={"acknowledge_risk": "true"}
    )
    assert risky.status_code == 403  # managing a RESTRICTED file is not reading it
    owner = await client.get(
        f"{URL}/{doc_id}/download", headers=manager, params={"acknowledge_risk": "true"}
    )
    assert owner.status_code == 200
