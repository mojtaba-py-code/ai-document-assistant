"""Documents API end to end: upload per format, listing, detail, updates, versions,
downloads, preview content, deletion, legal hold, grants and duplicate detection."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy import update

from docassist.core.enums import JobStatus
from docassist.db.models import Document
from docassist.db.session import DbContext
from docassist.documents.storage import storage_context
from tests.conftest import login
from tests.fixtures import sample_files as sf
from tests.helpers_documents import (
    audit_actions,
    jobs_for,
    make_ready,
    make_tenant,
    row_counts,
    unique_text,
    upload_file,
    version_row,
)

pytestmark = [pytest.mark.db]

URL = "/api/v1/documents"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _samples() -> dict[str, tuple[bytes, str]]:
    tag = uuid.uuid4().hex
    return {
        "pdf": (sf.build_pdf([f"Invoice {tag}"]), "application/pdf"),
        "docx": (sf.build_docx([f"Contract {tag}"]), DOCX_MIME),
        "xlsx": (sf.build_xlsx([("id", tag), ("amount", 10)]), XLSX_MIME),
        "csv": (f"id,amount\n{tag},10\n".encode(), "text/csv"),
        "txt": (f"Plain text {tag}\n".encode(), "text/plain"),
        "md": (f"# Heading {tag}\n\nBody.\n".encode("utf-16"), "text/markdown"),
    }


@pytest.mark.parametrize("fmt", ["pdf", "docx", "xlsx", "csv", "txt", "md"])
async def test_upload_happy_path_per_format(client, factory, container, fmt: str) -> None:
    tenant = await make_tenant(factory)
    headers = await login(client, tenant.alice)
    content, mime = _samples()[fmt]
    response = await upload_file(
        client,
        headers,
        content,
        f"Quarterly {fmt}.{fmt}",
        classification="CONFIDENTIAL",
        department_id=tenant.finance,
        tags="Finance, Q3",
        doc_type="report",
        content_type="application/x-anything",
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert response.headers["location"] == f"{URL}/{body['document_id']}"
    assert body["status"] == "processing" and body["version_status"] == "uploaded"
    assert body["version_number"] == 1
    assert body["detected_mime"] == mime
    assert body["findings"] == ["malware_scan_skipped"]
    assert body["size_bytes"] == len(content)

    # transactional outbox: exactly one ingestion job whose payload carries ids only
    [job] = [j for j in await jobs_for(container, tenant.org) if j.kind == "ingest_version"]
    assert job.payload == {"version_id": body["version_id"]}
    assert job.idempotency_key == f"ingest:{body['version_id']}"
    assert str(job.id) == body["job_id"]

    version = await version_row(container, tenant.org, uuid.UUID(body["version_id"]))
    assert version.original_filename == f"Quarterly {fmt}.{fmt}"
    assert version.declared_mime == "application/x-anything"  # recorded, never trusted
    assert version.detected_mime == mime
    assert version.sha256 == body["sha256"]

    # stored encrypted at rest and bound to this version
    stored = Path(container.settings.storage.root, *version.storage_key.split("/")).read_bytes()
    assert stored.startswith(b"DAENC1") and content[:64] not in stored
    plain = await container.storage.read_bytes(
        version.storage_key, storage_context(tenant.org, "version", version.id), len(content)
    )
    assert plain == content

    detail = (await client.get(f"{URL}/{body['document_id']}", headers=headers)).json()
    assert detail["title"] == f"Quarterly {fmt}"
    assert detail["tags"] == ["finance", "q3"]
    assert detail["doc_type"] == "report" and detail["doc_type_source"] == "user"
    actions = [a for a, _o, _d in await audit_actions(container, tenant.org)]
    assert "document.upload" in actions


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #
async def test_list_filters_flags_and_keyset_pagination(client, factory) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    carol = await login(client, tenant.carol)
    titles = ["alpha 100% done", "beta_report", "gamma", "delta", "epsilon"]
    for index, title in enumerate(titles):
        response = await upload_file(
            client,
            alice if index % 2 == 0 else carol,
            unique_text(title),
            f"{title}.txt",
            title=title,
            tags="shared" if index < 3 else "other",
        )
        assert response.status_code == 201, response.text

    seen: list[str] = []
    cursor = None
    while True:
        params = {"limit": 2, **({"cursor": cursor} if cursor else {})}
        page = (await client.get(URL, headers=alice, params=params)).json()
        seen += [item["title"] for item in page["items"]]
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == list(reversed(titles))  # newest first, every document exactly once

    percent = (await client.get(URL, headers=alice, params={"q": "100%"})).json()["items"]
    assert [d["title"] for d in percent] == ["alpha 100% done"]  # % is literal
    underscore = (await client.get(URL, headers=alice, params={"q": "a_"})).json()["items"]
    # "_" is literal too: as a wildcard "a_" would also match "alpha..." and "gamma"
    assert [d["title"] for d in underscore] == ["beta_report"]
    shared = (await client.get(URL, headers=alice, params={"tag": "SHARED"})).json()["items"]
    assert {d["title"] for d in shared} == set(titles[:3])
    mine = (await client.get(URL, headers=alice, params={"owner": "me"})).json()["items"]
    assert {d["title"] for d in mine} == {"alpha 100% done", "gamma", "epsilon"}
    assert all(d["can_manage"] for d in mine)
    assert not any(d["can_read"] for d in mine)  # still processing: not content-readable
    everything = (await client.get(URL, headers=alice)).json()["items"]
    others = [d for d in everything if d["title"] not in {m["title"] for m in mine}]
    assert others and not any(d["can_manage"] for d in others)
    processing = (await client.get(URL, headers=alice, params={"status": "processing"})).json()
    assert len(processing["items"]) == 5
    assert (await client.get(URL, headers=alice, params={"status": "deleted"})).status_code == 422
    bad_cursor = await client.get(URL, headers=alice, params={"cursor": "not-base64!"})
    assert bad_cursor.status_code == 422
    assert (await client.get(URL, headers=alice, params={"limit": 101})).status_code == 422


# --------------------------------------------------------------------------- #
# Detail, update, versions
# --------------------------------------------------------------------------- #
async def test_detail_hides_grants_and_findings_from_non_managers(
    client, factory, container
) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    carol = await login(client, tenant.carol)
    doc_id = (await upload_file(client, alice, unique_text(), "notes.txt")).json()["document_id"]
    await make_ready(
        container, tenant.org, uuid.UUID(doc_id), metadata={"author": "Alice", "bad key": "x"}
    )

    owner_view = (await client.get(f"{URL}/{doc_id}", headers=alice)).json()
    assert owner_view["can_manage"] and owner_view["can_read"]
    assert owner_view["grants"] == []
    assert owner_view["versions"][0]["findings"][0]["code"] == "malware_scan_skipped"
    assert owner_view["metadata"] == {"author": "Alice"}  # unsafe keys dropped
    assert owner_view["ingestion"] == {"version_number": 1, "status": "indexed", "error_code": None}

    reader_view = (await client.get(f"{URL}/{doc_id}", headers=carol)).json()
    assert reader_view["can_read"] and not reader_view["can_manage"]
    assert reader_view["grants"] is None
    assert reader_view["versions"][0]["findings"] is None
    assert reader_view["metadata"] == {"author": "Alice"}


async def test_update_metadata_and_permission_changes_are_audited(
    client, factory, container
) -> None:
    tenant = await make_tenant(factory)
    manager = await login(client, tenant.manager)
    doc_id = (
        await upload_file(client, manager, unique_text(), "plan.txt", department_id=tenant.finance)
    ).json()["document_id"]
    async with container.db.transaction(DbContext.system_for_org(tenant.org)) as session:
        await session.execute(
            update(Document)
            .where(Document.id == uuid.UUID(doc_id))
            .values(suggested_classification="CONFIDENTIAL")
        )

    response = await client.patch(
        f"{URL}/{doc_id}",
        headers=manager,
        json={
            "title": "  Budget plan 2027 ",
            "tags": ["Plan", "plan", "2027"],
            "doc_type": "financial",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["title"], body["tags"], body["doc_type"]) == (
        "Budget plan 2027",
        ["plan", "2027"],
        "financial",
    )

    response = await client.patch(
        f"{URL}/{doc_id}",
        headers=manager,
        json={
            "classification": "CONFIDENTIAL",
            "allowed_roles": ["department_manager", "employee"],
            "retention_until": "2031-12-31",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["classification"] == "CONFIDENTIAL"
    assert body["suggested_classification"] is None  # the accepted suggestion is cleared
    assert body["retention_until"] == "2031-12-31"

    same = await client.patch(
        f"{URL}/{doc_id}", headers=manager, json={"title": "Budget plan 2027"}
    )
    assert same.status_code == 200
    cleared = await client.patch(f"{URL}/{doc_id}", headers=manager, json={"department_id": None})
    assert cleared.json()["department_id"] is None
    injected = await client.patch(f"{URL}/{doc_id}", headers=manager, json={"owner_id": "x"})
    assert injected.status_code == 422

    events = [(a, d) for a, _o, d in await audit_actions(container, tenant.org)]
    updated = [d for a, d in events if a == "document.updated"]
    assert updated[0]["fields"] == ["title", "tags", "doc_type"]
    changes = [d for a, d in events if a == "document.permissions_changed"]
    assert changes[0]["fields"] == ["classification", "allowed_roles"]
    assert changes[0]["before"]["classification"] == "INTERNAL"
    assert changes[0]["after"]["classification"] == "CONFIDENTIAL"
    assert changes[0]["after"]["allowed_roles"] == ["department_manager", "employee"]
    assert changes[1]["fields"] == ["department_id"]


async def test_new_version_keeps_document_ready_on_current_version(
    client, factory, container
) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    doc_id = (await upload_file(client, alice, unique_text(), "policy.txt")).json()["document_id"]
    v1 = await make_ready(container, tenant.org, uuid.UUID(doc_id))

    response = await client.post(
        f"{URL}/{doc_id}/versions",
        headers=alice,
        data={"change_note": "Updated\nterms"},
        files={"file": ("policy-v2.pdf", sf.build_pdf(), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert (body["version_number"], body["status"], body["version_status"]) == (
        2,
        "ready",
        "uploaded",
    )
    detail = (await client.get(f"{URL}/{doc_id}", headers=alice)).json()
    assert detail["current_version_id"] == str(v1)  # the old version serves until indexed
    assert detail["version_count"] == 2
    assert [v["version_number"] for v in detail["versions"]] == [2, 1]
    assert detail["versions"][0]["change_note"] == "Updated terms"
    assert detail["ingestion"]["version_number"] == 2
    payloads = [
        j.payload for j in await jobs_for(container, tenant.org) if j.kind == "ingest_version"
    ]
    assert {"version_id": body["version_id"]} in payloads
    missing = await client.post(
        f"{URL}/{doc_id}/versions", headers=alice, data={"change_note": "x"}
    )
    assert missing.status_code == 422


# --------------------------------------------------------------------------- #
# Download & preview
# --------------------------------------------------------------------------- #
async def test_download_headers_and_bytes(client, factory, container) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    content = sf.build_pdf(["Signed contract"])
    name = "Contrat sign" + chr(0xE9) + "; final.pdf"  # ";" is replaced when stored
    doc_id = (await upload_file(client, alice, content, name)).json()["document_id"]

    not_ready = await client.get(f"{URL}/{doc_id}/download", headers=alice)
    assert not_ready.status_code == 403  # only ready documents are content-readable
    await make_ready(container, tenant.org, uuid.UUID(doc_id))

    response = await client.get(f"{URL}/{doc_id}/download", headers=alice)
    assert response.status_code == 200
    assert response.content == content
    headers = response.headers
    assert headers["content-type"] == "application/pdf"
    assert headers["content-length"] == str(len(content))
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["content-security-policy"].startswith("sandbox")
    assert headers["cache-control"] == "no-store"
    disposition = headers["content-disposition"]
    assert disposition.startswith('attachment; filename="Contrat signe_ final.pdf"; ')
    assert disposition.endswith("filename*=UTF-8''Contrat%20sign%C3%A9_%20final.pdf")

    by_number = await client.get(f"{URL}/{doc_id}/versions/1/download", headers=alice)
    assert by_number.content == content
    missing = await client.get(f"{URL}/{doc_id}/versions/9/download", headers=alice)
    assert missing.status_code == 404
    downloads = [
        d for a, _o, d in await audit_actions(container, tenant.org) if a == "document.download"
    ]
    assert len(downloads) == 2 and downloads[0]["quarantined"] is False


async def test_text_download_declares_charset(client, factory, container) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    content = ("UTF-16 notes " + uuid.uuid4().hex).encode("utf-16")
    doc_id = (await upload_file(client, alice, content, "n.txt")).json()["document_id"]
    await make_ready(container, tenant.org, uuid.UUID(doc_id))
    response = await client.get(f"{URL}/{doc_id}/download", headers=alice)
    assert response.headers["content-type"] == "text/plain; charset=utf-16"
    assert response.content == content


async def test_content_preview_pages_and_cursor(client, factory, container) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    doc_id = (await upload_file(client, alice, unique_text(), "book.txt")).json()["document_id"]
    tag_char = chr(0xE0041)
    await make_ready(
        container,
        tenant.org,
        uuid.UUID(doc_id),
        chunks=[
            ("Chapter one", 1),
            (f"Chapter two{tag_char}", 2),
            ("Chapter two, part 2", 2),
            ("End", 3),
        ],
    )
    page2 = (await client.get(f"{URL}/{doc_id}/content", headers=alice, params={"page": 2})).json()
    assert [c["text"] for c in page2["items"]] == ["Chapter two", "Chapter two, part 2"]
    assert page2["page_count"] == 3 and page2["version_number"] == 1

    collected: list[int] = []
    cursor = None
    while True:
        params = {"limit": 1, **({"cursor": cursor} if cursor else {})}
        page = (await client.get(f"{URL}/{doc_id}/content", headers=alice, params=params)).json()
        collected += [c["chunk_index"] for c in page["items"]]
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert collected == [0, 1, 2, 3]
    assert "document.view" in [a for a, _o, _d in await audit_actions(container, tenant.org)]

    await client.post(
        f"{URL}/{doc_id}/versions",
        headers=alice,
        files={"file": ("book2.txt", unique_text(), "text/plain")},
    )
    pending = await client.get(f"{URL}/{doc_id}/content", headers=alice, params={"version": 2})
    assert pending.status_code == 409 and pending.json()["reason"] == "version_not_indexed"


# --------------------------------------------------------------------------- #
# Deletion & legal hold
# --------------------------------------------------------------------------- #
async def test_delete_removes_retrievable_data_and_cancels_jobs(client, factory, container) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    upload = await upload_file(client, alice, unique_text(), "old.txt")
    doc_id = uuid.UUID(upload.json()["document_id"])
    await make_ready(container, tenant.org, doc_id, chunks=[("a", 1), ("b", 1)])
    await client.post(
        f"{URL}/{doc_id}/versions",
        headers=alice,
        files={"file": ("new.txt", unique_text(), "text/plain")},
    )
    counts = await row_counts(container, tenant.org, doc_id)
    assert counts == {"chunks": 2, "embeddings": 2, "fields": 1}

    assert (await client.delete(f"{URL}/{doc_id}", headers=alice)).status_code == 204
    counts = await row_counts(container, tenant.org, doc_id)
    assert counts == {"chunks": 0, "embeddings": 0, "fields": 0}
    jobs = [j for j in await jobs_for(container, tenant.org) if j.kind == "ingest_version"]
    assert len(jobs) == 2
    assert all(j.status == JobStatus.CANCELLED.value for j in jobs)
    assert (await client.get(f"{URL}/{doc_id}", headers=alice)).status_code == 404
    listed = (await client.get(URL, headers=alice)).json()["items"]
    assert all(d["id"] != str(doc_id) for d in listed)
    assert (await client.delete(f"{URL}/{doc_id}", headers=alice)).status_code == 404
    [deleted] = [
        d for a, _o, d in await audit_actions(container, tenant.org) if a == "document.deleted"
    ]
    assert deleted["chunks_deleted"] == 2 and deleted["jobs_cancelled"] == 2


async def test_legal_hold_blocks_deletion(client, factory, container) -> None:
    tenant = await make_tenant(factory)
    admin = await login(client, tenant.admin)
    manager = await login(client, tenant.manager)
    doc_id = (
        await upload_file(
            client, manager, unique_text(), "evidence.txt", department_id=tenant.finance
        )
    ).json()["document_id"]

    denied = await client.patch(f"{URL}/{doc_id}", headers=manager, json={"legal_hold": True})
    assert denied.status_code == 403  # only organisation admins place holds
    held = await client.patch(f"{URL}/{doc_id}", headers=admin, json={"legal_hold": True})
    assert held.status_code == 200 and held.json()["legal_hold"] is True
    blocked = await client.delete(f"{URL}/{doc_id}", headers=manager)
    assert blocked.status_code == 409 and blocked.json()["reason"] == "legal_hold"
    assert (await client.get(f"{URL}/{doc_id}", headers=manager)).status_code == 200

    await client.patch(f"{URL}/{doc_id}", headers=admin, json={"legal_hold": False})
    assert (await client.delete(f"{URL}/{doc_id}", headers=manager)).status_code == 204
    outcomes = {(a, o) for a, o, _d in await audit_actions(container, tenant.org)}
    assert ("document.delete_refused", "denied") in outcomes
    assert ("document.permissions_changed", "success") in outcomes


# --------------------------------------------------------------------------- #
# Grants
# --------------------------------------------------------------------------- #
async def test_grant_flow(client, factory, container) -> None:
    tenant = await make_tenant(factory)
    other_org_user = await factory.user(await factory.org(), "employee")
    alice = await login(client, tenant.alice)
    bob = await login(client, tenant.bob)
    doc_id = (
        await upload_file(
            client,
            alice,
            unique_text(),
            "salary-review.txt",
            classification="CONFIDENTIAL",
            department_id=tenant.finance,
        )
    ).json()["document_id"]
    await make_ready(container, tenant.org, uuid.UUID(doc_id))
    assert (await client.get(f"{URL}/{doc_id}", headers=bob)).status_code == 404

    grant = {"grantee_type": "user", "user_id": str(tenant.bob.id), "permission": "read"}
    created = await client.post(f"{URL}/{doc_id}/grants", headers=alice, json=grant)
    assert created.status_code == 201, created.text
    grant_id = created.json()["id"]
    assert created.json()["active"] is True
    duplicate = await client.post(f"{URL}/{doc_id}/grants", headers=alice, json=grant)
    assert duplicate.status_code == 409

    assert (await client.get(f"{URL}/{doc_id}/download", headers=bob)).status_code == 200
    assert (await client.get(f"{URL}/{doc_id}/grants", headers=bob)).status_code == 403
    listed = (await client.get(f"{URL}/{doc_id}/grants", headers=alice)).json()
    assert [g["id"] for g in listed] == [grant_id]

    past = {"grantee_type": "role", "role": "auditor", "expires_at": "2001-01-01T00:00:00Z"}
    assert (
        await client.post(f"{URL}/{doc_id}/grants", headers=alice, json=past)
    ).status_code == 422
    foreign = {"grantee_type": "user", "user_id": str(other_org_user.id)}
    foreign_grant = await client.post(f"{URL}/{doc_id}/grants", headers=alice, json=foreign)
    assert foreign_grant.status_code == 422
    ghost = {"grantee_type": "department", "department_id": str(uuid.uuid4())}
    assert (
        await client.post(f"{URL}/{doc_id}/grants", headers=alice, json=ghost)
    ).status_code == 422

    revoke = await client.delete(f"{URL}/{doc_id}/grants/{grant_id}", headers=alice)
    assert revoke.status_code == 204
    assert (await client.get(f"{URL}/{doc_id}", headers=bob)).status_code == 404
    again = await client.delete(f"{URL}/{doc_id}/grants/{grant_id}", headers=alice)
    assert again.status_code == 404
    actions = [a for a, _o, _d in await audit_actions(container, tenant.org)]
    assert "document.grant_added" in actions and "document.grant_revoked" in actions


async def test_department_manage_grant_lets_members_manage(client, factory, container) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    bob = await login(client, tenant.bob)
    doc_id = (
        await upload_file(
            client,
            alice,
            unique_text(),
            "shared.txt",
            classification="CONFIDENTIAL",
            department_id=tenant.finance,
        )
    ).json()["document_id"]
    await make_ready(container, tenant.org, uuid.UUID(doc_id))
    dept = {"grantee_type": "department", "department_id": str(tenant.hr), "permission": "manage"}
    assert (
        await client.post(f"{URL}/{doc_id}/grants", headers=alice, json=dept)
    ).status_code == 201
    detail = (await client.get(f"{URL}/{doc_id}", headers=bob)).json()
    assert detail["can_read"] and detail["can_manage"]
    retitled = await client.patch(f"{URL}/{doc_id}", headers=bob, json={"title": "Shared by HR"})
    assert retitled.status_code == 200


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #
async def test_duplicate_detection_only_considers_visible_documents(client, factory) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    carol = await login(client, tenant.carol)
    bob = await login(client, tenant.bob)
    content = unique_text("confidential")
    first = await upload_file(
        client, alice, content, "a.txt", classification="CONFIDENTIAL", department_id=tenant.finance
    )
    assert first.status_code == 201
    original = first.json()["document_id"]

    # carol (same department) can see the original -> conflict naming it
    finance_only = {"classification": "CONFIDENTIAL", "department_id": tenant.finance}
    conflict = await upload_file(client, carol, content, "b.txt", **finance_only)
    assert conflict.status_code == 409
    assert conflict.json()["duplicate_of"] == original
    allowed = await upload_file(
        client, carol, content, "b.txt", allow_duplicate=True, **finance_only
    )
    assert allowed.status_code == 201 and allowed.json()["duplicate_of"] == original

    # bob (HR) cannot see either finance document -> no conflict, no existence leak
    invisible = await upload_file(
        client, bob, content, "c.txt", classification="CONFIDENTIAL", department_id=tenant.hr
    )
    assert invisible.status_code == 201 and invisible.json()["duplicate_of"] is None
