"""Hostile uploads: size, type confusion, macros, zip bombs, filenames, malware, active
content, scanner outages, form abuse, rate limits, URL-import SSRF, cleanup guarantees,
and vector-store sync jobs."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from docassist.core.config import RateRule
from docassist.core.errors import (
    PermissionDenied,
    RateLimited,
    RejectedContent,
    ServiceUnavailable,
)
from docassist.db.session import DbContext
from docassist.documents.scanning import EICAR_TEST_SIGNATURE, ClamAvScanner
from docassist.documents.schemas import DocumentUpdate, GrantCreate
from docassist.documents.service import DocumentService
from docassist.documents.url_import import UrlImporter, UrlImportRefused
from docassist.security import ssrf
from docassist.security.ssrf import EgressPolicy, Target
from tests.conftest import login
from tests.fixtures import sample_files as sf
from tests.helpers_documents import (
    FakeClamd,
    audit_actions,
    jobs_for,
    make_tenant,
    service_upload,
    unique_text,
    upload_file,
    version_row,
)

pytestmark = [pytest.mark.db]

URL = "/api/v1/documents"


def _reason(response: httpx.Response) -> str:
    return str(response.json().get("reason"))


def _settings(container: Any, **sections: dict[str, Any]) -> Any:
    updates = {
        name: getattr(container.settings, name).model_copy(update=values)
        for name, values in sections.items()
    }
    return container.settings.model_copy(update=updates)


def _org_blobs(container: Any, org_id: uuid.UUID) -> list[Path]:
    root = Path(container.settings.storage.root) / org_id.hex
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


# --------------------------------------------------------------------------- #
# Size and type
# --------------------------------------------------------------------------- #
async def test_oversize_upload_is_refused_and_nothing_is_stored(client, factory, container) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    too_big = b"a" * (container.settings.upload.max_upload_bytes + 1)
    response = await upload_file(client, alice, too_big, "big.txt")
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"
    assert _org_blobs(container, tenant.org) == []
    rejected = [
        d
        for a, o, d in await audit_actions(container, tenant.org)
        if a == "document.upload_rejected"
    ]
    assert rejected == [{"reason": "too_large", "extension": "txt"}]


@pytest.mark.parametrize(
    ("content", "filename", "reason"),
    [
        (sf.build_pdf(), "contract.docx", "type_mismatch"),
        (sf.build_docx(), "contract.pdf", "type_mismatch"),
        (sf.build_xlsx(), "report.docx", "type_mismatch"),
        (sf.build_docx(), "notes.txt", "type_mismatch"),
        (b"\x7fELF\x02\x01\x01" + b"\x00" * 100, "readme.md", "type_mismatch"),
        (
            b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100,
            "old.xlsx",
            "legacy_or_encrypted_office",
        ),
        (b"MZ\x00\x00" + b"\x00" * 100, "tool.txt", "binary_content"),
        (b"<html></html>", "page.html", "extension_not_allowed"),
        (b"MZ", "setup.exe", "extension_not_allowed"),
        (b"plain", "noextension", "extension_not_allowed"),
    ],
    ids=[
        "pdf-as-docx",
        "docx-as-pdf",
        "xlsx-as-docx",
        "docx-as-txt",
        "elf-as-md",
        "ole2-as-xlsx",
        "pe-as-txt",
        "html",
        "exe",
        "no-extension",
    ],
)
async def test_content_must_match_an_allowed_extension(
    client, factory, content: bytes, filename: str, reason: str
) -> None:
    tenant = await make_tenant(factory)
    response = await upload_file(
        client, await login(client, tenant.alice), content, filename, content_type="application/pdf"
    )
    assert response.status_code == 415, response.text
    assert _reason(response) == reason


@pytest.mark.parametrize(
    ("builder", "reason"),
    [
        (sf.docm_disguised_as_docx, "ooxml_macro_enabled"),
        (
            lambda: sf.raw_zip([("word/document.xml", b"\x00" * (20 * sf.MIB))]),
            "zip_ratio_exceeded",
        ),
        (lambda: sf.raw_zip([("word/embeddings/x.zip", b"PK")]), "zip_nested_archive"),
        (
            lambda: sf.set_zip_flag(
                sf.raw_zip([("word/document.xml", b"<x/>")]), "word/document.xml", 1
            ),
            "zip_encrypted_entry",
        ),
    ],
    ids=["docm-as-docx", "zip-bomb", "nested-archive", "encrypted-entry"],
)
async def test_hostile_office_archives_are_rejected(
    client, factory, container, builder, reason
) -> None:  # type: ignore[no-untyped-def]
    tenant = await make_tenant(factory)
    response = await upload_file(client, await login(client, tenant.alice), builder(), "file.docx")
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "content_rejected" and _reason(response) == reason
    assert _org_blobs(container, tenant.org) == []


async def test_encrypted_pdf_is_rejected(client, factory, container) -> None:
    tenant = await make_tenant(factory)
    pdf = sf.build_pdf(trailer_extra=sf.ENCRYPT_TRAILER)
    response = await upload_file(client, await login(client, tenant.alice), pdf, "locked.pdf")
    assert response.status_code == 422 and _reason(response) == "pdf_encrypted"
    assert "password" in response.json()["detail"]
    assert _org_blobs(container, tenant.org) == []


# --------------------------------------------------------------------------- #
# Filenames
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "stored"),
    [
        ("../../etc/passwd.txt", "passwd.txt"),
        ("..\\..\\windows\\win.ini.txt", "win.ini.txt"),
        ("a" + chr(0x202E) + "txt.exe.txt", "atxt.exe.txt"),
        ("CON.txt", "_CON.txt"),
        ("x" * 400 + ".txt", "x" * 146 + ".txt"),
    ],
    ids=["posix-traversal", "windows-traversal", "bidi-override", "device-name", "too-long"],
)
async def test_filenames_are_neutralised(client, factory, container, raw: str, stored: str) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    response = await upload_file(client, alice, unique_text(), raw)
    assert response.status_code == 201, response.text
    version = await version_row(container, tenant.org, uuid.UUID(response.json()["version_id"]))
    assert version.original_filename == stored
    assert version.storage_key.startswith(tenant.org.hex + "/")  # server-generated key


# --------------------------------------------------------------------------- #
# Malware and active content
# --------------------------------------------------------------------------- #
async def test_eicar_is_quarantined_and_never_parsed(client, factory, container) -> None:
    tenant = await make_tenant(factory)
    response = await upload_file(
        client, await login(client, tenant.alice), EICAR_TEST_SIGNATURE, "eicar.txt"
    )
    assert response.status_code == 201
    body = response.json()
    assert (body["status"], body["version_status"], body["job_id"]) == (
        "quarantined",
        "quarantined",
        None,
    )
    assert "eicar_test_signature" in body["findings"]
    assert [j for j in await jobs_for(container, tenant.org) if j.kind == "ingest_version"] == []
    [blob] = _org_blobs(container, tenant.org)
    assert EICAR_TEST_SIGNATURE not in blob.read_bytes()  # stored encrypted
    actions = [a for a, _o, _d in await audit_actions(container, tenant.org)]
    assert "document.quarantined" in actions and "document.upload" not in actions


@pytest.mark.parametrize(
    ("pdf_extra", "code"),
    [(sf.JS_OPEN_ACTION, "pdf_javascript"), (sf.HEX_ESCAPED_JS, "pdf_obfuscated_name")],
    ids=["plain-javascript", "hex-escaped-javascript"],
)
async def test_pdf_with_javascript_is_quarantined(
    client, factory, pdf_extra: bytes, code: str
) -> None:
    tenant = await make_tenant(factory)
    pdf = sf.build_pdf([uuid.uuid4().hex], catalog_extra=pdf_extra)
    response = await upload_file(client, await login(client, tenant.alice), pdf, "invoice.pdf")
    assert response.status_code == 201
    assert response.json()["status"] == "quarantined"
    assert code in response.json()["findings"]


async def test_remote_template_docx_is_quarantined_but_links_are_fine(client, factory) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    template = await upload_file(client, alice, sf.docx_with_remote_template(), "memo.docx")
    assert template.json()["status"] == "quarantined"
    assert "ooxml_remote_template" in template.json()["findings"]
    linked = await upload_file(
        client,
        alice,
        sf.docx_with_external_relationship("hyperlink", f"https://example.com/{uuid.uuid4()}"),
        "links.docx",
    )
    assert linked.json()["status"] == "processing"
    assert "ooxml_external_hyperlink" in linked.json()["findings"]


async def test_active_content_policy_can_be_relaxed_but_malware_cannot(factory, container) -> None:
    tenant = await make_tenant(factory)
    principal = await factory.principal(tenant.alice)
    relaxed = DocumentService(
        container, settings=_settings(container, upload={"reject_active_content": False})
    )
    js = sf.build_pdf([uuid.uuid4().hex], catalog_extra=sf.JS_OPEN_ACTION)
    accepted = await service_upload(relaxed, principal, js, "js.pdf")
    assert accepted.status.value == "processing" and "pdf_javascript" in accepted.findings
    eicar = await service_upload(relaxed, principal, EICAR_TEST_SIGNATURE + b" ", "e.txt")
    assert eicar.status.value == "quarantined"


async def test_clamav_detection_quarantines(factory, container) -> None:
    tenant = await make_tenant(factory)
    principal = await factory.principal(tenant.alice)
    clamd = FakeClamd(reply=b"stream: Win.Trojan.Agent-123 FOUND\x00")
    async with clamd.running():
        service = DocumentService(
            container, malware_scanner=ClamAvScanner("127.0.0.1", clamd.port, 5)
        )
        content = unique_text("infected")
        result = await service_upload(service, principal, content, "report.txt")
    assert bytes(clamd.received) == content
    assert result.status.value == "quarantined"
    assert "malware_detected" in result.findings and "malware_scan_skipped" not in result.findings
    [details] = [
        d for a, _o, d in await audit_actions(container, tenant.org) if a == "document.quarantined"
    ]
    assert details["malware_signature"] == "Win.Trojan.Agent-123"


async def test_scanner_outage_fails_closed(factory, container) -> None:
    tenant = await make_tenant(factory)
    principal = await factory.principal(tenant.alice)
    clamd = FakeClamd()
    async with clamd.running():
        port = clamd.port  # the server is gone after the block
    service = DocumentService(container, malware_scanner=ClamAvScanner("127.0.0.1", port, 2))
    with pytest.raises(ServiceUnavailable):
        await service_upload(service, principal, unique_text(), "report.txt")
    assert _org_blobs(container, tenant.org) == []
    outcomes = [(a, o) for a, o, _d in await audit_actions(container, tenant.org)]
    assert ("document.upload_failed", "failure") in outcomes


# --------------------------------------------------------------------------- #
# Form abuse, rate limits, cleanup
# --------------------------------------------------------------------------- #
async def test_malformed_forms_are_rejected(client, factory) -> None:
    tenant = await make_tenant(factory)
    alice = await login(client, tenant.alice)
    extra = await upload_file(client, alice, unique_text(), "a.txt", owner_id=str(uuid.uuid4()))
    assert extra.status_code == 422
    two_files = await client.post(
        URL,
        headers=alice,
        data={"classification": "INTERNAL"},
        files=[("file", ("a.txt", b"a", "text/plain")), ("file", ("b.txt", b"b", "text/plain"))],
    )
    assert two_files.status_code in {400, 422}
    no_classification = await client.post(
        URL, headers=alice, files={"file": ("a.txt", b"a", "text/plain")}
    )
    assert no_classification.status_code == 422
    wrong_field = await client.post(
        URL,
        headers=alice,
        data={"classification": "INTERNAL"},
        files={"document": ("a.txt", b"a", "text/plain")},
    )
    assert wrong_field.status_code == 422
    not_multipart = await client.post(URL, headers=alice, json={"classification": "INTERNAL"})
    assert not_multipart.status_code == 422


async def test_upload_rate_limit(factory, container) -> None:
    tenant = await make_tenant(factory)
    principal = await factory.principal(tenant.alice)
    strict = DocumentService(
        container,
        settings=_settings(
            container, rate_limit={"upload_per_user": RateRule(requests=1, per_seconds=3600)}
        ),
    )
    await service_upload(strict, principal, unique_text(), "one.txt")
    with pytest.raises(RateLimited):
        await service_upload(strict, principal, unique_text(), "two.txt")


async def test_temp_files_and_blobs_are_cleaned_up(
    factory, container, tmp_path, monkeypatch
) -> None:
    tenant = await make_tenant(factory)
    principal = await factory.principal(tenant.alice)
    service = DocumentService(
        container,
        settings=_settings(container, storage={"temp_dir": tmp_path}),
        memory_spool_bytes=1024,
    )
    big = unique_text() + b"x" * 200_000
    await service_upload(service, principal, big, "big.txt")
    with pytest.raises(RejectedContent):
        await service_upload(
            service,
            principal,
            sf.build_pdf(trailer_extra=sf.ENCRYPT_TRAILER) + b" " * 5000,
            "e.pdf",
        )
    assert list(tmp_path.iterdir()) == []  # spooled plaintext never lingers
    blobs_before = len(_org_blobs(container, tenant.org))

    async def broken_insert(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("database went away")

    monkeypatch.setattr(service, "_insert_document", broken_insert)
    with pytest.raises(RuntimeError):
        await service_upload(service, principal, unique_text() + b"y" * 5000, "lost.txt")
    assert len(_org_blobs(container, tenant.org)) == blobs_before  # orphan blob removed
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# Vector-store synchronisation (Qdrant backend)
# --------------------------------------------------------------------------- #
async def test_qdrant_backend_gets_sync_and_delete_jobs(factory, container) -> None:
    tenant = await make_tenant(factory)
    principal = await factory.principal(tenant.alice)
    service = DocumentService(
        container, settings=_settings(container, retrieval={"backend": "qdrant"})
    )
    result = await service_upload(service, principal, unique_text(), "q.txt")
    doc_id = result.document_id

    async def vector_jobs() -> list[tuple[str, str]]:
        return sorted(
            (j.kind, j.idempotency_key or "")
            for j in await jobs_for(container, tenant.org)
            if j.kind.startswith("vector_")
        )

    assert await vector_jobs() == []  # nothing is indexed yet
    await service.update(principal, doc_id, DocumentUpdate(title="Renamed"))
    assert await vector_jobs() == []  # the title is not part of the payload
    await service.update(principal, doc_id, DocumentUpdate(tags=["x"]))
    grant = await service.add_grant(
        principal,
        doc_id,
        GrantCreate(grantee_type="role", role="auditor"),  # type: ignore[arg-type]
    )
    await service.revoke_grant(principal, doc_id, grant.id)
    await service.delete(principal, doc_id)
    jobs = await vector_jobs()
    assert [kind for kind, _key in jobs].count("vector_sync_document") == 4
    assert ("vector_delete_document", f"vdelete:{doc_id}") in jobs
    assert all(
        key.startswith(f"vsync:{doc_id}:") for kind, key in jobs if kind == "vector_sync_document"
    )
    payloads = {
        str(j.payload)
        for j in await jobs_for(container, tenant.org)
        if j.kind.startswith("vector_")
    }
    assert payloads == {str({"document_id": str(doc_id)})}


# --------------------------------------------------------------------------- #
# URL import
# --------------------------------------------------------------------------- #
async def test_url_import_is_disabled_by_default(client, factory) -> None:
    tenant = await make_tenant(factory)
    response = await client.post(
        f"{URL}/import-url",
        headers=await login(client, tenant.alice),
        json={"url": "https://docs.example.com/a.pdf", "classification": "INTERNAL"},
    )
    assert response.status_code == 404 and response.json()["code"] == "feature_disabled"


def _import_service(container: Any, handler: Any) -> DocumentService:
    settings = _settings(
        container,
        upload={"url_import_enabled": True, "url_import_allowed_domains": ["docs.example.com"]},
    )
    egress = EgressPolicy.build(["docs.example.com", "api.anthropic.com"])
    importer = UrlImporter(settings.upload, egress, transport=httpx.MockTransport(handler))
    return DocumentService(container, settings=settings, url_importer=importer)


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost/a.pdf",
        "https://127.0.0.1/a.pdf",
        "https://169.254.169.254/latest/meta-data/iam",
        "http://docs.example.com/a.pdf",
        "https://api.anthropic.com/v1/messages",
        "https://docs.example.com@evil.example/a.pdf",
        "file:///etc/passwd",
    ],
)
async def test_url_import_ssrf_refusals(factory, container, url: str) -> None:
    tenant = await make_tenant(factory)
    principal = await factory.principal(tenant.alice)
    calls: list[httpx.Request] = []
    service = _import_service(
        container, lambda request: calls.append(request) or httpx.Response(200)
    )
    with pytest.raises(UrlImportRefused):
        await service.import_url(principal, url, classification="INTERNAL")
    assert calls == []  # refused before any network access
    refused = [
        d
        for a, _o, d in await audit_actions(container, tenant.org)
        if a == "document.import_refused"
    ]
    assert len(refused) == 1


async def test_url_import_refuses_private_resolution_and_bad_redirects(
    factory, container, monkeypatch
) -> None:
    tenant = await make_tenant(factory)
    principal = await factory.principal(tenant.alice)

    async def private(target: Target) -> str:
        raise ssrf.EgressDenied(f"{target.host} resolves to a non-public address")

    monkeypatch.setattr(ssrf, "resolve_checked", private)
    service = _import_service(container, lambda request: httpx.Response(200))
    with pytest.raises(UrlImportRefused):
        await service.import_url(
            principal, "https://docs.example.com/a.pdf", classification="INTERNAL"
        )

    async def public(target: Target) -> str:
        return "93.184.216.34"

    monkeypatch.setattr(ssrf, "resolve_checked", public)
    redirect = _import_service(
        container,
        lambda request: httpx.Response(302, headers={"location": "https://10.0.0.1/admin"}),
    )
    with pytest.raises(UrlImportRefused):
        await redirect.import_url(
            principal, "https://docs.example.com/a.pdf", classification="INTERNAL"
        )


async def test_url_import_creates_a_scanned_document(factory, container, monkeypatch) -> None:
    tenant = await make_tenant(factory)
    principal = await factory.principal(tenant.alice)

    async def public(target: Target) -> str:
        return "93.184.216.34"

    monkeypatch.setattr(ssrf, "resolve_checked", public)
    pdf = sf.build_pdf([uuid.uuid4().hex])
    service = _import_service(
        container,
        lambda request: httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=pdf
        ),
    )
    result = await service.import_url(
        principal,
        "https://docs.example.com/policies/travel",
        classification="INTERNAL",
        tags=["Imported"],
    )
    assert result.status.value == "processing" and result.detected_mime == "application/pdf"
    version = await version_row(container, tenant.org, result.version_id)
    assert version.original_filename == "travel.pdf"
    [details] = [
        d for a, _o, d in await audit_actions(container, tenant.org) if a == "document.upload"
    ]
    assert details["source"] == "url" and details["source_host"] == "docs.example.com"

    js_pdf = sf.build_pdf([uuid.uuid4().hex], catalog_extra=sf.JS_OPEN_ACTION)
    hostile = _import_service(
        container,
        lambda request: httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=js_pdf
        ),
    )
    flagged = await hostile.import_url(
        principal, "https://docs.example.com/x.pdf", classification="INTERNAL"
    )
    assert flagged.status.value == "quarantined"


async def test_url_import_respects_clearance_before_fetching(factory, container) -> None:
    tenant = await make_tenant(factory)
    principal = await factory.principal(tenant.alice)
    calls: list[httpx.Request] = []
    service = _import_service(
        container, lambda request: calls.append(request) or httpx.Response(200)
    )
    with pytest.raises(PermissionDenied):
        await service.import_url(
            principal, "https://docs.example.com/a.pdf", classification="RESTRICTED"
        )
    assert calls == []


async def test_platform_context_cannot_see_documents(factory, container) -> None:
    """Defence in depth: even with raw SQL, the platform context reads no tenant documents."""
    from sqlalchemy import select

    from docassist.db.models import Document

    tenant = await make_tenant(factory)
    await service_upload(container, await factory.principal(tenant.alice), unique_text(), "p.txt")
    async with container.db.session(DbContext(org_id=None, platform=True)) as session:
        assert (await session.execute(select(Document))).scalars().all() == []
