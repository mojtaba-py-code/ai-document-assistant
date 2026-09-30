"""Deadline queries: window math, time zones, authorisation, version and duplicate rules."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from docassist.core.errors import ValidationFailed
from docassist.db.models import DocumentGrant
from docassist.db.session import DbContext
from tests.conftest import login
from tests.helpers_intelligence import CONTRACT, FieldSpec, seed_document, set_document, tenant

pytestmark = [pytest.mark.db]

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
TODAY = NOW.date()


def due(day: date, field: str = "due_date", **kw: Any) -> FieldSpec:
    return FieldSpec(field, value_date=day, evidence=f"due on {day.isoformat()}", **kw)


async def window(container: Any, principal: Any, **kw: Any) -> Any:
    kw.setdefault("now", NOW)
    return await container.intelligence.deadlines.window(principal, **kw)


async def test_window_boundaries_and_days_left(factory, container) -> None:
    t = await tenant(factory)
    doc = await seed_document(
        container,
        t.org,
        t.employee.id,
        doc_type="invoice",
        versions=[CONTRACT],
        fields={
            0: [
                due(TODAY - timedelta(days=1)),
                due(TODAY),
                due(TODAY + timedelta(days=30), field="expiration_date"),
                due(TODAY + timedelta(days=31), field="renewal_date"),
                FieldSpec("effective_date", value_date=TODAY + timedelta(days=2)),  # not a deadline
            ]
        },
    )
    principal = await factory.principal(t.employee)
    result = await window(container, principal, within_days=30)
    assert (result.today, result.window_start, result.window_end) == (
        TODAY,
        TODAY,
        TODAY + timedelta(days=30),
    )
    assert [(i.field, i.days_left) for i in result.items] == [
        ("due_date", 0),
        ("expiration_date", 30),
    ]
    assert all(i.document_id == doc.id for i in result.items)
    overdue = await window(container, principal, within_days=30, overdue_days=1)
    assert [i.days_left for i in overdue.items] == [-1, 0, 30]
    zero = await window(container, principal, within_days=0)
    assert [i.days_left for i in zero.items] == [0]
    explicit = await window(container, principal, within_days=5, fields=["effective_date"])
    assert [(i.field, i.days_left) for i in explicit.items] == [("effective_date", 2)]
    with pytest.raises(ValidationFailed):
        await window(container, principal, within_days=4_000)
    with pytest.raises(ValidationFailed):
        await window(container, principal, within_days=5, fields=["password"])


async def test_time_zone_decides_what_today_is(factory, container) -> None:
    t = await tenant(factory)
    await seed_document(
        container,
        t.org,
        t.employee.id,
        versions=[CONTRACT],
        fields={0: [due(date(2026, 9, 30)), due(date(2026, 10, 30), field="expiration_date")]},
    )
    principal = await factory.principal(t.employee)
    late_evening = datetime(2026, 9, 30, 22, 30, tzinfo=UTC)
    east = await window(container, principal, within_days=60, tz="+03:00", now=late_evening)
    assert east.today == date(2026, 10, 1) and east.timezone == "+03:00"
    assert [(i.field, i.days_left) for i in east.items] == [("expiration_date", 29)]
    west = await window(container, principal, within_days=60, tz="-05:00", now=late_evening)
    assert west.today == date(2026, 9, 30)
    assert [(i.field, i.days_left) for i in west.items] == [
        ("due_date", 0),
        ("expiration_date", 30),
    ]
    with pytest.raises(ValidationFailed):
        await window(container, principal, within_days=60, tz="Nowhere/Land")


async def test_only_readable_current_documents_are_included(factory, container) -> None:
    t = await tenant(factory)
    other = await tenant(factory)
    soon = TODAY + timedelta(days=10)
    visible = await seed_document(
        container, t.org, t.employee.id, title="Visible",
        versions=[CONTRACT, CONTRACT],
        fields={0: [due(soon - timedelta(days=1))], 1: [due(soon)]},
    )  # fmt: skip
    await seed_document(
        container, t.org, t.manager.id, title="Other department", classification="CONFIDENTIAL",
        department_id=t.other_dept, fields={0: [due(soon)]},
    )  # fmt: skip
    deleted = await seed_document(
        container, t.org, t.employee.id, title="Deleted", fields={0: [due(soon)]}
    )
    await set_document(container, t.org, deleted.id, status="deleted")
    await seed_document(
        container, other.org, other.admin.id, title="Foreign", fields={0: [due(soon)]}
    )
    granted = await seed_document(
        container, t.org, t.manager.id, title="Restricted granted", classification="RESTRICTED",
        fields={0: [due(soon + timedelta(days=1))]},
    )  # fmt: skip
    expired_grant = await seed_document(
        container, t.org, t.manager.id, title="Restricted expired grant", classification="RESTRICTED",
        fields={0: [due(soon)]},
    )  # fmt: skip
    restricted_employee = await factory.user(
        t.org, "employee", departments=[t.dept], clearance="RESTRICTED"
    )
    async with container.db.transaction(DbContext(org_id=t.org)) as session:
        for doc_id, expires in ((granted.id, None), (expired_grant.id, NOW - timedelta(days=1))):
            session.add(
                DocumentGrant(
                    organization_id=t.org,
                    document_id=doc_id,
                    grantee_type="user",
                    grantee_user_id=restricted_employee.id,
                    permission="read",
                    granted_by=t.admin.id,
                    expires_at=expires,
                )
            )
    principal = await factory.principal(restricted_employee)
    result = await window(container, principal, within_days=365)
    assert [(i.document_title, i.date) for i in result.items] == [
        ("Visible", soon),  # current version only: the v1 date is ignored
        ("Restricted granted", soon + timedelta(days=1)),
    ]
    assert result.items[0].version_id == visible.current_version_id


async def test_duplicate_extractions_are_merged(factory, container) -> None:
    t = await tenant(factory)
    day = TODAY + timedelta(days=5)
    await seed_document(
        container, t.org, t.employee.id,
        fields={
            0: [
                due(day, method="rules", confidence=0.7),
                due(day, method="llm", confidence=0.9),
                due(day + timedelta(days=1), method="rules", confidence=0.6),
            ]
        },
    )  # fmt: skip
    await seed_document(
        container, t.org, t.employee.id, title="Tie",
        fields={0: [due(day, method="rules", confidence=0.8), due(day, method="llm", confidence=0.8)]},
    )  # fmt: skip
    principal = await factory.principal(t.employee)
    result = await window(container, principal, within_days=30)
    assert [(i.date, i.method, i.confidence) for i in result.items] == [
        (day, "llm", 0.9),
        (day, "llm", 0.8),  # equal confidence: the evidence-verified LLM row wins the tie
        (day + timedelta(days=1), "rules", 0.6),
    ]


async def test_filters_limit_and_contract_method(factory, container) -> None:
    t = await tenant(factory)
    day = TODAY + timedelta(days=3)
    contract = await seed_document(
        container,
        t.org,
        t.employee.id,
        doc_type="contract",
        fields={0: [due(day, field="expiration_date")]},
    )
    await seed_document(
        container,
        t.org,
        t.employee.id,
        doc_type="invoice",
        fields={0: [due(day + timedelta(days=1))]},
    )
    principal = await factory.principal(t.employee)
    contracts = await window(container, principal, within_days=30, doc_type="contract")
    assert [i.document_id for i in contracts.items] == [contract.id]
    limited = await window(container, principal, within_days=30, limit=1)
    assert len(limited.items) == 1 and limited.truncated
    rows = await container.intelligence.upcoming_deadlines(principal, 3_650, doc_type="invoice")
    assert [r.field for r in rows] == ["due_date"]
    via_attribute = await container.intelligence.deadlines.upcoming(principal, within_days=3_650)
    assert len(via_attribute) == 2


async def test_deadlines_api(client, factory, container) -> None:
    t = await tenant(factory)
    today = datetime.now(UTC).date()
    await seed_document(
        container, t.org, t.employee.id, title="Supplier contract",
        fields={0: [due(today + timedelta(days=20), field="expiration_date", page=4)]},
    )  # fmt: skip
    headers = await login(client, t.employee)
    response = await client.get("/api/v1/intelligence/deadlines?within_days=90", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    (item,) = body["items"]
    assert item["document_title"] == "Supplier contract"
    assert item["days_left"] == 20 and item["page"] == 4
    assert item["evidence"].startswith("due on")
    params = "within_days=90&fields=expiration_date&fields=due_date&tz=%2B02:00&doc_type=contract"
    assert (
        await client.get(f"/api/v1/intelligence/deadlines?{params}", headers=headers)
    ).status_code == 200
    for bad in (
        "within_days=5000",
        "tz=Mars/Base",
        "fields=secret",
        "doc_type=spaceship",
        "limit=0",
    ):
        assert (
            await client.get(f"/api/v1/intelligence/deadlines?{bad}", headers=headers)
        ).status_code == 422, bad
    auditor = await login(client, t.auditor)
    assert (await client.get("/api/v1/intelligence/deadlines", headers=auditor)).status_code == 403
