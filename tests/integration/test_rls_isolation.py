"""Row-level security is enforced by PostgreSQL itself, independent of application code."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, ProgrammingError

from docassist.db.models import AuditEvent, Department, Document, Organization, User
from docassist.db.session import DbContext, verify_least_privilege

pytestmark = [pytest.mark.db]


async def test_app_role_is_least_privileged(container) -> None:
    assert await verify_least_privilege(container.db, "docassist_app") == []


async def test_no_context_sees_nothing(container, factory) -> None:
    org = await factory.org()
    await factory.department(org)
    async with container.db.session(DbContext.anonymous()) as session:
        assert (await session.execute(select(Department))).scalars().all() == []
        assert (await session.execute(select(Organization))).scalars().all() == []
        assert (await session.execute(select(User))).scalars().all() == []


async def test_tenant_cannot_see_other_tenant(container, factory) -> None:
    org_a, org_b = await factory.org(), await factory.org()
    dept_b = await factory.department(org_b, name="Secret B")
    async with container.db.session(DbContext(org_id=org_a)) as session:
        assert await session.get(Department, dept_b) is None
        orgs = (await session.execute(select(Organization.id))).scalars().all()
        assert orgs == [org_a]


async def test_cannot_write_into_other_tenant(container, factory) -> None:
    org_a, org_b = await factory.org(), await factory.org()
    with pytest.raises((DBAPIError, ProgrammingError)):
        async with container.db.transaction(DbContext(org_id=org_a)) as session:
            session.add(Department(organization_id=org_b, name="Injected", slug="injected"))


async def test_cannot_move_row_to_other_tenant(container, factory) -> None:
    org_a, org_b = await factory.org(), await factory.org()
    dept = await factory.department(org_a)
    with pytest.raises((DBAPIError, ProgrammingError)):
        async with container.db.transaction(DbContext(org_id=org_a)) as session:
            await session.execute(
                text("UPDATE departments SET organization_id = :b WHERE id = :d"),
                {"b": org_b, "d": dept},
            )


async def test_composite_fk_blocks_cross_tenant_reference(container, factory) -> None:
    """A document of org A can never reference a department of org B (composite FK)."""
    org_a, org_b = await factory.org(), await factory.org()
    user_a = await factory.user(org_a)
    dept_b = await factory.department(org_b)
    with pytest.raises((DBAPIError, ProgrammingError)):
        async with container.db.transaction(DbContext(org_id=org_a)) as session:
            session.add(
                Document(
                    organization_id=org_a,
                    department_id=dept_b,
                    owner_id=user_a.id,
                    title="x",
                    classification="INTERNAL",
                    doc_type="other",
                    status="processing",
                )
            )


async def test_platform_context_cannot_read_tenant_content(container, factory) -> None:
    org = await factory.org()
    await factory.department(org)
    async with container.db.session(DbContext(org_id=None, platform=True)) as session:
        assert (await session.execute(select(Department))).scalars().all() == []
        assert (await session.execute(select(Document))).scalars().all() == []
        # but the platform operator does see the organisation registry
        assert org in (await session.execute(select(Organization.id))).scalars().all()


async def test_audit_log_is_append_only(container, factory) -> None:
    org = await factory.org()
    async with container.db.transaction(DbContext(org_id=org)) as session:
        session.add(
            AuditEvent(organization_id=org, action="test.event", outcome="success", details={})
        )
    # the API role has no UPDATE/DELETE privilege at all
    for statement in ("UPDATE audit_events SET action = 'tampered'", "DELETE FROM audit_events"):
        with pytest.raises((DBAPIError, ProgrammingError)):
            async with container.db.transaction(DbContext(org_id=org)) as session:
                await session.execute(text(statement))
    # even the worker role (which may seal) cannot rewrite event content
    with pytest.raises((DBAPIError, ProgrammingError)):
        async with container.worker_db.transaction(DbContext(org_id=org)) as session:
            await session.execute(
                text("UPDATE audit_events SET action = 'tampered' WHERE organization_id = :o"),
                {"o": org},
            )


async def test_context_does_not_leak_across_pooled_connections(container, factory) -> None:
    org_a = await factory.org()
    await factory.department(org_a)
    for _ in range(5):
        async with container.db.session(DbContext(org_id=org_a)) as session:
            assert (await session.execute(select(Department))).scalars().all()
            await session.commit()
        async with container.db.session(DbContext.anonymous()) as session:
            assert (await session.execute(select(Department))).scalars().all() == []


async def test_worker_role_can_claim_across_tenants_but_app_cannot(container, factory) -> None:
    from docassist.jobs.queue import enqueue

    org_a, org_b = await factory.org(), await factory.org()
    async with container.db.transaction(DbContext(org_id=org_a)) as session:
        await enqueue(
            session,
            kind="noop",
            organization_id=org_a,
            payload={},
            idempotency_key=str(uuid.uuid4()),
        )
    async with container.db.session(DbContext(org_id=org_b)) as session:
        rows = (
            await session.execute(
                text("SELECT count(*) FROM jobs WHERE organization_id = :a"), {"a": org_a}
            )
        ).scalar()
        assert rows == 0
    async with container.worker_db.session(DbContext.anonymous()) as session:
        rows = (
            await session.execute(
                text("SELECT count(*) FROM jobs WHERE organization_id = :a"), {"a": org_a}
            )
        ).scalar()
        assert rows == 1
