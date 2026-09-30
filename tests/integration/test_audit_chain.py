"""Tamper-evident audit chain: sealing, verification, tamper detection, anchored purge."""

from __future__ import annotations

from datetime import timedelta

import asyncpg
import pytest

from docassist.audit.service import PLATFORM_CHAIN, Actor, AuditSealer, verify_chain
from docassist.core.context import utcnow
from docassist.core.enums import AuditOutcome
from docassist.db.session import DbContext

pytestmark = [pytest.mark.db]


def sealer(container) -> AuditSealer:
    key = container.settings.security.audit_hmac_key.get_secret_value().encode()
    return AuditSealer(container.worker_db, key)


def key(container) -> bytes:
    return container.settings.security.audit_hmac_key.get_secret_value().encode()


async def _write(container, org, n: int) -> None:
    async with container.db.transaction(DbContext(org_id=org)) as session:
        for i in range(n):
            container.audit.record(
                session,
                Actor(None, "employee", org),
                "test.event",
                details={"i": i, "note": "mail a@b.co"},
            )


async def test_seal_and_verify(container, factory) -> None:
    org = await factory.org()
    await _write(container, org, 5)
    while await sealer(container).seal_pending():
        pass
    async with container.db.session(DbContext(org_id=org)) as session:
        report = await verify_chain(session, key(container), org)
    assert report.valid and report.checked == 5 and report.unsealed == 0
    await _write(container, org, 3)
    while await sealer(container).seal_pending():
        pass
    async with container.db.session(DbContext(org_id=org)) as session:
        report = await verify_chain(session, key(container), org)
    assert report.valid and report.checked == 8 and report.head_seq == 8


async def test_details_are_redacted(container, factory) -> None:
    from sqlalchemy import select

    from docassist.db.models import AuditEvent

    org = await factory.org()
    await _write(container, org, 1)
    async with container.db.session(DbContext(org_id=org)) as session:
        event = (
            await session.execute(select(AuditEvent).where(AuditEvent.organization_id == org))
        ).scalar_one()
    assert "a@b.co" not in str(event.details)


async def test_tampering_is_detected(container, factory) -> None:
    from tests.conftest import TEST_DB_URL

    org = await factory.org()
    await _write(container, org, 4)
    while await sealer(container).seal_pending():
        pass
    db_name = container.settings.database.url.get_secret_value().rsplit("/", 1)[1]
    admin_dsn = TEST_DB_URL.rsplit("/", 1)[0] + "/" + db_name
    conn = await asyncpg.connect(admin_dsn)
    try:
        # A superuser bypasses triggers and privileges - the HMAC chain still exposes the edit.
        async with conn.transaction():
            await conn.execute("SET LOCAL session_replication_role = replica")
            await conn.execute(
                "UPDATE audit_events SET action = 'test.innocent' WHERE organization_id = $1 AND seal_seq = 2",
                org,
            )
    finally:
        await conn.close()
    async with container.db.session(DbContext(org_id=org)) as session:
        report = await verify_chain(session, key(container), org)
    assert not report.valid and report.first_bad_seq == 2 and report.reason == "hash mismatch"


async def test_deletion_is_detected(container, factory) -> None:
    from tests.conftest import TEST_DB_URL

    org = await factory.org()
    await _write(container, org, 4)
    while await sealer(container).seal_pending():
        pass
    db_name = container.settings.database.url.get_secret_value().rsplit("/", 1)[1]
    conn = await asyncpg.connect(TEST_DB_URL.rsplit("/", 1)[0] + "/" + db_name)
    try:
        async with conn.transaction():
            await conn.execute("SET LOCAL session_replication_role = replica")
            await conn.execute(
                "DELETE FROM audit_events WHERE organization_id = $1 AND seal_seq = 4", org
            )
    finally:
        await conn.close()
    async with container.db.session(DbContext(org_id=org)) as session:
        report = await verify_chain(session, key(container), org)
    assert not report.valid  # truncated tail


async def test_retention_purge_keeps_chain_verifiable(container, factory) -> None:
    org = await factory.org()
    await _write(container, org, 6)
    while await sealer(container).seal_pending():
        pass
    deleted = await sealer(container).purge_older_than(org, utcnow() + timedelta(seconds=1))
    assert deleted == 6
    await _write(container, org, 2)
    while await sealer(container).seal_pending():
        pass
    async with container.db.session(DbContext(org_id=org)) as session:
        report = await verify_chain(session, key(container), org)
    assert report.valid and report.checked == 2 and report.head_seq == 8


async def test_org_cannot_read_other_chain(container, factory) -> None:
    org_a, org_b = await factory.org(), await factory.org()
    await _write(container, org_b, 2)
    while await sealer(container).seal_pending():
        pass
    async with container.db.session(DbContext(org_id=org_a)) as session:
        report = await verify_chain(session, key(container), org_b)
    assert report.checked == 0 and report.head_seq == 0  # nothing visible across tenants


async def test_platform_chain(container) -> None:
    await container.audit.record_detached(
        Actor(None, None, None), "auth.login_failed", outcome=AuditOutcome.FAILURE
    )
    while await sealer(container).seal_pending():
        pass
    async with container.db.session(DbContext(org_id=None, platform=True)) as session:
        report = await verify_chain(session, key(container), PLATFORM_CHAIN)
    assert report.valid and report.checked >= 1
