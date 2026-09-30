"""Differential test: the Python policy and the SQL policy must agree on every document.

A randomised organisation (departments, users of every role/clearance, documents with random
classification/status/department/owner/allowed_roles, random user/department/role grants,
some expired) is written to PostgreSQL. For every principal we compare the set of document
ids returned by ``readable_clause`` / ``listable_clause`` / ``manageable_clause`` with the
Python evaluation of ``can_read`` / ``can_list`` / ``can_manage``. Any divergence would mean
search/RAG and single-document endpoints disagree about who may see what.
"""

from __future__ import annotations

import random
from datetime import timedelta

import pytest
from sqlalchemy import select

from docassist.authz.policy import (
    DocumentFacts,
    can_list,
    can_manage,
    can_read,
    listable_clause,
    manageable_clause,
    readable_clause,
)
from docassist.core.context import utcnow
from docassist.core.enums import Classification, DocumentStatus, GranteeType, GrantPermission, Role
from docassist.db.models import Document, DocumentGrant
from docassist.db.session import DbContext

pytestmark = [pytest.mark.db, pytest.mark.slow]

ROLES = [Role.ORGANIZATION_ADMIN, Role.DEPARTMENT_MANAGER, Role.EMPLOYEE, Role.AUDITOR]


@pytest.mark.parametrize("seed", [7, 1337, 2026])
async def test_python_and_sql_policies_agree(container, factory, seed: int) -> None:
    rng = random.Random(seed)
    org = await factory.org()
    other_org = await factory.org()
    depts = [await factory.department(org) for _ in range(3)]
    users = []
    for _ in range(14):
        role = rng.choice(ROLES)
        members = rng.sample(depts, rng.randint(0, 2))
        managed = [d for d in members if role is Role.DEPARTMENT_MANAGER and rng.random() < 0.7]
        clearance = rng.choice(list(Classification))
        users.append(
            await factory.user(
                org, role.value, departments=members, managed=managed, clearance=clearance.value
            )
        )
    outsider = await factory.user(other_org, "organization_admin")
    now = utcnow()

    async with container.db.transaction(DbContext(org_id=org)) as session:
        docs = []
        for i in range(80):
            owner = rng.choice(users)
            doc = Document(
                organization_id=org,
                department_id=rng.choice([None, *depts]),
                owner_id=owner.id,
                title=f"doc {i}",
                classification=rng.choice(list(Classification)).value,
                doc_type="other",
                status=rng.choice(list(DocumentStatus)).value,
                allowed_roles=rng.choice(
                    [
                        [],
                        [],
                        [Role.EMPLOYEE.value],
                        [Role.DEPARTMENT_MANAGER.value, Role.AUDITOR.value],
                    ]
                ),
            )
            session.add(doc)
            docs.append(doc)
        await session.flush()
        for doc in docs:
            for _ in range(rng.choice([0, 0, 1, 2])):
                kind = rng.choice(list(GranteeType))
                grant = DocumentGrant(
                    organization_id=org,
                    document_id=doc.id,
                    grantee_type=kind.value,
                    permission=rng.choice(list(GrantPermission)).value,
                    granted_by=doc.owner_id,
                    expires_at=rng.choice(
                        [None, None, now - timedelta(days=1), now + timedelta(days=5)]
                    ),
                )
                if kind is GranteeType.USER:
                    grant.grantee_user_id = rng.choice(users).id
                elif kind is GranteeType.DEPARTMENT:
                    grant.grantee_department_id = rng.choice(depts)
                else:
                    grant.grantee_role = rng.choice(ROLES).value
                session.add(grant)

    async with container.db.session(DbContext(org_id=org)) as session:
        all_docs = (await session.execute(select(Document))).scalars().all()
        all_grants = (await session.execute(select(DocumentGrant))).scalars().all()
    facts = [DocumentFacts.from_row(d, all_grants) for d in all_docs]

    for user in [*users, outsider]:
        principal = await factory.principal(user)
        async with container.db.session(principal.db_context) as session:
            sql = {}
            for name, clause in (
                ("read", readable_clause),
                ("list", listable_clause),
                ("manage", manageable_clause),
            ):
                rows = await session.execute(select(Document.id).where(clause(principal, now)))
                sql[name] = set(rows.scalars().all())
        py = {
            "read": {f.id for f in facts if can_read(principal, f, now)},
            "list": {f.id for f in facts if can_list(principal, f, now)},
            "manage": {f.id for f in facts if can_manage(principal, f, now)},
        }
        for name in ("read", "list", "manage"):
            assert sql[name] == py[name], (
                f"seed={seed} role={principal.role} clearance={principal.clearance} rule={name} "
                f"sql-only={sql[name] - py[name]} py-only={py[name] - sql[name]}"
            )
        if user is outsider:
            assert sql["read"] == sql["list"] == sql["manage"] == set()


async def test_restricted_requires_owner_or_grant(container, factory) -> None:
    org = await factory.org()
    hr = await factory.department(org)
    admin = await factory.user(org, "organization_admin")
    hr_employee = await factory.user(org, "employee", departments=[hr], clearance="RESTRICTED")
    async with container.db.transaction(DbContext(org_id=org)) as session:
        doc = Document(
            organization_id=org,
            department_id=hr,
            owner_id=admin.id,
            title="salaries",
            classification="RESTRICTED",
            doc_type="hr",
            status="ready",
        )
        session.add(doc)
    now = utcnow()
    principal = await factory.principal(hr_employee)
    async with container.db.session(principal.db_context) as session:
        visible = (
            await session.execute(select(Document.id).where(readable_clause(principal, now)))
        ).all()
    assert visible == []  # same department and enough clearance is still not enough
