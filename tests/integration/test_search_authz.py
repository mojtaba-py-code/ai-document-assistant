"""Authorisation-aware retrieval: every document ACL rule, through every retrieval path.

Each scenario checks the keyword path, the semantic (pgvector) path and the RAG ``retrieve``
path, because each one builds its own SQL and each must embed the policy.
"""

from __future__ import annotations

import random
import uuid
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import select

from docassist.authz.policy import DocumentFacts, can_read
from docassist.core.context import utcnow
from docassist.core.enums import Classification, DocumentStatus, GranteeType, GrantPermission, Role
from docassist.db.models import Document, DocumentGrant
from docassist.db.session import DbContext
from docassist.search.keyword import keyword_search
from docassist.search.types import SearchFilters
from tests.helpers_search import (
    EMBEDDER,
    add_grant,
    add_version,
    seed_document,
    soft_delete,
    update_version,
)

pytestmark = [pytest.mark.db]

SECRET_TERM = "zephyrquartz"


async def visible_documents(
    container: Any, principal: Any, query: str, **filters: Any
) -> set[uuid.UUID]:
    """Documents the principal can retrieve for ``query`` - checked across every path.

    * keyword search (``search(mode="keyword")``) finds the documents containing the term;
    * an unbounded pgvector query returns *every* authorised chunk that has an embedding
      (similarity has no threshold there), i.e. exactly the authorised set;
    * hybrid / semantic search and RAG ``retrieve`` may only ever return documents from those
      two sets, and must include every keyword match.

    Scenarios keep one organisation per test with the term in every relevant chunk, so the
    keyword set and the authorised set coincide; the function asserts that and returns it.
    """
    search_filters = SearchFilters(**filters)
    found: dict[str, set[uuid.UUID]] = {}
    for mode in ("keyword", "semantic", "hybrid"):
        response = await container.search.search(
            principal, query=query, mode=mode, filters=search_filters, limit=50
        )
        assert not response.degraded
        found[mode] = {r.document_id for r in response.results}
    retrieved = await container.search.retrieve(principal, query, filters=search_filters, top_k=50)
    found["retrieve"] = {c.document_id for c in retrieved}
    found["authorised"] = await authorised_documents(container, principal, search_filters)
    allowed = found["keyword"] | found["authorised"]
    for mode in ("semantic", "hybrid", "retrieve"):
        assert found[mode] <= allowed, (mode, found)
    assert found["keyword"] <= found["hybrid"] and found["keyword"] <= found["retrieve"], found
    assert found["keyword"] == found["authorised"], found
    return found["keyword"]


async def authorised_documents(
    container: Any, principal: Any, filters: SearchFilters
) -> set[uuid.UUID]:
    async with container.db.session(principal.db_context) as session:
        hits = await container.vector_store.query(
            session,
            principal,
            EMBEDDER.embed_one("anything"),
            filters=filters,
            limit=500,
            now=utcnow(),
        )
        if not hits:
            return set()
        from docassist.db.models import DocumentChunk

        rows = await session.execute(
            select(DocumentChunk.document_id).where(DocumentChunk.id.in_([c for c, _ in hits]))
        )
        return set(rows.scalars().all())


async def _org_with_dept(factory: Any) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    org = await factory.org()
    return org, await factory.department(org), await factory.department(org)


async def test_tenant_isolation(container, factory) -> None:
    org_a, org_b = await factory.org(), await factory.org()
    owner_a = await factory.user(org_a, "organization_admin")
    admin_b = await factory.user(org_b, "organization_admin")
    doc = await seed_document(
        container,
        org_id=org_a,
        owner_id=owner_a.id,
        classification="PUBLIC",
        chunks=[f"public announcement {SECRET_TERM} for everyone"],
    )
    principal_a = await factory.principal(owner_a)
    principal_b = await factory.principal(admin_b)
    assert await visible_documents(container, principal_a, SECRET_TERM) == {doc.id}
    assert await visible_documents(container, principal_b, SECRET_TERM) == set()
    # even an explicit document filter cannot reach across tenants
    assert (
        await visible_documents(container, principal_b, SECRET_TERM, document_ids=(doc.id,))
        == set()
    )


async def test_clearance_is_a_hard_ceiling_even_for_owners(container, factory) -> None:
    org, dept, _ = await _org_with_dept(factory)
    owner = await factory.user(org, "employee", departments=[dept], clearance="INTERNAL")
    doc = await seed_document(
        container,
        org_id=org,
        owner_id=owner.id,
        classification="CONFIDENTIAL",
        department_id=dept,
        chunks=[f"budget {SECRET_TERM} numbers"],
    )
    colleague = await factory.user(org, "employee", departments=[dept], clearance="CONFIDENTIAL")
    assert await visible_documents(container, await factory.principal(owner), SECRET_TERM) == set()
    assert await visible_documents(container, await factory.principal(colleague), SECRET_TERM) == {
        doc.id
    }


async def test_allowed_roles_filter(container, factory) -> None:
    org, dept, _ = await _org_with_dept(factory)
    author = await factory.user(org, "department_manager", departments=[dept], managed=[dept])
    doc = await seed_document(
        container,
        org_id=org,
        owner_id=author.id,
        classification="INTERNAL",
        allowed_roles=["department_manager"],
        chunks=[f"manager handbook {SECRET_TERM}"],
    )
    employee = await factory.user(org, "employee", departments=[dept])
    manager = await factory.user(org, "department_manager", departments=[dept], managed=[dept])
    assert (
        await visible_documents(container, await factory.principal(employee), SECRET_TERM) == set()
    )
    assert await visible_documents(container, await factory.principal(manager), SECRET_TERM) == {
        doc.id
    }
    # a grant does not bypass allowed_roles
    await add_grant(container, doc, user_id=employee.id)
    assert (
        await visible_documents(container, await factory.principal(employee), SECRET_TERM) == set()
    )


async def test_confidential_department_rule_and_department_grant(container, factory) -> None:
    org, finance, sales = await _org_with_dept(factory)
    owner = await factory.user(org, "employee", departments=[finance])
    doc = await seed_document(
        container,
        org_id=org,
        owner_id=owner.id,
        classification="CONFIDENTIAL",
        department_id=finance,
        chunks=[f"finance forecast {SECRET_TERM}"],
    )
    finance_member = await factory.user(org, "employee", departments=[finance])
    sales_member = await factory.user(org, "employee", departments=[sales])
    assert await visible_documents(
        container, await factory.principal(finance_member), SECRET_TERM
    ) == {doc.id}
    assert (
        await visible_documents(container, await factory.principal(sales_member), SECRET_TERM)
        == set()
    )
    await add_grant(container, doc, department_id=sales)
    assert await visible_documents(
        container, await factory.principal(sales_member), SECRET_TERM
    ) == {doc.id}


async def test_restricted_needs_owner_or_active_grant(container, factory) -> None:
    org, hr, _ = await _org_with_dept(factory)
    owner = await factory.user(org, "department_manager", departments=[hr], managed=[hr])
    doc = await seed_document(
        container,
        org_id=org,
        owner_id=owner.id,
        classification="RESTRICTED",
        department_id=hr,
        chunks=[f"salary table {SECRET_TERM}"],
    )
    hr_employee = await factory.user(org, "employee", departments=[hr], clearance="RESTRICTED")
    admin = await factory.user(org, "organization_admin")
    granted = await factory.user(org, "employee", clearance="RESTRICTED")
    expired = await factory.user(org, "employee", clearance="RESTRICTED")
    low_clearance = await factory.user(org, "employee", clearance="CONFIDENTIAL")
    now = utcnow()
    await add_grant(container, doc, user_id=granted.id, expires_at=now + timedelta(days=1))
    await add_grant(container, doc, user_id=expired.id, expires_at=now - timedelta(seconds=1))
    await add_grant(container, doc, user_id=low_clearance.id)

    assert await visible_documents(container, await factory.principal(owner), SECRET_TERM) == {
        doc.id
    }
    assert await visible_documents(container, await factory.principal(granted), SECRET_TERM) == {
        doc.id
    }
    for user in (hr_employee, admin, expired, low_clearance):
        principal = await factory.principal(user)
        assert await visible_documents(container, principal, SECRET_TERM) == set(), user.role


async def test_role_grant(container, factory) -> None:
    org, finance, sales = await _org_with_dept(factory)
    owner = await factory.user(org, "employee", departments=[finance])
    doc = await seed_document(
        container,
        org_id=org,
        owner_id=owner.id,
        classification="CONFIDENTIAL",
        department_id=finance,
        chunks=[f"travel policy {SECRET_TERM}"],
    )
    employee = await factory.user(org, "employee", departments=[sales])
    manager = await factory.user(org, "department_manager", departments=[sales])
    await add_grant(container, doc, role="employee")
    assert await visible_documents(container, await factory.principal(employee), SECRET_TERM) == {
        doc.id
    }
    assert (
        await visible_documents(container, await factory.principal(manager), SECRET_TERM) == set()
    )


@pytest.mark.parametrize("status", ["processing", "failed", "quarantined"])
async def test_non_ready_documents_are_invisible(container, factory, status: str) -> None:
    org = await factory.org()
    owner = await factory.user(org, "organization_admin")
    await seed_document(
        container,
        org_id=org,
        owner_id=owner.id,
        classification="PUBLIC",
        status=status,
        chunks=[f"draft {SECRET_TERM}"],
    )
    assert await visible_documents(container, await factory.principal(owner), SECRET_TERM) == set()


async def test_deleted_document_disappears_immediately(container, factory) -> None:
    org = await factory.org()
    owner = await factory.user(org, "organization_admin")
    doc = await seed_document(
        container,
        org_id=org,
        owner_id=owner.id,
        classification="PUBLIC",
        chunks=[f"obsolete memo {SECRET_TERM}"],
    )
    principal = await factory.principal(owner)
    assert await visible_documents(container, principal, SECRET_TERM) == {doc.id}
    await soft_delete(container, doc)
    assert await visible_documents(container, principal, SECRET_TERM) == set()


async def test_old_versions_only_on_request(container, factory) -> None:
    org = await factory.org()
    owner = await factory.user(org, "employee")
    doc = await seed_document(
        container,
        org_id=org,
        owner_id=owner.id,
        classification="INTERNAL",
        chunks=[f"old payment terms {SECRET_TERM} net 60"],
    )
    await add_version(container, doc, ["new payment terms net 30"])
    principal = await factory.principal(owner)
    default = await container.search.search(principal, query=SECRET_TERM, mode="keyword")
    assert default.results == []
    assert await visible_documents(
        container, principal, SECRET_TERM, include_old_versions=True
    ) == {doc.id}
    response = await container.search.search(
        principal, query=SECRET_TERM, filters=SearchFilters(include_old_versions=True)
    )
    [hit] = response.results
    assert hit.version_number == 1 and hit.is_current is False
    current = await container.search.search(principal, query="payment terms net 30")
    assert {(r.version_number, r.is_current) for r in current.results} == {(2, True)}


async def test_uploaded_but_unindexed_version_does_not_replace_current(container, factory) -> None:
    org = await factory.org()
    owner = await factory.user(org, "employee")
    doc = await seed_document(
        container, org_id=org, owner_id=owner.id, chunks=[f"approved text {SECRET_TERM}"]
    )
    draft = await add_version(container, doc, ["replacement draft text"], make_current=False)
    principal = await factory.principal(owner)
    assert await visible_documents(container, principal, SECRET_TERM) == {doc.id}
    for mode in ("keyword", "hybrid", "semantic"):
        response = await container.search.search(principal, query="replacement draft", mode=mode)
        assert not {r.chunk_id for r in response.results} & set(draft.chunk_ids)


async def test_quarantined_version_never_contributes(container, factory) -> None:
    org = await factory.org()
    owner = await factory.user(org, "employee")
    doc = await seed_document(
        container, org_id=org, owner_id=owner.id, chunks=[f"first version {SECRET_TERM}"]
    )
    await add_version(container, doc, ["second version content"])
    await update_version(container, doc, doc.versions[0].id, status="quarantined")
    principal = await factory.principal(owner)
    old_chunks = set(doc.versions[0].chunk_ids)
    everything = SearchFilters(include_old_versions=True)
    for mode in ("keyword", "hybrid", "semantic"):
        response = await container.search.search(
            principal, query=SECRET_TERM, mode=mode, filters=everything
        )
        assert not {r.chunk_id for r in response.results} & old_chunks
    retrieved = await container.search.retrieve(principal, SECRET_TERM, filters=everything)
    assert not {c.chunk_id for c in retrieved} & old_chunks


async def test_filters_only_narrow(container, factory) -> None:
    org, dept, other_dept = await _org_with_dept(factory)
    owner = await factory.user(org, "organization_admin", departments=[dept])
    base = utcnow() - timedelta(days=10)
    contract = await seed_document(
        container,
        org_id=org,
        owner_id=owner.id,
        doc_type="contract",
        department_id=dept,
        tags=["legal", "2026"],
        created_at=base,
        chunks=[f"contract {SECRET_TERM}"],
    )
    invoice = await seed_document(
        container,
        org_id=org,
        owner_id=owner.id,
        doc_type="invoice",
        classification="PUBLIC",
        department_id=other_dept,
        tags=["finance"],
        created_at=base + timedelta(days=5),
        chunks=[f"invoice {SECRET_TERM}"],
    )
    principal = await factory.principal(owner)
    both = {contract.id, invoice.id}
    assert await visible_documents(container, principal, SECRET_TERM) == both
    assert await visible_documents(container, principal, SECRET_TERM, doc_types=("contract",)) == {
        contract.id
    }
    assert await visible_documents(
        container, principal, SECRET_TERM, department_ids=(other_dept,)
    ) == {invoice.id}
    assert await visible_documents(
        container, principal, SECRET_TERM, classifications=("PUBLIC",)
    ) == {invoice.id}
    assert await visible_documents(
        container, principal, SECRET_TERM, document_ids=(contract.id,)
    ) == {contract.id}
    assert await visible_documents(container, principal, SECRET_TERM, tags=("finance", "nope")) == {
        invoice.id
    }
    assert await visible_documents(
        container, principal, SECRET_TERM, created_after=base + timedelta(days=1)
    ) == {invoice.id}
    assert await visible_documents(
        container, principal, SECRET_TERM, created_before=base + timedelta(days=1)
    ) == {contract.id}


async def test_filters_cannot_widen_access(container, factory) -> None:
    org, finance, sales = await _org_with_dept(factory)
    owner = await factory.user(org, "employee", departments=[finance])
    doc = await seed_document(
        container,
        org_id=org,
        owner_id=owner.id,
        classification="CONFIDENTIAL",
        department_id=finance,
        chunks=[f"hidden {SECRET_TERM}"],
    )
    outsider = await factory.principal(await factory.user(org, "employee", departments=[sales]))
    for filters in (
        {"document_ids": (doc.id,)},
        {"department_ids": (finance,)},
        {"classifications": ("CONFIDENTIAL",)},
        {"include_old_versions": True},
    ):
        assert await visible_documents(container, outsider, SECRET_TERM, **filters) == set()


@pytest.mark.parametrize("seed", [11, 4242])
async def test_random_scenario_matches_python_policy(container, factory, seed: int) -> None:
    """Keyword and pgvector retrieval return exactly the chunks ``can_read`` allows."""
    rng = random.Random(seed)
    org = await factory.org()
    depts = [await factory.department(org) for _ in range(3)]
    roles = [Role.ORGANIZATION_ADMIN, Role.DEPARTMENT_MANAGER, Role.EMPLOYEE]
    users = []
    for _ in range(7):
        role = rng.choice(roles)
        members = rng.sample(depts, rng.randint(0, 2))
        users.append(
            await factory.user(
                org,
                role.value,
                departments=members,
                clearance=rng.choice(list(Classification)).value,
            )
        )
    now = utcnow()
    docs = []
    for i in range(24):
        doc = await seed_document(
            container,
            org_id=org,
            owner_id=rng.choice(users).id,
            title=f"random doc {i}",
            classification=rng.choice(list(Classification)).value,
            department_id=rng.choice([None, *depts]),
            status=rng.choice(["ready", "ready", "ready", "processing", "deleted"]),
            allowed_roles=rng.choice([[], [], ["employee"], ["department_manager"]]),
            chunks=[f"common corpus {SECRET_TERM} item {i}"],
        )
        for _ in range(rng.choice([0, 1, 2])):
            kind = rng.choice(list(GranteeType))
            await add_grant(
                container,
                doc,
                user_id=rng.choice(users).id if kind is GranteeType.USER else None,
                department_id=rng.choice(depts) if kind is GranteeType.DEPARTMENT else None,
                role=rng.choice(roles).value if kind is GranteeType.ROLE else None,
                permission=rng.choice(list(GrantPermission)).value,
                expires_at=rng.choice([None, now - timedelta(hours=1), now + timedelta(days=2)]),
            )
        docs.append(doc)

    async with container.db.session(DbContext(org_id=org)) as session:
        rows = (await session.execute(select(Document))).scalars().all()
        grants = (await session.execute(select(DocumentGrant))).scalars().all()
    facts = {row.id: DocumentFacts.from_row(row, grants) for row in rows}
    vector = EMBEDDER.embed_one(SECRET_TERM)
    for user in users:
        principal = await factory.principal(user)
        expected = {c for d in docs if can_read(principal, facts[d.id], now) for c in d.chunk_ids}
        async with container.db.session(principal.db_context) as session:
            semantic = await container.vector_store.query(
                session, principal, vector, filters=SearchFilters(), limit=500, now=now
            )
            keyword = await keyword_search(
                session, principal, SECRET_TERM, filters=SearchFilters(), limit=500, now=now
            )
        assert {c for c, _ in semantic} == expected
        assert {c for c, _ in keyword.hits} == expected
        assert all(
            facts[d.id].status is DocumentStatus.READY for d in docs if d.chunk_ids[0] in expected
        )
