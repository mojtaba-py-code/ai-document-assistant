"""The Qdrant payload filter is a faithful compilation of the document read policy.

Randomised documents (every classification/status/department/owner/allowed_roles combination,
user/department/role grants with READ or MANAGE, some expired) are written as payloads to an
in-memory Qdrant; for random principals the set of points matched by :func:`policy_filter`
must equal the documents :func:`docassist.authz.policy.can_read` allows. (Grant expiry is
applied when payloads are built - still-valid grants are indexed and re-checked in PostgreSQL
at query time.)
"""

from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from qdrant_client import AsyncQdrantClient, models

from docassist.authz.policy import DocumentFacts, can_read
from docassist.authz.principal import Principal
from docassist.core.enums import Classification, DocumentStatus, GranteeType, GrantPermission, Role
from docassist.db.models import Document, DocumentGrant
from docassist.search.qdrant_store import document_payload, policy_filter, principal_grant_keys
from docassist.search.types import SearchFilters

NOW = datetime(2026, 9, 30, 12, tzinfo=UTC)
ROLES = [Role.ORGANIZATION_ADMIN, Role.DEPARTMENT_MANAGER, Role.EMPLOYEE, Role.AUDITOR]


def _principal(org: uuid.UUID, rng: random.Random, users: list, depts: list) -> Principal:
    members = frozenset(rng.sample(depts, rng.randint(0, 2)))
    return Principal(
        user_id=rng.choice(users),
        org_id=org,
        role=rng.choice(ROLES),
        clearance=rng.choice(list(Classification)),
        session_id=uuid.uuid4(),
        department_ids=members,
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 99])
async def test_policy_filter_matches_python_policy(seed: int) -> None:
    rng = random.Random(seed)
    org, other_org = uuid.uuid4(), uuid.uuid4()
    depts = [uuid.uuid4() for _ in range(3)]
    users = [uuid.uuid4() for _ in range(6)]
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        "t", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
    )
    docs: list[Document] = []
    grants: list[DocumentGrant] = []
    points = []
    for i in range(120):
        doc = Document(
            id=uuid.uuid4(),
            organization_id=org if i % 10 else other_org,
            department_id=rng.choice([None, *depts]),
            owner_id=rng.choice(users),
            title=f"d{i}",
            classification=rng.choice(list(Classification)).value,
            doc_type=rng.choice(["contract", "invoice", "other"]),
            status=rng.choice(list(DocumentStatus)).value,
            allowed_roles=rng.choice([[], [], ["employee"], ["department_manager", "auditor"]]),
            tags=rng.choice([[], ["a"], ["a", "b"]]),
            created_at=NOW - timedelta(days=i),
        )
        doc_grants = []
        for _ in range(rng.choice([0, 0, 1, 2])):
            kind = rng.choice(list(GranteeType))
            grant = DocumentGrant(
                id=uuid.uuid4(),
                organization_id=doc.organization_id,
                document_id=doc.id,
                grantee_type=kind.value,
                permission=rng.choice(list(GrantPermission)).value,
                expires_at=rng.choice([None, NOW - timedelta(hours=1), NOW + timedelta(days=1)]),
            )
            if kind is GranteeType.USER:
                grant.grantee_user_id = rng.choice(users)
            elif kind is GranteeType.DEPARTMENT:
                grant.grantee_department_id = rng.choice(depts)
            else:
                grant.grantee_role = rng.choice(ROLES[:3]).value
            doc_grants.append(grant)
        docs.append(doc)
        grants.extend(doc_grants)
        payload = {
            **document_payload(doc, doc_grants, NOW),
            "version_id": str(uuid.uuid4()),
            "version_status": "indexed",
            "is_current": True,
        }
        points.append(models.PointStruct(id=str(doc.id), vector=[1.0, 0.5], payload=payload))
    await client.upsert("t", points=points)

    facts = {d.id: DocumentFacts.from_row(d, grants) for d in docs}
    for _ in range(25):
        principal = _principal(org, rng, users, depts)
        query_filter = policy_filter(principal, SearchFilters())
        assert query_filter is not None
        hits = await client.query_points(
            "t", query=[1.0, 0.5], query_filter=query_filter, limit=500
        )
        found = {uuid.UUID(str(p.id)) for p in hits.points}
        expected = {d.id for d in docs if can_read(principal, facts[d.id], NOW)}
        assert found == expected, (principal.role, principal.clearance, found ^ expected)

    # facet filters narrow the same way the SQL filters do
    principal = Principal(
        user_id=users[0],
        org_id=org,
        role=Role.ORGANIZATION_ADMIN,
        clearance=Classification.INTERNAL,
        session_id=uuid.uuid4(),
    )
    narrowed = SearchFilters(
        doc_types=("contract",),
        tags=("b",),
        created_after=NOW - timedelta(days=60),
        created_before=NOW - timedelta(days=5),
    )
    query_filter = policy_filter(principal, narrowed)
    assert query_filter is not None
    hits = await client.query_points("t", query=[1.0, 0.5], query_filter=query_filter, limit=500)
    found = {uuid.UUID(str(p.id)) for p in hits.points}
    expected = {
        d.id
        for d in docs
        if can_read(principal, facts[d.id], NOW)
        and d.doc_type == "contract"
        and "b" in d.tags
        and NOW - timedelta(days=60) <= d.created_at < NOW - timedelta(days=5)
    }
    assert found == expected
    await client.close()


def test_policy_filter_short_circuits() -> None:
    platform = Principal(
        user_id=uuid.uuid4(),
        org_id=None,
        role=Role.PLATFORM_ADMIN,
        clearance=Classification.PUBLIC,
        session_id=uuid.uuid4(),
    )
    assert policy_filter(platform, SearchFilters()) is None
    low = Principal(
        user_id=uuid.uuid4(),
        org_id=uuid.uuid4(),
        role=Role.EMPLOYEE,
        clearance=Classification.INTERNAL,
        session_id=uuid.uuid4(),
    )
    # asking only for classifications above the clearance can never match anything
    assert policy_filter(low, SearchFilters(classifications=("RESTRICTED",))) is None


def test_old_versions_toggle_the_current_version_condition() -> None:
    principal = Principal(
        user_id=uuid.uuid4(),
        org_id=uuid.uuid4(),
        role=Role.EMPLOYEE,
        clearance=Classification.CONFIDENTIAL,
        session_id=uuid.uuid4(),
        department_ids=frozenset({uuid.uuid4()}),
    )

    def keys(flt: models.Filter | None) -> list[str]:
        assert flt is not None and isinstance(flt.must, list)
        return [c.key for c in flt.must if isinstance(c, models.FieldCondition)]

    assert "is_current" in keys(policy_filter(principal, SearchFilters()))
    assert "is_current" not in keys(
        policy_filter(principal, SearchFilters(include_old_versions=True))
    )
    grant_keys = principal_grant_keys(principal)
    assert grant_keys[0] == f"user:{principal.user_id}"
    assert "role:employee" in grant_keys
    assert any(k.startswith("dept:") for k in grant_keys)


def test_document_payload_drops_expired_and_unknown_grants() -> None:
    doc = Document(
        id=uuid.uuid4(),
        organization_id=uuid.uuid4(),
        department_id=None,
        owner_id=uuid.uuid4(),
        title="t",
        classification="RESTRICTED",
        doc_type="other",
        status="ready",
        allowed_roles=["employee"],
        tags=["x"],
        created_at=NOW,
    )
    user = uuid.uuid4()
    grants = [
        DocumentGrant(
            document_id=doc.id,
            grantee_type="user",
            grantee_user_id=user,
            permission="read",
            expires_at=NOW + timedelta(days=1),
        ),
        DocumentGrant(
            document_id=doc.id,
            grantee_type="role",
            grantee_role="auditor",
            permission="manage",
            expires_at=None,
        ),
        DocumentGrant(
            document_id=doc.id,
            grantee_type="user",
            grantee_user_id=uuid.uuid4(),
            permission="read",
            expires_at=NOW - timedelta(seconds=1),
        ),
        DocumentGrant(
            document_id=uuid.uuid4(),
            grantee_type="role",
            grantee_role="employee",
            permission="read",
        ),
    ]
    payload = document_payload(doc, grants, NOW)
    assert payload["grant_principals"] == sorted([f"user:{user}", "role:auditor"])
    assert payload["department_id"] is None
    assert payload["allowed_roles"] == ["employee"]
    assert payload["created_at"] == NOW.timestamp()
