"""Document access policy - one definition, two compilations.

The same rules are expressed twice:

* ``can_read`` / ``can_manage`` / ``can_list`` evaluate one already-loaded document in Python;
* ``readable_clause`` / ``manageable_clause`` / ``listable_clause`` compile to SQL predicates
  that are embedded **inside** every search, retrieval and listing query, so unauthorised
  rows are never fetched in the first place (no "retrieve then filter").

A property-based test (``tests/integration/test_policy_equivalence.py``) generates random
principals, documents and grants and asserts both compilations always agree.

Rules
-----
content (read) access requires ALL of
    same organisation, status READY,
    classification <= principal clearance            (hard ceiling, even for owners),
    allowed_roles empty or principal role listed      (hard filter),
    and ONE of: PUBLIC/INTERNAL | owner | CONFIDENTIAL in one of the principal's
                departments | an active grant (user / department / role).
RESTRICTED therefore needs ownership or an explicit grant.

management (change metadata, permissions, versions, delete) requires same organisation,
not deleted, and ONE of: organisation admin | department manager of the document's
department (within clearance) | owner (within clearance) | active MANAGE grant (within
clearance). Organisation admins can *manage* everything but do not *read* RESTRICTED
content without a grant - granting themselves access is itself an audited event.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import ColumnElement, and_, any_, exists, false, func, literal, or_, select, true
from sqlalchemy.orm import InstrumentedAttribute

from docassist.authz.principal import Principal
from docassist.core.enums import (
    Classification,
    DocumentStatus,
    GranteeType,
    GrantPermission,
    Role,
)
from docassist.db.models import Document, DocumentGrant


@dataclass(frozen=True, slots=True)
class GrantFacts:
    grantee_type: GranteeType
    permission: GrantPermission
    user_id: uuid.UUID | None = None
    department_id: uuid.UUID | None = None
    role: Role | None = None
    expires_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class DocumentFacts:
    id: uuid.UUID
    org_id: uuid.UUID
    department_id: uuid.UUID | None
    owner_id: uuid.UUID
    classification: Classification
    status: DocumentStatus
    allowed_roles: tuple[str, ...] = ()
    grants: tuple[GrantFacts, ...] = field(default_factory=tuple)

    @classmethod
    def from_row(cls, doc: Document, grants: Iterable[DocumentGrant] = ()) -> DocumentFacts:
        return cls(
            id=doc.id,
            org_id=doc.organization_id,
            department_id=doc.department_id,
            owner_id=doc.owner_id,
            classification=Classification(doc.classification),
            status=DocumentStatus(doc.status),
            allowed_roles=tuple(doc.allowed_roles or ()),
            grants=tuple(
                GrantFacts(
                    grantee_type=GranteeType(g.grantee_type),
                    permission=GrantPermission(g.permission),
                    user_id=g.grantee_user_id,
                    department_id=g.grantee_department_id,
                    role=Role(g.grantee_role) if g.grantee_role else None,
                    expires_at=g.expires_at,
                )
                for g in grants
                if g.document_id == doc.id
            ),
        )


# --------------------------------------------------------------------------- #
# Python evaluation
# --------------------------------------------------------------------------- #
def _grant_matches(principal: Principal, grant: GrantFacts, now: datetime) -> bool:
    if grant.expires_at is not None and grant.expires_at <= now:
        return False
    if grant.grantee_type is GranteeType.USER:
        return grant.user_id == principal.user_id
    if grant.grantee_type is GranteeType.DEPARTMENT:
        return grant.department_id in principal.department_ids
    return grant.role is principal.role


def _has_grant(
    principal: Principal, doc: DocumentFacts, now: datetime, permissions: set[GrantPermission]
) -> bool:
    return any(
        g.permission in permissions and _grant_matches(principal, g, now) for g in doc.grants
    )


def _within_ceilings(principal: Principal, doc: DocumentFacts) -> bool:
    if doc.classification.rank > principal.clearance.rank:
        return False
    return not doc.allowed_roles or principal.role.value in doc.allowed_roles


def _content_rule(principal: Principal, doc: DocumentFacts, now: datetime) -> bool:
    if principal.org_id is None or doc.org_id != principal.org_id:
        return False
    if not _within_ceilings(principal, doc):
        return False
    if doc.classification in (Classification.PUBLIC, Classification.INTERNAL):
        return True
    if doc.owner_id == principal.user_id:
        return True
    if (
        doc.classification is Classification.CONFIDENTIAL
        and doc.department_id is not None
        and doc.department_id in principal.department_ids
    ):
        return True
    return _has_grant(principal, doc, now, {GrantPermission.READ, GrantPermission.MANAGE})


def can_read(principal: Principal, doc: DocumentFacts, now: datetime) -> bool:
    return doc.status is DocumentStatus.READY and _content_rule(principal, doc, now)


def can_manage(principal: Principal, doc: DocumentFacts, now: datetime) -> bool:
    if principal.org_id is None or doc.org_id != principal.org_id:
        return False
    if doc.status is DocumentStatus.DELETED:
        return False
    if principal.role is Role.ORGANIZATION_ADMIN:
        return True
    within = doc.classification.rank <= principal.clearance.rank
    if not within:
        return False
    if (
        principal.role is Role.DEPARTMENT_MANAGER
        and doc.department_id is not None
        and doc.department_id in principal.managed_department_ids
    ):
        return True
    if doc.owner_id == principal.user_id:
        return True
    return _has_grant(principal, doc, now, {GrantPermission.MANAGE})


def can_list(principal: Principal, doc: DocumentFacts, now: datetime) -> bool:
    if doc.status is DocumentStatus.DELETED:
        return False
    return _content_rule(principal, doc, now) or can_manage(principal, doc, now)


# --------------------------------------------------------------------------- #
# SQL compilation
# --------------------------------------------------------------------------- #
def _ids(values: Iterable[uuid.UUID]) -> list[uuid.UUID]:
    return sorted(values, key=str)


def _grant_exists(
    principal: Principal,
    doc_id: InstrumentedAttribute[uuid.UUID] | ColumnElement[uuid.UUID],
    doc_org: InstrumentedAttribute[uuid.UUID] | ColumnElement[uuid.UUID],
    now: datetime,
    permissions: list[str],
) -> ColumnElement[bool]:
    g = DocumentGrant
    who: list[ColumnElement[bool]] = [
        and_(g.grantee_type == GranteeType.USER.value, g.grantee_user_id == principal.user_id),
        and_(g.grantee_type == GranteeType.ROLE.value, g.grantee_role == principal.role.value),
    ]
    if principal.department_ids:
        who.append(
            and_(
                g.grantee_type == GranteeType.DEPARTMENT.value,
                g.grantee_department_id.in_(_ids(principal.department_ids)),
            )
        )
    return exists(
        select(literal(1))
        .where(g.document_id == doc_id)
        .where(g.organization_id == doc_org)
        .where(g.permission.in_(permissions))
        .where(or_(g.expires_at.is_(None), g.expires_at > now))
        .where(or_(*who))
    )


def _ceilings_sql(principal: Principal, doc: type[Document]) -> ColumnElement[bool]:
    allowed = [c.value for c in Classification.at_most(principal.clearance)]
    return and_(
        doc.classification.in_(allowed),
        or_(
            func.cardinality(doc.allowed_roles) == 0,
            literal(principal.role.value) == any_(doc.allowed_roles),
        ),
    )


def _content_rule_sql(
    principal: Principal, now: datetime, doc: type[Document]
) -> ColumnElement[bool]:
    if principal.org_id is None:
        return false()
    branches: list[ColumnElement[bool]] = [
        doc.classification.in_([Classification.PUBLIC.value, Classification.INTERNAL.value]),
        doc.owner_id == principal.user_id,
        _grant_exists(
            principal,
            doc.id,
            doc.organization_id,
            now,
            [GrantPermission.READ.value, GrantPermission.MANAGE.value],
        ),
    ]
    if principal.department_ids:
        branches.append(
            and_(
                doc.classification == Classification.CONFIDENTIAL.value,
                doc.department_id.in_(_ids(principal.department_ids)),
            )
        )
    return and_(
        doc.organization_id == principal.org_id,
        _ceilings_sql(principal, doc),
        or_(*branches),
    )


def readable_clause(
    principal: Principal, now: datetime, doc: type[Document] = Document
) -> ColumnElement[bool]:
    return and_(doc.status == DocumentStatus.READY.value, _content_rule_sql(principal, now, doc))


def manageable_clause(
    principal: Principal, now: datetime, doc: type[Document] = Document
) -> ColumnElement[bool]:
    if principal.org_id is None:
        return false()
    base = and_(doc.organization_id == principal.org_id, doc.status != DocumentStatus.DELETED.value)
    if principal.role is Role.ORGANIZATION_ADMIN:
        return and_(base, true())
    allowed = [c.value for c in Classification.at_most(principal.clearance)]
    branches: list[ColumnElement[bool]] = [
        doc.owner_id == principal.user_id,
        _grant_exists(principal, doc.id, doc.organization_id, now, [GrantPermission.MANAGE.value]),
    ]
    if principal.role is Role.DEPARTMENT_MANAGER and principal.managed_department_ids:
        branches.append(doc.department_id.in_(_ids(principal.managed_department_ids)))
    return and_(base, doc.classification.in_(allowed), or_(*branches))


def listable_clause(
    principal: Principal, now: datetime, doc: type[Document] = Document
) -> ColumnElement[bool]:
    return and_(
        doc.status != DocumentStatus.DELETED.value,
        or_(_content_rule_sql(principal, now, doc), manageable_clause(principal, now, doc)),
    )
