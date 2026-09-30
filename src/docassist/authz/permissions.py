"""Role-based access control: the permission catalogue and the role -> permission matrix.

RBAC answers "may this *kind* of user attempt this *kind* of action?". Whether the user may
touch a *specific* document is answered afterwards by :mod:`docassist.authz.policy` (ABAC).
Both must pass (complete mediation); anything not granted here is denied (default deny).
"""

from __future__ import annotations

from enum import StrEnum

from docassist.core.enums import Classification, Role


class Permission(StrEnum):
    # platform operator
    ORG_CREATE = "org:create"
    ORG_READ_ANY = "org:read_any"
    ORG_UPDATE_ANY = "org:update_any"
    PLATFORM_HEALTH = "platform:health"
    # own organisation
    ORG_READ = "org:read"
    ORG_UPDATE = "org:update"
    DEPARTMENT_READ = "department:read"
    DEPARTMENT_MANAGE = "department:manage"
    USER_READ = "user:read"
    USER_MANAGE = "user:manage"
    # documents & AI
    DOCUMENT_UPLOAD = "document:upload"
    DOCUMENT_READ = "document:read"
    DOCUMENT_MANAGE = "document:manage"
    SEARCH_USE = "search:use"
    ASSISTANT_USE = "assistant:use"
    ASSISTANT_AGENT = "assistant:agent"
    INTELLIGENCE_USE = "intelligence:use"
    EXPORT_CREATE = "export:create"
    # governance
    AUDIT_READ = "audit:read"
    AUDIT_VERIFY = "audit:verify"
    JOBS_READ = "jobs:read"
    JOBS_MANAGE = "jobs:manage"
    LLM_CONFIGURE = "llm:configure"
    USAGE_READ = "usage:read"


P = Permission

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.PLATFORM_ADMIN: frozenset(
        {
            P.ORG_CREATE,
            P.ORG_READ_ANY,
            P.ORG_UPDATE_ANY,
            P.PLATFORM_HEALTH,
            P.AUDIT_READ,
            P.AUDIT_VERIFY,
        }
    ),
    Role.ORGANIZATION_ADMIN: frozenset(
        {
            P.ORG_READ,
            P.ORG_UPDATE,
            P.DEPARTMENT_READ,
            P.DEPARTMENT_MANAGE,
            P.USER_READ,
            P.USER_MANAGE,
            P.DOCUMENT_UPLOAD,
            P.DOCUMENT_READ,
            P.DOCUMENT_MANAGE,
            P.SEARCH_USE,
            P.ASSISTANT_USE,
            P.ASSISTANT_AGENT,
            P.INTELLIGENCE_USE,
            P.EXPORT_CREATE,
            P.AUDIT_READ,
            P.AUDIT_VERIFY,
            P.JOBS_READ,
            P.JOBS_MANAGE,
            P.LLM_CONFIGURE,
            P.USAGE_READ,
        }
    ),
    Role.DEPARTMENT_MANAGER: frozenset(
        {
            P.ORG_READ,
            P.DEPARTMENT_READ,
            P.USER_READ,
            P.DOCUMENT_UPLOAD,
            P.DOCUMENT_READ,
            P.DOCUMENT_MANAGE,
            P.SEARCH_USE,
            P.ASSISTANT_USE,
            P.ASSISTANT_AGENT,
            P.INTELLIGENCE_USE,
            P.EXPORT_CREATE,
            P.JOBS_READ,
        }
    ),
    Role.EMPLOYEE: frozenset(
        {
            P.ORG_READ,
            P.DEPARTMENT_READ,
            P.DOCUMENT_UPLOAD,
            P.DOCUMENT_READ,
            P.SEARCH_USE,
            P.ASSISTANT_USE,
            P.INTELLIGENCE_USE,
            P.JOBS_READ,
        }
    ),
    # Auditors review activity; they do not read document content (separation of duties).
    Role.AUDITOR: frozenset(
        {P.ORG_READ, P.DEPARTMENT_READ, P.USER_READ, P.AUDIT_READ, P.AUDIT_VERIFY, P.USAGE_READ}
    ),
}

DEFAULT_CLEARANCE: dict[Role, Classification] = {
    Role.PLATFORM_ADMIN: Classification.PUBLIC,
    Role.ORGANIZATION_ADMIN: Classification.RESTRICTED,
    Role.DEPARTMENT_MANAGER: Classification.RESTRICTED,
    Role.EMPLOYEE: Classification.CONFIDENTIAL,
    Role.AUDITOR: Classification.INTERNAL,
}

# Which roles an actor may assign. Nobody can mint platform admins through the tenant API,
# and nobody can grant a role more privileged than their own.
ASSIGNABLE_ROLES: dict[Role, frozenset[Role]] = {
    Role.PLATFORM_ADMIN: frozenset({Role.ORGANIZATION_ADMIN}),
    Role.ORGANIZATION_ADMIN: frozenset(
        {Role.ORGANIZATION_ADMIN, Role.DEPARTMENT_MANAGER, Role.EMPLOYEE, Role.AUDITOR}
    ),
}


def permissions_for(role: Role) -> frozenset[Permission]:
    return ROLE_PERMISSIONS.get(role, frozenset())
