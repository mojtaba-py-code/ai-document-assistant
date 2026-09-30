"""Administration endpoints.

* ``/api/v1/platform/organizations`` - platform operators manage tenants;
* ``/api/v1/organization``           - an organisation's own profile and AI/retention settings;
* ``/api/v1/departments``            - department catalogue;
* ``/api/v1/users``                  - user lifecycle (no passwords ever travel through here);
* ``/api/v1/usage``                  - monthly LLM usage and budget;
* ``/api/v1/admin/health``           - dependency status for administrators.

Every route declares its RBAC permission with :func:`require`; the service re-checks it and
applies the state-dependent rules (escalation, last administrator, tenant isolation).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, Response, status

from docassist.api.container import Container
from docassist.api.deps import get_container, get_principal, require
from docassist.audit.service import Actor
from docassist.authz.permissions import Permission
from docassist.authz.principal import Principal
from docassist.core.enums import AuditOutcome, OrganizationStatus, Role, UserStatus
from docassist.core.errors import PermissionDenied
from docassist.identity.admin import AdminService
from docassist.identity.schemas import (
    AdminHealth,
    DepartmentCreate,
    DepartmentList,
    DepartmentOut,
    DepartmentUpdate,
    OrganizationCreate,
    OrganizationCreated,
    OrganizationOut,
    OrganizationPage,
    OrganizationSettingsOut,
    OrganizationUpdate,
    OrgSettingsPatch,
    SessionsRevoked,
    UsageReport,
    UserCreate,
    UserOut,
    UserPage,
    UserUpdate,
)

router = APIRouter()

platform = APIRouter(prefix="/api/v1/platform/organizations", tags=["platform"])
organization = APIRouter(prefix="/api/v1/organization", tags=["organization"])
departments = APIRouter(prefix="/api/v1/departments", tags=["departments"])
users = APIRouter(prefix="/api/v1/users", tags=["users"])
usage = APIRouter(prefix="/api/v1/usage", tags=["usage"])
admin = APIRouter(prefix="/api/v1/admin", tags=["admin"])

_department_manage = require(Permission.DEPARTMENT_MANAGE)
_department_read = require(Permission.DEPARTMENT_READ)
_org_create = require(Permission.ORG_CREATE)
_org_read = require(Permission.ORG_READ)
_org_read_any = require(Permission.ORG_READ_ANY)
_org_update = require(Permission.ORG_UPDATE)
_org_update_any = require(Permission.ORG_UPDATE_ANY)
_usage_read = require(Permission.USAGE_READ)
_user_manage = require(Permission.USER_MANAGE)
_user_read = require(Permission.USER_READ)

_CURSOR = Query(default=None, max_length=512)
_LIMIT = Query(default=50, ge=1, le=200)
_SEARCH = Query(default=None, min_length=1, max_length=100)


def _service(container: Container) -> AdminService:
    return container.admin


# --------------------------------------------------------------------------- #
# Platform operators
# --------------------------------------------------------------------------- #
@platform.post("", response_model=OrganizationCreated, status_code=status.HTTP_201_CREATED)
async def create_organization(
    payload: OrganizationCreate,
    principal: Principal = Depends(_org_create),
    container: Container = Depends(get_container),
) -> OrganizationCreated:
    return await _service(container).create_organization(principal, payload)


@platform.get("", response_model=OrganizationPage)
async def list_organizations(
    *,
    status_filter: OrganizationStatus | None = Query(default=None, alias="status"),
    q: str | None = _SEARCH,
    cursor: str | None = _CURSOR,
    limit: int = _LIMIT,
    principal: Principal = Depends(_org_read_any),
    container: Container = Depends(get_container),
) -> OrganizationPage:
    return await _service(container).list_organizations(
        principal, status=status_filter, q=q, cursor=cursor, limit=limit
    )


@platform.get("/{organization_id}", response_model=OrganizationOut)
async def get_organization(
    organization_id: uuid.UUID,
    principal: Principal = Depends(_org_read_any),
    container: Container = Depends(get_container),
) -> OrganizationOut:
    return await _service(container).get_organization(principal, organization_id)


@platform.patch("/{organization_id}", response_model=OrganizationOut)
async def update_organization(
    organization_id: uuid.UUID,
    payload: OrganizationUpdate,
    principal: Principal = Depends(_org_update_any),
    container: Container = Depends(get_container),
) -> OrganizationOut:
    return await _service(container).update_organization(principal, organization_id, payload)


# --------------------------------------------------------------------------- #
# Own organisation
# --------------------------------------------------------------------------- #
@organization.get("", response_model=OrganizationSettingsOut)
async def get_organization_settings(
    principal: Principal = Depends(_org_read),
    container: Container = Depends(get_container),
) -> OrganizationSettingsOut:
    return await _service(container).get_organization_settings(principal)


@organization.patch("/settings", response_model=OrganizationSettingsOut)
async def update_organization_settings(
    payload: OrgSettingsPatch,
    principal: Principal = Depends(_org_update),
    container: Container = Depends(get_container),
) -> OrganizationSettingsOut:
    return await _service(container).update_organization_settings(principal, payload)


# --------------------------------------------------------------------------- #
# Departments
# --------------------------------------------------------------------------- #
@departments.get("", response_model=DepartmentList)
async def list_departments(
    principal: Principal = Depends(_department_read),
    container: Container = Depends(get_container),
) -> DepartmentList:
    return await _service(container).list_departments(principal)


@departments.post("", response_model=DepartmentOut, status_code=status.HTTP_201_CREATED)
async def create_department(
    payload: DepartmentCreate,
    principal: Principal = Depends(_department_manage),
    container: Container = Depends(get_container),
) -> DepartmentOut:
    return await _service(container).create_department(principal, payload)


@departments.patch("/{department_id}", response_model=DepartmentOut)
async def update_department(
    department_id: uuid.UUID,
    payload: DepartmentUpdate,
    principal: Principal = Depends(_department_manage),
    container: Container = Depends(get_container),
) -> DepartmentOut:
    return await _service(container).update_department(principal, department_id, payload)


@departments.delete("/{department_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_department(
    department_id: uuid.UUID,
    principal: Principal = Depends(_department_manage),
    container: Container = Depends(get_container),
) -> Response:
    await _service(container).delete_department(principal, department_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
@users.get("", response_model=UserPage)
async def list_users(
    *,
    role: Role | None = Query(default=None),
    status_filter: UserStatus | None = Query(default=None, alias="status"),
    department_id: uuid.UUID | None = Query(default=None),
    q: str | None = _SEARCH,
    cursor: str | None = _CURSOR,
    limit: int = _LIMIT,
    principal: Principal = Depends(_user_read),
    container: Container = Depends(get_container),
) -> UserPage:
    return await _service(container).list_users(
        principal,
        role=role,
        status=status_filter,
        department_id=department_id,
        q=q,
        cursor=cursor,
        limit=limit,
    )


@users.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    principal: Principal = Depends(_user_manage),
    container: Container = Depends(get_container),
) -> UserOut:
    return await _service(container).create_user(principal, payload)


@users.get("/{user_id}", response_model=UserOut)
async def get_user(
    user_id: uuid.UUID,
    principal: Principal = Depends(_user_read),
    container: Container = Depends(get_container),
) -> UserOut:
    return await _service(container).get_user(principal, user_id)


@users.patch("/{user_id}", response_model=UserOut)
async def update_user(
    user_id: uuid.UUID,
    payload: UserUpdate,
    principal: Principal = Depends(_user_manage),
    container: Container = Depends(get_container),
) -> UserOut:
    return await _service(container).update_user(principal, user_id, payload)


@users.post("/{user_id}/revoke-sessions", response_model=SessionsRevoked)
async def revoke_user_sessions(
    user_id: uuid.UUID,
    principal: Principal = Depends(_user_manage),
    container: Container = Depends(get_container),
) -> SessionsRevoked:
    return await _service(container).revoke_user_sessions(principal, user_id)


@users.post("/{user_id}/reset-mfa", status_code=status.HTTP_204_NO_CONTENT)
async def reset_user_mfa(
    user_id: uuid.UUID,
    principal: Principal = Depends(_user_manage),
    container: Container = Depends(get_container),
) -> Response:
    await _service(container).reset_user_mfa(principal, user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@users.post("/{user_id}/send-reset", status_code=status.HTTP_202_ACCEPTED)
async def send_password_reset(
    user_id: uuid.UUID,
    principal: Principal = Depends(_user_manage),
    container: Container = Depends(get_container),
) -> dict[str, bool]:
    sent = await _service(container).send_password_reset(principal, user_id)
    return {"email_sent": sent}


# --------------------------------------------------------------------------- #
# Usage & health
# --------------------------------------------------------------------------- #
@usage.get("", response_model=UsageReport)
async def usage_report(
    month: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}$"),
    principal: Principal = Depends(_usage_read),
    container: Container = Depends(get_container),
) -> UsageReport:
    return await _service(container).usage_report(principal, month=month)


async def _health_viewer(
    principal: Principal = Depends(get_principal), container: Container = Depends(get_container)
) -> Principal:
    """Platform admins and organisation admins only (denials are audited like ``require``)."""
    if not AdminService.may_view_health(principal):
        await container.audit.record_detached(
            Actor.of(principal),
            "authz.denied",
            outcome=AuditOutcome.DENIED,
            details={"permission": Permission.PLATFORM_HEALTH.value},
        )
        raise PermissionDenied(internal_detail=f"{principal.role} may not view admin health")
    return principal


@admin.get("/health", response_model=AdminHealth)
async def admin_health(
    response: Response,
    principal: Principal = Depends(_health_viewer),
    container: Container = Depends(get_container),
) -> AdminHealth:
    response.headers["Cache-Control"] = "no-store"
    return await _service(container).health(principal)


for _sub in (platform, organization, departments, users, usage, admin):
    router.include_router(_sub)
