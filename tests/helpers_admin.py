"""Shared helpers for the administration / audit / jobs / CLI tests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from docassist.audit.service import AuditSealer
from docassist.db.models import AuditEvent, User
from docassist.db.session import DbContext
from tests.conftest import Factory, TestUser, login


@dataclass
class Tenant:
    org_id: uuid.UUID
    admin: TestUser
    finance: uuid.UUID
    hr: uuid.UUID


async def make_tenant(factory: Factory, *, admin_clearance: str | None = None) -> Tenant:
    """A fresh organisation with two departments and one active organisation admin."""
    org = await factory.org()
    finance = await factory.department(org, slug="finance", name="Finance")
    hr = await factory.department(org, slug="hr", name="Human Resources")
    admin = await factory.user(org, "organization_admin", clearance=admin_clearance)
    return Tenant(org, admin, finance, hr)


async def headers(client: Any, user: TestUser) -> dict[str, str]:
    return await login(client, user)


async def login_pair(client: Any, user: TestUser) -> tuple[dict[str, str], str]:
    """Log in with body transport: (Authorization header, refresh token)."""
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": user.password, "token_transport": "body"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["refresh_token"]


def last_token_for(container: Any, email: str) -> str:
    """The reset/invitation token most recently emailed to ``email`` (MemoryEmailSender)."""
    for to, _subject, body in reversed(container.email.sent):
        if to == email:
            return str(body.split("token=")[1].split()[0])
    raise AssertionError(f"no email sent to {email}")


def emails_to(container: Any, email: str) -> list[tuple[str, str, str]]:
    return [m for m in container.email.sent if m[0] == email]


async def seal_all(container: Any) -> None:
    key = container.settings.security.audit_hmac_key.get_secret_value().encode()
    sealer = AuditSealer(container.worker_db, key)
    while await sealer.seal_pending():
        pass


async def audit_events(
    container: Any, org_id: uuid.UUID | None, *, action: str | None = None
) -> list[AuditEvent]:
    ctx = DbContext(org_id=org_id) if org_id else DbContext(org_id=None, platform=True)
    stmt = select(AuditEvent).order_by(AuditEvent.id)
    stmt = stmt.where(
        AuditEvent.organization_id == org_id if org_id else AuditEvent.organization_id.is_(None)
    )
    if action:
        stmt = stmt.where(AuditEvent.action == action)
    async with container.db.session(ctx) as session:
        return list((await session.execute(stmt)).scalars().all())


async def load_user(container: Any, org_id: uuid.UUID | None, user_id: uuid.UUID) -> User:
    ctx = DbContext(org_id=org_id) if org_id else DbContext(org_id=None, platform=True)
    async with container.db.session(ctx) as session:
        user = await session.get(User, user_id)
    assert user is not None
    return user


def invited_user(body: dict[str, Any], org_id: uuid.UUID | None, password: str) -> TestUser:
    """A ``TestUser`` for an account created through the API (``UserOut`` JSON)."""
    departments = [uuid.UUID(d["department_id"]) for d in body.get("departments", [])]
    return TestUser(
        uuid.UUID(body["id"]), org_id, body["email"], password, body["role"], departments
    )


async def set_password_via_link(client: Any, container: Any, email: str) -> str:
    """Complete an invitation/reset link; returns the new password."""
    password = f"Invited-{uuid.uuid4().hex[:12]}-Pass!"
    token = last_token_for(container, email)
    response = await client.post(
        "/api/v1/auth/password/reset", json={"token": token, "new_password": password}
    )
    assert response.status_code == 204, response.text
    return password
