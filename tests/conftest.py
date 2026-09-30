"""Shared test fixtures.

Database tests need a PostgreSQL 16 server with pgvector. Point
``DOCASSIST_TEST_DATABASE_URL`` at a *superuser* DSN (e.g. the CI service container or a
local ``pgserver``); the suite then

1. creates a throw-away database,
2. creates the least-privilege roles ``docassist_app`` / ``docassist_worker``,
3. runs the real Alembic migrations as the owner,
4. connects the application **as ``docassist_app``** - so every test also exercises RLS.

Without the variable, tests marked ``db`` are skipped and the rest still run.
"""

from __future__ import annotations

import asyncio
import base64
import os
import secrets
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parent.parent
TEST_DB_URL = os.environ.get("DOCASSIST_TEST_DATABASE_URL")
APP_PASSWORD = "app-" + secrets.token_hex(12)
WORKER_PASSWORD = "worker-" + secrets.token_hex(12)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if TEST_DB_URL:
        return
    skip = pytest.mark.skip(reason="DOCASSIST_TEST_DATABASE_URL not set")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)


def _random_secret() -> str:
    return secrets.token_urlsafe(48)


def make_settings(**overrides: Any) -> Any:
    """Settings for tests (no .env, cheap Argon2, generated secrets)."""
    from docassist.core.config import Settings

    key = base64.b64encode(secrets.token_bytes(32)).decode()
    base: dict[str, Any] = {
        "environment": "test",
        "database": {
            "url": "postgresql+asyncpg://unused@localhost/unused",
            "verify_role_privileges": False,
        },
        "security": {
            "jwt_signing_key": _random_secret(),
            "token_pepper": _random_secret(),
            "audit_hmac_key": _random_secret(),
            "encryption_keys": f"k1:{key}",
            "active_encryption_key_id": "k1",
            "argon2_time_cost": 1,
            "argon2_memory_kib": 8192,
            "argon2_parallelism": 1,
            "cookie_secure": False,
        },
        "observability": {"log_format": "console", "log_level": "WARNING"},
    }
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(base.get(section), dict):
            base[section] = {**base[section], **values}
        else:
            base[section] = values
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Database provisioning
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TestDatabase:
    admin_url: str  # superuser, points at the test database
    owner_url: str  # asyncpg DSN used for migrations
    app_url: str
    worker_url: str
    name: str


def _with_db(url: str, dbname: str, user: str | None = None, password: str | None = None) -> str:
    parts = urlsplit(url)
    netloc = parts.netloc.rsplit("@", 1)[-1]
    if user:
        netloc = f"{user}:{password}@{netloc}" if password else f"{user}@{netloc}"
    elif "@" in parts.netloc:
        netloc = parts.netloc
    return urlunsplit((parts.scheme, netloc, "/" + dbname, parts.query, parts.fragment))


async def _admin_exec(url: str, *statements: str) -> None:
    import asyncpg

    conn = await asyncpg.connect(url.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        for statement in statements:
            await conn.execute(statement)
    finally:
        await conn.close()


def _ensure_role_sql(role: str, password: str) -> str:
    return f"""
    DO $$
    BEGIN
      IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN
        CREATE ROLE {role} LOGIN PASSWORD '{password}' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
      ELSE
        ALTER ROLE {role} LOGIN PASSWORD '{password}' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
      END IF;
    EXCEPTION WHEN unique_violation OR duplicate_object THEN
      ALTER ROLE {role} LOGIN PASSWORD '{password}';
    END
    $$;
    """


@pytest.fixture(scope="session")
def test_database() -> Iterator[TestDatabase]:
    if not TEST_DB_URL:
        pytest.skip("DOCASSIST_TEST_DATABASE_URL not set")
    name = f"docassist_test_{uuid.uuid4().hex[:10]}"
    admin_root = TEST_DB_URL
    asyncio.run(_admin_exec(admin_root, f'CREATE DATABASE "{name}"'))
    asyncio.run(
        _admin_exec(
            admin_root,
            _ensure_role_sql("docassist_app", APP_PASSWORD),
            _ensure_role_sql("docassist_worker", WORKER_PASSWORD),
        )
    )
    admin_url = _with_db(admin_root, name)
    owner_url = admin_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    env = {**os.environ, "DOCASSIST_DATABASE__MIGRATION_URL": owner_url}
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("migration failed:\n" + result.stderr[-4000:])
    db = TestDatabase(
        admin_url=admin_url,
        owner_url=owner_url,
        app_url=_with_db(owner_url, name, "docassist_app", APP_PASSWORD),
        worker_url=_with_db(owner_url, name, "docassist_worker", WORKER_PASSWORD),
        name=name,
    )
    yield db
    asyncio.run(
        _admin_exec(
            admin_root,
            f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{name}'",
            f'DROP DATABASE IF EXISTS "{name}"',
        )
    )


@pytest.fixture(scope="session")
def settings(test_database: TestDatabase, tmp_path_factory: pytest.TempPathFactory) -> Any:
    storage = tmp_path_factory.mktemp("storage")
    return make_settings(
        database={
            "url": test_database.app_url,
            "worker_url": test_database.worker_url,
            "verify_role_privileges": True,
            "ssl": "disable",
        },
        storage={"root": str(storage)},
        rate_limit={
            "login_per_ip": {"requests": 10_000, "per_seconds": 60},
            "login_per_account": {"requests": 10_000, "per_seconds": 60},
            "api_per_user": {"requests": 100_000, "per_seconds": 60},
            "upload_per_user": {"requests": 10_000, "per_seconds": 60},
            "search_per_user": {"requests": 10_000, "per_seconds": 60},
            "llm_per_user": {"requests": 10_000, "per_seconds": 60},
            "llm_per_org": {"requests": 10_000, "per_seconds": 60},
            "export_per_user": {"requests": 10_000, "per_seconds": 60},
            "tool_calls_per_user": {"requests": 10_000, "per_seconds": 60},
            "password_reset_per_ip": {"requests": 10_000, "per_seconds": 60},
            "refresh_per_session": {"requests": 10_000, "per_seconds": 60},
        },
        upload={"max_upload_bytes": 5 * 1024 * 1024},
        parser={"timeout_seconds": 60},
    )


@pytest_asyncio.fixture(scope="session")
async def container(settings: Any) -> AsyncIterator[Any]:
    import fakeredis

    from docassist.api.container import build_container
    from docassist.identity.notifications import MemoryEmailSender

    c = build_container(
        settings,
        role="worker",
        overrides={"redis": fakeredis.FakeAsyncRedis(), "email": MemoryEmailSender()},
    )
    yield c
    await c.close()


@pytest_asyncio.fixture(scope="session")
async def app(settings: Any, container: Any) -> AsyncIterator[Any]:
    from asgi_lifespan import LifespanManager

    from docassist.api.app import create_app

    application = create_app(settings, container=container, configure_logs=False)
    async with LifespanManager(application):
        yield application


@pytest_asyncio.fixture
async def client(app: Any) -> AsyncIterator[Any]:
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        yield http


# --------------------------------------------------------------------------- #
# Tenant factory
# --------------------------------------------------------------------------- #
@dataclass
class TestUser:
    id: uuid.UUID
    org_id: uuid.UUID | None
    email: str
    password: str
    role: str
    department_ids: list[uuid.UUID]


class Factory:
    def __init__(self, container: Any) -> None:
        self.c = container

    async def org(self, slug: str | None = None, name: str = "Test Org") -> uuid.UUID:
        from docassist.db.models import Organization
        from docassist.db.session import DbContext

        slug = slug or f"org-{uuid.uuid4().hex[:8]}"
        async with self.c.db.transaction(DbContext(org_id=None, platform=True)) as session:
            org = Organization(slug=slug, name=name, status="active")
            session.add(org)
            await session.flush()
            return org.id  # type: ignore[no-any-return]

    async def department(
        self, org_id: uuid.UUID, slug: str | None = None, name: str = "Dept"
    ) -> uuid.UUID:
        from docassist.db.models import Department
        from docassist.db.session import DbContext

        slug = slug or f"dept-{uuid.uuid4().hex[:6]}"
        async with self.c.db.transaction(DbContext(org_id=org_id)) as session:
            dept = Department(organization_id=org_id, name=name, slug=slug)
            session.add(dept)
            await session.flush()
            return dept.id  # type: ignore[no-any-return]

    async def user(
        self,
        org_id: uuid.UUID | None,
        role: str = "employee",
        *,
        departments: list[uuid.UUID] | None = None,
        managed: list[uuid.UUID] | None = None,
        clearance: str | None = None,
        email: str | None = None,
        password: str | None = None,
        status: str = "active",
    ) -> TestUser:
        from docassist.authz.permissions import DEFAULT_CLEARANCE
        from docassist.core.enums import Role
        from docassist.db.models import User, UserDepartment
        from docassist.db.session import DbContext

        email = (email or f"{role}-{uuid.uuid4().hex[:8]}@example.test").lower()
        password = password or f"Str0ng-{secrets.token_hex(8)}-pass"
        clearance = clearance or DEFAULT_CLEARANCE[Role(role)].value
        departments = list(departments or [])
        managed = list(managed or [])
        ctx = DbContext(org_id=org_id, platform=org_id is None)
        async with self.c.db.transaction(ctx) as session:
            user = User(
                organization_id=org_id,
                email=email,
                full_name=f"Test {role.title()}",
                password_hash=self.c.passwords.hash(password),
                role=role,
                clearance=clearance,
                status=status,
            )
            session.add(user)
            await session.flush()
            for dept in {*departments, *managed}:
                session.add(
                    UserDepartment(
                        user_id=user.id,
                        department_id=dept,
                        organization_id=org_id,
                        is_manager=dept in managed,
                    )
                )
            user_id = user.id
        return TestUser(
            user_id, org_id, email, password, role, sorted({*departments, *managed}, key=str)
        )

    async def principal(self, user: TestUser) -> Any:
        from sqlalchemy import select

        from docassist.authz.principal import Principal
        from docassist.core.enums import Classification, Role
        from docassist.db.models import User, UserDepartment
        from docassist.db.session import DbContext

        async with self.c.db.session(
            DbContext(org_id=user.org_id, platform=user.org_id is None)
        ) as session:
            row = await session.get(User, user.id)
            memberships = (
                (
                    await session.execute(
                        select(UserDepartment).where(UserDepartment.user_id == user.id)
                    )
                )
                .scalars()
                .all()
            )
        assert row is not None
        return Principal(
            user_id=row.id,
            org_id=row.organization_id,
            role=Role(row.role),
            clearance=Classification(row.clearance),
            session_id=uuid.uuid4(),
            email=row.email,
            department_ids=frozenset(m.department_id for m in memberships),
            managed_department_ids=frozenset(m.department_id for m in memberships if m.is_manager),
        )


@pytest.fixture(scope="session")
def factory(container: Any) -> Factory:
    return Factory(container)


async def login(client: Any, user: TestUser) -> dict[str, str]:
    """Log in through the real API and return an Authorization header."""
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": user.password, "token_transport": "body"},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def auth_headers() -> Any:
    return login
